#!/usr/bin/env python3
"""Compute 2021 6-hourly nowcasting metrics from prediction Zarr stores.

This script is intentionally separate from inference. It can read either:
  - one yearly observation/forecast Zarr store per model, or
  - monthly observation/forecast Zarr stores.

The metric set follows the current pystepsval-style evaluation:
RMSE, mean error (ME/bias), MAPE, Brier score, CRPS, rank histograms,
reliability, FSS, histograms, and contingency-table scores POD/FAR/CSI/ETS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

THRESHOLDS = [0.1, 0.5, 1.0, 5.0]
RELIABILITY_BIN_EDGES = np.linspace(-0.000001, 1.000001, 11).astype(np.float32)
FSS_WINDOW_SIZES = [5, 10, 30, 60]
HIST_BINS = np.array([0.0, 0.1, 0.5, 1.0, 5.0, np.inf], dtype=np.float32)


def _event_month_from_start_time(start_time: str, fallback: int) -> int:
    try:
        return int(pd.Timestamp(start_time).month)
    except Exception:
        return int(fallback)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _valid_mask(obs, pred_mean, valid_mask):
    return np.asarray(valid_mask, dtype=bool) & np.isfinite(obs) & np.isfinite(pred_mean)


def _mape(obs_vals, pred_vals, min_obs=0.1):
    keep = np.isfinite(obs_vals) & np.isfinite(pred_vals) & (obs_vals >= min_obs)
    if not np.any(keep):
        return np.nan
    return float(np.mean(np.abs((pred_vals[keep] - obs_vals[keep]) / obs_vals[keep])))


def _crps_ensemble(samples, obs, mask):
    samples = np.asarray(samples, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if samples.shape[0] == 1:
        vals = np.abs(samples[0][mask] - obs[mask])
        return float(np.mean(vals)) if vals.size else np.nan
    M = samples.shape[0]
    term1 = np.mean(np.abs(samples[:, mask] - obs[mask][None, :]), axis=0)
    # Pairwise term without materialising [M, M, H, W].
    pair_sum = np.zeros(term1.shape, dtype=np.float32)
    for m in range(M):
        sm = samples[m, mask]
        for n in range(M):
            pair_sum += np.abs(sm - samples[n, mask])
    term2 = pair_sum / (2.0 * M * M)
    return float(np.mean(term1 - term2)) if term1.size else np.nan


def _contingency_scores(obs_event, fc_event, mask):
    obs_event = np.asarray(obs_event, dtype=bool)
    fc_event = np.asarray(fc_event, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    H = int(np.sum(mask & fc_event & obs_event))
    M = int(np.sum(mask & (~fc_event) & obs_event))
    F = int(np.sum(mask & fc_event & (~obs_event)))
    C = int(np.sum(mask & (~fc_event) & (~obs_event)))
    total = H + M + F + C
    pod = H / (H + M) if (H + M) > 0 else np.nan
    far = F / (H + F) if (H + F) > 0 else np.nan
    csi = H / (H + M + F) if (H + M + F) > 0 else np.nan
    h_rand = ((H + M) * (H + F) / total) if total > 0 else np.nan
    ets = (H - h_rand) / (H + M + F - h_rand) if total > 0 and (H + M + F - h_rand) != 0 else np.nan
    return H, M, F, C, pod, far, csi, ets


def _local_fraction(field, size):
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(field.astype(np.float32), size=size, mode="constant", cval=0.0)
    except Exception:
        # Fallback, slower but dependency-free.
        from numpy.lib.stride_tricks import sliding_window_view
        pad = size // 2
        padded = np.pad(field.astype(np.float32), pad, mode="constant", constant_values=0.0)
        win = sliding_window_view(padded, (size, size))
        return win.mean(axis=(-1, -2))[: field.shape[0], : field.shape[1]]


def _bbox(mask):
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _crop(arr, bbox):
    if bbox is None:
        return arr
    r0, r1, c0, c1 = bbox
    return arr[..., r0:r1, c0:c1]


def _fss(obs_event, fc_prob, mask, scale):
    bbox = _bbox(mask)
    if bbox is None:
        return np.nan
    obs_crop = _crop(obs_event.astype(np.float32), bbox)
    fc_crop = _crop(fc_prob.astype(np.float32), bbox)
    mask_crop = _crop(mask, bbox)
    obs_crop = np.where(mask_crop, obs_crop, 0.0)
    fc_crop = np.where(mask_crop, fc_crop, 0.0)
    O = _local_fraction(obs_crop, scale)
    P = _local_fraction(fc_crop, scale)
    denom = float(np.sum(P ** 2) + np.sum(O ** 2))
    return float(1.0 - np.sum((P - O) ** 2) / denom) if denom > 0 else np.nan


def compute_for_model_month(model_name, forecast_store, obs_store, month):
    fg = zarr.open_group(str(forecast_store), mode="r")
    og = zarr.open_group(str(obs_store), mode="r")
    forecast = fg["forecast"]
    truth = og["truth"]
    valid = og["valid_mask"]

    t_starts = np.asarray(fg["t_start"][:]).astype(int)
    start_times = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in fg["start_time"][:]]
    n_events, n_members, n_leads = forecast.shape[:3]

    deterministic_rows = []
    categorical_rows = []
    probabilistic_rows = []
    rank_rows = []
    reliability_rows = []
    fss_rows = []
    histogram_rows = []

    for e in range(n_events):
        event_month = _event_month_from_start_time(start_times[e], month)
        print(f"{model_name} store_month={month:02d} event_month={event_month:02d} event {e+1}/{n_events} t_start={t_starts[e]}")
        obs_all = truth[e].astype(np.float32)
        valid_all = valid[e].astype(bool)
        samples_all = forecast[e].astype(np.float32)
        pred_mean_all = np.nanmean(samples_all, axis=0)

        for lead in range(n_leads):
            obs = obs_all[lead]
            pred_mean = pred_mean_all[lead]
            samples = samples_all[:, lead]
            mask = _valid_mask(obs, pred_mean, valid_all[lead])
            if not np.any(mask):
                continue

            obs_vals = obs[mask]
            pred_vals = pred_mean[mask]
            err = pred_vals - obs_vals
            deterministic_rows.append(
                {
                    "model": model_name,
                    "month": event_month,
                    "event_index_in_month": e,
                    "t_start": int(t_starts[e]),
                    "start_time": start_times[e],
                    "lead_index": lead,
                    "lead_time_min": int((lead + 1) * 5),
                    "n_members": int(n_members),
                    "n_valid_pixels": int(mask.sum()),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "me": float(np.mean(err)),
                    "mape_obs_ge_0p1": _mape(obs_vals, pred_vals, min_obs=0.1),
                    "max_observation": float(np.nanmax(obs_vals)),
                    "max_prediction": float(np.nanmax(pred_vals)),
                    "mean_ensemble_std": float(np.nanmean(np.nanstd(samples[:, mask], axis=0))) if n_members > 1 else 0.0,
                }
            )

            # Histograms of observation and ensemble mean forecast.
            for field_name, vals in [("observation", obs_vals), ("forecast_mean", pred_vals)]:
                hist, edges = np.histogram(vals[np.isfinite(vals)], bins=HIST_BINS)
                for k, count in enumerate(hist):
                    histogram_rows.append(
                        {
                            "model": model_name,
                            "month": event_month,
                            "t_start": int(t_starts[e]),
                            "lead_index": lead,
                            "lead_time_min": int((lead + 1) * 5),
                            "field": field_name,
                            "bin_left_mm_h": float(edges[k]),
                            "bin_right_mm_h": float(edges[k + 1]),
                            "count": int(count),
                        }
                    )

            # Deterministic-style contingency scores from the ensemble mean.
            for q in THRESHOLDS:
                obs_event = obs >= q
                fc_event = pred_mean >= q
                H, M, F, C, pod, far, csi, ets = _contingency_scores(obs_event, fc_event, mask)
                categorical_rows.append(
                    {
                        "model": model_name,
                        "month": event_month,
                        "t_start": int(t_starts[e]),
                        "start_time": start_times[e],
                        "lead_index": lead,
                        "lead_time_min": int((lead + 1) * 5),
                        "threshold_mm_h": float(q),
                        "H": H,
                        "M": M,
                        "F": F,
                        "C": C,
                        "pod": _safe_float(pod),
                        "far": _safe_float(far),
                        "csi": _safe_float(csi),
                        "ets": _safe_float(ets),
                    }
                )

                if n_members > 1:
                    prob = np.nanmean(samples >= q, axis=0).astype(np.float32)
                    obs_binary = (obs >= q).astype(np.float32)
                    brier = float(np.mean((prob[mask] - obs_binary[mask]) ** 2))
                    probabilistic_rows.append(
                        {
                            "model": model_name,
                            "month": event_month,
                            "t_start": int(t_starts[e]),
                            "start_time": start_times[e],
                            "lead_index": lead,
                            "lead_time_min": int((lead + 1) * 5),
                            "metric": "brier",
                            "threshold_mm_h": float(q),
                            "value": brier,
                        }
                    )

                    # Reliability diagram counts.
                    prob_vals = prob[mask]
                    obs_bin_vals = obs_binary[mask]
                    for b0, b1 in zip(RELIABILITY_BIN_EDGES[:-1], RELIABILITY_BIN_EDGES[1:]):
                        in_bin = (prob_vals >= b0) & (prob_vals < b1)
                        n_bin = int(np.sum(in_bin))
                        reliability_rows.append(
                            {
                                "model": model_name,
                                "month": event_month,
                                "t_start": int(t_starts[e]),
                                "lead_index": lead,
                                "lead_time_min": int((lead + 1) * 5),
                                "threshold_mm_h": float(q),
                                "prob_bin_left": float(b0),
                                "prob_bin_right": float(b1),
                                "n": n_bin,
                                "observed_frequency": float(np.mean(obs_bin_vals[in_bin])) if n_bin > 0 else np.nan,
                            }
                        )

            # CRPS once per lead.
            if n_members > 1:
                probabilistic_rows.append(
                    {
                        "model": model_name,
                        "month": event_month,
                        "t_start": int(t_starts[e]),
                        "start_time": start_times[e],
                        "lead_index": lead,
                        "lead_time_min": int((lead + 1) * 5),
                        "metric": "crps",
                        "threshold_mm_h": np.nan,
                        "value": _crps_ensemble(samples, obs, mask),
                    }
                )

                ranks = np.sum(samples[:, mask] < obs[mask][None, :], axis=0).astype(np.int64)
                rank_counts = np.bincount(ranks, minlength=n_members + 1)
                for rank_bin, count in enumerate(rank_counts):
                    rank_rows.append(
                        {
                            "model": model_name,
                            "month": event_month,
                            "t_start": int(t_starts[e]),
                            "lead_index": lead,
                            "lead_time_min": int((lead + 1) * 5),
                            "rank_bin": int(rank_bin),
                            "count": int(count),
                            "n_members": int(n_members),
                            "rain_condition_mm_h": np.nan,
                        }
                    )

                rain_mask = mask & (obs >= 0.1)
                if np.any(rain_mask):
                    ranks_rain = np.sum(samples[:, rain_mask] < obs[rain_mask][None, :], axis=0).astype(np.int64)
                    rank_counts_rain = np.bincount(ranks_rain, minlength=n_members + 1)
                    for rank_bin, count in enumerate(rank_counts_rain):
                        rank_rows.append(
                            {
                                "model": model_name,
                                "month": event_month,
                                "t_start": int(t_starts[e]),
                                "lead_index": lead,
                                "lead_time_min": int((lead + 1) * 5),
                                "rank_bin": int(rank_bin),
                                "count": int(count),
                                "n_members": int(n_members),
                                "rain_condition_mm_h": 0.1,
                            }
                        )

            # FSS using probabilistic exceedance field for ensembles and binary field for deterministic forecasts.
            for q in THRESHOLDS:
                obs_event = obs >= q
                if n_members > 1:
                    fc_prob = np.nanmean(samples >= q, axis=0).astype(np.float32)
                else:
                    fc_prob = (samples[0] >= q).astype(np.float32)
                for scale in FSS_WINDOW_SIZES:
                    fss = _fss(obs_event, fc_prob, mask, scale)
                    fss_rows.append(
                        {
                            "model": model_name,
                            "month": event_month,
                            "t_start": int(t_starts[e]),
                            "start_time": start_times[e],
                            "lead_index": lead,
                            "lead_time_min": int((lead + 1) * 5),
                            "threshold_mm_h": float(q),
                            "window_size_px": int(scale),
                            "fss": _safe_float(fss),
                        }
                    )

    return {
        "deterministic": pd.DataFrame(deterministic_rows),
        "categorical": pd.DataFrame(categorical_rows),
        "probabilistic": pd.DataFrame(probabilistic_rows),
        "rank_histograms": pd.DataFrame(rank_rows),
        "reliability": pd.DataFrame(reliability_rows),
        "fss": pd.DataFrame(fss_rows),
        "histograms": pd.DataFrame(histogram_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--observation-root", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--store-mode", choices=["yearly", "monthly"], default="yearly")
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)), help="Months to use in monthly mode.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="full")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accum = {k: [] for k in ["deterministic", "categorical", "probabilistic", "rank_histograms", "reliability", "fss", "histograms"]}

    for model in args.models:
        if args.store_mode == "yearly":
            obs_store = Path(args.observation_root) / "observations_2021.zarr"
            pred_store = Path(args.prediction_root) / model / f"{model}_{args.split}_2021.zarr"
            if not obs_store.exists():
                print("Missing observation store, skipping:", obs_store)
                continue
            if not pred_store.exists():
                print("Missing forecast store, skipping:", pred_store)
                continue
            tables = compute_for_model_month(model, pred_store, obs_store, 0)
            for key, df in tables.items():
                if df is not None and not df.empty:
                    accum[key].append(df)
        else:
            for month in args.months:
                obs_store = Path(args.observation_root) / f"observations_2021_month{month:02d}.zarr"
                pred_store = Path(args.prediction_root) / model / f"{model}_{args.split}_2021_month{month:02d}.zarr"
                if not obs_store.exists():
                    print("Missing observation store, skipping:", obs_store)
                    continue
                if not pred_store.exists():
                    print("Missing forecast store, skipping:", pred_store)
                    continue
                tables = compute_for_model_month(model, pred_store, obs_store, month)
                for key, df in tables.items():
                    if df is not None and not df.empty:
                        accum[key].append(df)

    outputs = {}
    for key, parts in accum.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        outputs[key] = df
        out_csv = output_dir / f"{key}.csv"
        df.to_csv(out_csv, index=False)
        print("Wrote", out_csv, "rows=", len(df))

    if not outputs["deterministic"].empty:
        summary = (
            outputs["deterministic"]
            .groupby("model", as_index=False)
            .agg(
                mean_mae=("mae", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_me=("me", "mean"),
                mean_mape_obs_ge_0p1=("mape_obs_ge_0p1", "mean"),
                n_events=("t_start", "nunique"),
            )
            .sort_values("mean_rmse")
        )
        summary.to_csv(output_dir / "summary_main_metrics.csv", index=False)
        print(summary)


if __name__ == "__main__":
    main()
