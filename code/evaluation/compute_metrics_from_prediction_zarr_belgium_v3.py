#!/usr/bin/env python3
"""Compute 2021 6-hourly nowcasting metrics from prediction Zarr stores.

Key safeguards in this version
------------------------------
1. All primary scores can be restricted to one fixed Belgium mask.
2. The scoring mask depends on geography and observation validity only; it is
   never reduced because a model forecast is non-finite.
3. Forecast support is audited and written to coverage_diagnostics.csv.
4. MAE and CRPS are also computed for observed rain >= 0.25 mm h-1.
5. Categorical thresholds include 30 mm h-1.
6. Rank-histogram ties are randomized reproducibly.
7. FSS is mask-aware near the Belgian border and is saved both as a\n   deterministic-style ensemble-mean score and as probabilistic FSS.\n8. Weighted/pooled summary CSVs are written for plotting and reporting.

Expected Zarr layout
--------------------
Forecast store:
    forecast   [event, member, lead, y, x]
    t_start    [event]
    start_time [event]

Observation store:
    truth      [event, lead, y, x]
    valid_mask [event, lead, y, x]
    t_start    [event]              (used to align monthly forecasts with yearly observations)

The Belgium mask must be a boolean or 0/1 array aligned with the unpadded
RADCLIM grid or with the stored forecast grid. Supported formats are .npy,
.npz and .zarr. If a 700 x 700 mask is used with 704 x 704 forecasts, it is
symmetrically padded by two pixels on every side.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import zarr
from pysteps import verification as pysteps_verification

THRESHOLDS_MM_H = [0.1, 0.5, 1.0, 5.0, 30.0]
RAIN_CONDITION_MM_H = 0.25
RANK_X_MIN_MM_H = 0.1
RANK_LEAD_TIMES_MIN = [5, 50, 100]
RELIABILITY_BIN_EDGES = np.linspace(-0.000001, 1.000001, 11).astype(np.float32)
FSS_WINDOW_SIZES = [5, 10, 30, 60]
HIST_BINS_MM_H = np.array(
    [0.0, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0, 30.0, np.inf],
    dtype=np.float32,
)
# Models whose stored forecasts were produced by converting
# normalized/clipped dBZ outputs back to rain rate.
DBZ_OUTPUT_MODELS = {
    "convgru",
    "convgru_ens",
    "unet",
    "unet_ens",
    "ldcast",
}

ZR_A = 200.0
ZR_B = 1.6

# Exact rain rate corresponding mathematically to 0 dBZ.
ZERO_DBZ_RR_FLOAT32 = np.float32(
    (1.0 / ZR_A) ** (1.0 / ZR_B)
)

# Forecast Zarrs are stored as float16, so the artificial floor
# actually appears as this rounded value:
# 0.036468505859375 mm/h.
ZERO_DBZ_RR_FLOAT16 = np.float32(
    np.float16(ZERO_DBZ_RR_FLOAT32)
)

def _remove_exported_zero_dbz_floor(
    samples: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Convert the artificial 0-dBZ rain-rate floor back to dry.

    Neural and LDCast forecast Zarrs were exported after clipping
    dBZ to a lower bound of 0 dBZ. The old inverse transformation
    mapped that lower bound to approximately 0.03647 mm/h instead
    of 0 mm/h.

    PySTEPS is excluded because it produces rain-rate forecasts
    directly rather than normalized dBZ outputs.
    """

    samples = np.asarray(
        samples,
        dtype=np.float32,
    ).copy()

    if model_name not in DBZ_OUTPUT_MODELS:
        return samples

    finite = np.isfinite(samples)

    floor_mask = finite & (
        np.isclose(
            samples,
            ZERO_DBZ_RR_FLOAT16,
            rtol=0.0,
            atol=1e-7,
        )
        |
        np.isclose(
            samples,
            ZERO_DBZ_RR_FLOAT32,
            rtol=0.0,
            atol=1e-7,
        )
    )

    samples[floor_mask] = 0.0

    return samples

def _event_month_from_start_time(start_time: str, fallback: int) -> int:
    try:
        return int(pd.Timestamp(start_time).month)
    except Exception:
        return int(fallback)


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _decode_strings(values: Iterable) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _load_mask_array(path: str | Path) -> np.ndarray:
    """Load a static geographical mask from .npy, .npz or .zarr."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Belgium mask does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            preferred = ("belgium_mask", "mask", "valid_mask")
            key = next((name for name in preferred if name in data.files), None)
            if key is None:
                if len(data.files) != 1:
                    raise ValueError(
                        f"{path} contains multiple arrays {data.files}; "
                        "name the required array 'belgium_mask' or 'mask'."
                    )
                key = data.files[0]
            arr = data[key]
    elif suffix == ".zarr" or path.is_dir():
        obj = zarr.open(str(path), mode="r")
        if hasattr(obj, "shape"):
            arr = np.asarray(obj[:])
        else:
            preferred = ("belgium_mask", "mask", "valid_mask")
            key = next((name for name in preferred if name in obj), None)
            if key is None:
                keys = list(obj.array_keys())
                if len(keys) != 1:
                    raise ValueError(
                        f"{path} contains arrays {keys}; "
                        "name the required array 'belgium_mask' or 'mask'."
                    )
                key = keys[0]
            arr = np.asarray(obj[key][:])
    else:
        raise ValueError(
            f"Unsupported Belgium-mask format: {path}. "
            "Use .npy, .npz or .zarr."
        )

    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise ValueError(
            f"Belgium mask must be two-dimensional after squeeze; got {arr.shape}."
        )

    if arr.dtype == bool:
        mask = arr
    else:
        mask = np.isfinite(arr) & (arr > 0)

    if not np.any(mask):
        raise ValueError("Belgium mask contains no True pixels.")

    return mask.astype(bool, copy=False)


def _align_static_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Symmetrically pad/crop a static mask to the stored spatial shape."""
    mask = np.asarray(mask, dtype=bool)
    target_h, target_w = map(int, target_shape)
    h, w = mask.shape

    if (h, w) == (target_h, target_w):
        return mask

    out = mask

    # Symmetric crop when the supplied mask is larger.
    if out.shape[0] > target_h:
        diff = out.shape[0] - target_h
        top = diff // 2
        bottom = diff - top
        out = out[top : out.shape[0] - bottom, :]
    if out.shape[1] > target_w:
        diff = out.shape[1] - target_w
        left = diff // 2
        right = diff - left
        out = out[:, left : out.shape[1] - right]

    # Symmetric pad when the supplied mask is smaller.
    dh = target_h - out.shape[0]
    dw = target_w - out.shape[1]
    if dh < 0 or dw < 0:
        raise ValueError(
            f"Could not align mask shape {mask.shape} to target {target_shape}."
        )

    if dh or dw:
        pad_top = dh // 2
        pad_bottom = dh - pad_top
        pad_left = dw // 2
        pad_right = dw - pad_left
        out = np.pad(
            out,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=False,
        )

    if out.shape != (target_h, target_w):
        raise ValueError(
            f"Mask alignment failed: got {out.shape}, expected {target_shape}."
        )

    return out


def _base_score_mask(
    obs: np.ndarray,
    observation_valid_mask: np.ndarray,
    domain_mask: np.ndarray,
) -> np.ndarray:
    """Build a model-independent score mask.

    Forecast finiteness is deliberately not included here.
    """
    return (
        np.asarray(domain_mask, dtype=bool)
        & np.asarray(observation_valid_mask, dtype=bool)
        & np.isfinite(obs)
    )


def _mape_components(
    obs_vals: np.ndarray,
    pred_vals: np.ndarray,
    min_obs: float = 0.1,
) -> tuple[float, float, int]:
    keep = (
        np.isfinite(obs_vals)
        & np.isfinite(pred_vals)
        & (obs_vals >= float(min_obs))
    )
    n = int(np.sum(keep))
    if n == 0:
        return np.nan, 0.0, 0

    abs_pct = (
        100.0
        * np.abs(
            (
                pred_vals[keep]
                - obs_vals[keep]
            )
            / obs_vals[keep]
        )
    )
    return float(np.mean(abs_pct)), float(np.sum(abs_pct)), n


def _crps_pointwise(
    samples: np.ndarray,
    obs: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return empirical ensemble CRPS for each masked pixel.

    The pairwise spread term is evaluated from sorted ensemble members in
    O(M log M) time rather than explicitly looping over all M x M pairs.
    """
    samples = np.asarray(samples, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    ens = samples[:, mask]
    obs_vals = obs[mask]

    if obs_vals.size == 0:
        return np.empty(0, dtype=np.float32)

    if samples.shape[0] == 1:
        return np.abs(ens[0] - obs_vals).astype(np.float32)

    m_count = samples.shape[0]
    term1 = np.mean(np.abs(ens - obs_vals[None, :]), axis=0)

    sorted_ens = np.sort(ens, axis=0)
    rank = np.arange(1, m_count + 1, dtype=np.float32)
    weights = (2.0 * rank - m_count - 1.0)[:, None]
    term2 = np.sum(weights * sorted_ens, axis=0) / float(m_count * m_count)

    return (term1 - term2).astype(np.float32)


def _contingency_scores(
    obs_event: np.ndarray,
    fc_event: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, int, int, float, float, float, float]:
    obs_event = np.asarray(obs_event, dtype=bool)
    fc_event = np.asarray(fc_event, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    hits = int(np.sum(mask & fc_event & obs_event))
    misses = int(np.sum(mask & (~fc_event) & obs_event))
    false_alarms = int(np.sum(mask & fc_event & (~obs_event)))
    correct_negatives = int(np.sum(mask & (~fc_event) & (~obs_event)))

    total = hits + misses + false_alarms + correct_negatives

    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = (
        false_alarms / (hits + false_alarms)
        if (hits + false_alarms) > 0
        else np.nan
    )
    csi = (
        hits / (hits + misses + false_alarms)
        if (hits + misses + false_alarms) > 0
        else np.nan
    )

    if total > 0:
        hits_random = ((hits + misses) * (hits + false_alarms)) / total
        ets_denom = hits + misses + false_alarms - hits_random
        ets = (hits - hits_random) / ets_denom if ets_denom != 0 else np.nan
    else:
        ets = np.nan

    return (
        hits,
        misses,
        false_alarms,
        correct_negatives,
        pod,
        far,
        csi,
        ets,
    )


def _local_sum(field: np.ndarray, size: int) -> np.ndarray:
    """Neighbourhood sum with constant-zero support outside the array."""
    field = np.asarray(field, dtype=np.float32)
    size = int(size)

    try:
        from scipy.ndimage import uniform_filter

        return (
            uniform_filter(field, size=size, mode="constant", cval=0.0)
            * float(size * size)
        ).astype(np.float32)
    except Exception:
        from numpy.lib.stride_tricks import sliding_window_view

        before = size // 2
        after = size - 1 - before
        padded = np.pad(
            field,
            ((before, after), (before, after)),
            mode="constant",
            constant_values=0.0,
        )
        windows = sliding_window_view(padded, (size, size))
        return windows.sum(axis=(-1, -2), dtype=np.float64).astype(np.float32)


def _fss_components(
    obs_event: np.ndarray,
    fc_prob: np.ndarray,
    mask: np.ndarray,
    scale: int,
) -> tuple[float, float, float, int]:
    """Mask-aware FSS.

    Pixels outside the geographical/observation-valid mask are not treated as
    ordinary dry pixels. Local fractions are divided by the number of valid
    pixels in each neighbourhood.
    """
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.nan, 0.0, 0.0, 0

    obs_event = np.asarray(obs_event, dtype=np.float32)
    fc_prob = np.asarray(fc_prob, dtype=np.float32)

    valid_count = _local_sum(mask.astype(np.float32), scale)
    obs_count = _local_sum(obs_event * mask, scale)
    fc_count = _local_sum(fc_prob * mask, scale)

    obs_fraction = np.divide(
        obs_count,
        valid_count,
        out=np.full_like(obs_count, np.nan, dtype=np.float32),
        where=valid_count > 0,
    )
    fc_fraction = np.divide(
        fc_count,
        valid_count,
        out=np.full_like(fc_count, np.nan, dtype=np.float32),
        where=valid_count > 0,
    )

    center_mask = mask & np.isfinite(obs_fraction) & np.isfinite(fc_fraction)
    n_centers = int(np.sum(center_mask))
    if n_centers == 0:
        return np.nan, 0.0, 0.0, 0

    diff = fc_fraction[center_mask] - obs_fraction[center_mask]
    numerator = float(np.sum(diff * diff, dtype=np.float64))
    denominator = float(
        np.sum(
            fc_fraction[center_mask] ** 2 + obs_fraction[center_mask] ** 2,
            dtype=np.float64,
        )
    )

    score = 1.0 - numerator / denominator if denominator > 0 else np.nan
    return float(score), numerator, denominator, n_centers


def _randomized_ranks(
    samples: np.ndarray,
    obs: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    tie_atol: float,
) -> np.ndarray:
    """Rank observations among ensemble members with randomized ties."""
    samples = np.asarray(samples, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    ens = samples[:, mask]
    obs_vals = obs[mask]
    if obs_vals.size == 0:
        return np.empty(0, dtype=np.int64)

    diff = ens - obs_vals[None, :]
    n_less = np.sum(diff < -float(tie_atol), axis=0)
    n_equal = np.sum(np.abs(diff) <= float(tie_atol), axis=0)

    # Uniform integer in {0, ..., n_equal}, independently per pixel.
    tie_offset = np.floor(rng.random(n_equal.size) * (n_equal + 1)).astype(
        np.int64
    )
    return (n_less.astype(np.int64) + tie_offset).astype(np.int64)


def _probability_bin_mask(
    probabilities: np.ndarray,
    left: float,
    right: float,
    is_last: bool,
) -> np.ndarray:
    if is_last:
        return (probabilities >= left) & (probabilities <= right)
    return (probabilities >= left) & (probabilities < right)


def _check_store_shapes(
    model_name: str,
    forecast,
    truth,
    valid,
) -> None:
    if forecast.ndim != 5:
        raise ValueError(
            f"{model_name}: forecast must have shape "
            f"[event, member, lead, y, x], got {forecast.shape}."
        )
    if truth.ndim != 4:
        raise ValueError(
            f"{model_name}: truth must have shape [event, lead, y, x], "
            f"got {truth.shape}."
        )
    if valid.shape != truth.shape:
        raise ValueError(
            f"{model_name}: valid-mask shape {valid.shape} does not match "
            f"truth shape {truth.shape}."
        )

    if forecast.shape[2] != truth.shape[1]:
        raise ValueError(
            f"{model_name}: forecast has {forecast.shape[2]} leads but "
            f"observations have {truth.shape[1]}."
        )
    if tuple(forecast.shape[-2:]) != tuple(truth.shape[-2:]):
        raise ValueError(
            f"{model_name}: forecast spatial shape {forecast.shape[-2:]} "
            f"does not match observation shape {truth.shape[-2:]}."
        )


def _observation_event_indices(
    model_name: str,
    forecast_t_starts: np.ndarray,
    forecast_event_count: int,
    observation_group,
    observation_event_count: int,
) -> np.ndarray:
    """Align forecast events with either monthly or yearly observations."""
    forecast_t_starts = np.asarray(forecast_t_starts).astype(np.int64)

    if "t_start" in observation_group:
        observation_t_starts = np.asarray(
            observation_group["t_start"][:]
        ).astype(np.int64)

        if len(np.unique(observation_t_starts)) != len(observation_t_starts):
            raise ValueError(
                f"{model_name}: duplicate t_start values exist in the "
                "observation store."
            )

        lookup = {
            int(t_start): index
            for index, t_start in enumerate(observation_t_starts)
        }
        missing = [
            int(t_start)
            for t_start in forecast_t_starts
            if int(t_start) not in lookup
        ]
        if missing:
            raise ValueError(
                f"{model_name}: {len(missing)} forecast events were not "
                "found in the observation store. First missing t_start "
                f"values: {missing[:10]}"
            )

        return np.asarray(
            [lookup[int(t_start)] for t_start in forecast_t_starts],
            dtype=np.int64,
        )

    if forecast_event_count != observation_event_count:
        raise ValueError(
            f"{model_name}: the observation store has no t_start array and "
            f"contains {observation_event_count} events, whereas the "
            f"forecast store contains {forecast_event_count}. Event "
            "alignment cannot be verified."
        )

    return np.arange(forecast_event_count, dtype=np.int64)


def compute_for_model_month(
    model_name: str,
    forecast_store: Path,
    obs_store: Path,
    month: int,
    static_domain_mask: np.ndarray | None,
    evaluation_domain: str,
    rng: np.random.Generator,
    rank_tie_atol: float,
) -> dict[str, pd.DataFrame]:
    fg = zarr.open_group(str(forecast_store), mode="r")
    og = zarr.open_group(str(obs_store), mode="r")

    forecast = fg["forecast"]
    truth = og["truth"]
    valid = og["valid_mask"]

    _check_store_shapes(model_name, forecast, truth, valid)

    t_starts = np.asarray(fg["t_start"][:]).astype(int)
    start_times = _decode_strings(fg["start_time"][:])

    if len(t_starts) != forecast.shape[0]:
        raise ValueError(
            f"{model_name}: t_start length {len(t_starts)} does not match "
            f"{forecast.shape[0]} forecast events."
        )
    if len(start_times) != forecast.shape[0]:
        raise ValueError(
            f"{model_name}: start_time length {len(start_times)} does not "
            f"match {forecast.shape[0]} forecast events."
        )

    n_events, n_members, n_leads = forecast.shape[:3]
    observation_indices = _observation_event_indices(
        model_name=model_name,
        forecast_t_starts=t_starts,
        forecast_event_count=n_events,
        observation_group=og,
        observation_event_count=truth.shape[0],
    )
    spatial_shape = tuple(map(int, forecast.shape[-2:]))

    if evaluation_domain == "belgium":
        if static_domain_mask is None:
            raise ValueError(
                "--belgium-mask is required when "
                "--evaluation-domain=belgium."
            )
        domain_mask = _align_static_mask(static_domain_mask, spatial_shape)
    else:
        domain_mask = np.ones(spatial_shape, dtype=bool)

    deterministic_rows: list[dict] = []
    categorical_rows: list[dict] = []
    probabilistic_rows: list[dict] = []
    rank_rows: list[dict] = []
    reliability_rows: list[dict] = []
    fss_rows: list[dict] = []
    histogram_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for event_index in range(n_events):
        event_month = _event_month_from_start_time(
            start_times[event_index],
            month,
        )
        print(
            f"{model_name} store_month={month:02d} "
            f"event_month={event_month:02d} "
            f"event {event_index + 1}/{n_events} "
            f"t_start={t_starts[event_index]}"
        )

        observation_event_index = int(observation_indices[event_index])

        for lead in range(n_leads):
            # Load one lead at a time to avoid materialising a complete
            # [member, lead, y, x] event in memory.
            obs = truth[observation_event_index, lead].astype(np.float32)
            observation_valid = valid[
                observation_event_index,
                lead,
            ].astype(bool)
            samples = forecast[
                event_index,
                :,
                lead,
            ].astype(np.float32)
            
            samples = _remove_exported_zero_dbz_floor(
                samples,
                model_name=model_name,
            )

            score_mask = _base_score_mask(
                obs,
                observation_valid,
                domain_mask,
            )

            member_finite = np.all(np.isfinite(samples), axis=0)
            bad_forecast = score_mask & (~member_finite)

            n_domain = int(np.sum(domain_mask))
            n_obs_valid = int(np.sum(score_mask))
            n_nonfinite = int(np.sum(bad_forecast))
            finite_fraction = (
                float(np.sum(score_mask & member_finite) / n_obs_valid)
                if n_obs_valid > 0
                else np.nan
            )

            coverage_rows.append(
                {
                    "model": model_name,
                    "evaluation_domain": evaluation_domain,
                    "month": event_month,
                    "event_index_in_month": event_index,
                    "t_start": int(t_starts[event_index]),
                    "start_time": start_times[event_index],
                    "lead_index": lead,
                    "lead_time_min": int((lead + 1) * 5),
                    "n_members": int(n_members),
                    "n_domain_pixels": n_domain,
                    "n_obs_valid_domain_pixels": n_obs_valid,
                    "n_nonfinite_forecast_pixels": n_nonfinite,
                    "forecast_finite_fraction": finite_fraction,
                    "status": (
                        "ok"
                        if n_obs_valid > 0 and n_nonfinite == 0
                        else "no_valid_observation_pixels"
                        if n_obs_valid == 0
                        else "nonfinite_forecast"
                    ),
                }
            )

            if n_obs_valid == 0 or n_nonfinite > 0:
                # Never shrink the score mask to accommodate a model.
                continue

            pred_mean = np.mean(samples, axis=0, dtype=np.float32)
            obs_vals = obs[score_mask]
            pred_vals = pred_mean[score_mask]
            err = pred_vals - obs_vals

            abs_err = np.abs(err)
            sq_err = err * err

            rain_mask = score_mask & (obs >= RAIN_CONDITION_MM_H)
            rain_err = pred_mean[rain_mask] - obs[rain_mask]
            n_rain = int(np.sum(rain_mask))
            rain_sum_abs_error = (
                float(np.sum(np.abs(rain_err), dtype=np.float64))
                if n_rain > 0
                else 0.0
            )

            mape, sum_abs_pct, n_mape = _mape_components(
                obs_vals,
                pred_vals,
                min_obs=0.1,
            )

            deterministic_rows.append(
                {
                    "model": model_name,
                    "evaluation_domain": evaluation_domain,
                    "month": event_month,
                    "event_index_in_month": event_index,
                    "t_start": int(t_starts[event_index]),
                    "start_time": start_times[event_index],
                    "lead_index": lead,
                    "lead_time_min": int((lead + 1) * 5),
                    "n_members": int(n_members),
                    "n_valid_pixels": n_obs_valid,
                    "sum_abs_error": float(
                        np.sum(abs_err, dtype=np.float64)
                    ),
                    "sum_squared_error": float(
                        np.sum(sq_err, dtype=np.float64)
                    ),
                    "sum_error": float(np.sum(err, dtype=np.float64)),
                    "mae": float(np.mean(abs_err)),
                    "rmse": float(np.sqrt(np.mean(sq_err))),
                    "me": float(np.mean(err)),
                    "mape_obs_ge_0p1": mape,
                    "sum_abs_pct_error_obs_ge_0p1": sum_abs_pct,
                    "n_obs_ge_0p1": n_mape,
                    "mae_obs_ge_0p25": (
                        rain_sum_abs_error / n_rain
                        if n_rain > 0
                        else np.nan
                    ),
                    "sum_abs_error_obs_ge_0p25": rain_sum_abs_error,
                    "n_obs_ge_0p25": n_rain,
                    "max_observation": float(np.max(obs_vals)),
                    "max_prediction": float(np.max(pred_vals)),
                    "mean_ensemble_std": (
                        float(
                            np.mean(
                                np.std(samples[:, score_mask], axis=0)
                            )
                        )
                        if n_members > 1
                        else 0.0
                    ),
                }
            )

            # Histograms of observations and ensemble-mean forecasts.
            for field_name, values in (
                ("observation", obs_vals),
                ("forecast_mean", pred_vals),
            ):
                hist, edges = np.histogram(
                    values,
                    bins=HIST_BINS_MM_H,
                )
                for bin_index, count in enumerate(hist):
                    histogram_rows.append(
                        {
                            "model": model_name,
                            "evaluation_domain": evaluation_domain,
                            "month": event_month,
                            "t_start": int(t_starts[event_index]),
                            "lead_index": lead,
                            "lead_time_min": int((lead + 1) * 5),
                            "field": field_name,
                            "bin_left_mm_h": float(edges[bin_index]),
                            "bin_right_mm_h": float(edges[bin_index + 1]),
                            "count": int(count),
                        }
                    )

            # Deterministic-style contingency scores from the ensemble mean.
            for threshold in THRESHOLDS_MM_H:
                obs_event = obs >= threshold
                fc_event = pred_mean >= threshold
                (
                    hits,
                    misses,
                    false_alarms,
                    correct_negatives,
                    pod,
                    far,
                    csi,
                    ets,
                ) = _contingency_scores(obs_event, fc_event, score_mask)

                categorical_rows.append(
                    {
                        "model": model_name,
                        "evaluation_domain": evaluation_domain,
                        "month": event_month,
                        "t_start": int(t_starts[event_index]),
                        "start_time": start_times[event_index],
                        "lead_index": lead,
                        "lead_time_min": int((lead + 1) * 5),
                        "threshold_mm_h": float(threshold),
                        "H": hits,
                        "M": misses,
                        "F": false_alarms,
                        "C": correct_negatives,
                        "pod": _safe_float(pod),
                        "far": _safe_float(far),
                        "csi": _safe_float(csi),
                        "ets": _safe_float(ets),
                        "n_observed_exceedances": hits + misses,
                        "n_forecast_exceedances": hits + false_alarms,
                    }
                )

                if n_members > 1:
                    probability = np.mean(
                        samples >= threshold,
                        axis=0,
                    ).astype(np.float32)
                    obs_binary = (obs >= threshold).astype(np.float32)

                    brier_values = (
                        probability[score_mask] - obs_binary[score_mask]
                    ) ** 2

                    probabilistic_rows.append(
                        {
                            "model": model_name,
                            "evaluation_domain": evaluation_domain,
                            "month": event_month,
                            "t_start": int(t_starts[event_index]),
                            "start_time": start_times[event_index],
                            "lead_index": lead,
                            "lead_time_min": int((lead + 1) * 5),
                            "metric": "brier",
                            "threshold_mm_h": float(threshold),
                            "value": float(np.mean(brier_values)),
                            "sum_value": float(
                                np.sum(brier_values, dtype=np.float64)
                            ),
                            "n_pixels": int(brier_values.size),
                        }
                    )

                    # Reliability counts and weighted sums.
                    prob_vals = probability[score_mask]
                    obs_bin_vals = obs_binary[score_mask]
                    edges = RELIABILITY_BIN_EDGES

                    for bin_index, (left, right) in enumerate(
                        zip(edges[:-1], edges[1:])
                    ):
                        in_bin = _probability_bin_mask(
                            prob_vals,
                            float(left),
                            float(right),
                            is_last=bin_index == len(edges) - 2,
                        )
                        n_bin = int(np.sum(in_bin))
                        observed_count = (
                            int(np.sum(obs_bin_vals[in_bin]))
                            if n_bin > 0
                            else 0
                        )
                        probability_sum = (
                            float(
                                np.sum(
                                    prob_vals[in_bin],
                                    dtype=np.float64,
                                )
                            )
                            if n_bin > 0
                            else 0.0
                        )

                        reliability_rows.append(
                            {
                                "model": model_name,
                                "evaluation_domain": evaluation_domain,
                                "month": event_month,
                                "t_start": int(t_starts[event_index]),
                                "lead_index": lead,
                                "lead_time_min": int((lead + 1) * 5),
                                "threshold_mm_h": float(threshold),
                                "prob_bin_left": float(left),
                                "prob_bin_right": float(right),
                                "n": n_bin,
                                "observed_count": observed_count,
                                "forecast_probability_sum": probability_sum,
                                "observed_frequency": (
                                    observed_count / n_bin
                                    if n_bin > 0
                                    else np.nan
                                ),
                                "mean_forecast_probability": (
                                    probability_sum / n_bin
                                    if n_bin > 0
                                    else np.nan
                                ),
                            }
                        )

            if n_members > 1:
                pointwise_crps = _crps_pointwise(
                    samples,
                    obs,
                    score_mask,
                )
                probabilistic_rows.append(
                    {
                        "model": model_name,
                        "evaluation_domain": evaluation_domain,
                        "month": event_month,
                        "t_start": int(t_starts[event_index]),
                        "start_time": start_times[event_index],
                        "lead_index": lead,
                        "lead_time_min": int((lead + 1) * 5),
                        "metric": "crps",
                        "threshold_mm_h": np.nan,
                        "value": float(np.mean(pointwise_crps)),
                        "sum_value": float(
                            np.sum(pointwise_crps, dtype=np.float64)
                        ),
                        "n_pixels": int(pointwise_crps.size),
                    }
                )

                pointwise_crps_rain = pointwise_crps[
                    obs_vals >= RAIN_CONDITION_MM_H
                ]
                probabilistic_rows.append(
                    {
                        "model": model_name,
                        "evaluation_domain": evaluation_domain,
                        "month": event_month,
                        "t_start": int(t_starts[event_index]),
                        "start_time": start_times[event_index],
                        "lead_index": lead,
                        "lead_time_min": int((lead + 1) * 5),
                        "metric": "crps_obs_ge_0p25",
                        "threshold_mm_h": RAIN_CONDITION_MM_H,
                        "value": (
                            float(np.mean(pointwise_crps_rain))
                            if pointwise_crps_rain.size
                            else np.nan
                        ),
                        "sum_value": float(
                            np.sum(
                                pointwise_crps_rain,
                                dtype=np.float64,
                            )
                        ),
                        "n_pixels": int(pointwise_crps_rain.size),
                    }
                )

                # ============================================================
                # Rank histogram
                #
                # - +5, +50, and +100 min
                # - Belgium / observation-valid domain
                # - PySTEPS verification implementation
                # - X_min = 0.1 mm/h
                #
                # IMPORTANT:
                # X_min is handled by PySTEPS itself. Do NOT additionally
                # condition on obs >= 0.1.
                # ============================================================
                
                lead_time_min = int(
                    (lead + 1) * 5
                )
                
                if (
                    n_members > 1
                    and lead_time_min in RANK_LEAD_TIMES_MIN
                ):
                
                    # score_mask already contains:
                    # Belgium geography
                    # AND RADCLIM observation validity
                    # AND finite observation.
                    #
                    # Forecast finiteness has already been audited above.
                    rank_fc = samples[
                        :,
                        score_mask,
                    ]
                
                    rank_obs = obs[
                        score_mask
                    ]
                
                    rankhist = (
                        pysteps_verification.rankhist_init(
                            n_members,
                            RANK_X_MIN_MM_H,
                        )
                    )
                
                    pysteps_verification.rankhist_accum(
                        rankhist,
                        rank_fc,
                        rank_obs,
                    )
                
                    counts = np.asarray(
                        pysteps_verification.rankhist_compute(
                            rankhist,
                            normalize=False,
                        )
                    )
                
                    if counts.shape[0] != n_members + 1:
                        raise RuntimeError(
                            "Unexpected rank-histogram size: "
                            f"{counts.shape}; expected "
                            f"{n_members + 1} bins."
                        )
                
                    for rank_bin, count in enumerate(counts):
                
                        rank_rows.append(
                            {
                                "model": model_name,
                                "evaluation_domain":
                                    evaluation_domain,
                
                                "month": event_month,
                                "t_start":
                                    int(t_starts[event_index]),
                
                                "lead_index": lead,
                                "lead_time_min":
                                    lead_time_min,
                
                                "rank_bin":
                                    int(rank_bin),
                
                                "count":
                                    int(count),
                
                                "n_members":
                                    int(n_members),
                
                                "rank_x_min_mm_h":
                                    float(RANK_X_MIN_MM_H),
                            }
                        )

            # FSS is stored in two clearly distinguished forms:
            # 1. deterministic_style: direct forecast for deterministic models
            #    and thresholded ensemble mean for ensemble models;
            # 2. probabilistic_exceedance: member exceedance probabilities
            #    (ensemble models only).
            for threshold in THRESHOLDS_MM_H:
                obs_event = obs >= threshold

                forecast_fields = [
                    (
                        "deterministic_style",
                        (pred_mean >= threshold).astype(np.float32),
                    )
                ]
                if n_members > 1:
                    forecast_fields.append(
                        (
                            "probabilistic_exceedance",
                            np.mean(
                                samples >= threshold,
                                axis=0,
                            ).astype(np.float32),
                        )
                    )

                for forecast_type, fc_fraction_field in forecast_fields:
                    for scale in FSS_WINDOW_SIZES:
                        (
                            fss,
                            fss_numerator,
                            fss_denominator,
                            n_centers,
                        ) = _fss_components(
                            obs_event,
                            fc_fraction_field,
                            score_mask,
                            scale,
                        )
                        fss_rows.append(
                            {
                                "model": model_name,
                                "evaluation_domain": evaluation_domain,
                                "forecast_type": forecast_type,
                                "month": event_month,
                                "t_start": int(t_starts[event_index]),
                                "start_time": start_times[event_index],
                                "lead_index": lead,
                                "lead_time_min": int((lead + 1) * 5),
                                "threshold_mm_h": float(threshold),
                                "window_size_px": int(scale),
                                "fss": _safe_float(fss),
                                "fss_numerator": fss_numerator,
                                "fss_denominator": fss_denominator,
                                "n_center_pixels": n_centers,
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
        "coverage_diagnostics": pd.DataFrame(coverage_rows),
    }


def _weighted_deterministic_summary(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n_valid = int(group["n_valid_pixels"].sum())
        n_rain = int(group["n_obs_ge_0p25"].sum())
        n_mape = int(group["n_obs_ge_0p1"].sum())

        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_valid_pixels": n_valid,
                "mae": (
                    group["sum_abs_error"].sum() / n_valid
                    if n_valid > 0
                    else np.nan
                ),
                "rmse": (
                    math.sqrt(
                        group["sum_squared_error"].sum() / n_valid
                    )
                    if n_valid > 0
                    else np.nan
                ),
                "me": (
                    group["sum_error"].sum() / n_valid
                    if n_valid > 0
                    else np.nan
                ),
                "mape_obs_ge_0p1": (
                    group["sum_abs_pct_error_obs_ge_0p1"].sum()
                    / n_mape
                    if n_mape > 0
                    else np.nan
                ),
                "n_obs_ge_0p1": n_mape,
                "mae_obs_ge_0p25": (
                    group["sum_abs_error_obs_ge_0p25"].sum()
                    / n_rain
                    if n_rain > 0
                    else np.nan
                ),
                "n_obs_ge_0p25": n_rain,
                "n_events": int(group["t_start"].nunique()),
                "n_case_leads": int(len(group)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _pooled_categorical_summary(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        hits = int(group["H"].sum())
        misses = int(group["M"].sum())
        false_alarms = int(group["F"].sum())
        correct_negatives = int(group["C"].sum())
        total = hits + misses + false_alarms + correct_negatives

        pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
        far = (
            false_alarms / (hits + false_alarms)
            if (hits + false_alarms) > 0
            else np.nan
        )
        csi = (
            hits / (hits + misses + false_alarms)
            if (hits + misses + false_alarms) > 0
            else np.nan
        )

        if total > 0:
            hits_random = (
                (hits + misses) * (hits + false_alarms) / total
            )
            denom = hits + misses + false_alarms - hits_random
            ets = (
                (hits - hits_random) / denom
                if denom != 0
                else np.nan
            )
        else:
            ets = np.nan

        row = dict(zip(group_columns, keys))
        row.update(
            {
                "H": hits,
                "M": misses,
                "F": false_alarms,
                "C": correct_negatives,
                "pod": pod,
                "far": far,
                "csi": csi,
                "ets": ets,
                "n_observed_exceedances": hits + misses,
                "n_forecast_exceedances": hits + false_alarms,
                "n_case_leads_with_observed_event": int(
                    np.sum((group["H"] + group["M"]) > 0)
                ),
                "n_case_leads_with_forecast_event": int(
                    np.sum((group["H"] + group["F"]) > 0)
                ),
                "n_case_leads": int(len(group)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _weighted_probabilistic_summary(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n_pixels = int(group["n_pixels"].sum())
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "value": (
                    group["sum_value"].sum() / n_pixels
                    if n_pixels > 0
                    else np.nan
                ),
                "n_pixels": n_pixels,
                "n_events": int(group["t_start"].nunique()),
                "n_case_leads": int(len(group)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _pooled_fss_summary(
    df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        numerator = float(group["fss_numerator"].sum())
        denominator = float(group["fss_denominator"].sum())
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "fss": (
                    1.0 - numerator / denominator
                    if denominator > 0
                    else np.nan
                ),
                "fss_numerator": numerator,
                "fss_denominator": denominator,
                "n_center_pixels": int(
                    group["n_center_pixels"].sum()
                ),
                "n_events": int(group["t_start"].nunique()),
                "n_case_leads": int(len(group)),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _pooled_reliability_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_columns = [
        "model",
        "evaluation_domain",
        "threshold_mm_h",
        "prob_bin_left",
        "prob_bin_right",
    ]
    summary = (
        df.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            n=("n", "sum"),
            observed_count=("observed_count", "sum"),
            forecast_probability_sum=("forecast_probability_sum", "sum"),
        )
    )
    summary["observed_frequency"] = np.divide(
        summary["observed_count"],
        summary["n"],
        out=np.full(len(summary), np.nan, dtype=float),
        where=summary["n"].to_numpy() > 0,
    )
    summary["mean_forecast_probability"] = np.divide(
        summary["forecast_probability_sum"],
        summary["n"],
        out=np.full(len(summary), np.nan, dtype=float),
        where=summary["n"].to_numpy() > 0,
    )
    return summary


def _pooled_rank_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_columns = [
        "model",
        "evaluation_domain",
        "n_members",
        "rank_x_min_mm_h",
        "lead_time_min",
        "rank_bin",
    ]
    summary = (
        df.groupby(group_columns, as_index=False, dropna=False)
        .agg(count=("count", "sum"))
    )
    totals = summary.groupby(
        [
            "model",
            "evaluation_domain",
            "n_members",
            "rank_x_min_mm_h",
            "lead_time_min",
        ],
        dropna=False,
    )["count"].transform("sum")
    summary["relative_frequency"] = np.divide(
        summary["count"],
        totals,
        out=np.full(len(summary), np.nan, dtype=float),
        where=totals.to_numpy() > 0,
    )
    summary["expected_relative_frequency"] = 1.0 / (
        summary["n_members"] + 1
    )
    return summary


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print("Wrote", path, "rows=", len(df))


def _write_summaries(
    outputs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    deterministic = outputs["deterministic"]
    categorical = outputs["categorical"]
    probabilistic = outputs["probabilistic"]
    fss = outputs["fss"]
    reliability = outputs["reliability"]
    rank_histograms = outputs["rank_histograms"]

    det_overall = _weighted_deterministic_summary(
        deterministic,
        ["model", "evaluation_domain"],
    )
    det_by_lead = _weighted_deterministic_summary(
        deterministic,
        ["model", "evaluation_domain", "lead_time_min"],
    )
    _write_csv(
        det_overall,
        output_dir / "summary_deterministic_overall.csv",
    )
    _write_csv(
        det_by_lead,
        output_dir / "summary_deterministic_by_lead.csv",
    )

    cat_overall = _pooled_categorical_summary(
        categorical,
        ["model", "evaluation_domain", "threshold_mm_h"],
    )
    cat_by_lead = _pooled_categorical_summary(
        categorical,
        [
            "model",
            "evaluation_domain",
            "threshold_mm_h",
            "lead_time_min",
        ],
    )
    _write_csv(
        cat_overall,
        output_dir / "summary_categorical_pooled_overall.csv",
    )
    _write_csv(
        cat_by_lead,
        output_dir / "summary_categorical_pooled_by_lead.csv",
    )

    prob_overall = _weighted_probabilistic_summary(
        probabilistic,
        [
            "model",
            "evaluation_domain",
            "metric",
            "threshold_mm_h",
        ],
    )
    prob_by_lead = _weighted_probabilistic_summary(
        probabilistic,
        [
            "model",
            "evaluation_domain",
            "metric",
            "threshold_mm_h",
            "lead_time_min",
        ],
    )
    _write_csv(
        prob_overall,
        output_dir / "summary_probabilistic_overall.csv",
    )
    _write_csv(
        prob_by_lead,
        output_dir / "summary_probabilistic_by_lead.csv",
    )

    fss_overall = _pooled_fss_summary(
        fss,
        [
            "model",
            "evaluation_domain",
            "forecast_type",
            "threshold_mm_h",
            "window_size_px",
        ],
    )
    fss_by_lead = _pooled_fss_summary(
        fss,
        [
            "model",
            "evaluation_domain",
            "forecast_type",
            "threshold_mm_h",
            "window_size_px",
            "lead_time_min",
        ],
    )
    _write_csv(
        fss_overall,
        output_dir / "summary_fss_pooled_overall.csv",
    )
    _write_csv(
        fss_by_lead,
        output_dir / "summary_fss_pooled_by_lead.csv",
    )

    _write_csv(
        _pooled_reliability_summary(reliability),
        output_dir / "summary_reliability_weighted.csv",
    )
    _write_csv(
        _pooled_rank_summary(rank_histograms),
        output_dir / "summary_rank_histograms_by_lead.csv",
    )

    # Backward-compatible compact table for the main deterministic metrics.
    if not det_overall.empty:
        compact = det_overall.rename(
            columns={
                "mae": "mean_mae",
                "rmse": "mean_rmse",
                "me": "mean_me",
                "mape_obs_ge_0p1": "mean_mape_obs_ge_0p1",
            }
        )
        compact = compact.sort_values("mean_rmse")
        _write_csv(
            compact,
            output_dir / "summary_main_metrics.csv",
        )
        print(compact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--observation-root", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--store-mode",
        choices=["yearly", "monthly"],
        default="yearly",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
        help="Months to use in monthly mode.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="full")
    parser.add_argument(
        "--evaluation-domain",
        choices=["belgium", "radar"],
        default="belgium",
        help=(
            "Use the fixed Belgium mask for primary results, or the full "
            "observation-valid radar domain for a sensitivity check."
        ),
    )
    parser.add_argument(
        "--belgium-mask",
        default=None,
        help=(
            "Path to a Belgium mask in .npy, .npz or .zarr format. "
            "Required when --evaluation-domain=belgium."
        ),
    )
    parser.add_argument(
        "--rank-seed",
        type=int,
        default=42,
        help="Random seed used only for randomized rank-histogram ties.",
    )
    parser.add_argument(
        "--rank-tie-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance used to identify tied ranks.",
    )
    parser.add_argument(
        "--allow-nonfinite-forecasts",
        action="store_true",
        help=(
            "Write partial diagnostic outputs and exit successfully even "
            "when forecasts are non-finite inside the score domain. "
            "Do not use partial metrics as final thesis results."
        ),
    )
    args = parser.parse_args()

    if args.evaluation_domain == "belgium" and not args.belgium_mask:
        parser.error(
            "--belgium-mask is required when "
            "--evaluation-domain=belgium."
        )

    static_domain_mask = (
        _load_mask_array(args.belgium_mask)
        if args.belgium_mask
        else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_names = [
        "deterministic",
        "categorical",
        "probabilistic",
        "rank_histograms",
        "reliability",
        "fss",
        "histograms",
        "coverage_diagnostics",
    ]
    accum: dict[str, list[pd.DataFrame]] = {
        key: [] for key in table_names
    }

    rng = np.random.default_rng(args.rank_seed)

    for model in args.models:
        if args.store_mode == "yearly":
            obs_store = (
                Path(args.observation_root) / "observations_2021.zarr"
            )
            pred_store = (
                Path(args.prediction_root)
                / model
                / f"{model}_{args.split}_2021.zarr"
            )

            if not obs_store.exists():
                print("Missing observation store, skipping:", obs_store)
                continue
            if not pred_store.exists():
                print("Missing forecast store, skipping:", pred_store)
                continue

            tables = compute_for_model_month(
                model_name=model,
                forecast_store=pred_store,
                obs_store=obs_store,
                month=0,
                static_domain_mask=static_domain_mask,
                evaluation_domain=args.evaluation_domain,
                rng=rng,
                rank_tie_atol=args.rank_tie_atol,
            )
            for key, dataframe in tables.items():
                if dataframe is not None and not dataframe.empty:
                    accum[key].append(dataframe)
        else:
            for month in args.months:
                monthly_obs_store = (
                    Path(args.observation_root)
                    / f"observations_2021_month{month:02d}.zarr"
                )
                yearly_obs_store = (
                    Path(args.observation_root) / "observations_2021.zarr"
                )
                obs_store = (
                    monthly_obs_store
                    if monthly_obs_store.exists()
                    else yearly_obs_store
                )
                pred_store = (
                    Path(args.prediction_root)
                    / model
                    / f"{model}_{args.split}_2021_month{month:02d}.zarr"
                )

                if not obs_store.exists():
                    print(
                        "Missing both monthly and yearly observation stores; "
                        "skipping month:",
                        month,
                    )
                    continue
                if not pred_store.exists():
                    print("Missing forecast store, skipping:", pred_store)
                    continue

                tables = compute_for_model_month(
                    model_name=model,
                    forecast_store=pred_store,
                    obs_store=obs_store,
                    month=month,
                    static_domain_mask=static_domain_mask,
                    evaluation_domain=args.evaluation_domain,
                    rng=rng,
                    rank_tie_atol=args.rank_tie_atol,
                )
                for key, dataframe in tables.items():
                    if dataframe is not None and not dataframe.empty:
                        accum[key].append(dataframe)

    outputs: dict[str, pd.DataFrame] = {}
    for key, parts in accum.items():
        dataframe = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame()
        )
        outputs[key] = dataframe
        _write_csv(dataframe, output_dir / f"{key}.csv")

    _write_summaries(outputs, output_dir)

    diagnostics = outputs["coverage_diagnostics"]
    if not diagnostics.empty:
        bad = diagnostics[
            diagnostics["n_nonfinite_forecast_pixels"] > 0
        ]
        if not bad.empty:
            bad_summary = (
                bad.groupby("model", as_index=False)
                .agg(
                    bad_case_leads=(
                        "n_nonfinite_forecast_pixels",
                        "size",
                    ),
                    nonfinite_pixels=(
                        "n_nonfinite_forecast_pixels",
                        "sum",
                    ),
                    minimum_finite_fraction=(
                        "forecast_finite_fraction",
                        "min",
                    ),
                )
            )
            _write_csv(
                bad_summary,
                output_dir / "summary_nonfinite_forecasts.csv",
            )
            print(
                "\nNon-finite forecast values were found inside the "
                "evaluation domain:\n",
                bad_summary,
            )
            if not args.allow_nonfinite_forecasts:
                raise RuntimeError(
                    "Non-finite forecasts were found inside the fixed "
                    "score domain. Diagnostic CSVs were written, but the "
                    "affected metric rows were skipped. Fix the forecast "
                    "export before using these metrics as final results, "
                    "or rerun with --allow-nonfinite-forecasts only for "
                    "diagnostic exploration."
                )


if __name__ == "__main__":
    main()
