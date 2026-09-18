#!/usr/bin/env python3
"""Select observation-based qualitative validation events for radar nowcasting.

The script screens full-domain validation sequences at a fixed temporal stride,
computes observation-only descriptors, and ranks candidates for three regimes:

1. intense convection,
2. widespread organized rainfall,
3. rapidly evolving rainfall.

It does not use any model predictions, so the event selection remains independent
of model performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr-path", required=True)
    ap.add_argument("--var-name", default="precip_intensity_EDK")
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--tout", type=int, default=20)
    ap.add_argument(
        "--candidate-spacing-steps",
        type=int,
        default=72,
        help="Minimum spacing between screened starts in 5-minute steps. "
             "72 steps = 6 hours.",
    )
    ap.add_argument(
        "--selection-separation-steps",
        type=int,
        default=288,
        help="Minimum separation between the three suggested events. "
             "288 steps = 24 hours.",
    )
    ap.add_argument("--max-candidates", type=int, default=None)
    return ap.parse_args()


def choose_spaced_candidates(values: np.ndarray, spacing: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    values = np.unique(values)
    values.sort()

    selected: list[int] = []
    last = -10**18
    for value in values:
        value = int(value)
        if value - last >= spacing:
            selected.append(value)
            last = value
    return np.asarray(selected, dtype=np.int64)


def weighted_centroid(field: np.ndarray, threshold: float = 1.0) -> tuple[float, float]:
    arr = np.asarray(field, dtype=np.float32)
    valid = np.isfinite(arr) & (arr >= threshold)
    if not valid.any():
        return np.nan, np.nan

    weights = np.where(valid, arr, 0.0).astype(np.float64)
    total = float(weights.sum())
    if total <= 0:
        return np.nan, np.nan

    y, x = np.indices(arr.shape)
    cy = float((weights * y).sum() / total)
    cx = float((weights * x).sum() / total)
    return cy, cx


def event_descriptors(seq: np.ndarray, tin: int, tout: int) -> dict[str, float]:
    seq = np.asarray(seq, dtype=np.float32)
    past = seq[:tin]
    future = seq[tin:tin + tout]

    finite = future[np.isfinite(future)]
    if finite.size == 0:
        raise ValueError("No finite future pixels.")

    area01 = np.nanmean(future >= 0.1, axis=(1, 2))
    area1 = np.nanmean(future >= 1.0, axis=(1, 2))
    area5 = np.nanmean(future >= 5.0, axis=(1, 2))
    area10 = np.nanmean(future >= 10.0, axis=(1, 2))
    mean_by_lead = np.nanmean(future, axis=(1, 2))

    first = future[0]
    last = future[-1]
    structural_change = float(np.nanmean(np.abs(last - first)))

    cy0, cx0 = weighted_centroid(first, threshold=1.0)
    cy1, cx1 = weighted_centroid(last, threshold=1.0)
    if np.isfinite([cy0, cx0, cy1, cx1]).all():
        centroid_displacement = float(np.hypot(cy1 - cy0, cx1 - cx0))
    else:
        centroid_displacement = np.nan

    last_input = past[-1]
    input_to_end_change = float(np.nanmean(np.abs(last - last_input)))

    return {
        "mean_future_rain_mm_h": float(np.mean(finite)),
        "p95_future_rain_mm_h": float(np.percentile(finite, 95)),
        "p99_future_rain_mm_h": float(np.percentile(finite, 99)),
        "maximum_future_rain_mm_h": float(np.max(finite)),
        "mean_fraction_above_0.1": float(np.mean(area01)),
        "mean_fraction_above_1": float(np.mean(area1)),
        "mean_fraction_above_5": float(np.mean(area5)),
        "mean_fraction_above_10": float(np.mean(area10)),
        "maximum_fraction_above_1": float(np.max(area1)),
        "maximum_fraction_above_10": float(np.max(area10)),
        "absolute_area1_change": float(abs(area1[-1] - area1[0])),
        "area1_variability": float(np.std(area1)),
        "absolute_mean_rain_change": float(abs(mean_by_lead[-1] - mean_by_lead[0])),
        "structural_change_mm_h": structural_change,
        "input_to_end_change_mm_h": input_to_end_change,
        "centroid_displacement_px": centroid_displacement,
    }


def robust_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()

    if not np.isfinite(mad) or mad == 0:
        std = values.std()
        if not np.isfinite(std) or std == 0:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values - values.mean()) / std

    return 0.67448975 * (values - median) / mad


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "p99_future_rain_mm_h",
        "mean_fraction_above_10",
        "mean_fraction_above_1",
        "mean_future_rain_mm_h",
        "structural_change_mm_h",
        "absolute_area1_change",
        "area1_variability",
        "centroid_displacement_px",
    ]

    for col in numeric_cols:
        df[f"z_{col}"] = robust_zscore(df[col].fillna(df[col].median()))

    df["intense_score"] = (
        df["z_p99_future_rain_mm_h"]
        + 0.60 * df["z_mean_fraction_above_10"]
    )

    df["widespread_score"] = (
        df["z_mean_fraction_above_1"]
        + 0.35 * df["z_mean_future_rain_mm_h"]
    )

    df["evolving_score"] = (
        df["z_structural_change_mm_h"]
        + 0.50 * df["z_absolute_area1_change"]
        + 0.35 * df["z_area1_variability"]
        + 0.25 * df["z_centroid_displacement_px"]
    )

    for score in ["intense_score", "widespread_score", "evolving_score"]:
        df[f"rank_{score.removesuffix('_score')}"] = (
            df[score].rank(method="min", ascending=False).astype(int)
        )

    return df


def choose_nonoverlapping(
    df: pd.DataFrame,
    score_col: str,
    chosen_t: list[int],
    minimum_separation: int,
) -> pd.Series:
    for _, row in df.sort_values(score_col, ascending=False).iterrows():
        t_start = int(row["t_start"])
        if all(abs(t_start - other) >= minimum_separation for other in chosen_t):
            return row
    raise RuntimeError(f"Could not find a separated candidate for {score_col}.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(args.val_csv)
    if "t" not in val.columns:
        raise KeyError(f"Validation CSV has no 't' column: {args.val_csv}")

    starts = choose_spaced_candidates(
        val["t"].dropna().astype(np.int64).to_numpy(),
        spacing=args.candidate_spacing_steps,
    )

    ds = xr.open_zarr(args.zarr_path)
    if args.var_name not in ds:
        raise KeyError(f"{args.var_name!r} not found. Variables: {list(ds.data_vars)}")

    da = ds[args.var_name]
    time_dim = "time" if "time" in da.dims else da.dims[0]
    n_time = int(da.sizes[time_dim])
    total = args.tin + args.tout

    starts = starts[starts + total <= n_time]
    if args.max_candidates is not None:
        starts = starts[: args.max_candidates]

    print("Validation CSV:", args.val_csv)
    print("Zarr:", args.zarr_path)
    print("Variable:", args.var_name)
    print("Screened starts:", len(starts))
    print("Candidate spacing:", args.candidate_spacing_steps, "steps")

    rows: list[dict[str, object]] = []

    for i, t_start in enumerate(starts, start=1):
        if i == 1 or i % 25 == 0 or i == len(starts):
            print(f"[{i}/{len(starts)}] t_start={int(t_start)}", flush=True)

        try:
            seq = da.isel({time_dim: slice(int(t_start), int(t_start) + total)}).values
            seq = np.asarray(seq, dtype=np.float32)

            if seq.shape[0] != total:
                continue

            row: dict[str, object] = {
                "t_start": int(t_start),
            }

            if time_dim in da.coords:
                start_time = da[time_dim].isel({time_dim: int(t_start)}).values
                row["start_time"] = str(np.asarray(start_time).item())
            else:
                row["start_time"] = ""

            row.update(event_descriptors(seq, tin=args.tin, tout=args.tout))
            rows.append(row)

        except Exception as exc:
            print(f"Skipping t_start={int(t_start)}: {exc!r}", flush=True)

    if not rows:
        raise RuntimeError("No candidate descriptors were produced.")

    descriptors = add_scores(pd.DataFrame(rows))
    descriptors = descriptors.sort_values("t_start").reset_index(drop=True)
    descriptors.to_csv(output_dir / "qualitative_event_descriptors.csv", index=False)

    top_tables = []
    for category, score_col in [
        ("intense convection", "intense_score"),
        ("widespread organized rainfall", "widespread_score"),
        ("rapidly evolving rainfall", "evolving_score"),
    ]:
        top = descriptors.sort_values(score_col, ascending=False).head(20).copy()
        top.insert(0, "category", category)
        top.insert(1, "category_score", top[score_col])
        top_tables.append(top)

    top_candidates = pd.concat(top_tables, ignore_index=True)
    top_candidates.to_csv(output_dir / "qualitative_top_candidates.csv", index=False)

    chosen_rows = []
    chosen_t: list[int] = []

    selection_plan = [
        ("intense convection", "intense_score"),
        ("widespread organized rainfall", "widespread_score"),
        ("rapidly evolving rainfall", "evolving_score"),
    ]

    for category, score_col in selection_plan:
        row = choose_nonoverlapping(
            descriptors,
            score_col=score_col,
            chosen_t=chosen_t,
            minimum_separation=args.selection_separation_steps,
        )
        chosen_t.append(int(row["t_start"]))
        out = row.to_dict()
        out["category"] = category
        out["selection_score"] = float(row[score_col])
        chosen_rows.append(out)

    suggested = pd.DataFrame(chosen_rows)
    first_cols = [
        "category",
        "t_start",
        "start_time",
        "selection_score",
        "p99_future_rain_mm_h",
        "mean_fraction_above_1",
        "mean_fraction_above_10",
        "structural_change_mm_h",
        "absolute_area1_change",
        "centroid_displacement_px",
    ]
    remaining = [c for c in suggested.columns if c not in first_cols]
    suggested = suggested[first_cols + remaining]
    suggested.to_csv(output_dir / "qualitative_suggested_events.csv", index=False)

    print("\nSuggested events:")
    print(suggested[first_cols].to_string(index=False))
    print("\nWrote:")
    for name in [
        "qualitative_event_descriptors.csv",
        "qualitative_top_candidates.csv",
        "qualitative_suggested_events.csv",
    ]:
        print(" ", output_dir / name)


if __name__ == "__main__":
    main()
