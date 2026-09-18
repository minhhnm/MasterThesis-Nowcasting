#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from Minh_read_mlcast_yaml import load_radar_cfg


BASE_TIME = pd.Timestamp("2017-01-01 00:00:00")


def t_to_datetime(t):
    return BASE_TIME + pd.Timedelta(minutes=5 * int(t))


def compute_event_metrics(da, t, tin, tout):
    # Score the target period, not the past context.
    rr = da.isel(time=slice(t + tin, t + tin + tout)).values.astype(np.float32)

    valid = np.isfinite(rr)
    if not valid.any():
        return None

    valid_values = rr[valid]

    max_rr = float(np.nanmax(valid_values))
    p99_rr = float(np.nanpercentile(valid_values, 99.0))
    p999_rr = float(np.nanpercentile(valid_values, 99.9))

    wet1 = float(((rr >= 1.0) & valid).sum() / valid.sum())
    wet5 = float(((rr >= 5.0) & valid).sum() / valid.sum())
    wet10 = float(((rr >= 10.0) & valid).sum() / valid.sum())

    # Heuristic score: strong cores + spatially large rain.
    score = p999_rr + 0.25 * max_rr + 100.0 * wet5 + 200.0 * wet10

    return {
        "t_start": int(t),
        "datetime": str(t_to_datetime(t)),
        "max_rr": max_rr,
        "p99_rr": p99_rr,
        "p999_rr": p999_rr,
        "wet1_frac": wet1,
        "wet5_frac": wet5,
        "wet10_frac": wet10,
        "score": score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_config", required=True)
    ap.add_argument("--region", default="belgium")
    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--tout", type=int, default=20)
    ap.add_argument("--coarse_every", type=int, default=12)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--refine_radius", type=int, default=12)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    cfg = load_radar_cfg(args.data_config, args.region)

    print("Zarr:", cfg["zarr_path"])
    print("Variable:", cfg["var_name"])
    print("Val CSV:", cfg["val_csv_path"])

    val = pd.read_csv(cfg["val_csv_path"]).sort_values("t")
    unique_t = np.array(sorted(val["t"].unique()), dtype=int)

    # Keep only safe t_start values.
    ds = xr.open_zarr(cfg["zarr_path"])
    da = ds[cfg["var_name"]]
    max_safe_t = da.sizes["time"] - (args.tin + args.tout)
    unique_t = unique_t[unique_t <= max_safe_t]

    print("Unique validation t values:", len(unique_t))

    coarse_t = unique_t[:: args.coarse_every]
    print("Coarse candidates:", len(coarse_t))

    coarse_rows = []
    for i, t in enumerate(coarse_t, start=1):
        if i % 100 == 0:
            print(f"Coarse scan {i}/{len(coarse_t)}")

        row = compute_event_metrics(da, t, args.tin, args.tout)
        if row is not None:
            coarse_rows.append(row)

    coarse_df = pd.DataFrame(coarse_rows).sort_values("score", ascending=False)
    print("\nTop coarse candidates:")
    print(coarse_df.head(args.top_k))

    # Refine around the best coarse candidates using exact 5-minute candidates.
    best_coarse = coarse_df.head(args.top_k)["t_start"].to_numpy(dtype=int)

    refine_candidates = set()
    unique_t_set = set(unique_t.tolist())

    for t0 in best_coarse:
        for dt in range(-args.refine_radius, args.refine_radius + 1):
            t = int(t0 + dt)
            if t in unique_t_set:
                refine_candidates.add(t)

    refine_candidates = sorted(refine_candidates)
    print("Refine candidates:", len(refine_candidates))

    refine_rows = []
    for i, t in enumerate(refine_candidates, start=1):
        if i % 50 == 0:
            print(f"Refine scan {i}/{len(refine_candidates)}")

        row = compute_event_metrics(da, t, args.tin, args.tout)
        if row is not None:
            refine_rows.append(row)

    final_df = pd.DataFrame(refine_rows).sort_values("score", ascending=False)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_csv, index=False)

    print("\nTop refined candidates:")
    print(final_df.head(20))

    print("\nRecommended t_start:")
    print(int(final_df.iloc[0]["t_start"]))
    print("Datetime:")
    print(final_df.iloc[0]["datetime"])

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
