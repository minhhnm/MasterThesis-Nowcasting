#!/usr/bin/env python3
"""
Lightweight data audit for RADCLIM Zarr (streaming, low memory).
- does NOT load huge stacks into RAM
- samples N frames, reads a small window, optional downsampling stride
- writes JSON summary

Usage:
python scripts/00_data_audit_light.py \
  --zarr "$VSC_DATA_VO/radar/RADCLIMrates.zarr" \
  --var precip_intensity_EDK \
  --out results/data_audit_light.json \
  --n_frames 30 \
  --window 256 \
  --stride 5
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import zarr


def infer_time_divisor(time_int: np.ndarray) -> float:
    """Infer whether time is in ns/us/ms/s by matching dt~300s."""
    t = time_int[:2000].astype(np.float64)
    dt_raw = np.median(np.diff(t))
    candidates = [1.0, 1e3, 1e6, 1e9]
    best = min(candidates, key=lambda div: abs(dt_raw / div - 300.0))
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--var", required=True)
    ap.add_argument("--out", default="results/data_audit_light.json")
    ap.add_argument("--n_frames", type=int, default=30, help="how many time indices to sample")
    ap.add_argument("--window", type=int, default=256, help="spatial window size (square)")
    ap.add_argument("--stride", type=int, default=5, help="spatial downsampling stride")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    store = args.zarr
    var = args.var

    # open consolidated if available (faster)
    opener = zarr.open_consolidated if os.path.exists(os.path.join(store, ".zmetadata")) else zarr.open
    root = opener(store, mode="r")

    if var not in root.array_keys():
        raise KeyError(f"Variable '{var}' not found. array_keys={list(root.array_keys())}")

    arr = root[var]          # (T, H, W)
    time_arr = root["time"]  # (T,)

    T, H, W = arr.shape
    dtype = str(arr.dtype)

    # infer dt
    time_vals = time_arr[:min(T, 3000)]
    div = infer_time_divisor(time_vals)
    dt_raw = float(np.median(np.diff(time_vals[:2000].astype(np.float64))))
    dt_sec = dt_raw / div

    # choose sample indices evenly spaced (more stable than random)
    n = min(args.n_frames, T)
    idxs = np.linspace(0, T - 1, n, dtype=int)

    # choose a central window
    win = min(args.window, H, W)
    y0 = (H - win) // 2
    x0 = (W - win) // 2
    ys = slice(y0, y0 + win, args.stride)
    xs = slice(x0, x0 + win, args.stride)

    # streaming stats
    finite_count = 0
    nan_count = 0
    inf_count = 0
    neg_count = 0
    s1 = 0.0
    s2 = 0.0
    vmin = np.inf
    vmax = -np.inf

    for t in idxs:
        frame = np.asarray(arr[t, ys, xs], dtype=np.float32)
        nan_count += int(np.isnan(frame).sum())
        inf_count += int(np.isinf(frame).sum())

        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        neg_count += int((frame < 0).sum())

        finite_count += frame.size
        s1 += float(frame.sum())
        s2 += float((frame * frame).sum())
        vmin = min(vmin, float(frame.min()))
        vmax = max(vmax, float(frame.max()))

    mean = s1 / max(1, finite_count)
    var_ = s2 / max(1, finite_count) - mean * mean
    std = float(np.sqrt(max(0.0, var_)))

    summary = {
        "zarr_path": store,
        "variable": var,
        "shape": [int(T), int(H), int(W)],
        "dtype": dtype,
        "sample_n_frames": int(n),
        "sample_window": [int(win), int(win)],
        "sample_stride": int(args.stride),
        "dt_seconds_median": float(dt_sec),
        "nan_fraction_percent": float(100.0 * nan_count / max(1, finite_count)),
        "inf_fraction_percent": float(100.0 * inf_count / max(1, finite_count)),
        "negative_fraction_percent": float(100.0 * neg_count / max(1, finite_count)),
        "sample_mean": float(mean),
        "sample_std": float(std),
        "sample_min": float(vmin),
        "sample_max": float(vmax),
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()