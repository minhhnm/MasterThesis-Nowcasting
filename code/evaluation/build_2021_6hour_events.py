#!/usr/bin/env python3
"""Build the fixed 2021 6-hourly RADCLIM test-event list.

The event starts are 00:00, 06:00, 12:00 and 18:00 UTC/local dataset time
for every day in 2021. Each event uses 4 input frames and 20 future frames,
so the script removes starts for which the full 24-frame window is not
available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _fallback_time_index_2021_5min() -> dict[pd.Timestamp, int]:
    """Fallback mapping for 5-min data starting at 2017-01-01 00:00."""
    dataset_start = pd.Timestamp("2017-01-01 00:00:00")
    starts = pd.date_range("2021-01-01 00:00:00", "2021-12-31 18:00:00", freq="6h")
    return {t: int((t - dataset_start).total_seconds() // 300) for t in starts}


def build_event_table(
    zarr_path: str,
    var_name: str,
    output_csv: str,
    tin: int = 4,
    tout: int = 20,
    year: int = 2021,
) -> pd.DataFrame:
    ds = xr.open_zarr(zarr_path)
    n_time = int(ds.sizes["time"])
    total_frames = int(tin + tout)

    starts = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 18:00:00", freq="6h")

    if "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        time_values = pd.to_datetime(ds["time"].values)
        index_by_time = {pd.Timestamp(t).floor("min"): i for i, t in enumerate(time_values)}
        t_lookup = {}
        missing = []
        for start in starts:
            key = pd.Timestamp(start).floor("min")
            if key in index_by_time:
                t_lookup[key] = int(index_by_time[key])
            else:
                missing.append(str(key))
        if missing:
            raise ValueError(
                f"Could not find {len(missing)} requested 6-hour timestamps in the Zarr time coordinate. "
                f"First missing timestamp: {missing[0]}"
            )
    else:
        print("WARNING: no datetime time coordinate found; using 5-minute fallback from 2017-01-01.")
        t_lookup = _fallback_time_index_2021_5min()

    rows = []
    for event_index, start_time in enumerate(starts):
        start_time = pd.Timestamp(start_time)
        t_start = int(t_lookup[start_time])
        valid = (t_start >= 0) and (t_start + total_frames <= n_time)
        if not valid:
            continue
        rows.append(
            {
                "event_index": int(event_index),
                "t_start": int(t_start),
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "year": int(start_time.year),
                "month": int(start_time.month),
                "day": int(start_time.day),
                "hour": int(start_time.hour),
                "input_start_index": int(t_start),
                "input_end_index": int(t_start + tin - 1),
                "first_target_index": int(t_start + tin),
                "last_target_index": int(t_start + tin + tout - 1),
                "tin": int(tin),
                "tout": int(tout),
            }
        )

    df = pd.DataFrame(rows)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} events to {out}")
    print(df.head())
    print(df.tail())
    expected = 365 * 4
    if len(df) != expected:
        print(f"WARNING: expected {expected} events for a complete non-leap year; got {len(df)}.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--var-name", default="precip_intensity_EDK")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--tin", type=int, default=4)
    parser.add_argument("--tout", type=int, default=20)
    parser.add_argument("--year", type=int, default=2021)
    args = parser.parse_args()

    build_event_table(
        zarr_path=args.zarr_path,
        var_name=args.var_name,
        output_csv=args.output_csv,
        tin=args.tin,
        tout=args.tout,
        year=args.year,
    )


if __name__ == "__main__":
    main()
