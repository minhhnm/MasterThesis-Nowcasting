#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_TIME = pd.Timestamp("2017-01-01 00:00:00")


def t_to_datetime(t):
    return BASE_TIME + pd.Timedelta(minutes=5 * int(t))


def audit(train_csv, val_csv, tin, tout, event_t=None):
    seq_len = tin + tout
    max_overlap = seq_len - 1

    print("Train CSV:", train_csv)
    print("Val CSV:", val_csv)
    print("tin:", tin, "tout:", tout, "seq_len:", seq_len)

    train = pd.read_csv(train_csv, usecols=["t", "x", "y"])
    val = pd.read_csv(val_csv, usecols=["t", "x", "y"])

    print("train rows:", len(train))
    print("val rows:", len(val))

    train_t = np.array(sorted(train["t"].unique()), dtype=np.int64)
    val_t = np.array(sorted(val["t"].unique()), dtype=np.int64)

    print("unique train t:", len(train_t))
    print("unique val t:", len(val_t))

    # 1. Same t in train and validation.
    same_t = np.intersect1d(train_t, val_t)
    print("\nSame t in train and val:", len(same_t))
    if len(same_t) > 0:
        print("First same t examples:")
        for t in same_t[:10]:
            print(int(t), t_to_datetime(t))

    # 2. Temporal window overlap:
    # train sequence [t, t+seq_len-1]
    # val sequence   [v, v+seq_len-1]
    # overlap exists if train_t is within [val_t-max_overlap, val_t+max_overlap]
    leak_mask = np.zeros(len(train_t), dtype=bool)

    for i, t in enumerate(train_t):
        lo = t - max_overlap
        hi = t + max_overlap
        j = np.searchsorted(val_t, lo, side="left")
        if j < len(val_t) and val_t[j] <= hi:
            leak_mask[i] = True

    leak_t = train_t[leak_mask]
    print("\nTrain t values whose 24-frame window overlaps a val window:", len(leak_t))

    if len(leak_t) > 0:
        train_counts = train.groupby("t").size()
        leaking_rows = int(train_counts.loc[leak_t].sum())
        print("Train rows affected by temporal overlap:", leaking_rows)
        print("First temporal-overlap examples:")
        for t in leak_t[:10]:
            print(int(t), t_to_datetime(t))

    # 3. Event-specific check.
    if event_t is not None:
        event_t = int(event_t)
        near = train_t[(train_t >= event_t - max_overlap) & (train_t <= event_t + max_overlap)]
        print("\nEvent t:", event_t, t_to_datetime(event_t))
        print(f"Train t within +/- {max_overlap} timesteps of event:", len(near))
        if len(near) > 0:
            for t in near[:20]:
                print(int(t), t_to_datetime(t))

    if len(same_t) == 0 and len(leak_t) == 0:
        print("\nRESULT: No temporal train/validation leakage detected.")
    else:
        print("\nRESULT: Potential leakage detected. Inspect the examples above.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--tout", type=int, default=20)
    ap.add_argument("--event_t", type=int, default=None)
    args = ap.parse_args()

    audit(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        tin=args.tin,
        tout=args.tout,
        event_t=args.event_t,
    )


if __name__ == "__main__":
    main()
