#!/bin/bash
#SBATCH --job-name=split_sampled_train_val
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/split_sampled_train_val_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

export INPUT_CSV="/data/brussel/114/vsc11442/sampled_datacubes.csv"
export TRAIN_CSV="/data/brussel/114/vsc11442/train_datacubes.csv"
export VAL_CSV="/data/brussel/114/vsc11442/val_datacubes.csv"
export REPORT_TXT="/data/brussel/114/vsc11442/train_val_split_report.txt"

# Fixed validation block each month: 8th-12th inclusive.
export VAL_START_DAY=8
export VAL_END_DAY=12

# 24 timesteps * 5 minutes = 2 hours buffer on each side for training exclusion.
export BUFFER_STEPS=24

# Each sampled cube has length 24, so the latest valid start in a raw block is E - 23.
export TIME_DEPTH=24

python - <<'PY'
import csv
import os
from bisect import bisect_right
from datetime import datetime, timedelta

INPUT_CSV = os.environ["INPUT_CSV"]
TRAIN_CSV = os.environ["TRAIN_CSV"]
VAL_CSV = os.environ["VAL_CSV"]
REPORT_TXT = os.environ["REPORT_TXT"]
VAL_START_DAY = int(os.environ["VAL_START_DAY"])
VAL_END_DAY = int(os.environ["VAL_END_DAY"])
BUFFER_STEPS = int(os.environ["BUFFER_STEPS"])
TIME_DEPTH = int(os.environ["TIME_DEPTH"])

BASE = datetime(2017, 1, 1, 0, 0)
STEP = timedelta(minutes=5)


def dt_to_t(dt: datetime) -> int:
    return int((dt - BASE) / STEP)


def t_to_dt(t: int) -> datetime:
    return BASE + t * STEP


# Build fixed monthly validation windows for 2017-01 to 2023-11, excluding 2021.
windows = []
for year in range(2017, 2024):
    if year == 2021:
        continue
    last_month = 11 if year == 2023 else 12
    for month in range(1, last_month + 1):
        raw_start = datetime(year, month, VAL_START_DAY, 0, 0)
        raw_end = datetime(year, month, VAL_END_DAY, 23, 55)

        raw_start_t = dt_to_t(raw_start)
        raw_end_t = dt_to_t(raw_end)

        # Validation are full 5 days each month.
        val_start_t = raw_start_t
        val_end_t = raw_end_t

        # Training exclusion = raw validation block + 2h buffer before and after.
        train_excl_start_t = raw_start_t - BUFFER_STEPS
        train_excl_end_t = raw_end_t + BUFFER_STEPS

        windows.append({
            "label": f"{year}-{month:02d}",
            "raw_start_t": raw_start_t,
            "raw_end_t": raw_end_t,
            "val_start_t": val_start_t,
            "val_end_t": val_end_t,
            "train_excl_start_t": train_excl_start_t,
            "train_excl_end_t": train_excl_end_t,
        })

windows.sort(key=lambda w: w["raw_start_t"])

val_intervals = [(w["val_start_t"], w["val_end_t"]) for w in windows]
excl_intervals = [(w["train_excl_start_t"], w["train_excl_end_t"]) for w in windows]

val_starts = [a for a, _ in val_intervals]
val_ends = [b for _, b in val_intervals]
excl_starts = [a for a, _ in excl_intervals]
excl_ends = [b for _, b in excl_intervals]


def in_intervals(t: int, starts: list[int], ends: list[int]) -> bool:
    i = bisect_right(starts, t) - 1
    return i >= 0 and t <= ends[i]


os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
os.makedirs(os.path.dirname(VAL_CSV), exist_ok=True)

n_total = 0
n_train = 0
n_val = 0
n_dropped = 0

with open(INPUT_CSV, "r", newline="") as f_in, \
     open(TRAIN_CSV, "w", newline="") as f_train, \
     open(VAL_CSV, "w", newline="") as f_val:

    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError("Input CSV has no header.")

    required = {"t", "x", "y"}
    if not required.issubset(fieldnames):
        raise ValueError(f"Input CSV must contain columns {required}, got {fieldnames}")

    train_writer = csv.DictWriter(f_train, fieldnames=fieldnames)
    val_writer = csv.DictWriter(f_val, fieldnames=fieldnames)
    train_writer.writeheader()
    val_writer.writeheader()

    for row in reader:
        n_total += 1
        t = int(row["t"])

        if in_intervals(t, val_starts, val_ends):
            val_writer.writerow(row)
            n_val += 1
        elif in_intervals(t, excl_starts, excl_ends):
            n_dropped += 1
        else:
            train_writer.writerow(row)
            n_train += 1

        if n_total % 1000000 == 0:
            print(f"Processed {n_total:,} rows | train={n_train:,} val={n_val:,} dropped={n_dropped:,}")

with open(REPORT_TXT, "w") as f:
    f.write("Fixed monthly validation split\n")
    f.write(f"Input CSV: {INPUT_CSV}\n")
    f.write(f"Train CSV: {TRAIN_CSV}\n")
    f.write(f"Val CSV: {VAL_CSV}\n")
    f.write(f"VAL_START_DAY: {VAL_START_DAY}\n")
    f.write(f"VAL_END_DAY: {VAL_END_DAY}\n")
    f.write(f"BUFFER_STEPS: {BUFFER_STEPS}\n")
    f.write(f"TIME_DEPTH: {TIME_DEPTH}\n\n")
    f.write("Monthly windows\n")
    for w in windows:
        f.write(
            f"{w['label']}: raw_val={t_to_dt(w['raw_start_t'])} -> {t_to_dt(w['raw_end_t'])} | "
            f"val_starts={t_to_dt(w['val_start_t'])} -> {t_to_dt(w['val_end_t'])} | "
            f"train_exclude={t_to_dt(w['train_excl_start_t'])} -> {t_to_dt(w['train_excl_end_t'])}\n"
        )
    f.write("\n")
    f.write(f"Total rows: {n_total:,}\n")
    f.write(f"Train rows: {n_train:,}\n")
    f.write(f"Val rows: {n_val:,}\n")
    f.write(f"Dropped rows: {n_dropped:,}\n")

print(f"Total rows:   {n_total:,}")
print(f"Train rows:   {n_train:,}")
print(f"Val rows:     {n_val:,}")
print(f"Dropped rows: {n_dropped:,}")
print(f"Report: {REPORT_TXT}")
PY
