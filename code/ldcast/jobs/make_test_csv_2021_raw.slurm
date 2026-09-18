#!/bin/bash
#SBATCH --job-name=make_test_2021_raw
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

cd $VSC_DATA/code/mlcast-ldcast

export ZARR_PATH="/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"
export OUTPUT_CSV="/data/brussel/114/vsc11442/test_datacubes_2021_raw.csv"
export REPORT_TXT="/data/brussel/114/vsc11442/test_datacubes_2021_raw_report.txt"

export TEST_YEAR_START_T=420768
export TEST_YEAR_END_T=525864

export TIME_DEPTH=24
export T_STRIDE=3
export XY_STRIDE=16
export CROP_SIZE=256

python - <<'PY'
import csv
import os
import xarray as xr

ZARR_PATH = os.environ["ZARR_PATH"]
OUTPUT_CSV = os.environ["OUTPUT_CSV"]
REPORT_TXT = os.environ["REPORT_TXT"]

TEST_YEAR_START_T = int(os.environ["TEST_YEAR_START_T"])
TEST_YEAR_END_T = int(os.environ["TEST_YEAR_END_T"])

TIME_DEPTH = int(os.environ["TIME_DEPTH"])
T_STRIDE = int(os.environ["T_STRIDE"])
XY_STRIDE = int(os.environ["XY_STRIDE"])
CROP_SIZE = int(os.environ["CROP_SIZE"])

ds = xr.open_zarr(ZARR_PATH)

y_size = int(ds.sizes["y"])
x_size = int(ds.sizes["x"])
time_size = int(ds.sizes["time"])

if TEST_YEAR_END_T + TIME_DEPTH > time_size:
    raise ValueError(
        f"TEST_YEAR_END_T={TEST_YEAR_END_T} with TIME_DEPTH={TIME_DEPTH} exceeds time_size={time_size}"
    )

x_values = list(range(0, x_size - CROP_SIZE + 1, XY_STRIDE))
y_values = list(range(0, y_size - CROP_SIZE + 1, XY_STRIDE))
t_values = list(range(TEST_YEAR_START_T, TEST_YEAR_END_T + 1, T_STRIDE))

n_rows = 0

with open(OUTPUT_CSV, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["t", "x", "y"])

    for t in t_values:
        for y in y_values:
            for x in x_values:
                writer.writerow([t, x, y])
                n_rows += 1

with open(REPORT_TXT, "w") as f:
    f.write("Raw 2021 test CSV\n")
    f.write(f"ZARR_PATH: {ZARR_PATH}\n")
    f.write(f"OUTPUT_CSV: {OUTPUT_CSV}\n")
    f.write(f"time_size: {time_size}\n")
    f.write(f"y_size: {y_size}\n")
    f.write(f"x_size: {x_size}\n")
    f.write(f"TEST_YEAR_START_T: {TEST_YEAR_START_T}\n")
    f.write(f"TEST_YEAR_END_T: {TEST_YEAR_END_T}\n")
    f.write(f"TIME_DEPTH: {TIME_DEPTH}\n")
    f.write(f"T_STRIDE: {T_STRIDE}\n")
    f.write(f"XY_STRIDE: {XY_STRIDE}\n")
    f.write(f"CROP_SIZE: {CROP_SIZE}\n")
    f.write(f"n_t_values: {len(t_values)}\n")
    f.write(f"n_y_values: {len(y_values)}\n")
    f.write(f"n_x_values: {len(x_values)}\n")
    f.write(f"Total rows: {n_rows}\n")

print("Done.")
print(f"time_size={time_size}, y_size={y_size}, x_size={x_size}")
print(f"n_t_values={len(t_values)}")
print(f"n_y_values={len(y_values)}")
print(f"n_x_values={len(x_values)}")
print(f"Total rows written: {n_rows}")
print(f"CSV: {OUTPUT_CSV}")
print(f"Report: {REPORT_TXT}")
PY