#!/bin/bash
#SBATCH --job-name=split_train_by_year
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/split_train_by_year_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

export INPUT_CSV="/data/brussel/114/vsc11442/valid_datacubes_no_2021.csv"
export OUTDIR="/data/brussel/114/vsc11442/year_splits"
export SUFFIX="24x256x256_3x16x16_10000"

python - <<'PY'
import os
import csv
from datetime import datetime, timedelta

input_csv = os.environ["INPUT_CSV"]
outdir = os.environ["OUTDIR"]
suffix = os.environ["SUFFIX"]

os.makedirs(outdir, exist_ok=True)

base = datetime(2017, 1, 1)

year_to_range = {
    2017: ("2017-01-01", "2017-12-31"),
    2018: ("2018-01-01", "2018-12-31"),
    2019: ("2019-01-01", "2019-12-31"),
    2020: ("2020-01-01", "2020-12-31"),
    2022: ("2022-01-01", "2022-12-31"),
    2023: ("2023-01-01", "2023-11-30"),
}

writers = {}
files = {}
counts = {y: 0 for y in year_to_range}

with open(input_csv, newline="") as f_in:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames

    for year, (start_date, end_date) in year_to_range.items():
        out_csv = os.path.join(
            outdir,
            f"valid_datacubes_{start_date}-{end_date}_{suffix}.csv"
        )
        f_out = open(out_csv, "w", newline="")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        files[year] = f_out
        writers[year] = writer

    for row in reader:
        t = int(row["t"])
        year = (base + timedelta(minutes=5 * t)).year

        if year in writers:
            writers[year].writerow(row)
            counts[year] += 1

for f in files.values():
    f.close()

print("Done.")
for y in sorted(counts):
    print(y, counts[y])
PY
