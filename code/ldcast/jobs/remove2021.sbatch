#!/bin/bash
#SBATCH --job-name=radclim_remove_2021
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/remove_2021_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

export INPUT_CSV="/data/brussel/114/vsc11442/valid_datacubes_2017-01-01-2023-11-30_24x256x256_3x16x16_10000.csv"
export OUTPUT_CSV="/data/brussel/114/vsc11442/valid_datacubes_no_2021.csv"

python - <<'PY'
import os
import csv

input_csv = os.environ["INPUT_CSV"]
output_csv = os.environ["OUTPUT_CSV"]

T2021 = 420768
T2022 = 525888
TIME_DEPTH = 24

n_in = 0
n_out = 0

with open(input_csv, newline="") as f_in, open(output_csv, "w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        n_in += 1
        t = int(row["t"])

        # keep only cubes that do NOT touch 2021
        if (t <= T2021 - TIME_DEPTH) or (t >= T2022):
            writer.writerow(row)
            n_out += 1

print("source CSV:", input_csv)
print("original rows:", n_in)
print("rows after removing 2021:", n_out)
print("saved to:", output_csv)
PY
