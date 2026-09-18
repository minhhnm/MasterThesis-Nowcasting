#!/bin/bash
#SBATCH --job-name=mk2021ev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"
ZARR_PATH="/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"
OUT_CSV="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/events_2021_6hourly.csv"

python "$CODE_DIR/build_2021_6hour_events.py" \
  --zarr-path "$ZARR_PATH" \
  --var-name "precip_intensity_EDK" \
  --output-csv "$OUT_CSV" \
  --tin 4 \
  --tout 20 \
  --year 2021
