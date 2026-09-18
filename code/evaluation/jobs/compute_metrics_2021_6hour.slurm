#!/bin/bash
#SBATCH --job-name=met2021yr
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a
source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"
OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"
PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"
OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/metrics"

python "$CODE_DIR/compute_metrics_from_prediction_zarr.py" \
  --store-mode yearly \
  --observation-root "$OBS_ROOT" \
  --prediction-root "$PRED_ROOT" \
  --models convgru convgru_ens unet unet_ens pysteps ldcast \
  --split full \
  --output-dir "$OUT_DIR"
