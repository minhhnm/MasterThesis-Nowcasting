#!/bin/bash
#SBATCH --job-name=metunetens10pct
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"
OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"
PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"
OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/metrics_unet_ens_10pct"

echo "OBS_ROOT=$OBS_ROOT"
echo "PRED_ROOT=$PRED_ROOT"
echo "OUT_DIR=$OUT_DIR"

test -d "$OBS_ROOT/observations_2021.zarr"
test -d "$PRED_ROOT/unet_ens/unet_ens_10pct_2021.zarr"

mkdir -p "$OUT_DIR"

python "$CODE_DIR/compute_metrics_from_prediction_zarr.py" \
  --store-mode yearly \
  --observation-root "$OBS_ROOT" \
  --prediction-root "$PRED_ROOT" \
  --models unet_ens\
  --split 10pct \
  --output-dir "$OUT_DIR"
