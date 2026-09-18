#!/bin/bash
#SBATCH --job-name=metfullbe
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# The logs directory must already exist when sbatch is called:
# mkdir -p logs
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
METRIC_SCRIPT="$CODE_DIR/compute_metrics_from_prediction_zarr_belgium_v3.py"
BELGIUM_MASK="$VSC_DATA/belgium_mask_700.npy"
OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"
PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"

test -f "$METRIC_SCRIPT"
test -f "$BELGIUM_MASK"
test -d "$OBS_ROOT/observations_2021.zarr"

python - <<'PY'
import scipy
print("SciPy:", scipy.__version__)
PY

BASE_OUT="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/metrics_belgium_full"

for MODEL in convgru convgru_ens unet unet_ens pysteps ldcast; do
  YEARLY_STORE="$PRED_ROOT/$MODEL/${MODEL}_full_2021.zarr"
  MONTH01_STORE="$PRED_ROOT/$MODEL/${MODEL}_full_2021_month01.zarr"

  if [[ -d "$YEARLY_STORE" ]]; then
    STORE_MODE="yearly"
  elif [[ -d "$MONTH01_STORE" ]]; then
    STORE_MODE="monthly"
    for M in 01 02 03 04 05 06 07 08 09 10 11 12; do
      test -d "$PRED_ROOT/$MODEL/${MODEL}_full_2021_month${M}.zarr"
    done
  else
    echo "No complete yearly or monthly store found for $MODEL" >&2
    exit 1
  fi

  OUT_DIR="$BASE_OUT/${MODEL}_full"
  mkdir -p "$OUT_DIR"

  echo "Running MODEL=$MODEL STORE_MODE=$STORE_MODE OUT_DIR=$OUT_DIR"

  python "$METRIC_SCRIPT" \
    --store-mode "$STORE_MODE" \
    --observation-root "$OBS_ROOT" \
    --prediction-root "$PRED_ROOT" \
    --models "$MODEL" \
    --split full \
    --evaluation-domain belgium \
    --belgium-mask "$BELGIUM_MASK" \
    --output-dir "$OUT_DIR"
done
