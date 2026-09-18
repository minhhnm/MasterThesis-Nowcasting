#!/bin/bash
#SBATCH --job-name=metpy
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# Run once before sbatch if needed:
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

OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/metrics_belgium_pysteps_full"


# ============================================================
# Basic checks
# ============================================================

test -f "$METRIC_SCRIPT"
test -f "$BELGIUM_MASK"

python - <<'PY'
import scipy
print("SciPy:", scipy.__version__)
PY


# ============================================================
# Check all 12 monthly observation and PySTEPS stores
# ============================================================

echo "Checking monthly stores..."

for M in $(seq -w 1 12); do

    OBS="$OBS_ROOT/observations_2021_month${M}.zarr"
    PRED="$PRED_ROOT/pysteps/pysteps_full_2021_month${M}.zarr"

    echo "Month ${M}:"

    if [ ! -d "$OBS" ]; then
        echo "ERROR: missing observation store:"
        echo "  $OBS"
        exit 1
    fi

    if [ ! -d "$PRED" ]; then
        echo "ERROR: missing prediction store:"
        echo "  $PRED"
        exit 1
    fi

    echo "  observations OK"
    echo "  predictions  OK"

done

echo "All 12 monthly stores found."


# ============================================================
# Output
# ============================================================

mkdir -p "$OUT_DIR"


# ============================================================
# Belgium-only PySTEPS audit
# ============================================================

python "$METRIC_SCRIPT" \
    --store-mode monthly \
    --months 1 2 3 4 5 6 7 8 9 10 11 12 \
    --observation-root "$OBS_ROOT" \
    --prediction-root "$PRED_ROOT" \
    --models pysteps \
    --split full \
    --evaluation-domain belgium \
    --belgium-mask "$BELGIUM_MASK" \
    --output-dir "$OUT_DIR"


echo "============================================================"
echo "PySTEPS monthly Belgium audit finished"
echo "Output: $OUT_DIR"
echo "End: $(date)"
echo "============================================================"