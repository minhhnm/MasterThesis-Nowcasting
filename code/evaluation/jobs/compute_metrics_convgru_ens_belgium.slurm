#!/bin/bash
#SBATCH --job-name=metcge20
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

SPLIT="${1:?Usage: sbatch script.slurm 10pct|full}"

if [[ "$SPLIT" != "10pct" && "$SPLIT" != "full" ]]; then
    echo "ERROR: SPLIT must be 10pct or full"
    exit 1
fi

MODEL="convgru_ens"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"

METRIC_SCRIPT="$CODE_DIR/compute_metrics_from_prediction_zarr_belgium_v3.py"

BELGIUM_MASK="$VSC_DATA/belgium_mask_700.npy"

OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"

PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"

OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/metrics_belgium_${MODEL}_${SPLIT}_m20"

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

test -f "$METRIC_SCRIPT"
test -f "$BELGIUM_MASK"

if [[ ! -d "$OBS_ROOT/observations_2021.zarr" && \
      ! -d "$OBS_ROOT/observations_2021_month01.zarr" ]]; then
    echo "ERROR: no yearly or monthly observation Zarr found"
    exit 1
fi

# Make sure ALL twelve forecast stores exist.
for MONTH in {1..12}; do
    MM=$(printf "%02d" "$MONTH")

    STORE="$PRED_ROOT/$MODEL/${MODEL}_${SPLIT}_2021_month${MM}.zarr"

    if [[ ! -d "$STORE" ]]; then
        echo "ERROR: missing forecast store:"
        echo "$STORE"
        exit 1
    fi
done

# ------------------------------------------------------------
# Verify Zarr shape / ensemble size
# ------------------------------------------------------------

python - "$PRED_ROOT" "$MODEL" "$SPLIT" <<'PY'
import sys
import zarr

pred_root, model, split = sys.argv[1:4]

print("Checking monthly forecast stores...")

for month in range(1, 13):

    path = (
        f"{pred_root}/{model}/"
        f"{model}_{split}_2021_month{month:02d}.zarr"
    )

    g = zarr.open_group(path, mode="r")
    shape = g["forecast"].shape

    print(
        f"month {month:02d}: "
        f"forecast shape = {shape}"
    )

    if len(shape) != 5:
        raise RuntimeError(
            f"Unexpected forecast rank for {path}: {shape}"
        )

    if shape[1] != 20:
        raise RuntimeError(
            f"{path}: expected 20 members, got {shape[1]}"
        )

    if shape[2] != 20:
        raise RuntimeError(
            f"{path}: expected 20 lead times, got {shape[2]}"
        )

print("All 12 monthly stores passed.")
PY

# ------------------------------------------------------------
# Run metrics
# ------------------------------------------------------------

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "MODEL=$MODEL"
echo "SPLIT=$SPLIT"
echo "STORE_MODE=monthly"
echo "EXPECTED_MEMBERS=20"
echo "BELGIUM_MASK=$BELGIUM_MASK"
echo "OUT_DIR=$OUT_DIR"
echo "============================================================"

python "$METRIC_SCRIPT" \
    --store-mode monthly \
    --months 1 2 3 4 5 6 7 8 9 10 11 12 \
    --observation-root "$OBS_ROOT" \
    --prediction-root "$PRED_ROOT" \
    --models "$MODEL" \
    --split "$SPLIT" \
    --evaluation-domain belgium \
    --belgium-mask "$BELGIUM_MASK" \
    --output-dir "$OUT_DIR"

echo "============================================================"
echo "Finished."
echo "Output: $OUT_DIR"
echo "============================================================"