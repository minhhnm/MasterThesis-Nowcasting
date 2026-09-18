#!/bin/bash
#SBATCH --job-name=metue20
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=hopper_gpu
#SBATCH --mem=128G
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ============================================================
# Environment
# ============================================================

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

# ============================================================
# Configuration
# ============================================================

MODEL="unet_ens"
SPLIT="full"

# ============================================================
# Paths
# ============================================================

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"

METRIC_SCRIPT="$CODE_DIR/compute_metrics_from_prediction_zarr_belgium_v3.py"

BELGIUM_MASK="$VSC_DATA/belgium_mask_700.npy"

OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"

PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"

PRED_STORE="$PRED_ROOT/$MODEL/${MODEL}_${SPLIT}_2021.zarr"

OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/metrics_belgium_${MODEL}_${SPLIT}"

# ============================================================
# Basic checks
# ============================================================

if [[ ! -f "$METRIC_SCRIPT" ]]; then
    echo "ERROR: metric script not found:"
    echo "$METRIC_SCRIPT"
    exit 1
fi

if [[ ! -f "$BELGIUM_MASK" ]]; then
    echo "ERROR: Belgium mask not found:"
    echo "$BELGIUM_MASK"
    exit 1
fi

if [[ ! -d "$OBS_ROOT/observations_2021.zarr" ]]; then
    echo "ERROR: yearly observation Zarr not found:"
    echo "$OBS_ROOT/observations_2021.zarr"
    exit 1
fi

if [[ ! -d "$PRED_STORE" ]]; then
    echo "ERROR: yearly forecast Zarr not found:"
    echo "$PRED_STORE"
    exit 1
fi

# ============================================================
# Verify forecast store
# ============================================================

python - "$PRED_STORE" <<'PY'
import sys
import zarr

path = sys.argv[1]

print("Checking forecast store:")
print(path)

g = zarr.open_group(path, mode="r")

required = ["forecast", "t_start", "start_time"]

for key in required:
    if key not in g:
        raise RuntimeError(
            f"Required array '{key}' missing from {path}"
        )

shape = g["forecast"].shape

print("forecast shape =", shape)

if len(shape) != 5:
    raise RuntimeError(
        "Expected forecast shape "
        "[event, member, lead, y, x], "
        f"got {shape}"
    )

if shape[1] != 20:
    raise RuntimeError(
        f"Expected 20 ensemble members, got {shape[1]}"
    )

if shape[2] != 20:
    raise RuntimeError(
        f"Expected 20 forecast lead times, got {shape[2]}"
    )

if g["t_start"].shape[0] != shape[0]:
    raise RuntimeError(
        f"t_start has {g['t_start'].shape[0]} entries "
        f"but forecast has {shape[0]} events"
    )

if g["start_time"].shape[0] != shape[0]:
    raise RuntimeError(
        f"start_time has {g['start_time'].shape[0]} entries "
        f"but forecast has {shape[0]} events"
    )

print("Forecast store passed.")
print("Events:  ", shape[0])
print("Members: ", shape[1])
print("Leads:   ", shape[2])
print("Spatial: ", shape[3:])
PY

# ============================================================
# Run metrics
# ============================================================

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "U-Net ensemble 2021 evaluation"
echo "MODEL:             $MODEL"
echo "SPLIT:             $SPLIT"
echo "STORE_MODE:        yearly"
echo "EXPECTED_MEMBERS:  20"
echo "FORECAST:          $PRED_STORE"
echo "OBSERVATIONS:      $OBS_ROOT/observations_2021.zarr"
echo "BELGIUM_MASK:      $BELGIUM_MASK"
echo "OUTPUT:            $OUT_DIR"
echo "START:             $(date)"
echo "============================================================"

python "$METRIC_SCRIPT" \
    --store-mode yearly \
    --observation-root "$OBS_ROOT" \
    --prediction-root "$PRED_ROOT" \
    --models "$MODEL" \
    --split "$SPLIT" \
    --evaluation-domain belgium \
    --belgium-mask "$BELGIUM_MASK" \
    --output-dir "$OUT_DIR"

echo "============================================================"
echo "Finished."
echo "End:    $(date)"
echo "Output: $OUT_DIR"
echo "============================================================"