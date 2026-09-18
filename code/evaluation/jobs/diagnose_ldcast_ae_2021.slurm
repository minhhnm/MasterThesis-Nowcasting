#!/bin/bash
#SBATCH --job-name=ae21diag
#SBATCH --partition=ampere_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

echo "============================================================"
echo "LDCast AE 2021 temporal diagnostic"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "============================================================"

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

cd "$VSC_DATA/code/mlcast_eval2021_6hourly"

AE_NAMES=(
    "10pct"
    "full"
)

AE="${AE_NAMES[$SLURM_ARRAY_TASK_ID]}"

OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"

BELGIUM_MASK="$VSC_DATA/belgium_mask_700.npy"

OUT_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly/results/ldcast_ae_2021_diagnostic"

mkdir -p "$OUT_DIR"

echo "AE: $AE"
echo "Observation root: $OBS_ROOT"
echo "Output: $OUT_DIR"

nvidia-smi

python diagnose_ldcast_ae_2021.py \
    --ae "$AE" \
    --observation-root "$OBS_ROOT" \
    --belgium-mask "$BELGIUM_MASK" \
    --mlcast-dir "$VSC_DATA/code/mlcast-ldcast" \
    --output-dir "$OUT_DIR"

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
