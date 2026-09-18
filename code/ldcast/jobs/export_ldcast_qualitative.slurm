#!/bin/bash
#SBATCH --job-name=ldqual
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-7%2
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

cd "$VSC_DATA/code/mlcast-ldcast"

export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ============================================================
# Final qualitative 2021 test events
# ============================================================

EVENTS="465264,428616,473904,476568"

IFS=',' read -r -a EVENT_ARRAY <<< "$EVENTS"

if [[ "${#EVENT_ARRAY[@]}" -ne 4 ]]; then
    echo "ERROR: expected exactly four qualitative events."
    exit 1
fi


# ============================================================
# Map array task -> event + split
#
# task 0 = event 1, 10%
# task 1 = event 1, full
# task 2 = event 2, 10%
# task 3 = event 2, full
# ...
# task 7 = event 4, full
# ============================================================

EVENT_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
SPLIT_INDEX=$((SLURM_ARRAY_TASK_ID % 2))

T_START="${EVENT_ARRAY[$EVENT_INDEX]}"


# ============================================================
# Model configuration
# ============================================================

if [[ "$SPLIT_INDEX" -eq 0 ]]; then

    SPLIT_NAME="10pct"
    EPOCH="029"

    AUTOENC="$VSC_DATA/code/mlcast-ldcast/results/ldcast/autoenc_10pct_crop256_b32/epoch029.ckpt"

    CKPT="$VSC_SCRATCH/mlcast_runs_archive/ldcast/genforecast_10pct_crop128_ae256_b32/epoch029.ckpt"

    MODEL_DIR="$VSC_SCRATCH/mlcast_runs_archive/ldcast/genforecast_10pct_crop128_ae256_b32"

else

    SPLIT_NAME="full"
    EPOCH="002"

    AUTOENC="$VSC_DATA/code/mlcast-ldcast/results/ldcast/autoenc_full_crop256_b32/epoch002.ckpt"

    CKPT="$VSC_DATA/code/mlcast-ldcast/results/ldcast/genforecast_full_crop128_ae256_b32/epoch002.ckpt"

    MODEL_DIR="$VSC_DATA/code/mlcast-ldcast/results/ldcast/genforecast_full_crop128_ae256_b32"

fi


# ============================================================
# Output
# ============================================================

OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/ldcast_visuals"

OUT_NPZ="$OUT_DIR/ldcast_${SPLIT_NAME}_epoch${EPOCH}_t${T_START}_m20_plms100.npz"

mkdir -p "$OUT_DIR"


echo "============================================================"
echo "LDCast qualitative inference"
echo "Node:        $(hostname)"
echo "Task:        $SLURM_ARRAY_TASK_ID"
echo "Event index: $EVENT_INDEX"
echo "Event:       $T_START"
echo "Split:       $SPLIT_NAME"
echo "Checkpoint:  $CKPT"
echo "Autoencoder: $AUTOENC"
echo "Output:      $OUT_NPZ"
echo "Start:       $(date)"
echo "============================================================"


# ============================================================
# Sanity checks
# ============================================================

nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: no usable CUDA GPU")
PY

test -f "$AUTOENC"
test -f "$CKPT"


# ============================================================
# Skip if already completed
# ============================================================

if [[ -f "$OUT_NPZ" ]]; then
    echo "Output already exists:"
    echo "$OUT_NPZ"
    echo "Skipping this task."
    exit 0
fi


# ============================================================
# Run inference
# ============================================================

python scripts/Minh_export_ldcast_npz.py \
    --data_config="$VSC_DATA/code/mlcast-ldcast/src/new_config_data_sampled.yaml" \
    --region="belgium" \
    --autoenc_weights_fn="$AUTOENC" \
    --ckpt_path="$CKPT" \
    --model_dir="$MODEL_DIR" \
    --out_npz="$OUT_NPZ" \
    --tin=4 \
    --future_timesteps=20 \
    --target_size=704 \
    --t_start="$T_START" \
    --lr=1e-5 \
    --precision="bf16-mixed" \
    --n_members=20 \
    --sampling_steps=100


echo "============================================================"
echo "Finished"
echo "Event:  $T_START"
echo "Split:  $SPLIT_NAME"
echo "Output: $OUT_NPZ"
echo "End:    $(date)"
echo "============================================================"