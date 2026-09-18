#!/bin/bash
#SBATCH --job-name=unetens0
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=80:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch/2.6.0-foss-2024a-CUDA-12.6.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"

ZARR_PATH="/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"

EVENTS_CSV="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/events_2021_6hourly.csv"

CKPT="$VSC_DATA/code/mlcast-unet/logs/stoch_unet_full_crop128_b32/version_0/checkpoints/epoch=0-step=222943-v1.ckpt"

PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"

OUT="$PRED_ROOT/unet_ens/unet_ens_full_2021_epoch0.zarr"

mkdir -p "$PRED_ROOT/unet_ens"

echo "============================================================"
echo "U-Net ensemble full — final 2021 inference"
echo "Checkpoint: $CKPT"
echo "Output:     $OUT"
echo "============================================================"

test -f "$CKPT"
test -f "$EVENTS_CSV"
test -d "$ZARR_PATH"

nvidia-smi

python - <<PY
import torch

p = "$CKPT"
ckpt = torch.load(
    p,
    map_location="cpu",
    weights_only=False,
)

print("checkpoint epoch:", ckpt.get("epoch"))
print("global step:", ckpt.get("global_step"))
print("CUDA available:", torch.cuda.is_available())

assert int(ckpt.get("epoch")) == 0
assert torch.cuda.is_available()
PY

python "$CODE_DIR/export_predictions_2021_zarr.py" \
  --model unet_ens \
  --split full \
  --events-csv "$EVENTS_CSV" \
  --out-zarr "$OUT" \
  --zarr-path "$ZARR_PATH" \
  --var-name "precip_intensity_EDK" \
  --checkpoint "$CKPT" \
  --n-members 20 \
  --tin 4 \
  --tout 20 \
  --target-size 704 \
  --unet-noise-channels 4 \
  --unet-root "$VSC_DATA/code/mlcast-unet" \
  --seed 42 \
  --overwrite
