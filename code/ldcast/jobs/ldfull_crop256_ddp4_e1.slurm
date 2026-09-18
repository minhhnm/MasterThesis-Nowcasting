#!/bin/bash
#SBATCH --job-name=ldc256ddp
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=192G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

cd "$VSC_DATA/code/mlcast-ldcast"
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== SLURM INFO ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "===== CUDA CHECK BEFORE SRUN ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

AUTOENC_WEIGHTS="$VSC_DATA/code/mlcast-ldcast/results/ldcast/autoenc_full_crop256_b32/epoch002.ckpt"
MODEL_DIR="$VSC_DATA/code/mlcast-ldcast/results/ldcast/genforecast_full_crop256_b32_ddp4"

if [[ ! -f "$AUTOENC_WEIGHTS" ]]; then
  echo "ERROR: AE checkpoint not found:"
  echo "$AUTOENC_WEIGHTS"
  exit 1
fi

mkdir -p "$MODEL_DIR"

echo "Using AE checkpoint:"
echo "$AUTOENC_WEIGHTS"

echo "Saving generator to:"
echo "$MODEL_DIR"

python scripts/Minh_train_genforecast_ddp.py \
  --data_config="$VSC_DATA/code/mlcast-ldcast/src/new_config_data_sampled_scratch.yaml" \
  --region="belgium" \
  --autoenc_weights_fn="$AUTOENC_WEIGHTS" \
  --model_dir="$MODEL_DIR" \
  --future_timesteps=20 \
  --crop_size=256 \
  --batch_size=8 \
  --num_workers=4 \
  --max_epochs=3 \
  --lr=1e-5 \
  --precision="bf16-mixed"
