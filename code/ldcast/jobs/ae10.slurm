#!/bin/bash
#SBATCH --job-name=ae10
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=110:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

source $VSC_DATA/venvs/mlcast_ldcast/bin/activate

cd $VSC_DATA/code/mlcast-ldcast
export PYTHONPATH=$PWD/src:$PWD/scripts:${PYTHONPATH:-}

nvidia-smi || true

python scripts/Minh_train_autoenc.py \
  --data_config="$VSC_DATA/code/mlcast-ldcast/src/new_config_data_sampled_10pct_scratch.yaml" \
  --region="belgium" \
  --model_dir="results/ldcast/autoenc_10pct_crop256_b32" \
  --batch_size=32 \
  --num_workers=10 \
  --pin_memory=True \
  --total_steps=4 \
  --autoenc_steps=4 \
  --crop_size=256 \
  --max_epochs=30 \
  --precision="16-mixed"
