#!/bin/bash
#SBATCH --job-name=ldfull
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

source $VSC_DATA/venvs/mlcast_ldcast/bin/activate

cd $VSC_DATA/code/mlcast-ldcast
export PYTHONPATH=$PWD/src:$PWD/scripts:${PYTHONPATH:-}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
nvidia-smi || true

python scripts/Minh_train_genforecast.py \
  --data_config="$VSC_DATA/code/mlcast-ldcast/src/new_config_data_sampled.yaml" \
  --region="belgium" \
  --autoenc_weights_fn="$VSC_DATA/code/mlcast-ldcast/results/ldcast/autoenc_full_crop256_b32/epoch002.ckpt" \
  --model_dir="results/ldcast/genforecast_full_crop128_ae256_b32_new" \
  --batch_size=32 \
  --num_workers=10 \
  --pin_memory=True \
  --tin=4 \
  --future_timesteps=20 \
  --crop_size=128 \
  --lr=1e-5 \
  --max_epochs=3 \
  --precision="bf16-mixed"\
  --ckpt_path="results/ldcast/genforecast_full_crop128_ae256_b32_new/epoch001.ckpt"
