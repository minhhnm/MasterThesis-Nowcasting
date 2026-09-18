#!/bin/bash
#SBATCH --job-name=ldfullvis
#SBATCH --partition=ampere_gpu
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=72G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

source $VSC_DATA/venvs/mlcast_ldcast/bin/activate

cd $VSC_DATA/code/mlcast-ldcast
export PYTHONPATH=$PWD/src:$PWD/scripts:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi || true

OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/ldcast_visuals"
mkdir -p "$OUT_DIR"

python scripts/Minh_export_ldcast_npz.py \
  --data_config="$VSC_DATA/code/mlcast-ldcast/src/new_config_data_sampled.yaml" \
  --region="belgium" \
  --autoenc_weights_fn="$VSC_DATA/code/mlcast-ldcast/results/ldcast/autoenc_full_crop256_b32/epoch002.ckpt" \
  --ckpt_path="$VSC_DATA/code/mlcast-ldcast/results/ldcast/genforecast_full_crop128_ae256_b32/epoch002.ckpt" \
  --model_dir="$VSC_DATA/code/mlcast-ldcast/results/ldcast/genforecast_full_crop128_ae256_b32" \
  --out_npz="$OUT_DIR/ldcast_full_epoch002_t704352_m20_plms100.npz" \
  --tin=4 \
  --future_timesteps=20 \
  --target_size=704 \
  --t_start=704352 \
  --lr=1e-5 \
  --precision="bf16-mixed" \
  --n_members=20 \
  --sampling_steps=100