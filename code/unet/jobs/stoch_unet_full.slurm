#!/bin/bash
#SBATCH --job-name=sunetfull
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

cd "$VSC_DATA/code/mlcast-unet"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY
nvidia-smi || true

python -m mlcast train \
  --config set:ckpt_path="'$VSC_DATA/code/mlcast-unet/logs/stoch_unet_full_crop128_b32/version_1/checkpoints/epoch=2-step=668829.ckpt'" \
  --config set:data.dataset_factory.zarr_path="'/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr'" \
  --config set:data.dataset_factory.csv_path="'/data/brussel/114/vsc11442/train_datacubes.csv'" \
  --config set:data.dataset_factory.standard_names="['precip_intensity_EDK']" \
  --config set:data.dataset_factory.normalization_names="['rainfall_rate']" \
  --config set:data.dataset_factory.input_steps=4 \
  --config set:data.dataset_factory.forecast_steps=20 \
  --config set:data.dataset_factory.width=128 \
  --config set:data.dataset_factory.height=128 \
  --config "fiddler:use_stochastic_unet(base_channels=32,num_blocks=4,noise_channels=4,ensemble_size=4)" \
  --config set:pl_module.optimizer.lr=1e-5 \
  --config set:data.batch_size=32 \
  --config set:data.num_workers=10 \
  --config set:trainer.max_epochs=5 \
  --config set:trainer.num_sanity_val_steps=0 \
  --config set:trainer.limit_test_batches=0 \
  --config set:trainer.precision="'bf16-mixed'" \
  --config set:trainer.logger.name="'stoch_unet_full_crop128_b32'"
