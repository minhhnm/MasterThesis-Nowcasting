#!/bin/bash
#SBATCH --job-name=convgru_full
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch/2.6.0-foss-2024a-CUDA-12.6.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

source $VSC_DATA/venvs/mlcast_pl/bin/activate
cd $VSC_DATA/code/mlcast-ldcast
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python -c "import torch; print('torch', torch.__version__)"
python -c "import pytorch_lightning; print('pl', pytorch_lightning.__version__)"
python -c "import torch; print('cuda available', torch.cuda.is_available()); print('cuda count', torch.cuda.device_count())"
nvidia-smi

python scripts/02_train_convgru_epoch_multiproc_safe.py \
  --config src/new_config_data_sampled.yaml \
  --region belgium \
  --out_dir results/convgru/convgru_full_b32_crop256 \
  --tin 4 \
  --tout 20 \
  --crop 256 \
  --batch 32 \
  --workers 10 \
  --max_epochs 5 \
  --precision 16-mixed \
  --use_dbz_norm \
  --allowed_nan_fraction 1.0 \
  --lr 1e-5 \
  --scheduler_factor 0.5 \
  --scheduler_patience 1 \
  --min_lr 1e-6 \
  --resume "$VSC_DATA/code/mlcast-ldcast/results/convgru/convgru_full_b32_crop256/latest-epoch02-step1021218.ckpt" \
  "$@"
