#!/bin/bash

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch/2.6.0-foss-2024a-CUDA-12.6.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

source $VSC_DATA/venvs/mlcast_pl/bin/activate
cd $VSC_DATA/code/mlcast-ldcast
export PYTHONPATH=$PWD/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print('torch', torch.__version__)"
python -c "import pytorch_lightning; print('pl', pytorch_lightning.__version__)"
python -c "import torch; print('cuda available', torch.cuda.is_available()); print('cuda count', torch.cuda.device_count())"
nvidia-smi

python scripts/02_train_convgru_epoch.py \
  --config src/new_config_data_sampled_10pct.yaml \
  --region belgium \
  --out_dir results/convgru/ensemble_10pct_run1 \
  --tin 4 \
  --tout 20 \
  --crop 256 \
  --batch 32 \
  --workers 9 \
  --max_epochs 30 \
  --precision 16-mixed \
  --use_dbz_norm \
  --allowed_nan_fraction 1.0 \
  --lr 1e-3 \
  --scheduler_factor 0.5 \
  --scheduler_patience 1 \
  --min_lr 1e-6 \
  --loss_name crps \
  --ensemble_members 4 \
  "$@"
