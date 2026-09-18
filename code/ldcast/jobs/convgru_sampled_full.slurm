#!/bin/bash
#SBATCH --job-name=convgru_run1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
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
export PYTHONPATH=$PWD/src:$PYTHONPATH

python -c "import torch; print('torch', torch.__version__)"
python -c "import pytorch_lightning; print('pl', pytorch_lightning.__version__)"

python scripts/03_train_convgru_sampled.py \
  --config src/new_config_data_sampled.yaml \
  --region belgium \
  --out_dir results/convgru/run1 \
  --tin 4 \
  --tout 20 \
  --crop 256 \
  --batch 16 \
  --workers 16 \
  --precision 16-mixed \
  --use_dbz_norm \
  --allowed_nan_fraction 1.0
