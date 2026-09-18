#!/bin/bash
#SBATCH --job-name=probe_batch
#SBATCH --cpus-per-task=12
#SBATCH --partition=hopper_gpu
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
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
python -c "import torch; print('cuda available', torch.cuda.is_available()); print('cuda count', torch.cuda.device_count())"
nvidia-smi

BATCHES=(700)

for B in "${BATCHES[@]}"; do
  echo "=============================="
  echo "Testing batch size $B"
  echo "=============================="
  if python scripts/01_probe_convgru_batch_size.py       --config src/new_config_data_sampled.yaml       --region belgium       --batch "$B"       --workers 8       --tin 4       --tout 20       --crop 256       --steps 100       --precision 16-mixed       --use_dbz_norm       --allowed_nan_fraction 1.0
  then
      echo "BATCH $B OK"
  else
      echo "BATCH $B FAILED"
      exit 1
  fi
done
