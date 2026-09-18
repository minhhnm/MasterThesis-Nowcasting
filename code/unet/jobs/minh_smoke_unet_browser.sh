#!/bin/bash
set -euo pipefail

cd "$VSC_DATA/code/mlcast-unet"

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "===== LOCATION ====="
pwd

echo ""
echo "===== PYTHON / TORCH ====="
python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo ""
echo "===== GIT STATUS ====="
git status --short

echo ""
echo "===== CHECK UNET IMPORT AND FORWARD ====="
python - <<'PY'
import torch
from mlcast.models import UNetModel

model = UNetModel(
    input_channels=1,
    input_steps=4,
    base_channels=16,
    num_blocks=3,
)

x = torch.randn(2, 4, 1, 128, 128)
y = model(x, steps=20, ensemble_size=10)

print("input:", tuple(x.shape))
print("output:", tuple(y.shape))

assert tuple(y.shape) == (2, 20, 1, 128, 128)
print("UNet forward smoke OK")
PY

echo ""
echo "===== CHECK FIDDLE CONFIG ====="
python -m mlcast train \
  --config set:data.dataset_factory.input_steps=4 \
  --config set:data.dataset_factory.forecast_steps=20 \
  --config set:data.dataset_factory.width=128 \
  --config set:data.dataset_factory.height=128 \
  --config "fiddler:use_unet(base_channels=16,num_blocks=3)" \
  --config set:trainer.max_epochs=1 \
  --config set:trainer.limit_train_batches=2 \
  --config set:trainer.limit_val_batches=1 \
  --config set:data.batch_size=1 \
  --config set:data.num_workers=0 \
  --print_config_and_exit

echo ""
echo "===== CHECK NORMALIZATION_NAMES SUPPORT ====="
if grep -q "normalization_names" src/mlcast/data/source_data_datasets.py; then
    echo "normalization_names support exists."
else
    echo "normalization_names support does NOT exist yet."
    echo "Real RADCLIM training with standard_names=['precip_intensity_EDK'] will probably fail until we add it."
fi

echo ""
echo "===== DONE ====="
echo "Browser U-Net smoke passed."
