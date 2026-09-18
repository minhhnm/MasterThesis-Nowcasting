#!/bin/bash
set -euo pipefail

cd "$VSC_DATA/code/mlcast-unet"

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "===== LOCATION ====="
pwd

echo ""
echo "===== GIT STATUS ====="
git status --short

echo ""
echo "===== GPU ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

python -m mlcast train \
  --config set:data.dataset_factory.zarr_path="'/scratch/brussel/114/vsc11442/radar_data/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr'" \
  --config set:data.dataset_factory.csv_path="'/data/brussel/114/vsc11442/train_datacubes_10pct.csv'" \
  --config set:data.dataset_factory.standard_names="['precip_intensity_EDK']" \
  --config set:data.dataset_factory.normalization_names="['rainfall_rate']" \
  --config set:data.dataset_factory.input_steps=4 \
  --config set:data.dataset_factory.forecast_steps=20 \
  --config set:data.dataset_factory.width=128 \
  --config set:data.dataset_factory.height=128 \
  --config "fiddler:use_unet(base_channels=16,num_blocks=3)" \
  --config set:trainer.max_epochs=1 \
  --config set:trainer.limit_train_batches=2 \
  --config set:trainer.limit_val_batches=1 \
  --config set:trainer.limit_test_batches=1 \
  --config set:data.batch_size=1 \
  --config set:data.num_workers=0

echo ""
echo "===== DONE ====="