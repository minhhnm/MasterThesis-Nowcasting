#!/bin/bash
# ConvGRU deterministic smoke test using the multiprocessing-safe dataset file.
#
# Run:
#   cd "$VSC_DATA/code/mlcast-ldcast"
#   bash jobs/smoke_convgru_safe_workers10.sh
#
# Useful options:
#   SPLIT=10pct WORKERS=10 BATCH=8 bash jobs/smoke_convgru_safe_workers10.sh
#   SPLIT=full  WORKERS=10 BATCH=8 bash jobs/smoke_convgru_safe_workers10.sh
#   SPLIT=10pct WORKERS=10 BATCH=32 bash jobs/smoke_convgru_safe_workers10.sh

set -euo pipefail

MLCAST_DIR="${MLCAST_DIR:-$VSC_DATA/code/mlcast-ldcast}"
cd "$MLCAST_DIR"

echo "============================================================"
echo "ConvGRU safe-multiprocessing smoke test"
echo "Node:       $(hostname)"
echo "PWD:        $(pwd)"
echo "Date:       $(date)"
echo "============================================================"

if command -v module >/dev/null 2>&1; then
  module purge
  module load Python/3.12.3-GCCcore-13.3.0
  module load PyTorch/2.6.0-foss-2024a-CUDA-12.6.0
  module load zarr/2.18.4-foss-2024a
  module load xarray/2024.11.0-gfbf-2024a
fi

if [[ -d "$VSC_DATA/venvs/mlcast_pl" ]]; then
  source "$VSC_DATA/venvs/mlcast_pl/bin/activate"
elif [[ -d "$VSC_DATA/venvs/mlcast_ldcast" ]]; then
  source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"
else
  echo "ERROR: no mlcast_pl or mlcast_ldcast venv found under $VSC_DATA/venvs"
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLIT="${SPLIT:-10pct}"       # 10pct or full
WORKERS="${WORKERS:-10}"      # default now tests workers=10
BATCH="${BATCH:-8}"           # use 8 first; then try 32
CROP="${CROP:-256}"
TIN="${TIN:-4}"
TOUT="${TOUT:-20}"

# Path to the multiprocessing-safe dataset file.
# Put the uploaded file here:
#   $VSC_DATA/code/mlcast-ldcast/radar_dataset_sampled_multiproc_safe.py
SAFE_DATASET="${SAFE_DATASET:-$MLCAST_DIR/src/radar_dataset_sampled_multiproc_safe.py}"

# Optional: just check this exists, but the smoke test does not import it.
SAFE_TRAIN_A="$MLCAST_DIR/scripts/02_train_convgru_epoch_multiproc_safe.py"
SAFE_TRAIN_B="$MLCAST_DIR/02_train_convgru_epoch_multiproc_safe.py"

if [[ ! -f "$SAFE_DATASET" ]]; then
  echo "ERROR: safe dataset file not found:"
  echo "  $SAFE_DATASET"
  echo
  echo "Copy/upload it to:"
  echo "  $MLCAST_DIR/radar_dataset_sampled_multiproc_safe.py"
  exit 1
fi

if [[ -f "$SAFE_TRAIN_A" ]]; then
  SAFE_TRAIN="$SAFE_TRAIN_A"
elif [[ -f "$SAFE_TRAIN_B" ]]; then
  SAFE_TRAIN="$SAFE_TRAIN_B"
else
  SAFE_TRAIN=""
fi

if [[ "$SPLIT" == "10pct" ]]; then
  if [[ -f "src/new_config_data_sampled_10pct.yaml" ]]; then
    BASE_CONFIG="src/new_config_data_sampled_10pct.yaml"
  else
    BASE_CONFIG="src/new_config_data_sampled_10pct_scratch.yaml"
  fi
else
  if [[ -f "src/new_config_data_sampled.yaml" ]]; then
    BASE_CONFIG="src/new_config_data_sampled.yaml"
  else
    BASE_CONFIG="src/new_config_data_sampled_scratch.yaml"
  fi
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG"
  echo "Available sampled configs:"
  ls -lh src/*sampled*.yaml || true
  exit 1
fi

TMP_CONFIG="/tmp/convgru_safe_smoke_${SPLIT}_$$_data.yaml"
ZARR_OLD="/scratch/brussel/114/vsc11442/radar_data/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"
ZARR_NEW="/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"

cp "$BASE_CONFIG" "$TMP_CONFIG"
sed -i "s#$ZARR_OLD#$ZARR_NEW#g" "$TMP_CONFIG"

echo "Environment:"
which python
python - <<'PY'
import sys
print("python", sys.executable, flush=True)
import torch
print("torch", torch.__version__, flush=True)
print("cuda available", torch.cuda.is_available(), flush=True)
print("cuda count", torch.cuda.device_count(), flush=True)
import pytorch_lightning as pl
print("pytorch_lightning", pl.__version__, flush=True)
PY

nvidia-smi || true

echo "Smoke settings:"
echo "SPLIT:        $SPLIT"
echo "BASE_CONFIG:  $BASE_CONFIG"
echo "TMP_CONFIG:   $TMP_CONFIG"
echo "SAFE_DATASET: $SAFE_DATASET"
echo "SAFE_TRAIN:   ${SAFE_TRAIN:-not found / not used in smoke}"
echo "WORKERS:      $WORKERS"
echo "BATCH:        $BATCH"
echo "CROP:         $CROP"

echo "============================================================"
echo "Starting Python smoke test. Outer timeout: 30 minutes."
echo "============================================================"

timeout 30m python - <<PY
import importlib.util
import os
import time
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlcast.modules import ConvGRU

CONFIG = "$TMP_CONFIG"
SAFE_DATASET = "$SAFE_DATASET"
REGION = "belgium"
TIN = int("$TIN")
TOUT = int("$TOUT")
CROP = int("$CROP")
BATCH = int("$BATCH")
WORKERS = int("$WORKERS")

# Load the safe dataset file explicitly by path.
# This avoids accidentally importing the old radar_dataset_sampled.py from the repo root.
spec = importlib.util.spec_from_file_location("radar_dataset_sampled_safe", SAFE_DATASET)
safe_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(safe_module)
RadarDatasetNew = safe_module.RadarDatasetNew

print("Using RadarDatasetNew from:", SAFE_DATASET, flush=True)

def radar_pair_collate(batch):
    x = torch.stack([sample["radar_past"] for sample in batch], dim=0)
    y = torch.stack([sample["radar_future"] for sample in batch], dim=0)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    y = y.permute(0, 2, 1, 3, 4).contiguous()
    return x, y

class ConvGRUForecastWrapper(nn.Module):
    def __init__(self, net: nn.Module, steps: int):
        super().__init__()
        self.net = net
        self.steps = steps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, self.steps)

print("Reading config:", CONFIG, flush=True)
with open(CONFIG, "r") as f:
    data_sources = yaml.safe_load(f)

data_sources[REGION]["radar"]["transform_precip_to_dbz"] = True

print("Radar config:", data_sources[REGION]["radar"], flush=True)

print("Building train dataset...", flush=True)
t0 = time.time()
train_ds = RadarDatasetNew(
    mode="train",
    regions=[REGION],
    is_primary_source=True,
    inputs=["radar_past"],
    targets=["radar_future"],
    data_sources=data_sources,
    number_of_past_radar_time_steps=TIN,
    number_of_future_radar_time_steps=TOUT,
    crop_size=CROP,
    allowed_nan_fraction=1.0,
)
print(f"Dataset built in {time.time() - t0:.2f}s; len={len(train_ds)}", flush=True)

print("Testing train_ds[0] in main process...", flush=True)
t0 = time.time()
sample = train_ds[0]
print(f"train_ds[0] loaded in {time.time() - t0:.2f}s", flush=True)
print("radar_past:", tuple(sample["radar_past"].shape), flush=True)
print("radar_future:", tuple(sample["radar_future"].shape), flush=True)
print("metadata:", sample["t"], sample["x"], sample["y"], flush=True)

loader_extra = {"persistent_workers": False}
if WORKERS > 0:
    loader_extra["timeout"] = 300
    loader_extra["prefetch_factor"] = 2

print("Building DataLoader...", flush=True)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
    collate_fn=radar_pair_collate,
    drop_last=True,
    **loader_extra,
)

print("Testing first DataLoader batch...", flush=True)
t0 = time.time()
xb, yb = next(iter(train_loader))
print(f"First batch loaded in {time.time() - t0:.2f}s", flush=True)
print("xb:", tuple(xb.shape), "yb:", tuple(yb.shape), flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("Testing ConvGRU forward/backward on GPU...", flush=True)
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

model = ConvGRUForecastWrapper(ConvGRU(), steps=TOUT).to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-5)

xb = xb.to(device, non_blocking=True)
yb = yb.to(device, non_blocking=True)

torch.cuda.synchronize()
t0 = time.time()
pred = model(xb)
loss = loss_fn(pred, yb)
loss.backward()
opt.step()
torch.cuda.synchronize()

print(f"GPU forward/backward finished in {time.time() - t0:.2f}s", flush=True)
print("pred:", tuple(pred.shape), "loss:", float(loss.detach().cpu()), flush=True)
print("SMOKE TEST PASSED", flush=True)
PY

echo "============================================================"
echo "Smoke test finished successfully."
echo "============================================================"
