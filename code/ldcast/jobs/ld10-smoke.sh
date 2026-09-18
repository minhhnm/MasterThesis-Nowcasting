#!/bin/bash

set -euo pipefail

PROJECT_DIR="$VSC_DATA/code/mlcast-ldcast"

DATA_CONFIG="$PROJECT_DIR/src/new_config_data_sampled_10pct.yaml"

AUTOENC_CKPT="$PROJECT_DIR/results/ldcast/autoenc_10pct_crop256_b32/epoch029.ckpt"

GEN_CKPT="/scratch/brussel/114/vsc11442/mlcast_runs_archive/ldcast/genforecast_10pct_crop128_ae256_b32/epoch029.ckpt"

# Separate debug output: do not write into the production continuation folder.
MODEL_DIR="$PROJECT_DIR/results/ldcast/genforecast_10pct_resume_anansi_debug"

LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/ldcast_resume_anansi_debug_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
mkdir -p "$MODEL_DIR"

module purge
module load Python/3.12.3-GCCcore-13.3.0

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

cd "$PROJECT_DIR"

export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "============================================================"
echo "LDCast generator resume debug test"
echo "Host:        $(hostname)"
echo "Start time:  $(date)"
echo "Autoencoder: $AUTOENC_CKPT"
echo "Generator:   $GEN_CKPT"
echo "Output:      $MODEL_DIR"
echo "Log:         $LOG_FILE"
echo "============================================================"

# Fail early if a required checkpoint is missing.
for f in \
    "$DATA_CONFIG" \
    "$AUTOENC_CKPT" \
    "$GEN_CKPT"
do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing required file:"
        echo "  $f"
        exit 1
    fi
done

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible in this browser session.")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
    "GiB",
)
print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

python scripts/Minh_train_genforecast.py \
  --data_config="$DATA_CONFIG" \
  --region="belgium" \
  --autoenc_weights_fn="$AUTOENC_CKPT" \
  --model_dir="$PROJECT_DIR/results/ldcast/genforecast_10pct_anansi_fresh_smoke_$(date +%Y%m%d_%H%M%S)" \
  --batch_size=1 \
  --num_workers=0 \
  --pin_memory=False \
  --tin=4 \
  --future_timesteps=20 \
  --crop_size=128 \
  --lr=1e-5 \
  --max_epochs=1 \
  --precision="bf16-mixed" \
  --limit_train_batches=1 \
  --limit_val_batches=1
  2>&1 | tee "$LOG_FILE"

echo
echo "============================================================"
echo "Debug test finished."
echo "Log: $LOG_FILE"
echo "============================================================"