#!/bin/bash
#SBATCH --job-name=pred2021
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=72G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${MODEL:?Set MODEL=convgru|convgru_ens|unet|unet_ens|pysteps|ldcast}"

SPLIT="${SPLIT:-full}"
CHECKPOINT="${CHECKPOINT:-}"
N_MEMBERS="${N_MEMBERS:-}"

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"
ZARR_PATH="/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"
EVENTS_CSV="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/events_2021_6hourly.csv"
PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"

LDCAST_EXTRA_ARGS=()

if [[ "$MODEL" == "ldcast" ]]; then
  if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: MODEL=ldcast requires CHECKPOINT=/path/to/generator.ckpt"
    exit 1
  fi

  AUTOENC="${AUTOENC:-}"

  if [[ -z "$AUTOENC" ]]; then
    if [[ "$SPLIT" == "10pct" ]]; then
      AUTOENC="$VSC_DATA/code/mlcast-ldcast/results/ldcast/autoenc_10pct_crop256_b32/epoch029.ckpt"
    else
      AUTOENC="$VSC_DATA/code/mlcast-ldcast/results/ldcast/autoenc_full_crop256_b32/epoch002.ckpt"
    fi
  fi

  LDCAST_MODEL_DIR="${LDCAST_MODEL_DIR:-$(dirname "$CHECKPOINT")}"
  LDC_PLMS_STEPS="${LDC_PLMS_STEPS:-100}"

  LDCAST_EXTRA_ARGS+=(
    --ldcast-autoenc-weights-fn "$AUTOENC"
    --ldcast-model-dir "$LDCAST_MODEL_DIR"
    --ldcast-plms-steps "$LDC_PLMS_STEPS"
  )

  echo "LDCast autoencoder: $AUTOENC"
  echo "LDCast generator: $CHECKPOINT"
  echo "LDCast model dir: $LDCAST_MODEL_DIR"
  echo "LDCast PLMS steps: $LDC_PLMS_STEPS"
fi

ARGS=(
  --model "$MODEL"
  --split "$SPLIT"
  --events-csv "$EVENTS_CSV"
  --out-zarr "$PRED_ROOT/$MODEL/${MODEL}_${SPLIT}_2021.zarr"
  --zarr-path "$ZARR_PATH"
  --var-name "precip_intensity_EDK"
  --overwrite
)

if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

if [[ -n "$N_MEMBERS" ]]; then
  ARGS+=(--n-members "$N_MEMBERS")
fi

ARGS+=("${LDCAST_EXTRA_ARGS[@]}")

echo "Running:"
echo python "$CODE_DIR/export_predictions_2021_zarr.py" "${ARGS[@]}"

python "$CODE_DIR/export_predictions_2021_zarr.py" "${ARGS[@]}"