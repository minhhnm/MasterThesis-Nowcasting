#!/bin/bash
#SBATCH --job-name=pred2021m2
#SBATCH --cpus-per-task=32
#SBATCH --partition=zen5_mpi
#SBATCH --mem=96G
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

MODEL="${1:?Missing MODEL}"
SPLIT="${2:?Missing SPLIT}"
CHECKPOINT="${3:?Missing CHECKPOINT}"
N_MEMBERS="${4:-20}"
MONTH_START="${5:?Missing MONTH_START}"
MONTH_END="${6:?Missing MONTH_END}"

echo "MODEL=$MODEL"
echo "SPLIT=$SPLIT"
echo "CHECKPOINT=$CHECKPOINT"
echo "N_MEMBERS=$N_MEMBERS"
echo "MONTH_START=$MONTH_START"
echo "MONTH_END=$MONTH_END"

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"
ZARR_PATH="/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"
EVENTS_CSV="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/events_2021_6hourly.csv"
PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"

mkdir -p "$PRED_ROOT/$MODEL"

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

echo "Processing months ${MONTH_START}-${MONTH_END}"

for MONTH in $(seq "$MONTH_START" "$MONTH_END"); do
  MONTH2=$(printf "%02d" "$MONTH")
  OUT_ZARR="$PRED_ROOT/$MODEL/${MODEL}_${SPLIT}_2021_month${MONTH2}.zarr"

  echo "============================================================"
  echo "Running month $MONTH2"
  echo "Output: $OUT_ZARR"
  echo "============================================================"

  ARGS=(
    --model "$MODEL"
    --split "$SPLIT"
    --events-csv "$EVENTS_CSV"
    --month "$MONTH"
    --out-zarr "$OUT_ZARR"
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

  echo python "$CODE_DIR/export_predictions_2021_zarr.py" "${ARGS[@]}"
  python "$CODE_DIR/export_predictions_2021_zarr.py" "${ARGS[@]}"

  echo "Finished month $MONTH2"
done

echo "All requested months finished."