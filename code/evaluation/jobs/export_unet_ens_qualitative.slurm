#!/bin/bash
#SBATCH --job-name=qual_unet20
#SBATCH --gpus=1
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ============================================================
# Environment
# ============================================================

source /data/brussel/114/vsc11442/venvs/mlcast_ldcast/bin/activate

cd /data/brussel/114/vsc11442/code/mlcast_eval2021_6hourly

mkdir -p logs

# ============================================================
# Required variables supplied with sbatch --export
# ============================================================

: "${SPLIT:?Set SPLIT=10pct or full}"
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${OUT_ZARR:?Set OUT_ZARR}"

# ============================================================
# Paths
# ============================================================

EVENTS_CSV="/data/brussel/114/vsc11442/code/mlcast_eval2021_6hourly/qualitative_events_2021.csv"

RADAR_ZARR="/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"

EXPORTER="/data/brussel/114/vsc11442/code/mlcast_eval2021_6hourly/export_predictions_2021_zarr.py"

# ============================================================
# Diagnostics
# ============================================================

echo "============================================================"
echo "U-Net ensemble qualitative inference"
echo "Host:        $(hostname)"
echo "Split:       $SPLIT"
echo "Checkpoint:  $CHECKPOINT"
echo "Output:      $OUT_ZARR"
echo "Members:     20"
echo "Events:      4"
echo "============================================================"

nvidia-smi

# ============================================================
# Run
# ============================================================

python "$EXPORTER" \
    --model unet_ens \
    --split "$SPLIT" \
    --events-csv "$EVENTS_CSV" \
    --out-zarr "$OUT_ZARR" \
    --zarr-path "$RADAR_ZARR" \
    --var-name precip_intensity_EDK \
    --checkpoint "$CHECKPOINT" \
    --n-members 20 \
    --tin 4 \
    --tout 20 \
    --target-size 704 \
    --device cuda \
    --overwrite

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
