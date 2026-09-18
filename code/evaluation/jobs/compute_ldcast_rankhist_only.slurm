#!/bin/bash
#SBATCH --job-name=rankld20
#SBATCH --cpus-per-task=16
#SBATCH --partition=zen5_mpi
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SPLIT="${1:?Usage: sbatch compute_ldcast_rankhist_only.slurm 10pct|full}"

if [[ "$SPLIT" != "10pct" && "$SPLIT" != "full" ]]; then
    echo "ERROR: split must be 10pct or full"
    exit 1
fi

mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"

SCRIPT="$CODE_DIR/compute_ldcast_rankhist_only.py"

BELGIUM_MASK="$VSC_DATA/belgium_mask_700.npy"

OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"

PRED_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/predictions"

OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/metrics_belgium_ldcast_${SPLIT}"

test -f "$SCRIPT"
test -f "$BELGIUM_MASK"
test -d "$OBS_ROOT"
test -d "$PRED_ROOT/ldcast"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "LDCast rank histogram only"
echo "Split:       $SPLIT"
echo "Members:     20"
echo "Leads:       +5, +50, +100 min"
echo "X_min:       0.1 mm/h"
echo "Output:      $OUT_DIR"
echo "Host:        $(hostname)"
echo "============================================================"


# ------------------------------------------------------------
# Preserve the old rank-histogram files once.
# ------------------------------------------------------------

if [[ -f "$OUT_DIR/rank_histograms.csv" && \
      ! -f "$OUT_DIR/rank_histograms_old.csv" ]]; then

    cp \
        "$OUT_DIR/rank_histograms.csv" \
        "$OUT_DIR/rank_histograms_old.csv"
fi


if [[ -f "$OUT_DIR/summary_rank_histograms_by_lead.csv" && \
      ! -f "$OUT_DIR/summary_rank_histograms_by_lead_old.csv" ]]; then

    cp \
        "$OUT_DIR/summary_rank_histograms_by_lead.csv" \
        "$OUT_DIR/summary_rank_histograms_by_lead_old.csv"
fi


python "$SCRIPT" \
    --prediction-root "$PRED_ROOT" \
    --observation-root "$OBS_ROOT" \
    --belgium-mask "$BELGIUM_MASK" \
    --split "$SPLIT" \
    --output-dir "$OUT_DIR"


echo "============================================================"
echo "Finished LDCast $SPLIT rank histograms"
echo "End: $(date)"
echo "============================================================"
