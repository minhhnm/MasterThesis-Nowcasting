#!/bin/bash
#SBATCH --job-name=qualtestbe
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

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

SCRIPT="$CODE_DIR/select_qualitative_test_events_belgium.py"

OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"

BELGIUM_MASK="$VSC_DATA/belgium_mask_700.npy"

OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/evaluation/qualitative_test_event_selection_belgium"


echo "============================================================"
echo "2021 qualitative-event selection over Belgium"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Observation root: $OBS_ROOT"
echo "Belgium mask: $BELGIUM_MASK"
echo "Output: $OUT_DIR"
echo "============================================================"


test -f "$SCRIPT"
test -f "$BELGIUM_MASK"

for M in $(seq -w 1 12); do

    STORE="$OBS_ROOT/observations_2021_month${M}.zarr"

    if [ ! -d "$STORE" ]; then
        echo "ERROR: missing $STORE"
        exit 1
    fi

done

mkdir -p "$OUT_DIR"


python "$SCRIPT" \
    --observation-root "$OBS_ROOT" \
    --belgium-mask "$BELGIUM_MASK" \
    --output-dir "$OUT_DIR" \
    --selection-separation-steps 288 \
    --top-k 20 \
    --flood-start "2021-07-13 00:00:00" \
    --flood-end "2021-07-17 00:00:00"


echo "============================================================"
echo "Finished"
echo "End: $(date)"
echo "============================================================"