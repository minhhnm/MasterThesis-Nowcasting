#!/bin/bash
#SBATCH --job-name=obs2021mon
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
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
ZARR_PATH="/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"
EVENTS_CSV="$VSC_DATA/code/mlcast-ldcast/results/evaluation/test_2021_6hourly/events_2021_6hourly.csv"
OBS_ROOT="$VSC_SCRATCH/mlcast_predictions/test_2021_6hourly/observations"

mkdir -p "$OBS_ROOT"

for M in {1..12}; do
  MM=$(printf "%02d" "$M")
  OUT="$OBS_ROOT/observations_2021_month${MM}.zarr"

  echo "============================================================"
  echo "Exporting monthly observations: month ${MM}"
  echo "Output: $OUT"
  echo "============================================================"

  python "$CODE_DIR/export_predictions_2021_zarr.py" \
    --model observations \
    --events-csv "$EVENTS_CSV" \
    --month "$M" \
    --out-zarr "$OUT" \
    --zarr-path "$ZARR_PATH" \
    --var-name "precip_intensity_EDK" \
    --overwrite
done
