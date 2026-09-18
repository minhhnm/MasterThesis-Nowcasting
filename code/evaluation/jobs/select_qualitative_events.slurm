#!/bin/bash
#SBATCH --job-name=qualevents
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

source "$VSC_DATA/venvs/mlcast_ldcast/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

CODE_DIR="$VSC_DATA/code/mlcast_eval2021_6hourly"
OUT_DIR="$VSC_DATA/code/mlcast-ldcast/results/evaluation/qualitative_event_selection"

python "$CODE_DIR/select_qualitative_events.py" \
  --zarr-path "/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr" \
  --var-name "precip_intensity_EDK" \
  --val-csv "$VSC_DATA/val_datacubes.csv" \
  --output-dir "$OUT_DIR" \
  --tin 4 \
  --tout 20 \
  --candidate-spacing-steps 72 \
  --selection-separation-steps 288
