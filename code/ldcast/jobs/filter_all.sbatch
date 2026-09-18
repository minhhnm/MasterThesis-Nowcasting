#!/bin/bash
#SBATCH --job-name=radclim_filter_all
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/filter_all_%j.out

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load uv/0.2.30-GCCcore-13.3.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

mkdir -p logs
cd "$VSC_DATA"

DATA=/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr

uvx --from "git+https://github.com/mlcast-community/mlcast-dataset-sampler" \
mlcast.sample_dataset filter-nan "$DATA" \
  --start-date 2017-01-01 \
  --end-date 2023-11-30 \
  --time-depth 24 \
  --width 256 \
  --height 256 \
  --max-nan 10000 \
  --workers 16 \
  --data-var precip_intensity_EDK \
  --time-var time
