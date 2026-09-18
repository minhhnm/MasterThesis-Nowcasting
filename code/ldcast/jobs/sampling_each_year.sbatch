#!/bin/bash
#SBATCH --job-name=radclim_importance_sampling
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --output=logs/importance_sampling_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load uv/0.2.30-GCCcore-13.3.0
module load zarr/2.18.4-foss-2024a
module load xarray/2024.11.0-gfbf-2024a

cd "$VSC_DATA"

DATA=/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr
CSV="${CSV:?CSV not set}"

uvx --from "git+https://github.com/mlcast-community/mlcast-dataset-sampler" \
mlcast.sample_dataset sample "$DATA" "$CSV" \
  --q-min 1e-4 \
  --mean-weight 0.1 \
  --data-var precip_intensity_EDK
