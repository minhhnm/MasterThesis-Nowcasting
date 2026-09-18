#!/bin/bash
#SBATCH --job-name=make_smoke_10pct
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd $VSC_DATA/code/mlcast-ldcast

python scripts/make_smoke_split.py   --train_in /data/brussel/114/vsc11442/train_datacubes.csv   --val_in /data/brussel/114/vsc11442/val_datacubes.csv   --train_out /data/brussel/114/vsc11442/train_datacubes_10pct.csv   --val_out /data/brussel/114/vsc11442/val_datacubes_10pct.csv   --fraction 0.1   --seed 42   --report /data/brussel/114/vsc11442/train_val_10pct_report.txt
