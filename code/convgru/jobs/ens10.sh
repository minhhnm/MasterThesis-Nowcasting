#!/bin/bash

set -euo pipefail

module purge
module load Python/3.13.1-GCCcore-14.2.0
source $VSC_DATA/venvs/convgru_ensemble3131/bin/activate

cd $VSC_DATA/code/convgru-ensemble

python -m convgru_ensemble.train \
  --config config:experiment \
  --config set:datamodule.zarr_path='"/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"' \
  --config set:datamodule.variable_name='"precip_intensity_EDK"' \
  --config set:datamodule.train_csv_path='"/data/brussel/114/vsc11442/train_datacubes_10pct.csv"' \
  --config set:datamodule.val_csv_path='"/data/brussel/114/vsc11442/val_datacubes_10pct.csv"' \
  --config set:datamodule.test_csv_path='"/data/brussel/114/vsc11442/test_datacubes.csv"' \
  --config set:datamodule.steps=24 \
  --config set:datamodule.batch_size=12 \
  --config set:datamodule.num_workers=4 \
  --config set:model.forecast_steps=20 \
  --config set:model.num_blocks=4 \
  --config set:model.ensemble_size=5 \
  --config set:model.noisy_decoder=True \
  --config set:model.loss_class='"crps"' \
  --config set:trainer.precision='"32-true"' \
  --config set:float32_matmul_precision='"high"' \
  --config set:trainer.num_sanity_val_steps=0
  --config set:trainer.max_epochs=30
