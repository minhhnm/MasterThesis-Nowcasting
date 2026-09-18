#!/bin/bash
#SBATCH --job-name=ensfull
#SBATCH --partition=hopper_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

module purge
module load Python/3.13.1-GCCcore-14.2.0
source $VSC_DATA/venvs/convgru_ensemble3131/bin/activate

cd $VSC_DATA/code/convgru-ensemble

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

CKPT_PATH="$VSC_DATA/code/convgru-ensemble/logs/CRPS_4past-20fut_noise/job12369418_4past-20fut_noise_bs32_lr1e-05/checkpoints/last.ckpt"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: checkpoint not found:"
  echo "$CKPT_PATH"
  exit 1
fi

echo "Resuming FULL ensemble from:"
echo "$CKPT_PATH"
echo "Target total epochs: 5"

python - <<PY
import torch
ckpt = torch.load("$CKPT_PATH", map_location="cpu", weights_only=False)
print("checkpoint epoch:", ckpt.get("epoch"))
print("global_step:", ckpt.get("global_step"))
PY

python -m convgru_ensemble.train \
  --config config:experiment \
  --config set:checkpoint_path='"'"$CKPT_PATH"'"' \
  --config set:datamodule.zarr_path='"/data/brussel/vo/000/bvo00029/data/observations/RADCLIMrates_2017_2023_5m_be_v1/RADCLIMrates_f16.zarr"' \
  --config set:datamodule.variable_name='"precip_intensity_EDK"' \
  --config set:datamodule.train_csv_path='"/data/brussel/114/vsc11442/train_datacubes.csv"' \
  --config set:datamodule.val_csv_path='"/data/brussel/114/vsc11442/val_datacubes.csv"' \
  --config set:datamodule.test_csv_path='"/data/brussel/114/vsc11442/test_datacubes.csv"' \
  --config set:datamodule.steps=24 \
  --config set:datamodule.batch_size=32 \
  --config set:datamodule.num_workers=10 \
  --config set:model.forecast_steps=20 \
  --config set:model.num_blocks=4 \
  --config set:model.ensemble_size=4 \
  --config set:model.noisy_decoder=True \
  --config set:model.loss_class='"crps"' \
  --config set:model.optimizer_params='{"lr": 1e-5, "fused": True}' \
  --config set:trainer.precision='"16-mixed"' \
  --config set:trainer.num_sanity_val_steps=0 \
  --config set:callbacks.checkpoint_val.save_last=True \
  --config set:loggers.tensorboard.version='"resume_to3_from_job12369418"' \
  --config set:trainer.max_epochs=3

