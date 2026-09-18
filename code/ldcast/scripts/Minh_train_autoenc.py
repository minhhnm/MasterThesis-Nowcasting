import gc
import os

from fire import Fire
from omegaconf import OmegaConf
import pytorch_lightning as pl
import collections
import torch
from torch.serialization import add_safe_globals
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import CSVLogger

add_safe_globals([
    Adam,
    ReduceLROnPlateau,
    collections.defaultdict,
])

from mlcast.models.ldcast.autoenc.autoenc import (
    Autoencoder,
    AutoencoderKLNet,
    AutoencoderLoss,
)
from mlcast.models.ldcast.autoenc.encoder import (
    SimpleConvEncoder,
    SimpleConvDecoder,
)

from Minh_read_mlcast_yaml import load_radar_cfg
from Minh_radclim_csv_data import RadarAutoencoderDataModule


def setup_model(
    model_dir: str,
    max_epochs: int = 1000,
    precision: str = "16-mixed",
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
):
    os.makedirs(model_dir, exist_ok=True)

    logger = CSVLogger(
        save_dir=model_dir,
        name="lightning_logs",
    )
    
    enc = SimpleConvEncoder()
    dec = SimpleConvDecoder()
    net = AutoencoderKLNet(encoder=enc, decoder=dec)
    loss = AutoencoderLoss(kl_weight=0.01)

    model = Autoencoder(
        net=net,
        loss=loss,
        antialiaser=None,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-4},
        lr_scheduler_config=None,
    )

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"
    devices = num_gpus if num_gpus > 0 else 1
    strategy = "dp" if num_gpus > 1 else "auto"

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val/rec_loss",
            patience=6,
            verbose=True,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_dir,
            filename="epoch{epoch:03d}",
            monitor="val/rec_loss",
            mode="min",
            every_n_epochs=1,
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]

    trainer = pl.Trainer(
        default_root_dir=model_dir,
        logger=logger,        
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=max_epochs,
        precision=precision,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    return model, trainer


def train(
    data_config,
    region="belgium",
    model_dir="models/autoenc",
    ckpt_path=None,
    batch_size=8,
    num_workers=0,
    pin_memory=True,
    total_steps=4,
    autoenc_steps=4,
    crop_size=256,
    max_epochs=1000,
    precision="16-mixed",
    limit_train_batches=1.0,
    limit_val_batches=1.0,
):
    cfg = load_radar_cfg(data_config, region)

    print("Using data config:", data_config)
    print("Region:", region)
    print("Train CSV:", cfg["train_csv_path"])
    print("Val CSV:", cfg["val_csv_path"])
    print("Test CSV:", cfg["test_csv_path"])
    print("Variable:", cfg["var_name"])
    print("DBZ transform:", cfg["transform_precip_to_dbz"])
    print("Autoencoder total_steps:", total_steps)
    print("Autoencoder autoenc_steps:", autoenc_steps)

    datamodule = RadarAutoencoderDataModule(
        zarr_path=cfg["zarr_path"],
        var_name=cfg["var_name"],
        train_csv_path=cfg["train_csv_path"],
        val_csv_path=cfg["val_csv_path"],
        test_csv_path=cfg["test_csv_path"],
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        total_steps=total_steps,
        autoenc_steps=autoenc_steps,
        crop_size=crop_size,
        transform_precip_to_dbz=cfg["transform_precip_to_dbz"],
    )

    model, trainer = setup_model(
        model_dir=model_dir,
        max_epochs=max_epochs,
        precision=precision,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )

    gc.collect()
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    cfg = OmegaConf.load(config) if config is not None else {}
    cfg.update(kwargs)
    train(**cfg)


if __name__ == "__main__":
    Fire(main)
