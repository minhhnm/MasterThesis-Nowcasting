from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fiddle as fdl
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

def autoenc_experiment(
    *,
    datamodule_cfg: fdl.Config,
    model_cfg: fdl.Config,
    output_dir: str,
    logger_name: str = "ldcast_autoenc",
    max_epochs: int = 30,
    precision: str = "16-mixed",
    accelerator: str = "gpu",
    devices: int = 1,
    monitor: str = "val/rec_loss",
    mode: str = "min",
    early_stopping_patience: int = 10,
    log_every_n_steps: int = 50,
) -> fdl.Config:
    """
    General LDcast autoencoder experiment builder.
    """
    cfg = fdl.Config(dict)

    cfg.stage = "autoenc"
    cfg.output_dir = output_dir

    cfg.datamodule = datamodule_cfg
    cfg.model = model_cfg

    cfg.callbacks = _callbacks_config(
        model_dir=output_dir,
        monitor=monitor,
        mode=mode,
        early_stopping_patience=early_stopping_patience,
    )

    cfg.loggers = _loggers_config(
        save_dir="lightning_logs",
        name=logger_name,
    )

    cfg.trainer = fdl.Config(
        pl.Trainer,
        max_epochs=max_epochs,
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
    )

    return cfg

def config_to_dict(cfg: Any) -> Any:
    """Convert a nested Fiddle config tree into plain Python objects for saving."""
    if isinstance(cfg, fdl.Config):
        result = {}
        for key, value in fdl.ordered_arguments(cfg).items():
            result[key] = config_to_dict(value)
        return result
    elif isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    elif isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    else:
        return cfg


def _callbacks_config(
    model_dir: str,
    monitor: str,
    mode: str = "min",
    early_stopping_patience: int = 10,
) -> fdl.Config:
    cfg = fdl.Config(dict)

    cfg.checkpoint = fdl.Config(
        ModelCheckpoint,
        dirpath=model_dir,
        filename="{epoch}-{step}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=True,
    )

    cfg.early_stopping = fdl.Config(
        EarlyStopping,
        monitor=monitor,
        mode=mode,
        patience=early_stopping_patience,
    )

    cfg.lr_monitor = fdl.Config(
        LearningRateMonitor,
        logging_interval="epoch",
    )

    return cfg


def _loggers_config(
    save_dir: str = "lightning_logs",
    name: str = "ldcast",
) -> fdl.Config:
    """Common logger config, following the ConvGRU style."""
    cfg = fdl.Config(dict)

    cfg.tensorboard = fdl.Config(
        TensorBoardLogger,
        save_dir=save_dir,
        name=name,
    )

    return cfg


def train(cfg: fdl.Config) -> None:
    """
    Generic train runner for LDcast experiments.

    Expected config tree:
      cfg.datamodule : fdl.Config(...)
      cfg.model      : fdl.Config(...)
      cfg.trainer    : fdl.Config(...)
      cfg.callbacks  : fdl.Config(dict)
      cfg.loggers    : fdl.Config(dict)
      cfg.output_dir : str
    """
    # Build plain dict-like top-level config
    built = fdl.build(cfg)

    # Build subparts
    datamodule = fdl.build(cfg.datamodule)
    model = fdl.build(cfg.model)

    callbacks_dict = fdl.build(cfg.callbacks)
    loggers_dict = fdl.build(cfg.loggers)

    callbacks = list(callbacks_dict.values())
    loggers = list(loggers_dict.values())

    trainer = fdl.build(cfg.trainer)
    trainer.callbacks = callbacks
    trainer.logger = loggers[0] if len(loggers) == 1 else loggers

    # Save resolved config
    output_dir = Path(built["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2, default=str)

    trainer.fit(model, datamodule=datamodule)

