"""Fiddler mutators for high-level semantic configuration changes.

Fiddlers are functions that accept a ``fdl.Config`` and mutate it in place.
They are the right tool when a change spans multiple config parameters that
must stay in sync — for example, switching the dataset class while preserving
its existing parameters, or enabling masking consistently across both the data
pipeline and the loss function.

Use fiddlers from the CLI via ``--config fiddler:<name>`` or
``--config "fiddler:<name>(arg=value)"``, or call them directly on a buildable
config in Python before passing it to ``fdl.build()``.
"""

import inspect
import os

import fiddle as fdl
from loguru import logger
from pytorch_lightning.loggers import MLFlowLogger

from ..callbacks import LogSystemInfoCallback
from ..data.source_data_datasets import SourceDataRandomSamplingDataset
from ..models.unet import StochasticUNetModel, UNetModel


def set_variables(cfg: fdl.Config, standard_names: list[str]) -> None:
    """Fiddler to synchronize dataset variables with the network's input channels.

    Sets ``dataset_factory.standard_names`` on the data config and, when the
    network config exposes an ``input_channels`` parameter (e.g.
    ``ConvGruModel``), keeps it in sync.  Networks that use a different
    parameter name for the channel count (e.g. ``HalfUNet`` uses
    ``in_channels``) are left unchanged — callers are responsible for keeping
    that parameter consistent when swapping in an external architecture.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    standard_names : list of str
        The new list of standard names to load.
    """
    cfg.data.dataset_factory.standard_names = standard_names
    network_cls = cfg.pl_module.network.__fn_or_cls__
    sig = inspect.signature(network_cls.__init__)
    if "input_channels" in sig.parameters:
        cfg.pl_module.network.input_channels = len(standard_names)
    else:
        logger.warning(
            "set_variables: network {} has no 'input_channels' parameter; "
            "channel count not updated. Set it manually on the network config.",
            network_cls.__name__,
        )


def toggle_masking(cfg: fdl.Config, enabled: bool) -> None:
    """Fiddler to synchronize dataset mask yielding with masked loss computation.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    enabled : bool
        Whether to enable masking or not.
    """
    cfg.data.dataset_factory.return_mask = enabled
    cfg.pl_module.masked_loss = enabled


def use_random_sampler(cfg: fdl.Config) -> None:
    """Fiddler to switch the dataset factory to use the random sampler.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    """
    # Keep the existing parameters but change the underlying class
    cfg.data.dataset_factory = fdl.Partial(
        SourceDataRandomSamplingDataset,
        zarr_path=cfg.data.dataset_factory.zarr_path,
        standard_names=cfg.data.dataset_factory.standard_names,
        input_steps=cfg.data.dataset_factory.input_steps,
        forecast_steps=cfg.data.dataset_factory.forecast_steps,
        return_mask=cfg.data.dataset_factory.return_mask,
        storage_options=getattr(cfg.data.dataset_factory, "storage_options", None),
    )


def use_ratio_splits(cfg: fdl.Config, train: float, val: float) -> None:
    """Fiddler to set fraction-based train/val/test splits on the data module."""
    cfg.data.splits = {"time": {"train": train, "val": val, "test": 1.0 - train - val}}


def use_anon_s3_dataset(cfg: fdl.Buildable, zarr_path: str, endpoint_url: str) -> None:
    """Configure the dataset factory to read anonymously from an S3 object store.

    Parameters
    ----------
    cfg : fdl.Buildable
        The Fiddle configuration to mutate.
    zarr_path : str
        The S3 URI path to the Zarr dataset (e.g., s3://bucket/path.zarr).
    endpoint_url : str
        The endpoint URL for the S3 object store.
    """
    cfg.data.dataset_factory.zarr_path = zarr_path
    cfg.data.dataset_factory.storage_options = {
        "anon": True,
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "verify": False,
        },
        "config_kwargs": {"signature_version": "s3v4"},
    }


def use_mlflow_logger(cfg: fdl.Config) -> None:
    """Fiddler to switch the trainer logger to MLflow.

    Replaces the default TensorBoardLogger with an MLFlowLogger, inheriting
    the experiment name from the existing logger config. The tracking URI and
    run name are left unset, deferring to the ``MLFLOW_TRACKING_URI`` and
    ``MLFLOW_RUN_NAME`` environment variables (or MLflow defaults).

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.warning(
            "MLFLOW_TRACKING_URI is not set. MLflow will log to a local './mlruns' directory. "
            "Set MLFLOW_TRACKING_URI to point to a remote tracking server, "
            "e.g. export MLFLOW_TRACKING_URI=http://localhost:5000"
        )
    cfg.trainer.logger = fdl.Config(MLFlowLogger, experiment_name=cfg.trainer.logger.name)
    cfg.trainer.callbacks.append(fdl.Config(LogSystemInfoCallback))


def use_unet(cfg: fdl.Config, base_channels: int = 32, num_blocks: int = 4) -> None:
    """Fiddler to switch the configured network to a deterministic U-Net.

    The U-Net is deterministic, so this fiddler also sets ``ensemble_size=1``
    and uses MSE loss by default. Dataset variables and input steps are read
    from the existing data configuration so the fiddler remains dataset-general.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    base_channels : int, optional
        Number of channels in the first U-Net level. Default is ``32``.
    num_blocks : int, optional
        Number of U-Net downsampling blocks. Default is ``4``.
    """
    cfg.pl_module.network = fdl.Config(
        UNetModel,
        input_channels=len(cfg.data.dataset_factory.standard_names),
        input_steps=cfg.data.dataset_factory.input_steps,
        base_channels=base_channels,
        num_blocks=num_blocks,
    )

    cfg.pl_module.ensemble_size = 1
    cfg.pl_module.loss_class = "mse"
    cfg.pl_module.loss_params = None



def use_stochastic_unet(
    cfg: fdl.Config,
    base_channels: int = 32,
    num_blocks: int = 4,
    noise_channels: int = 4,
    noise_scale: float = 1.0,
    ensemble_size: int = 4,
    temporal_lambda: float = 0.0,
) -> None:
    """Fiddler to switch the configured network to a stochastic U-Net.

    The stochastic U-Net generates multiple ensemble members by conditioning
    each autoregressive forecast on independent random noise channels. It is
    intended to be trained with CRPS.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    base_channels : int, optional
        Number of channels in the first U-Net level. Default is ``32``.
    num_blocks : int, optional
        Number of U-Net downsampling blocks. Default is ``4``.
    noise_channels : int, optional
        Number of random noise channels. Default is ``4``.
    noise_scale : float, optional
        Standard deviation multiplier for Gaussian noise. Default is ``1.0``.
    ensemble_size : int, optional
        Number of ensemble members used during training. Default is ``4``.
    temporal_lambda : float, optional
        Optional temporal smoothness penalty passed to CRPS. Default is ``0.0``.
    """
    if ensemble_size < 2:
        raise ValueError("use_stochastic_unet requires ensemble_size >= 2 for CRPS training.")

    cfg.pl_module.network = fdl.Config(
        StochasticUNetModel,
        input_channels=len(cfg.data.dataset_factory.standard_names),
        input_steps=cfg.data.dataset_factory.input_steps,
        base_channels=base_channels,
        num_blocks=num_blocks,
        noise_channels=noise_channels,
        noise_scale=noise_scale,
    )

    cfg.pl_module.ensemble_size = ensemble_size
    cfg.pl_module.loss_class = "crps"
    cfg.pl_module.loss_params = {"temporal_lambda": temporal_lambda}
