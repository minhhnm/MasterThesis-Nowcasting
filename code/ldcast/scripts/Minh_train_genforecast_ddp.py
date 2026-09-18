import gc
import os

from fire import Fire
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger

from mlcast.models.ldcast.autoenc.autoenc import (
    Autoencoder,
    AutoencoderKLNet,
    AutoencoderLoss,
)
from mlcast.models.ldcast.autoenc.encoder import (
    SimpleConvEncoder,
    SimpleConvDecoder,
)
from mlcast.models.ldcast.context.context import AFNONowcastNetCascade
from mlcast.models.ldcast.diffusion.diffusion import (
    LatentDiffusion,
    LatentDiffusionNet,
)
from mlcast.models.ldcast.diffusion.unet import UNetModel

from Minh_read_mlcast_yaml import load_radar_cfg
from Minh_radclim_csv_data import RadarGenforecastDataModule


class LinearBetaScheduler:
    def __init__(
        self,
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
    ):
        self.timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

    def schedule(self, dtype, device):
        betas = torch.linspace(
            self.linear_start,
            self.linear_end,
            self.timesteps,
            dtype=dtype,
            device=device,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=dtype, device=device), alphas_cumprod[:-1]], dim=0
        )

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        )
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        return {
            "betas": betas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
            "log_one_minus_alphas_cumprod": torch.log(
                torch.clamp(1.0 - alphas_cumprod, min=1e-20)
            ),
            "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
            "sqrt_recipm1_alphas_cumprod": torch.sqrt(
                torch.clamp(1.0 / alphas_cumprod - 1.0, min=0.0)
            ),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": torch.log(posterior_variance),
            "posterior_mean_coef1": betas
            * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - alphas_cumprod),
            "posterior_mean_coef2": (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        }


def _strip_prefix_if_all(state_dict: dict, prefix: str) -> dict:
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def load_state_dict_any(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    for prefix in ("module.", "model."):
        state = _strip_prefix_if_all(state, prefix)

    return state

def build_autoencoder_from_weights(autoenc_weights_fn: str) -> Autoencoder:
    enc = SimpleConvEncoder()
    dec = SimpleConvDecoder()
    net = AutoencoderKLNet(encoder=enc, decoder=dec)
    loss = AutoencoderLoss(kl_weight=0.01)

    autoencoder = Autoencoder(
        net=net,
        loss=loss,
        antialiaser=None,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-4},
        lr_scheduler_config=None,
    )
    autoencoder.load_state_dict(load_state_dict_any(autoenc_weights_fn), strict=True)
    autoencoder.eval()
    return autoencoder


def setup_model(
    autoenc_weights_fn: str,
    model_dir: str,
    future_timesteps: int = 20,
    lr: float = 1e-4,
    max_epochs: int = 1000,
    precision: str = "16-mixed",
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
):
    os.makedirs(model_dir, exist_ok=True)
    
    autoencoder_obs = build_autoencoder_from_weights(autoenc_weights_fn)

    logger = CSVLogger(
        save_dir=model_dir,
        name="lightning_logs",
    )
    
    conditioner = AFNONowcastNetCascade(
        autoencoder_dim=autoencoder_obs.net.hidden_width,
        embed_dim=128,
        analysis_depth=4,
        forecast_depth=4,
        input_patches=1,
        input_size_ratios=1,
        output_patches=future_timesteps // 4,
        cascade_depth=3,
    )

    denoiser = UNetModel(
        in_channels=autoencoder_obs.net.hidden_width,
        model_channels=256,
        out_channels=autoencoder_obs.net.hidden_width,
        num_res_blocks=2,
        attention_resolutions=(1, 2),
        dims=3,
        channel_mult=(1, 2, 4),
        num_heads=8,
        num_timesteps=future_timesteps // 4,
        context_ch=conditioner.cascade_dims,
    )

    net = LatentDiffusionNet(
        conditioner=conditioner,
        denoiser=denoiser,
        parametrization="eps",
    )

    loss = nn.MSELoss()
    scheduler = LinearBetaScheduler(
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
    )

    ldm = LatentDiffusion(
        net=net,
        loss=loss,
        scheduler=scheduler,
        autoencoder=autoencoder_obs,
        ema_config={"use": True, "kwargs": {"decay": 0.9999, "store_device": "cpu"}},
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": lr},
        lr_scheduler_config=None,
    )

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if num_gpus > 0 else "cpu"

    # DDP version:
    # - For Slurm multi-task jobs, each task/rank should use one GPU.
    # - For a single process with multiple visible GPUs, Lightning can still use DDP.
    slurm_tasks = int(os.environ.get("SLURM_NTASKS", "1"))

    if num_gpus == 0:
        devices = 1
        strategy = "auto"
    elif slurm_tasks > 1:
        devices = 1
        strategy = "ddp"
    elif num_gpus > 1:
        devices = num_gpus
        strategy = "ddp"
    else:
        devices = 1
        strategy = "auto"

    print("num_gpus:", num_gpus)
    print("SLURM_NTASKS:", slurm_tasks)
    print("accelerator:", accelerator)
    print("devices:", devices)
    print("strategy:", strategy)

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val/loss",
            patience=6,
            verbose=True,
            mode="min",
            check_finite=False,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_dir,
            filename="epoch{epoch:03d}",
            monitor="val/loss",
            mode="min",
            every_n_epochs=1,
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
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

    gc.collect()
    return ldm, trainer


def train(
    data_config,
    autoenc_weights_fn,
    region="belgium",
    model_dir="models/genforecast",
    ckpt_path=None,
    initial_weights=None,
    strict_weights=True,
    batch_size=2,
    num_workers=0,
    pin_memory=True,
    tin=4,
    future_timesteps=20,
    crop_size=256,
    lr=1e-4,
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
    print("Autoencoder weights:", autoenc_weights_fn)

    datamodule = RadarGenforecastDataModule(
        zarr_path=cfg["zarr_path"],
        var_name=cfg["var_name"],
        train_csv_path=cfg["train_csv_path"],
        val_csv_path=cfg["val_csv_path"],
        test_csv_path=cfg["test_csv_path"],
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        tin=tin,
        tout=future_timesteps,
        crop_size=crop_size,
        transform_precip_to_dbz=cfg["transform_precip_to_dbz"],
    )

    model, trainer = setup_model(
        autoenc_weights_fn=autoenc_weights_fn,
        model_dir=model_dir,
        future_timesteps=future_timesteps,
        lr=lr,
        max_epochs=max_epochs,
        precision=precision,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )

    if initial_weights is not None:
        print("Loading initial weights from:", initial_weights)
        model.load_state_dict(load_state_dict_any(initial_weights), strict=strict_weights)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    cfg = OmegaConf.load(config) if config is not None else {}
    cfg.update(kwargs)
    train(**cfg)


if __name__ == "__main__":
    Fire(main)
