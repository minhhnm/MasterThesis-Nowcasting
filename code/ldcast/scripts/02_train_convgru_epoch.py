import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from mlcast.models.base import NowcastingLightningModule
from mlcast.modules import ConvGRU
from radar_dataset_sampled import RadarDatasetNew


def radar_pair_collate(batch):
    x = torch.stack([sample["radar_past"] for sample in batch], dim=0)
    y = torch.stack([sample["radar_future"] for sample in batch], dim=0)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    y = y.permute(0, 2, 1, 3, 4).contiguous()
    return x, y

class ConvGRUForecastWrapper(nn.Module):
    """
    Wrap ConvGRU/EncoderDecoder so it can be called as net(x).

    The mlcast base Lightning module calls:
        self.net(x)

    But ConvGRU EncoderDecoder expects:
        self.net(x, steps)

    This wrapper stores the forecast horizon.
    """

    def __init__(self, net: nn.Module, steps: int):
        super().__init__()
        self.net = net
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, self.steps)

class NowcastingLightningModuleWithScheduler(NowcastingLightningModule):
    def __init__(
        self,
        net,
        loss,
        lr: float = 1e-3,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 1,
        min_lr: float = 1e-6,
    ):
        super().__init__(net=net, loss=loss)
        self.lr = lr
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--region", default="belgium")
    ap.add_argument("--out_dir", default="results/convgru/det_full_run1")
    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--tout", type=int, default=20)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_epochs", type=int, default=3)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--use_dbz_norm", action="store_true")
    ap.add_argument("--allowed_nan_fraction", type=float, default=1.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scheduler_factor", type=float, default=0.5)
    ap.add_argument("--scheduler_patience", type=int, default=1)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--loss_name", choices=["mse", "crps"], default="mse")
    ap.add_argument("--ensemble_members", type=int, default=4)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")

    with open(args.config, "r") as f:
        data_sources = yaml.safe_load(f)

    if args.region not in data_sources:
        raise KeyError(f"Region {args.region!r} not found in config")

    if args.use_dbz_norm:
        data_sources[args.region]["radar"]["transform_precip_to_dbz"] = True

    train_ds = RadarDatasetNew(
        mode="train",
        regions=[args.region],
        is_primary_source=True,
        inputs=["radar_past"],
        targets=["radar_future"],
        data_sources=data_sources,
        number_of_past_radar_time_steps=args.tin,
        number_of_future_radar_time_steps=args.tout,
        crop_size=args.crop,
        allowed_nan_fraction=args.allowed_nan_fraction,
    )
    val_ds = RadarDatasetNew(
        mode="val",
        regions=[args.region],
        is_primary_source=True,
        inputs=["radar_past"],
        targets=["radar_future"],
        data_sources=data_sources,
        number_of_past_radar_time_steps=args.tin,
        number_of_future_radar_time_steps=args.tout,
        crop_size=args.crop,
        allowed_nan_fraction=args.allowed_nan_fraction,
    )

    sample = train_ds[0]
    print("train[0] radar_past:", tuple(sample["radar_past"].shape))
    print("train[0] radar_future:", tuple(sample["radar_future"].shape))
    print("train[0] metadata:", sample["t"], sample["x"], sample["y"])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0), 
        collate_fn=radar_pair_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=radar_pair_collate,
        drop_last=False,
    )

    if args.loss_name == "mse":
        raw_net = ConvGRU()
        net = ConvGRUForecastWrapper(raw_net, steps=args.tout)
        loss = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss_name: {args.loss_name}")
    
    model = NowcastingLightningModuleWithScheduler(
        net=net,
        loss=loss,
        lr=args.lr,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    best_ckpt = ModelCheckpoint(
        dirpath=args.out_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        filename="best-epoch{epoch:02d}-step{step}",
        auto_insert_metric_name=False,
        save_last=False,
        save_on_train_epoch_end=False,
    )
    
    latest_ckpt = ModelCheckpoint(
        dirpath=args.out_dir,
        monitor=None,
        save_top_k=-1,
        every_n_epochs=1,
        filename="latest-epoch{epoch:02d}-step{step}",
        auto_insert_metric_name=False,
        save_last=False,
        save_on_train_epoch_end=False,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    trainer = Trainer(
        default_root_dir=args.out_dir,
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[best_ckpt, latest_ckpt, lr_monitor],
        log_every_n_steps=200,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )
    
if __name__ == "__main__":
    main()