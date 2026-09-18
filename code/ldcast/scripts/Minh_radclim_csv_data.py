import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from radar_dataset_sampled import RadarDatasetNew


def _build_data_sources(
    zarr_path: str,
    var_name: str,
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: str | None = None,
    origin: str = "upper",
    transform_precip_to_dbz: bool = True,
):
    radar_cfg = {
        "data_dir": zarr_path,
        "variables": [var_name],
        "origin": origin,
        "transform_precip_to_dbz": transform_precip_to_dbz,
        "train_csv": train_csv_path,
        "val_csv": val_csv_path,
    }
    if test_csv_path is not None:
        radar_cfg["test_csv"] = test_csv_path

    return {
        "belgium": {
            "radar": radar_cfg,
        }
    }


class RadarAutoencoderDataset(Dataset):
    """
    Thin wrapper around RadarDatasetNew for autoencoder training.

    Uses only past radar frames and returns:
        x: (C, T, H, W)
        y: (C, T, H, W)
    """

    def __init__(
        self,
        mode: str,
        zarr_path: str,
        var_name: str,
        train_csv_path: str,
        val_csv_path: str,
        test_csv_path: str | None = None,
        autoenc_steps: int = 4,
        crop_size: int = 256,
        transform_precip_to_dbz: bool = True,
        allowed_nan_fraction: float = 1.0,
        origin: str = "upper",
        region: str = "belgium",
    ):
        data_sources = _build_data_sources(
            zarr_path=zarr_path,
            var_name=var_name,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            test_csv_path=test_csv_path,
            origin=origin,
            transform_precip_to_dbz=transform_precip_to_dbz,
        )

        self.base = RadarDatasetNew(
            mode=mode,
            regions=[region],
            is_primary_source=True,
            inputs=["radar_past"],
            targets=["radar_past"],
            data_sources=data_sources,
            is_mask_enabled_by_region={region: False},
            number_of_past_radar_time_steps=autoenc_steps,
            number_of_future_radar_time_steps=0,
            allowed_nan_fraction=allowed_nan_fraction,
            crop_size=crop_size,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        x = sample["radar_past"]          # (C, T, H, W)
        y = x.clone()
        return x, y


class RadarGenforecastDataset(Dataset):
    """
    Thin wrapper around RadarDatasetNew for LDcast training.

    Returns:
        x: [(past, t_rel)]
        y: future
    where
        past:   (C, Tin, H, W)
        t_rel:  (Tin,)
        future: (C, Tout, H, W)
    """

    def __init__(
        self,
        mode: str,
        zarr_path: str,
        var_name: str,
        train_csv_path: str,
        val_csv_path: str,
        test_csv_path: str | None = None,
        tin: int = 4,
        tout: int = 20,
        crop_size: int = 256,
        transform_precip_to_dbz: bool = True,
        allowed_nan_fraction: float = 1.0,
        origin: str = "upper",
        region: str = "belgium",
    ):
        data_sources = _build_data_sources(
            zarr_path=zarr_path,
            var_name=var_name,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            test_csv_path=test_csv_path,
            origin=origin,
            transform_precip_to_dbz=transform_precip_to_dbz,
        )

        self.base = RadarDatasetNew(
            mode=mode,
            regions=[region],
            is_primary_source=True,
            inputs=["radar_past"],
            targets=["radar_future"],
            data_sources=data_sources,
            is_mask_enabled_by_region={region: False},
            number_of_past_radar_time_steps=tin,
            number_of_future_radar_time_steps=tout,
            allowed_nan_fraction=allowed_nan_fraction,
            crop_size=crop_size,
        )
        self.t_rel = torch.arange(-tin + 1, 1, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        past = sample["radar_past"]       # (C, Tin, H, W)
        future = sample["radar_future"]   # (C, Tout, H, W)
        x = past
        y = future
        return x, y


class RadarAutoencoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zarr_path: str,
        var_name: str,
        train_csv_path: str,
        val_csv_path: str,
        test_csv_path: str | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        total_steps: int = 4,   # kept for interface compatibility
        autoenc_steps: int = 4,
        crop_size: int = 256,
        transform_precip_to_dbz: bool = True,
        allowed_nan_fraction: float = 1.0,
        origin: str = "upper",
        region: str = "belgium",
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.var_name = var_name
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.total_steps = total_steps
        self.autoenc_steps = autoenc_steps
        self.crop_size = crop_size
        self.transform_precip_to_dbz = transform_precip_to_dbz
        self.allowed_nan_fraction = allowed_nan_fraction
        self.origin = origin
        self.region = region

    def setup(self, stage=None):
        self.train_dataset = RadarAutoencoderDataset(
            mode="train",
            zarr_path=self.zarr_path,
            var_name=self.var_name,
            train_csv_path=self.train_csv_path,
            val_csv_path=self.val_csv_path,
            test_csv_path=self.test_csv_path,
            autoenc_steps=self.autoenc_steps,
            crop_size=self.crop_size,
            transform_precip_to_dbz=self.transform_precip_to_dbz,
            allowed_nan_fraction=self.allowed_nan_fraction,
            origin=self.origin,
            region=self.region,
        )
        self.val_dataset = RadarAutoencoderDataset(
            mode="val",
            zarr_path=self.zarr_path,
            var_name=self.var_name,
            train_csv_path=self.train_csv_path,
            val_csv_path=self.val_csv_path,
            test_csv_path=self.test_csv_path,
            autoenc_steps=self.autoenc_steps,
            crop_size=self.crop_size,
            transform_precip_to_dbz=self.transform_precip_to_dbz,
            allowed_nan_fraction=self.allowed_nan_fraction,
            origin=self.origin,
            region=self.region,
        )
        self.test_dataset = None
        if self.test_csv_path is not None:
            self.test_dataset = RadarAutoencoderDataset(
                mode="test",
                zarr_path=self.zarr_path,
                var_name=self.var_name,
                train_csv_path=self.train_csv_path,
                val_csv_path=self.val_csv_path,
                test_csv_path=self.test_csv_path,
                autoenc_steps=self.autoenc_steps,
                crop_size=self.crop_size,
                transform_precip_to_dbz=self.transform_precip_to_dbz,
                allowed_nan_fraction=self.allowed_nan_fraction,
                origin=self.origin,
                region=self.region,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )


class RadarGenforecastDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zarr_path: str,
        var_name: str,
        train_csv_path: str,
        val_csv_path: str,
        test_csv_path: str | None = None,
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = True,
        tin: int = 4,
        tout: int = 20,
        crop_size: int = 256,
        transform_precip_to_dbz: bool = True,
        allowed_nan_fraction: float = 1.0,
        origin: str = "upper",
        region: str = "belgium",
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.var_name = var_name
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tin = tin
        self.tout = tout
        self.crop_size = crop_size
        self.transform_precip_to_dbz = transform_precip_to_dbz
        self.allowed_nan_fraction = allowed_nan_fraction
        self.origin = origin
        self.region = region

    def setup(self, stage=None):
        self.train_dataset = RadarGenforecastDataset(
            mode="train",
            zarr_path=self.zarr_path,
            var_name=self.var_name,
            train_csv_path=self.train_csv_path,
            val_csv_path=self.val_csv_path,
            test_csv_path=self.test_csv_path,
            tin=self.tin,
            tout=self.tout,
            crop_size=self.crop_size,
            transform_precip_to_dbz=self.transform_precip_to_dbz,
            allowed_nan_fraction=self.allowed_nan_fraction,
            origin=self.origin,
            region=self.region,
        )
        self.val_dataset = RadarGenforecastDataset(
            mode="val",
            zarr_path=self.zarr_path,
            var_name=self.var_name,
            train_csv_path=self.train_csv_path,
            val_csv_path=self.val_csv_path,
            test_csv_path=self.test_csv_path,
            tin=self.tin,
            tout=self.tout,
            crop_size=self.crop_size,
            transform_precip_to_dbz=self.transform_precip_to_dbz,
            allowed_nan_fraction=self.allowed_nan_fraction,
            origin=self.origin,
            region=self.region,
        )
        self.test_dataset = None
        if self.test_csv_path is not None:
            self.test_dataset = RadarGenforecastDataset(
                mode="test",
                zarr_path=self.zarr_path,
                var_name=self.var_name,
                train_csv_path=self.train_csv_path,
                val_csv_path=self.val_csv_path,
                test_csv_path=self.test_csv_path,
                tin=self.tin,
                tout=self.tout,
                crop_size=self.crop_size,
                transform_precip_to_dbz=self.transform_precip_to_dbz,
                allowed_nan_fraction=self.allowed_nan_fraction,
                origin=self.origin,
                region=self.region,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )
