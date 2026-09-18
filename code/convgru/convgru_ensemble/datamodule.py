import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .utils import rainrate_to_normalized


class SampledRadarDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        csv_path: str,
        steps: int,
        return_mask: bool = False,
        deterministic: bool = False,
        augment: bool = False,
        indices=None,
        variable_name: str = "RR",
    ):
        self.coords = pd.read_csv(csv_path).sort_values("t")
        if indices is not None:
            self.coords = self.coords.iloc[list(indices)].reset_index(drop=True)

        self.zg = xr.open_zarr(zarr_path)

        # Support either RR already in (time, x, y) or the original RADCLIM variable
        da = self.zg[variable_name]
        self.RR = da.transpose("time", "x", "y")

        self.rng = np.random.default_rng(seed=42) if deterministic else np.random.default_rng(int(time.time()))
        self.return_mask = return_mask
        self.augment = augment

        self.w = 256
        self.h = 256
        self.dt = 24
        self.steps = steps

        if augment:
            print("Data augmentation is enabled.")

        if self.steps > self.dt:
            print(f"Warning: requested steps ({self.steps}) > sampled time window ({self.dt})")

    def __len__(self):
        return len(self.coords)

    def shape(self):
        return (len(self.coords), self.steps, 1, self.w, self.h)

    def _apply_augmentations(self, *tensors, rotate_prob=0.5, hflip_prob=0.5, vflip_prob=0.5):
        if self.rng.random() < rotate_prob:
            k = self.rng.integers(1, 4)
            tensors = [torch.rot90(t, k, dims=[-2, -1]) for t in tensors]
        if self.rng.random() < hflip_prob:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]
        if self.rng.random() < vflip_prob:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]
        tensors = [t.contiguous() for t in tensors]
        return tensors[0] if len(tensors) == 1 else tuple(tensors)

    def __getitem__(self, idx: int):
        t0, x0, y0 = self.coords.iloc[idx]
        t0 = int(t0); x0 = int(x0); y0 = int(y0)

        x_slice = slice(x0, x0 + self.w)
        y_slice = slice(y0, y0 + self.h)

        if self.steps < self.dt:
            t_start = int(self.rng.integers(t0, t0 + self.dt - self.steps + 1))
        else:
            t_start = t0

        t_slice = slice(t_start, t_start + self.steps)

        data = rainrate_to_normalized(self.RR[t_slice, x_slice, y_slice])

        if self.return_mask:
            mask = (~(np.isnan(data).any(axis=0, keepdims=True))).astype(np.float32)

        data = np.nan_to_num(data, nan=-1.0)
        data = torch.from_numpy(data[:, np.newaxis, :, :])

        if self.return_mask:
            mask = torch.from_numpy(np.asarray(mask)[:, np.newaxis, :, :])

        if self.augment:
            if self.return_mask:
                data, mask = self._apply_augmentations(data, mask)
            else:
                data = self._apply_augmentations(data)

        if self.return_mask:
            return {"data": data, "mask": mask}
        return {"data": data}


class RadarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zarr_path,
        steps,
        csv_path=None,
        train_csv_path=None,
        val_csv_path=None,
        test_csv_path=None,
        variable_name="RR",
        train_ratio=0.7,
        val_ratio=0.15,
        return_mask=False,
        deterministic=False,
        augment=True,
        **dataloader_kwargs,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.steps = steps
        self.csv_path = csv_path
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.variable_name = variable_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.return_mask = return_mask
        self.deterministic = deterministic
        self.augment = augment
        self.dataloader_kwargs = dataloader_kwargs

    def setup(self, stage=None):
        use_explicit_splits = all([
            self.train_csv_path is not None,
            self.val_csv_path is not None,
            self.test_csv_path is not None,
        ])

        if use_explicit_splits:
            self.train_dataset = SampledRadarDataset(
                self.zarr_path,
                self.train_csv_path,
                self.steps,
                self.return_mask,
                self.deterministic,
                augment=self.augment,
                variable_name=self.variable_name,
            )
            self.val_dataset = SampledRadarDataset(
                self.zarr_path,
                self.val_csv_path,
                self.steps,
                self.return_mask,
                self.deterministic,
                augment=False,
                variable_name=self.variable_name,
            )
            self.test_dataset = SampledRadarDataset(
                self.zarr_path,
                self.test_csv_path,
                self.steps,
                self.return_mask,
                self.deterministic,
                augment=False,
                variable_name=self.variable_name,
            )
        else:
            if self.csv_path is None:
                raise ValueError("Provide either csv_path or train/val/test_csv_path")

            coords = pd.read_csv(self.csv_path).sort_values("t")
            n = len(coords)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))

            self.train_dataset = SampledRadarDataset(
                self.zarr_path,
                self.csv_path,
                self.steps,
                self.return_mask,
                self.deterministic,
                augment=self.augment,
                indices=range(0, train_end),
                variable_name=self.variable_name,
            )
            self.val_dataset = SampledRadarDataset(
                self.zarr_path,
                self.csv_path,
                self.steps,
                self.return_mask,
                self.deterministic,
                augment=False,
                indices=range(train_end, val_end),
                variable_name=self.variable_name,
            )
            self.test_dataset = SampledRadarDataset(
                self.zarr_path,
                self.csv_path,
                self.steps,
                self.return_mask,
                self.deterministic,
                augment=False,
                indices=range(val_end, n),
                variable_name=self.variable_name,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_kwargs)
