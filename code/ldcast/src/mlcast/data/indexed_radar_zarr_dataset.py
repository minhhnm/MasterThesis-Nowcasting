from pathlib import Path
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

def r_to_norm_dbz(R: np.ndarray, a=200.0, b=1.6, eps=1e-6) -> np.ndarray:
    # R in mm/h -> dBZ-like -> clamp -> normalize ~[-1,1]
    R = np.asarray(R, dtype=np.float32)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = np.maximum(R, 0.0)

    Z = a * np.power(R, b) + eps
    dbz = 10.0 * np.log10(Z)
    dbz = np.clip(dbz, 0.0, 60.0)
    return ((dbz - 30.0) / 30.0).astype(np.float32)

class IndexedRadarZarrDataset(Dataset):
    """
    Uses anchor indices (last input timestep) for efficient sampling.
    x: [anchor-tin+1 ... anchor]
    y: [anchor+1 ... anchor+tout]
    Returns:
      x: (Tin, 1, H, W), y: (Tout, 1, H, W)
    """
    def __init__(
        self,
        zarr_dir: Path,
        variable_name: str,
        anchors_npy: Path,
        tin: int = 12,
        tout: int = 12,
        crop: int = 256,
        use_dbz_norm: bool = True,
        seed: int = 42,
    ):
        self.root = zarr.open_consolidated(str(zarr_dir), mode="r") if (zarr_dir / ".zmetadata").exists() else zarr.open(str(zarr_dir), mode="r")
        self.arr = self.root[variable_name]  # (T,H,W)
        self.anchors = np.load(str(anchors_npy)).astype(np.int64)
        self.tin, self.tout = tin, tout
        self.crop = crop
        self.use_dbz_norm = use_dbz_norm
        self.rng = np.random.default_rng(seed)
        self.H, self.W = self.arr.shape[1], self.arr.shape[2]

    def __len__(self):
        return len(self.anchors)

    def _crop_slices(self):
        if self.crop is None or self.crop >= self.H or self.crop >= self.W:
            return slice(None), slice(None)
        y0 = int(self.rng.integers(0, self.H - self.crop))
        x0 = int(self.rng.integers(0, self.W - self.crop))
        return slice(y0, y0 + self.crop), slice(x0, x0 + self.crop)

    def __getitem__(self, i: int):
        anchor = int(self.anchors[i])
        start = anchor - self.tin + 1
        end = anchor + self.tout  # inclusive future end
        seq = np.asarray(self.arr[start : end + 1], dtype=np.float32)  # (Tin+Tout,H,W)

        sy, sx = self._crop_slices()
        seq = seq[:, sy, sx]

        if self.use_dbz_norm:
            seq = r_to_norm_dbz(seq)

        x = seq[: self.tin]
        y = seq[self.tin :]

        x = torch.from_numpy(x).unsqueeze(1)  # (Tin,1,H,W)
        y = torch.from_numpy(y).unsqueeze(1)  # (Tout,1,H,W)
        return x, y
