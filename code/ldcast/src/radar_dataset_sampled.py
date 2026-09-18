import csv
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import xarray as xr
from zarr.storage import ZipStore
from pathlib import Path
from typing import Dict, List, Any, Tuple


class RadarDatasetNew(Dataset):
    def __init__(self,
        mode: str,
        regions: List[str],
        is_primary_source: bool,
        inputs: List[str],
        targets: List[str],
        data_sources: Dict[str, Dict[str, Dict]],
        is_mask_enabled_by_region: Dict[str, bool] = None,
        number_of_past_radar_time_steps: int = 12,
        number_of_future_radar_time_steps: int = 12,\
        allowed_nan_fraction: float = 0.1,
        crop_size: int = 256
        ) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if number_of_past_radar_time_steps < 0 or number_of_future_radar_time_steps < 0:
            raise ValueError("Past and future radar time steps must be >= 0")
        if crop_size <= 0:
            raise ValueError("crop_size must be > 0")
        
        self.mode = mode
        self.is_primary_source = is_primary_source
        # Because the setup with multiple data sources is more complex, we need to check the provided input
        # If we pass the check, we have an internal variable called self.regions which contains the enabled regions
        self.regions = regions
        self.inputs = inputs
        self.targets = targets
        requested = set(inputs + targets)
        self.need_radar_past = "radar_past" in requested
        self.need_radar_future = "radar_future" in requested
        if is_mask_enabled_by_region is None:
            # Default everything to False if no dict is provided
            self.is_mask_enabled_by_region = {region: False for region in regions}
        else:
            self.is_mask_enabled_by_region = is_mask_enabled_by_region

        # TODO: check of this check is still needed with changes to the code
        #  We will most definitely need to update this check code!
        # self.check_provided_input(mode, dataset_sources, accumulation_minutes)
        # Set up an index_map for the valid indices for each region
        self.dataset_sources = data_sources
        if self.is_primary_source:
            self.index_map = self.setup_index_map(mode)
        
        # Initialise data_radar
        self.data_radar = {}
        for region in self.regions:
            ## TODO: check the fileds' order (should be region, source, ...)
            self.data_radar[region] = self.open_dataset(data_sources[region]["radar"]["data_dir"])
        
        self.allowed_nan_fraction = allowed_nan_fraction
        self.number_of_past_radar_time_steps = int(number_of_past_radar_time_steps)
        self.number_of_future_radar_time_steps = int(number_of_future_radar_time_steps)
        self.total_time_steps = (
            self.number_of_past_radar_time_steps + self.number_of_future_radar_time_steps
        )        
        self.crop_size = crop_size
        
        # Initiate the global statistics for the dataset
        ## TODO: rephrase it / eventually: consider to add in the above Initialise data_radar
        self.global_statistics_radar = {}
        for region in self.regions:
            self.global_statistics_radar[region] = self.data_radar[region][
                self.dataset_sources[region]["radar"]["variables"][0]].attrs
            
        # Empty tensor to return if no data is available, should be bigger than the crop size or can not crop
        self.empty_data = {}
        for region in self.regions:
            shape = self.data_radar[region][data_sources[region]["radar"]["variables"][0]].shape
            channels = len(data_sources[region]["radar"]["variables"])
            T_total = self.number_of_past_radar_time_steps + self.number_of_future_radar_time_steps
            self.empty_data[region] = (channels, T_total, shape[1], shape[2])
        print("Init done")

    def __len__(self) -> int:
        if not self.is_primary_source:
            raise RuntimeError("RadarDatasetNew.__len__ called on non-primary source.")
        return len(self.index_map)

    def get_time(self, idx: int) -> np.datetime64:
        """ Return reference time.
        Force to conver out datetime into [ns] precision to cover
        back while converting into integer later on.
        Args:
            idx: (int) dataset index.
        Return:
            datetime [ns].
        """
        region, t_start, _, _ = self.index_map[idx]
        ref_idx = t_start + self.number_of_past_radar_time_steps - 1
        time_size = int(self.data_radar[region].sizes["time"])
        if ref_idx < 0 or ref_idx >= time_size:
            raise IndexError(
                f"Reference time index {ref_idx} is out of bounds for time size {time_size}."
            )
        time = self.data_radar[region]["time"].isel(time=ref_idx).values
        return np.asarray(time).astype("datetime64[ns]")


    def get_ref_time(self, idx: int) -> int:
        """ Return reference time.
        Args:
            idx: (int) dataset index.
        Return:
            integer value corresponding to the actual datetime.            
        """
        time = self.get_time(idx)
        return int(time.astype(np.int64))

    def _build_sample_for_row(self, region: str, t_start: int, x_start: int, y_start: int) -> Dict[str, Any]:
        radar_cfg = self.dataset_sources[region]["radar"]
        radar_data = self.read_in_radar(
            region = region,
            t_start = t_start,
            x_start = x_start,
            y_start = y_start,
            variables = radar_cfg["variables"],
            origin = radar_cfg.get("origin", "upper"),
        )

        ## TODO: rewrite 
        sample: Dict[str, Any] = {}

        if self.is_mask_enabled_by_region.get(region, False):
            radar_mask = self.compute_radar_mask(radar_data)
            sample["radar_mask"] = radar_mask.unsqueeze(0)

        if radar_cfg.get("transform_precip_to_dbz", True):
            radar_data = self.transform_precip_to_dbZ(radar_data)

        radar_normalized_tensor = self.normalize_radar(
            radar_data,
            region=region,
        ).to(torch.float32)

        radar_past, radar_future = self._split_past_future(radar_normalized_tensor)

        if radar_past is not None and self.need_radar_past:
            sample["radar_past"] = radar_past
        if radar_future is not None and self.need_radar_future:
            sample["radar_future"] = radar_future

        return sample

    def get_data_at_time(self,
        actual_time: np.datetime64,
        region: str
        ) -> Dict[str, Any]:
        """Yield current radar sample if not using radar as primary source.
        """
        ds = self.data_radar[region]
        closest_past_time = ds.sel(time=actual_time, method="pad")["time"].values
        matches = np.where(ds["time"].values == closest_past_time)[0]
        if len(matches) == 0:
            raise ValueError(f"No matching time found for time {actual_time} in region {region}")
        t_start = int(matches[0]) - self.number_of_past_radar_time_steps + 1
        if t_start < 0:
            raise ValueError(f"Computed t_start={t_start} is negative for time {actual_time}")
        return self._build_sample_for_row(region, t_start, 0, 0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Returns ground_truth, input_model, ref_time (for debugging)
        region, t_start, x_start, y_start = self.index_map[idx]
        sample = self._build_sample_for_row(region, t_start, x_start, y_start)
        if self.is_primary_source:
            sample["t"] = int(t_start)
            sample["x"] = int(x_start)
            sample["y"] = int(y_start)
        return sample

    def _split_past_future(self, data: Tensor) -> tuple[Tensor | None, Tensor | None]:
        """
        data: (C, T_total, H, W)
        returns (past, future) where either can be None if length is zero.
        """
        C, T, H, W = data.shape
        T_past = self.number_of_past_radar_time_steps
        T_future = self.number_of_future_radar_time_steps

        assert T == T_past + T_future, (
            f"Time dim mismatch: got {T}, expected {T_past + T_future}"
        )

        past = data[:, :T_past] if T_past > 0 else None
        future = data[:, T_past:] if T_future > 0 else None
        return past, future

    def below_nan_fraction(self, data: Tensor) -> bool:
        # For full domain training, we are not interested in this nan fraction check, it can also manualy be done by setting the allowed_nan_fraction to None or >=1
        if self.allowed_nan_fraction is None or self.allowed_nan_fraction >= 1:
            return True
        channels, time, y, x = data.shape
        nan_fraction = torch.isnan(data[0, 0]).sum().item() / (x * y)
        if nan_fraction <= self.allowed_nan_fraction:
            return True
        else:
            return False

    def read_in_radar(self, region: str, t_start: int, x_start: int, y_start: int, variables: List[str],
        origin: str,) -> Tensor | None:
        # We add +1 to the next line because we want to look accumulation_time*timesteps back but not include
        # the earliest time step as this is still part of the accumulation period of the previous time step
        ds = self.data_radar[region]

        var0 = ds[variables[0]]
        time_size = int(var0.sizes["time"])
        y_size = int(var0.sizes["y"])
        x_size = int(var0.sizes["x"])

        t_end = t_start + self.total_time_steps
        y_end = y_start + self.crop_size
        x_end = x_start + self.crop_size

        if t_start < 0 or t_end > time_size:
            raise IndexError(
                f"Requested time slice [{t_start}:{t_end}] is out of bounds for time size {time_size}."
            )
        if y_start < 0 or y_end > y_size:
            raise IndexError(
                f"Requested y slice [{y_start}:{y_end}] is out of bounds for y size {y_size}."
            )
        if x_start < 0 or x_end > x_size:
            raise IndexError(
                f"Requested x slice [{x_start}:{x_end}] is out of bounds for x size {x_size}."
            )

        dataset_subset = ds.isel({
            "time": slice(t_start, t_end),
            "y": slice(y_start, y_end),
            "x": slice(x_start, x_end),
        })

        tensors = []
        for variable in variables:
            arr = dataset_subset[variable].values.astype(np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0)
            tensors.append(tensor)
        data_tensor = torch.cat(tensors, dim=0)

        if origin == "lower":
            data_tensor = torch.flip(data_tensor, dims=[2])
        elif origin != "upper":
            raise ValueError("Origin not supported. Use 'upper' or 'lower'.")

        if not self.below_nan_fraction(data_tensor):
            raise ValueError(
                f"Crop at region={region}, t={t_start}, x={x_start}, y={y_start} exceeds allowed_nan_fraction."
            )

        return data_tensor

    def normalize_radar(self, data: Tensor, region: str) -> Tensor:
        # The data here is in dBZ between 0 and 60, we transform this to a min/max beween -1 and 1
        normalized_data = ((data - 30) / 30)
        return normalized_data

    def transform_precip_to_dbZ(self, data: Tensor, a=200, b=1.6) -> Tensor:
        eps = 1e-16
        data = data.to(torch.float32)  # Clamp does not work on float16
        dbz = 10.0 * torch.log10(a * torch.clamp(data, min=0) ** b + eps)
        dbz = torch.clamp(dbz, 0, 60)  # 0…60 dBZ
        return torch.nan_to_num(dbz, nan=0.0)

    def compute_radar_mask(self, data: Tensor) -> Tensor:
        """
        Computes a spatial union mask across selected time steps and all channels.

        Args:
            data: (C, T, H, W) raw radar tensor containing NaNs.
        Returns:
            (H, W) float32 tensor where 1.0 is valid and 0.0 is masked.
        """
        # Slice the data based on past timesteps
        # If self.number_of_past_radar_time_steps is 8, we take data[:, :8, :, :]
        if self.number_of_past_radar_time_steps > 0:
            mask_data = data[:, :self.number_of_past_radar_time_steps, :, :]
        else:
            mask_data = data

        nan_map = torch.isnan(mask_data)             # (C, T_subset, H, W)
        nan_union_c = nan_map.any(dim=0)             # (T_subset, H, W) -> union over channels
        nan_union = nan_union_c.any(dim=0)           # (H, W) -> union over the selected time steps
        return (~nan_union).to(torch.float32)


    """
    General preparation methods for:
     - checking the input
     - constructing a global the index map over all regions
     - reading in datasets
    """
    """
    def check_provided_input(self, mode: str, dataset_sources: dict,
                             accumulation_minutes: int):
        # Check if datasplit is valid, if not raise ValueError
        if mode not in ["train_val", "test"]:
            raise ValueError(f"Data split {mode} not supported! Only train_val and test are supported.")

        # Check per source which of the regions are enabled
        def source_region_check(data_source: str):
            if data_source not in dataset_sources.keys():
                raise ValueError(f"{data_source} data source not provided in dataset_sources!")
            regions = []
            for region_name, region_data in dataset_sources[data_source].items():
                if region_data["enabled"]:
                    regions.append(region_name)
            if len(regions) == 0:
                raise ValueError(f"Required data source '{data_source}' is not enabled for any region!")
            return regions

        regions_radar = source_region_check("radar")
        self.regions = regions_radar
        if len(self.regions) > 1:
            variable_baseline = dataset_sources["radar"][self.regions[0]]["variables"]
            for region in self.regions:
                variable_other_region = dataset_sources[region]["radar"]["variables"]
                if variable_baseline != variable_other_region:
                    raise ValueError(f"Variables for region {region} do not match the variables of {self.regions[0]}!")

        # Check if accumulation_minutes is valid, if not raise ValueError
        if int(accumulation_minutes) % 5 != 0:
            raise ValueError(f"Accumulation minutes {accumulation_minutes} is not a multiple of 5 minutes!")

        print("Input checks passed")
    """

    def setup_index_map(self, mode: str):
        index_map: List[Tuple[str, int, int, int]] = []
        csv_key = {
            "train": "train_csv",
            "val": "val_csv",
            "test": "test_csv",
        }[mode]

        for region in self.regions:
            radar_cfg = self.dataset_sources[region]["radar"]
            csv_path = radar_cfg.get(csv_key)
            if not csv_path:
                raise ValueError(f"Missing '{csv_key}' for region '{region}'")

            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)

            with csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                required = {"t", "x", "y"}
                if not required.issubset(fieldnames):
                    raise ValueError(
                        f"CSV {csv_path} must contain columns {required}. Found {fieldnames}."
                    )
                for row in reader:
                    index_map.append((region, int(row["t"]), int(row["x"]), int(row["y"])))

        return index_map

    def open_dataset(self, filename) -> xr.Dataset:
        extension_filename = filename.split(".")[-1]
        if extension_filename == "zip":
            store = ZipStore(filename, mode="r")
            return xr.open_zarr(store)
        if extension_filename == "zarr":
            return xr.open_zarr(filename)
        if extension_filename == "nc":
            return xr.open_dataset(filename, engine="netcdf4")
        if extension_filename == "tif":
            return xr.open_dataset(filename, engine="rasterio")
        raise ValueError(f"Extension {extension_filename} not supported to read in datasets!")
