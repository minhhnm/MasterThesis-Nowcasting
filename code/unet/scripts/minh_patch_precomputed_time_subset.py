from pathlib import Path

p = Path("src/mlcast/data/source_data_datasets.py")
s = p.read_text()

old = """        time_range: tuple[str, str] | None = (subset or {}).get("time")
        if time_range is not None:
            self._time_index_slice: slice | None = _time_range_to_index_slice(zarr_path, time_range, storage_options)
        else:
            self._time_index_slice = None
        super().__init__(
"""

new = """        time_range: tuple[str, str] | None = (subset or {}).get("time")
        if time_range is not None:
            coord_time_index_slice: slice | None = _time_range_to_index_slice(zarr_path, time_range, storage_options)
        else:
            coord_time_index_slice = None

        # Precomputed CSV coordinates use absolute integer indices into the
        # original Zarr time dimension. Therefore, we filter the CSV rows using
        # the requested time subset, but keep the Zarr itself unsliced so that
        # the stored ``t`` coordinates remain valid.
        self._time_index_slice = None
        super().__init__(
"""

if old not in s:
    raise SystemExit("Could not find precomputed time subset block.")

s = s.replace(old, new, 1)

old = """        self.coords = pd.read_csv(csv_path).sort_values("t")
        if self._time_index_slice is not None:
            t_start = self._time_index_slice.start
            t_stop = self._time_index_slice.stop
            self.coords = self.coords[(self.coords["t"] >= t_start) & (self.coords["t"] < t_stop)].reset_index(
                drop=True
            )
"""

new = """        self.coords = pd.read_csv(csv_path).sort_values("t")
        if coord_time_index_slice is not None:
            t_start = coord_time_index_slice.start
            t_stop = coord_time_index_slice.stop
            self.coords = self.coords[(self.coords["t"] >= t_start) & (self.coords["t"] < t_stop)].reset_index(
                drop=True
            )
"""

if old not in s:
    raise SystemExit("Could not find precomputed CSV filtering block.")

s = s.replace(old, new, 1)

p.write_text(s)
print("Updated", p)
