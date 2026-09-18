from pathlib import Path
import yaml


def load_radar_cfg(data_config: str, region: str = "belgium") -> dict:
    data_config = Path(data_config)
    with open(data_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    radar = cfg[region]["radar"]
    variables = radar["variables"]
    var_name = variables[0] if isinstance(variables, list) else variables

    return {
        "zarr_path": radar["data_dir"],
        "train_csv_path": radar["train_csv"],
        "val_csv_path": radar["val_csv"],
        "test_csv_path": radar["test_csv"],
        "var_name": var_name,
        "time_delta_minutes": radar.get("time_delta_minutes", 5),
        "units": radar.get("units", "mm/h"),
        "origin": radar.get("origin", "upper"),
        "transform_precip_to_dbz": radar.get("transform_precip_to_dbz", False),
    }