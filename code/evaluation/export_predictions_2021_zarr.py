#!/usr/bin/env python3
"""Export 2021 6-hourly full-domain nowcast predictions to Zarr stores.

By default, the script writes all selected events to one Zarr store. If
``--month`` is supplied, only that calendar month is exported. The output format
is intentionally simple:

Forecast stores:
    forecast[event, member, lead, y, x] in rain-rate units (mm/h)

Observation stores:
    past[event, input_time, y, x] in rain-rate units (mm/h)
    truth[event, lead, y, x] in rain-rate units (mm/h)
    valid_mask[event, lead, y, x]

All prediction fields are saved without masking by future truth availability.
Validity masks are saved separately for scoring.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

try:
    import torch
except Exception:
    torch = None

try:
    import zarr
    from numcodecs import Blosc
except Exception as e:  # pragma: no cover
    raise ImportError("This exporter needs zarr and numcodecs in the active environment.") from e


DBZ_MEAN = 30.0
DBZ_STD = 30.0
ZR_A = 200.0
ZR_B = 1.6
DBZ_MIN = 0.0
DBZ_MAX = 60.0
NO_RAIN_NORM_VALUE = -1.0


def norm_to_dbz_np(x_norm: np.ndarray) -> np.ndarray:
    return np.asarray(x_norm, dtype=np.float32) * DBZ_STD + DBZ_MEAN


def dbz_to_rainrate_np(dbz: np.ndarray) -> np.ndarray:
    dbz = np.asarray(dbz, dtype=np.float32)

    rr = np.full_like(
        dbz,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(dbz)

    # The preprocessing clips all values below 0 dBZ to 0 dBZ.
    # Therefore the lower clipped bound is treated as dry during
    # inverse transformation.
    rr[valid] = 0.0

    wet = valid & (dbz > DBZ_MIN)

    if np.any(wet):
        dbz_wet = np.clip(
            dbz[wet],
            DBZ_MIN,
            DBZ_MAX,
        )

        z = 10.0 ** (
            dbz_wet / 10.0
        )

        rr[wet] = (
            z / ZR_A
        ) ** (
            1.0 / ZR_B
        )

    return rr


def norm_to_rainrate_np(x_norm: np.ndarray) -> np.ndarray:
    return dbz_to_rainrate_np(norm_to_dbz_np(x_norm))


def rainrate_to_norm_dbz_np(rr: np.ndarray) -> np.ndarray:
    rr = np.asarray(rr, dtype=np.float32)
    norm = np.full_like(rr, np.nan, dtype=np.float32)
    valid = np.isfinite(rr)
    positive = valid & (rr > 0)
    dbz = np.zeros_like(rr, dtype=np.float32)
    dbz[positive] = 10.0 * np.log10(ZR_A * np.power(rr[positive], ZR_B))
    dbz = np.clip(dbz, DBZ_MIN, DBZ_MAX)
    norm[valid] = (dbz[valid] - DBZ_MEAN) / DBZ_STD
    return norm.astype(np.float32)


def pad_last_two_dims_to_square(arr: np.ndarray, target_size: int, fill_value=np.nan):
    arr = np.asarray(arr)
    h, w = arr.shape[-2], arr.shape[-1]
    if h > target_size or w > target_size:
        raise ValueError(f"Input domain {h}x{w} is larger than target {target_size}x{target_size}")
    pad_h = target_size - h
    pad_w = target_size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_spec = [(0, 0)] * arr.ndim
    pad_spec[-2] = (pad_top, pad_bottom)
    pad_spec[-1] = (pad_left, pad_right)
    padded = np.pad(arr, pad_spec, mode="constant", constant_values=fill_value)
    return padded, {
        "original_h": int(h),
        "original_w": int(w),
        "pad_top": int(pad_top),
        "pad_bottom": int(pad_bottom),
        "pad_left": int(pad_left),
        "pad_right": int(pad_right),
        "target_size": int(target_size),
    }


def load_event_sequence(
    zarr_path: str,
    var_name: str,
    t_start: int,
    tin: int,
    tout: int,
    target_size: int,
) -> dict[str, Any]:

    ds = xr.open_zarr(zarr_path)

    # Original RADCLIM rain rate in mm/h.
    rr = (
        ds[var_name]
        .isel(
            time=slice(
                int(t_start),
                int(t_start) + tin + tout,
            )
        )
        .values
        .astype(np.float32)
    )

    # Preserve the original rain-rate values for observations
    # and PySTEPS.
    rr_padded, pad_info = pad_last_two_dims_to_square(
        rr,
        target_size=target_size,
        fill_value=np.nan,
    )

    # Neural models still use the transformed/normalized dBZ input.
    norm = rainrate_to_norm_dbz_np(rr)

    norm_padded, _ = pad_last_two_dims_to_square(
        norm,
        target_size=target_size,
        fill_value=np.nan,
    )

    valid_mask = np.isfinite(rr_padded)

    model_input = np.where(
        valid_mask,
        norm_padded,
        NO_RAIN_NORM_VALUE,
    ).astype(np.float32)

    return {
        "t_start": int(t_start),

        # Original physical rain rate.
        "orig_rr": rr_padded.astype(np.float32),

        # Normalized representation used by neural models.
        "orig_norm": norm_padded.astype(np.float32),

        "model_input": model_input,
        "valid_mask": valid_mask,
        "pad_info": pad_info,
    }


def _compressor():
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def open_group_compat(out_zarr: Path):
    """Open a writable Zarr group.

    The HPC environment may import either Zarr 2 or Zarr 3 depending on the
    active virtual environment.  Zarr 3 removed Group.create_dataset(), while
    older code commonly uses it.  This helper keeps the output compatible with
    both APIs and requests a v2 store when the installed Zarr supports the
    zarr_format argument.
    """
    try:
        return zarr.open_group(str(out_zarr), mode="w", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(out_zarr), mode="w")


def create_dataset_compat(root, name: str, **kwargs):
    """Create a Zarr array with both Zarr 2 and Zarr 3 APIs.

    Zarr 3 create_array() does not allow data=... together with dtype=...,
    while the old create_dataset() style commonly used both.  For metadata
    arrays created from data, infer dtype from the data and remove dtype before
    calling create_array().
    """
    if hasattr(root, "create_dataset"):
        return root.create_dataset(name, **kwargs)

    kwargs_z3 = dict(kwargs)
    if "data" in kwargs_z3:
        kwargs_z3.pop("dtype", None)

    try:
        return root.create_array(name, **kwargs_z3)
    except Exception as first_error:
        kwargs_no_comp = dict(kwargs_z3)
        kwargs_no_comp.pop("compressor", None)
        try:
            return root.create_array(name, **kwargs_no_comp)
        except Exception:
            raise first_error


def init_forecast_store(out_zarr: Path, events: pd.DataFrame, n_members: int, tout: int, target_size: int, args) -> Any:
    if out_zarr.exists():
        if args.overwrite:
            shutil.rmtree(out_zarr)
        else:
            raise FileExistsError(f"Output Zarr already exists: {out_zarr}. Use --overwrite to replace it.")
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = open_group_compat(out_zarr)
    comp = _compressor()
    n_events = len(events)
    create_dataset_compat(root, 
        "forecast",
        shape=(n_events, n_members, tout, target_size, target_size),
        chunks=(1, min(n_members, 4), tout, 352, 352),
        dtype="float16",
        compressor=comp,
        fill_value=np.nan,
    )
    create_dataset_compat(root, "t_start", data=events["t_start"].to_numpy(np.int64), dtype="int64", compressor=comp)
    create_dataset_compat(root, "event_index", data=events["event_index"].to_numpy(np.int64), dtype="int64", compressor=comp)
    create_dataset_compat(root, "start_time", data=events["start_time"].astype("S19").to_numpy(), dtype="S19", compressor=comp)
    root.attrs.update(
        {
            "kind": "forecast",
            "model": args.model,
            "split": args.split,
            "checkpoint": str(args.checkpoint or ""),
            "unit": "mm/h",
            "tin": int(args.tin),
            "tout": int(args.tout),
            "target_size": int(args.target_size),
            "n_members": int(n_members),
            "month": int(args.month) if args.month is not None else -1,
            "event_scope": "month" if args.month is not None else "full_year",
            "prediction_masking": "none; validity masks are stored in the observation Zarr",
        }
    )
    return root


def init_observation_store(out_zarr: Path, events: pd.DataFrame, tin: int, tout: int, target_size: int, args) -> Any:
    if out_zarr.exists():
        if args.overwrite:
            shutil.rmtree(out_zarr)
        else:
            raise FileExistsError(f"Output Zarr already exists: {out_zarr}. Use --overwrite to replace it.")
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = open_group_compat(out_zarr)
    comp = _compressor()
    n_events = len(events)
    create_dataset_compat(root, 
        "past",
        shape=(n_events, tin, target_size, target_size),
        chunks=(1, tin, 352, 352),
        dtype="float16",
        compressor=comp,
        fill_value=np.nan,
    )
    create_dataset_compat(root, 
        "truth",
        shape=(n_events, tout, target_size, target_size),
        chunks=(1, tout, 352, 352),
        dtype="float16",
        compressor=comp,
        fill_value=np.nan,
    )
    create_dataset_compat(root, 
        "valid_mask",
        shape=(n_events, tout, target_size, target_size),
        chunks=(1, tout, 352, 352),
        dtype="bool",
        compressor=comp,
        fill_value=False,
    )
    create_dataset_compat(root, "t_start", data=events["t_start"].to_numpy(np.int64), dtype="int64", compressor=comp)
    create_dataset_compat(root, "event_index", data=events["event_index"].to_numpy(np.int64), dtype="int64", compressor=comp)
    create_dataset_compat(root, "start_time", data=events["start_time"].astype("S19").to_numpy(), dtype="S19", compressor=comp)
    root.attrs.update(
        {
            "kind": "observations",
            "unit": "mm/h",
            "tin": int(tin),
            "tout": int(tout),
            "target_size": int(target_size),
            "month": int(args.month) if args.month is not None else -1,
            "event_scope": "month" if args.month is not None else "full_year",
        }
    )
    return root


def extract_net_state_dict(ckpt):
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        new_k = k
        for prefix in ["net.net.", "net.", "model.net.", "model.", "module."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        cleaned[new_k] = v
    return cleaned


def load_convgru_model(args, device: str):
    if torch is None:
        raise ImportError("torch is required for ConvGRU inference")
    mlcast_dir = Path(args.mlcast_dir)
    for p in [mlcast_dir, mlcast_dir / "src"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from mlcast.modules import ConvGRU

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ConvGRU()
    model.load_state_dict(extract_net_state_dict(ckpt), strict=True)
    model.to(device).eval()
    return model


def load_convgru_ensemble_model(args, device: str):
    if torch is None:
        raise ImportError("torch is required for ConvGRU ensemble inference")
    ens_root = Path(args.convgru_ensemble_dir)
    if str(ens_root) not in sys.path:
        sys.path.insert(0, str(ens_root))
    from convgru_ensemble.lightning_model import RadarLightningModel

    model = RadarLightningModel.from_checkpoint(str(args.checkpoint), device=device)
    model.to(device).eval()
    return model


def convgru_ensemble_output_to_members(pred, tout: int):
    pred = pred.detach().float().cpu()
    if pred.ndim == 5:
        pred0 = pred[0]
        if pred0.shape[0] == tout:      # [T, M, H, W]
            return pred0.permute(1, 0, 2, 3).contiguous().numpy()
        if pred0.shape[1] == tout:      # [M, T, H, W]
            return pred0.contiguous().numpy()
    if pred.ndim == 4:
        if pred.shape[1] == tout:
            return pred.contiguous().numpy()
        if pred.shape[0] == tout:
            return pred.permute(1, 0, 2, 3).contiguous().numpy()
    raise ValueError(f"Cannot infer ensemble layout from prediction shape {tuple(pred.shape)}")


def predict_convgru_samples_norm(
    model,
    event_data,
    args,
    ensemble: bool,
) -> np.ndarray:

    device = args.device

    x_np = event_data["model_input"][: args.tin]

    x = (
        torch.from_numpy(x_np[:, None])
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():

        if ensemble:
            y_hat = model(
                x,
                forecast_steps=args.tout,
                ensemble_size=args.n_members,
            )
        else:
            y_hat = model(
                x,
                args.tout,
            )

    if ensemble:
        samples = convgru_ensemble_output_to_members(
            y_hat,
            args.tout,
        )

    else:
        y_hat = y_hat.detach().float().cpu()

        samples = (
            y_hat[0, :, 0]
            .numpy()[None, ...]
        )

    del x, y_hat

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return samples.astype(np.float32)


def load_unet_network(args, device: str, ensemble: bool):
    if torch is None:
        raise ImportError("torch is required for U-Net inference")
    unet_src = Path(args.unet_root) / "src"
    sys.path.insert(0, str(unet_src))
    for module_name in list(sys.modules):
        if module_name == "mlcast" or module_name.startswith("mlcast."):
            del sys.modules[module_name]
    from mlcast.models import UNetModel, StochasticUNetModel

    if ensemble:
        network = StochasticUNetModel(
            input_channels=1,
            input_steps=args.tin,
            base_channels=32,
            num_blocks=4,
            noise_channels=args.unet_noise_channels,
        )
    else:
        network = UNetModel(
            input_channels=1,
            input_steps=args.tin,
            base_channels=32,
            num_blocks=4,
        )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    network_state = {k.replace("network.", "", 1): v for k, v in state_dict.items() if k.startswith("network.")}
    if not network_state:
        raise ValueError(f"No keys starting with 'network.' found in checkpoint: {args.checkpoint}")
    missing, unexpected = network.load_state_dict(network_state, strict=False)
    print(f"Loaded U-Net network. missing={len(missing)} unexpected={len(unexpected)}")
    network.to(device).eval()
    return network


def predict_unet_samples_norm(network, event_data, args, ensemble: bool) -> np.ndarray:
    device = args.device
    x_np = event_data["model_input"][: args.tin]
    x = torch.from_numpy(x_np[:, None]).unsqueeze(0).float().to(device)
    with torch.no_grad():
        y_hat = network(x, steps=args.tout, ensemble_size=args.n_members if ensemble else 1).detach().float().cpu()
    if y_hat.ndim == 5 and y_hat.shape[2] == 1:
        samples = y_hat[0, :, 0].numpy()[None, ...]
    elif y_hat.ndim == 5 and y_hat.shape[2] > 1:
        samples = y_hat[0].permute(1, 0, 2, 3).numpy()
    elif y_hat.ndim == 6:
        samples = y_hat[0, :, :, 0].permute(1, 0, 2, 3).numpy()
    else:
        raise ValueError(f"Unsupported U-Net output shape: {tuple(y_hat.shape)}")
    del x, y_hat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return samples.astype(np.float32)


def predict_pysteps_rr(
    event_data,
    args,
) -> np.ndarray:

    import inspect

    from pysteps import motion, nowcasts
    from pysteps.utils import transformation

    # ============================================================
    # Reference-style PySTEPS preprocessing
    # ============================================================

    RAIN_THRESHOLD_MM_H = 0.1
    DRY_VALUE_DBR = -15.0
    RAIN_THRESHOLD_DBR = -10.0  # 10*log10(0.1)

    # Common experiment has four input fields:
    # t-15, t-10, t-5, t.
    #
    # STEPS with AR(2) requires ar_order + 1 = 3 fields,
    # therefore use t-10, t-5, t.
    past_rr = np.asarray(
        event_data["orig_rr"][: args.tin],
        dtype=np.float32,
    )[-3:].copy()

    # ------------------------------------------------------------
    # PySTEPS logarithmic rain-rate transform
    # ------------------------------------------------------------

    past_dbr, _ = transformation.dB_transform(
        past_rr,
        threshold=RAIN_THRESHOLD_MM_H,
        zerovalue=DRY_VALUE_DBR,
    )

    past_dbr = np.asarray(
        past_dbr,
        dtype=np.float32,
    )

    # Follow the PySTEPS example: missing input values are represented
    # by the dry transformed value before optical flow / STEPS.
    past_dbr[~np.isfinite(past_dbr)] = DRY_VALUE_DBR

    # ============================================================
    # Lucas-Kanade optical flow
    # ============================================================

    oflow_method = motion.get_method("LK")

    velocity = np.asarray(
        oflow_method(past_dbr),
        dtype=np.float32,
    )

    if velocity.shape != (
        2,
        args.target_size,
        args.target_size,
    ):
        raise ValueError(
            "Unexpected PySTEPS velocity shape: "
            f"{velocity.shape}"
        )

    if not np.isfinite(velocity).all():
        raise RuntimeError(
            "PySTEPS Lucas-Kanade velocity contains "
            "non-finite values."
        )

    # ============================================================
    # STEPS
    # ============================================================

    nowcast_method = nowcasts.get_method("steps")

    steps_signature = inspect.signature(
        nowcast_method
    )

    # PySTEPS <=1.4 uses R_thr, newer versions use precip_thr.
    if "R_thr" in steps_signature.parameters:
        threshold_kw = {
            "R_thr": RAIN_THRESHOLD_DBR,
        }

    elif "precip_thr" in steps_signature.parameters:
        threshold_kw = {
            "precip_thr": RAIN_THRESHOLD_DBR,
        }

    else:
        raise RuntimeError(
            "Installed PySTEPS STEPS implementation has neither "
            "'R_thr' nor 'precip_thr'."
        )

    print(
        "Running reference-style PySTEPS:",
        f"members={args.n_members}",
        "inputs=3",
        "threshold=0.1 mm/h",
    )

    try:
        forecast_dbr = nowcast_method(
            past_dbr,
            velocity,
            args.tout,
            n_ens_members=args.n_members,
            n_cascade_levels=6,
            kmperpixel=1.0,
            timestep=5.0,
            decomp_method="fft",
            bandpass_filter_method="gaussian",
            noise_method="nonparametric",
            vel_pert_method="bps",
            mask_method="incremental",
            probmatching_method="cdf",
            seed=args.seed,
            num_workers=1,
            **threshold_kw,
        )
    
    except RuntimeError as exc:

        if "nonstationary AR" not in str(exc):
            raise

        print(
            "WARNING: probabilistic STEPS failed for "
            f"t_start={event_data['t_start']} because of "
            f"nonstationary AR process: {exc}"
        )

        print(
            "Using deterministic semi-Lagrangian extrapolation "
            "for this case."
        )

        extrapolation_method = nowcasts.get_method(
            "extrapolation"
        )

        deterministic_dbr = extrapolation_method(
            past_dbr[-1],
            velocity,
            args.tout,
            extrap_method="semilagrangian",
            extrap_kwargs={
                "outval": "min",
            },
        )

        # [lead, H, W] -> [member, lead, H, W]
        forecast_dbr = np.repeat(
            deterministic_dbr[None, ...],
            args.n_members,
            axis=0,
        )

    forecast_dbr = np.asarray(
        forecast_dbr,
        dtype=np.float32,
    )

    # PySTEPS' default semi-Lagrangian boundary treatment can return
    # non-finite values for trajectories originating outside the domain.
    # Preserve all finite STEPS predictions and assign only these undefined
    # boundary values to the transformed dry/background value.
    n_boundary_nonfinite = int(
        np.count_nonzero(
            ~np.isfinite(forecast_dbr)
        )
    )
    
    if n_boundary_nonfinite > 0:
        print(
            "PySTEPS undefined boundary values before dry fill:",
            n_boundary_nonfinite,
        )
    
    forecast_dbr = np.where(
        np.isfinite(forecast_dbr),
        forecast_dbr,
        DRY_VALUE_DBR,
    ).astype(np.float32)

    expected_shape = (
        args.n_members,
        args.tout,
        args.target_size,
        args.target_size,
    )

    if forecast_dbr.shape != expected_shape:
        raise ValueError(
            "Unexpected PySTEPS transformed forecast shape: "
            f"{forecast_dbr.shape}; "
            f"expected {expected_shape}"
        )

    # ============================================================
    # Back-transform exactly as in the PySTEPS reference example
    # ============================================================

    forecast_rr, _ = transformation.dB_transform(
        forecast_dbr,
        threshold=RAIN_THRESHOLD_DBR,
        inverse=True,
    )

    forecast_rr = np.asarray(
        forecast_rr,
        dtype=np.float32,
    )
    
    # Boundary values returned as non-finite by the default
    # semi-Lagrangian STEPS extrapolation were assigned the
    # transformed dry/background value above. The inverse-transformed
    # forecast should therefore now be fully finite.
    
    if not np.isfinite(forecast_rr).all():
        raise RuntimeError(
            "PySTEPS still contains non-finite values after "
            "explicit dry-boundary handling."
        )
    
    forecast_rr = np.maximum(
        forecast_rr,
        0.0,
    ).astype(np.float32)

    # ============================================================
    # Diagnostics only
    # ============================================================

    finite = np.isfinite(forecast_rr)

    n_bad = int(
        np.count_nonzero(~finite)
    )

    bad_by_lead = (
        (~finite)
        .sum(axis=(0, 2, 3))
        .astype(int)
    )

    finite_values = forecast_rr[finite]

    member_spread = float(
        np.nanmean(
            np.nanstd(
                forecast_rr,
                axis=0,
            )
        )
    )

    print(
        "PySTEPS output:",
        f"shape={forecast_rr.shape}",
        f"nonfinite={n_bad}",
        f"min={float(np.min(finite_values)) if finite_values.size else np.nan}",
        f"max={float(np.max(finite_values)) if finite_values.size else np.nan}",
        f"mean member std={member_spread:.6f}",
    )

    if n_bad > 0:
        print(
            "Non-finite values by lead:",
            bad_by_lead.tolist(),
        )

    return forecast_rr

def clear_torch_memory():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def ae_decode(ae, z):
    """Decode latent LDCast samples for both wrapped and unwrapped AE objects."""
    if hasattr(ae, "decode"):
        return ae.decode(z)
    if hasattr(ae, "net") and hasattr(ae.net, "decode"):
        return ae.net.decode(z)
    raise AttributeError("Could not find decode method on autoencoder.")


def default_ldcast_autoenc(args) -> str:
    """Return the default AE checkpoint used with the final AE256 LDCast runs."""
    base = Path(args.mlcast_dir) / "results" / "ldcast"
    if args.split == "10pct":
        return str(base / "autoenc_10pct_crop256_b32" / "epoch029.ckpt")
    return str(base / "autoenc_full_crop256_b32" / "epoch002.ckpt")


def load_ldcast_model(args, device):
    """Load LDCast generator + autoencoder for direct inference-to-Zarr.
    """
    if torch is None:
        raise ImportError("torch is required for LDCast inference")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for MODEL=ldcast")

    mlcast_dir = Path(args.mlcast_dir)
    for p in [mlcast_dir, mlcast_dir / "src", mlcast_dir / "scripts"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from Minh_train_genforecast import setup_model, load_state_dict_any, LinearBetaScheduler

    # Needed when loading checkpoints that refer to this scheduler class.
    import __main__
    __main__.LinearBetaScheduler = LinearBetaScheduler

    autoenc_weights = args.ldcast_autoenc_weights_fn or default_ldcast_autoenc(args)
    model_dir = args.ldcast_model_dir or str(Path(args.checkpoint).parent)

    if not Path(autoenc_weights).exists():
        raise FileNotFoundError(f"LDCast autoencoder checkpoint not found: {autoenc_weights}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"LDCast generator checkpoint not found: {args.checkpoint}")

    print("Building LDCast model...")
    print("LDCast autoencoder:", autoenc_weights)
    print("LDCast generator:", args.checkpoint)
    print("LDCast model_dir:", model_dir)

    model, _trainer = setup_model(
        autoenc_weights_fn=autoenc_weights,
        model_dir=model_dir,
        future_timesteps=args.tout,
        lr=args.ldcast_lr,
        max_epochs=1,
        precision=args.ldcast_precision,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    state = load_state_dict_any(args.checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded LDCast checkpoint. missing keys:", len(missing), "unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()
    clear_torch_memory()
    return model


@torch.no_grad()
def sample_ldcast_plms_norm(model, x, y_for_shape, n_members: int, plms_steps: int, device: str) -> np.ndarray:
    """Sample LDCast members in normalized dBZ space.

    Returns [member, lead, y, x].
    """
    model.eval()
    x = x.to(device)
    y_for_shape = y_for_shape.to(device)

    from mlcast.models.ldcast.diffusion.plms import PLMSSampler

    latent_inputs = model.autoencoder.encode(x)
    latent_shape_ref = model.autoencoder.encode(y_for_shape)
    gen_shape = tuple(latent_shape_ref.shape[1:])

    condition = model.net.conditioner(latent_inputs)
    sampler = PLMSSampler(model.net.denoiser)

    members = []
    print("LDCast latent input shape:", tuple(latent_inputs.shape))
    print("LDCast latent forecast shape:", tuple(latent_shape_ref.shape))
    print("LDCast PLMS steps:", plms_steps, "members:", n_members)

    for m in range(n_members):
        print(f"  LDCast sampling member {m + 1}/{n_members}")
        latent_sample, intermediates = sampler.sample(
            S=plms_steps,
            batch_size=1,
            shape=gen_shape,
            conditioning=condition,
            progbar=False,
            verbose=False,
        )
        decoded = ae_decode(model.autoencoder, latent_sample).detach().float().cpu()
        members.append(decoded[0, 0].numpy())  # [lead, y, x]
        del latent_sample, intermediates, decoded
        clear_torch_memory()

    del x, y_for_shape, latent_inputs, latent_shape_ref, condition, sampler
    clear_torch_memory()
    return np.stack(members, axis=0).astype(np.float32)


def predict_ldcast_samples_rr(ldcast_model, event_data, args):
    """Run LDCast directly and return rain-rate samples [member, lead, y, x]."""
    past = event_data["model_input"][: args.tin]
    future_shape_ref = event_data["model_input"][args.tin : args.tin + args.tout]

    # LDCast expects [B, C, T, H, W].
    x = torch.from_numpy(past[None, None]).float()
    y_for_shape = torch.from_numpy(future_shape_ref[None, None]).float()

    samples_norm = sample_ldcast_plms_norm(
        model=ldcast_model,
        x=x,
        y_for_shape=y_for_shape,
        n_members=args.n_members,
        plms_steps=args.ldcast_plms_steps,
        device=args.device,
    )

    # The decoded LDCast samples are in the same normalized dBZ space used by
    # the validation NPZ exporter. Convert them to mm/h before writing Zarr.
    samples_rr = norm_to_rainrate_np(samples_norm)
    samples_rr = np.where(np.isfinite(samples_rr), np.maximum(samples_rr, 0.0), np.nan)
    return samples_rr.astype(np.float32)


def load_predictor(args):
    if args.model == "convgru":
        return load_convgru_model(args, args.device)
    if args.model == "convgru_ens":
        return load_convgru_ensemble_model(args, args.device)
    if args.model == "unet":
        return load_unet_network(args, args.device, ensemble=False)
    if args.model == "unet_ens":
        return load_unet_network(args, args.device, ensemble=True)
    if args.model == "ldcast":
        return load_ldcast_model(args, args.device)
    return None


def predict_samples_rr(predictor, event_data, event_row, args) -> np.ndarray:
    if args.model == "convgru":
        return norm_to_rainrate_np(predict_convgru_samples_norm(predictor, event_data, args, ensemble=False))
    if args.model == "convgru_ens":
        return norm_to_rainrate_np(predict_convgru_samples_norm(predictor, event_data, args, ensemble=True))
    if args.model == "unet":
        return norm_to_rainrate_np(predict_unet_samples_norm(predictor, event_data, args, ensemble=False))
    if args.model == "unet_ens":
        return norm_to_rainrate_np(predict_unet_samples_norm(predictor, event_data, args, ensemble=True))
    if args.model == "pysteps":
        return predict_pysteps_rr(event_data, args)
    if args.model == "ldcast":
        return predict_ldcast_samples_rr(predictor, event_data, args)
    raise ValueError(f"Unsupported model: {args.model}")


def default_n_members(model: str, requested: int | None) -> int:
    if requested is not None:
        return int(requested)

    if model in {"convgru", "unet"}:
        return 1

    if model in {
        "convgru_ens",
        "unet_ens",
        "pysteps",
        "ldcast",
    }:
        return 20

    if model == "observations":
        return 0

    raise ValueError(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["observations", "convgru", "convgru_ens", "unet", "unet_ens", "pysteps", "ldcast"])
    parser.add_argument("--split", default="full", choices=["full", "10pct"])
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--month", type=int, default=None, help="Optional month filter (1-12). If omitted, all events in the CSV are exported.")
    parser.add_argument("--out-zarr", required=True)
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--var-name", default="precip_intensity_EDK")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--n-members", type=int, default=None)
    parser.add_argument("--tin", type=int, default=4)
    parser.add_argument("--tout", type=int, default=20)
    parser.add_argument("--target-size", type=int, default=704)
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-events", type=int, default=None, help="Smoke-test limit after optional month filtering.")

    parser.add_argument("--mlcast-dir", default="/data/brussel/114/vsc11442/code/mlcast-ldcast")
    parser.add_argument("--convgru-ensemble-dir", default="/data/brussel/114/vsc11442/code/convgru-ensemble")
    parser.add_argument("--unet-root", default="/data/brussel/114/vsc11442/code/mlcast-unet")
    parser.add_argument("--unet-noise-channels", type=int, default=4)
    parser.add_argument("--ldcast-autoenc-weights-fn", default="", help="Autoencoder checkpoint for LDCast. If omitted, a split-specific AE256 default is used.")
    parser.add_argument("--ldcast-model-dir", default="", help="Model directory passed to Minh_train_genforecast.setup_model. Defaults to the generator checkpoint directory.")
    parser.add_argument("--ldcast-plms-steps", type=int, default=100)
    parser.add_argument("--ldcast-lr", type=float, default=1e-5)
    parser.add_argument("--ldcast-precision", default="bf16-mixed")
    args = parser.parse_args()

    if args.checkpoint == "":
        args.checkpoint = None
    elif not Path(args.checkpoint).exists() and args.model not in {"observations", "pysteps"}:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.model == "ldcast" and args.checkpoint is None:
        raise ValueError("MODEL=ldcast requires --checkpoint pointing to the generator checkpoint")

    events = pd.read_csv(args.events_csv)
    if args.month is not None:
        events = events[events["month"].astype(int) == int(args.month)].reset_index(drop=True)
    else:
        events = events.reset_index(drop=True)
    if args.max_events is not None:
        events = events.head(args.max_events).copy()
    if len(events) == 0:
        scope = f"month {args.month}" if args.month is not None else "the full event list"
        raise ValueError(f"No events found for {scope} in {args.events_csv}")

    n_members = default_n_members(args.model, args.n_members)
    out_zarr = Path(args.out_zarr)

    print("Model:", args.model, "split:", args.split, "month:", args.month if args.month is not None else "ALL")
    print("Events:", len(events))
    print("Output:", out_zarr)

    if args.model == "observations":
        root = init_observation_store(out_zarr, events, args.tin, args.tout, args.target_size, args)
        for row_i, row in events.iterrows():
            print(f"[{row_i + 1}/{len(events)}] observations t_start={row.t_start} {row.start_time}")
            ev = load_event_sequence(args.zarr_path, args.var_name, int(row.t_start), args.tin, args.tout, args.target_size)
            rr = ev["orig_rr"]
            root["past"][row_i] = rr[: args.tin].astype(np.float16)
            root["truth"][row_i] = rr[args.tin : args.tin + args.tout].astype(np.float16)
            root["valid_mask"][row_i] = ev["valid_mask"][args.tin : args.tin + args.tout]
        print("Done:", out_zarr)
        return

    predictor = load_predictor(args)
    root = init_forecast_store(out_zarr, events, n_members, args.tout, args.target_size, args)

    for row_i, row in events.iterrows():
        print(f"[{row_i + 1}/{len(events)}] {args.model} t_start={row.t_start} {row.start_time}")
        ev = load_event_sequence(args.zarr_path, args.var_name, int(row.t_start), args.tin, args.tout, args.target_size)
        samples_rr = predict_samples_rr(predictor, ev, row, args)
        if samples_rr.ndim != 4:
            raise ValueError(f"Expected samples_rr [member, lead, y, x], got {samples_rr.shape}")
        if samples_rr.shape[1:] != (args.tout, args.target_size, args.target_size):
            raise ValueError(f"Unexpected forecast shape {samples_rr.shape}; expected [M,{args.tout},{args.target_size},{args.target_size}]")
        if samples_rr.shape[0] != n_members:
            raise ValueError(
                f"{args.model}: requested {n_members} members "
                f"but inference returned {samples_rr.shape[0]}."
            )
        
        root["forecast"][row_i] = (
            samples_rr.astype(np.float16)
        )

    print("Done:", out_zarr)


if __name__ == "__main__":
    main()
