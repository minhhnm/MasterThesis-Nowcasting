#!/usr/bin/env python

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from mlcast.models.ldcast.diffusion.plms import PLMSSampler
from Minh_read_mlcast_yaml import load_radar_cfg
from Minh_train_genforecast import (
    setup_model,
    load_state_dict_any,
    LinearBetaScheduler,
)

import __main__
__main__.LinearBetaScheduler = LinearBetaScheduler


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def rainrate_to_norm_dbz(rr):
    rr = rr.astype(np.float32)
    valid = np.isfinite(rr)

    rr_nonneg = np.where(valid, np.maximum(rr, 0.0), 0.0).astype(np.float32)

    # Match training dataset preprocessing:
    # Z = 200 R^1.6, then dBZ clipped to [0, 60].
    z = 200.0 * np.power(rr_nonneg, 1.6)
    dbz = 10.0 * np.log10(z + 1e-16)
    dbz = np.clip(dbz, 0.0, 60.0)

    norm = ((dbz - 30.0) / 30.0).astype(np.float32)

    # Keep no-data as NaN in saved truth/past, but later model_input fills it with -1.
    norm = np.where(valid, norm, np.nan).astype(np.float32)

    return norm, valid


def pad_or_center_crop(arr, target_size):
    """
    arr: [T, H, W]
    returns arr_target, valid_mask_target, pad_info
    """
    t, h, w = arr.shape

    if target_size >= h and target_size >= w:
        pad_top = (target_size - h) // 2
        pad_bottom = target_size - h - pad_top
        pad_left = (target_size - w) // 2
        pad_right = target_size - w - pad_left

        out = np.pad(
            arr,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=np.nan,
        )

        pad_info = {
            "mode": "pad",
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }

        return out, pad_info

    # Center crop fallback, useful for target_size=640.
    y0 = (h - target_size) // 2
    x0 = (w - target_size) // 2
    out = arr[:, y0:y0 + target_size, x0:x0 + target_size]

    pad_info = {
        "mode": "center_crop",
        "y0": y0,
        "x0": x0,
    }

    return out, pad_info


def ae_decode(ae, z):
    if hasattr(ae, "decode"):
        return ae.decode(z)
    if hasattr(ae, "net") and hasattr(ae.net, "decode"):
        return ae.net.decode(z)
    raise AttributeError("Could not find decode method on autoencoder.")


def load_full_domain_batch(data_config, region, t_start, tin, tout, target_size):
    cfg = load_radar_cfg(str(data_config), region)

    print("Using data config:", data_config)
    print("Region:", region)
    print("Zarr:", cfg["zarr_path"])
    print("Variable:", cfg["var_name"])
    print("t_start:", t_start)
    print("target_size:", target_size)

    ds = xr.open_zarr(cfg["zarr_path"])
    da = ds[cfg["var_name"]]

    total = tin + tout
    raw = da.isel(time=slice(t_start, t_start + total)).values.astype(np.float32)
    print("Raw full-domain rain-rate shape:", raw.shape)

    raw_target, pad_info = pad_or_center_crop(raw, target_size=target_size)
    print("Target rain-rate shape:", raw_target.shape)
    print("pad/crop info:", pad_info)

    norm, valid = rainrate_to_norm_dbz(raw_target)

    # Use -1 for invalid/no-data model input, but keep valid mask for final output.
    model_input = np.where(valid, norm, -1.0).astype(np.float32)

    past = model_input[:tin]
    future = model_input[tin:tin + tout]

    x = torch.from_numpy(past[None, None])      # [B, C, Tin, H, W]
    y = torch.from_numpy(future[None, None])    # [B, C, Tout, H, W]

    print("x shape:", tuple(x.shape))
    print("y shape:", tuple(y.shape))

    return x, y, norm, valid, pad_info, cfg


@torch.no_grad()
def sample_ldcast_plms(model, x, y_for_shape, n_members, plms_steps, device):
    model.eval()

    x = x.to(device)
    y_for_shape = y_for_shape.to(device)

    latent_inputs = model.autoencoder.encode(x)
    latent_shape_ref = model.autoencoder.encode(y_for_shape)
    gen_shape = tuple(latent_shape_ref.shape[1:])

    print("latent input shape:", tuple(latent_inputs.shape))
    print("latent forecast shape:", tuple(latent_shape_ref.shape))
    print("PLMS gen_shape:", gen_shape)
    print("PLMS steps:", plms_steps)
    print("members:", n_members)

    condition = model.net.conditioner(latent_inputs)
    sampler = PLMSSampler(model.net.denoiser)

    members = []

    for m in range(n_members):
        print(f"Sampling member {m + 1}/{n_members}")

        latent_sample, intermediates = sampler.sample(
            S=plms_steps,
            batch_size=1,
            shape=gen_shape,
            conditioning=condition,
            progbar=False,
            verbose=False,
        )

        decoded = ae_decode(model.autoencoder, latent_sample)
        decoded = decoded.detach().float().cpu()

        members.append(decoded[0, 0])  # [T, H, W]

        del latent_sample, intermediates, decoded
        clear_memory()

    return torch.stack(members, dim=0)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_config", required=True)
    ap.add_argument("--region", default="belgium")
    ap.add_argument("--autoenc_weights_fn", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_npz", required=True)

    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--future_timesteps", type=int, default=20)
    ap.add_argument("--target_size", type=int, default=704)
    ap.add_argument("--t_start", type=int, required=True)

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--n_members", type=int, default=1)
    ap.add_argument("--sampling_steps", type=int, default=25)

    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    x, y, norm_full, valid_mask, pad_info, cfg = load_full_domain_batch(
        data_config=args.data_config,
        region=args.region,
        t_start=args.t_start,
        tin=args.tin,
        tout=args.future_timesteps,
        target_size=args.target_size,
    )

    clear_memory()

    print("Building LDCast model...")
    model, _trainer = setup_model(
        autoenc_weights_fn=args.autoenc_weights_fn,
        model_dir=args.model_dir,
        future_timesteps=args.future_timesteps,
        lr=args.lr,
        max_epochs=1,
        precision=args.precision,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    print("Loading checkpoint:", args.ckpt_path)
    state = load_state_dict_any(args.ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()

    samples = sample_ldcast_plms(
        model=model,
        x=x,
        y_for_shape=y,
        n_members=args.n_members,
        plms_steps=args.sampling_steps,
        device=device,
    )

    samples_np = samples.numpy().astype(np.float32)

    # Keep future truth availability for evaluation only.
    # Do NOT mask model predictions here, otherwise the forecast figure
    # appears to know future radar availability.
    future_valid = valid_mask[args.tin:args.tin + args.future_timesteps].astype(bool)

    pred_mean = samples_np.mean(axis=0).astype(np.float32)
    pred_std = (
        samples_np.std(axis=0).astype(np.float32)
        if args.n_members > 1
        else np.zeros_like(pred_mean, dtype=np.float32)
    )

    np.savez_compressed(
        out_npz,
        past=norm_full[:args.tin].astype(np.float32),
        truth=norm_full[args.tin:args.tin + args.future_timesteps].astype(np.float32),
        samples=samples_np,
        pred_mean=pred_mean,
        pred_std=pred_std,
        valid_mask=valid_mask.astype(bool),
        future_valid=future_valid.astype(bool),
        t_start=np.array(args.t_start),
        target_size=np.array(args.target_size),
        tin=np.array(args.tin),
        tout=np.array(args.future_timesteps),
        n_members=np.array(args.n_members),
        sampling_steps=np.array(args.sampling_steps),
        checkpoint=str(args.ckpt_path),
        autoencoder=str(args.autoenc_weights_fn),
        mode="true_full_domain_one_pass_unmasked_prediction",
        pad_info=str(pad_info),
    )

    print("Saved:", out_npz)


if __name__ == "__main__":
    main()