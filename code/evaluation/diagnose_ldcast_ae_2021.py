#!/usr/bin/env python3

"""
Systematic LDCast autoencoder temporal-position diagnostic.

For every 2021 six-hourly test case:

    20 observed future rain-rate frames
        -> normalized dBZ
        -> LDCast AE encode
        -> latent sequence
        -> LDCast AE decode
        -> 20 reconstructed frames

Metrics are computed:
    - by event and lead
    - pooled by lead
    - pooled by temporal position within each four-frame decoder block

Both:
    - Belgium-only valid pixels
    - full valid RADCLIM domain

are reported.

This is an AE-only diagnostic: no diffusion / PLMS sampling is run.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr


# ============================================================
# Constants: same preprocessing as final 2021 exporter
# ============================================================

DBZ_MEAN = 30.0
DBZ_STD = 30.0
DBZ_MIN = 0.0
DBZ_MAX = 60.0

ZR_A = 200.0
ZR_B = 1.6

TOUT = 20


# ============================================================
# Rain-rate / normalized-dBZ transformations
# ============================================================

def rainrate_to_norm_dbz_np(rr):
    rr = np.asarray(rr, dtype=np.float32)

    norm = np.full_like(
        rr,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(rr)
    positive = valid & (rr > 0.0)

    dbz = np.zeros_like(
        rr,
        dtype=np.float32,
    )

    if np.any(positive):
        dbz[positive] = (
            10.0
            * np.log10(
                ZR_A
                * np.power(
                    rr[positive],
                    ZR_B,
                )
            )
        )

    dbz = np.clip(
        dbz,
        DBZ_MIN,
        DBZ_MAX,
    )

    norm[valid] = (
        dbz[valid] - DBZ_MEAN
    ) / DBZ_STD

    return norm.astype(np.float32)


def norm_to_rainrate_np(x_norm):
    x_norm = np.asarray(
        x_norm,
        dtype=np.float32,
    )

    dbz = (
        x_norm * DBZ_STD
        + DBZ_MEAN
    )

    rr = np.full_like(
        dbz,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(dbz)

    # Lower clipped bound represents dry pixels.
    rr[valid] = 0.0

    wet = valid & (dbz > DBZ_MIN)

    if np.any(wet):
        dbz_wet = np.clip(
            dbz[wet],
            DBZ_MIN,
            DBZ_MAX,
        )

        z = np.power(
            10.0,
            dbz_wet / 10.0,
        )

        rr[wet] = np.power(
            z / ZR_A,
            1.0 / ZR_B,
        )

    return rr.astype(np.float32)


# ============================================================
# LDCast AE loading
# ============================================================

def strip_prefix_if_all(state_dict, prefix):
    if (
        state_dict
        and all(
            k.startswith(prefix)
            for k in state_dict
        )
    ):
        return {
            k[len(prefix):]: v
            for k, v in state_dict.items()
        }

    return state_dict


def load_state_dict_any(path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(obj, dict)
        and "state_dict" in obj
    ):
        state = obj["state_dict"]
    else:
        state = obj

    for prefix in (
        "module.",
        "model.",
    ):
        state = strip_prefix_if_all(
            state,
            prefix,
        )

    return state


def build_autoencoder(
    mlcast_dir,
    checkpoint,
):

    mlcast_dir = Path(mlcast_dir)

    for p in [
        mlcast_dir / "src",
        mlcast_dir / "scripts",
        mlcast_dir,
    ]:
        p = str(p)

        if p not in sys.path:
            sys.path.insert(0, p)

    from mlcast.models.ldcast.autoenc.autoenc import (
        Autoencoder,
        AutoencoderKLNet,
        AutoencoderLoss,
    )

    from mlcast.models.ldcast.autoenc.encoder import (
        SimpleConvEncoder,
        SimpleConvDecoder,
    )

    enc = SimpleConvEncoder()
    dec = SimpleConvDecoder()

    net = AutoencoderKLNet(
        encoder=enc,
        decoder=dec,
    )

    loss = AutoencoderLoss(
        kl_weight=0.01,
    )

    ae = Autoencoder(
        net=net,
        loss=loss,
        antialiaser=None,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-4},
        lr_scheduler_config=None,
    )

    state = load_state_dict_any(
        checkpoint
    )

    ae.load_state_dict(
        state,
        strict=True,
    )

    ae.eval()

    return ae


def ae_decode(ae, z):

    if hasattr(ae, "decode"):
        return ae.decode(z)

    if (
        hasattr(ae, "net")
        and hasattr(ae.net, "decode")
    ):
        return ae.net.decode(z)

    raise AttributeError(
        "Could not find AE decode method."
    )


@torch.inference_mode()
def ae_reconstruct(ae, x):
    z = ae.encode(x)
    rec = ae_decode(ae, z)
    return rec, z


# ============================================================
# Helpers
# ============================================================

def load_belgium_mask(path):

    mask = np.load(path).astype(bool)

    if mask.shape == (700, 700):
        mask = np.pad(
            mask,
            ((2, 2), (2, 2)),
            constant_values=False,
        )

    if mask.shape != (704, 704):
        raise ValueError(
            f"Unexpected Belgium mask shape: "
            f"{mask.shape}"
        )

    return mask


def find_month_store(
    root,
    month,
):
    root = Path(root)

    candidates = [
        root
        / f"observations_2021_month{month:02d}.zarr",

        root
        / f"observations_2021_month{month}.zarr",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No observation Zarr found "
        f"for month {month}"
    )


def metric_components(
    obs_rr,
    rec_rr,
    obs_norm,
    rec_norm,
    mask,
):

    n = int(mask.sum())

    if n == 0:
        return {
            "n": 0,
            "sum_obs_rr": 0.0,
            "sum_rec_rr": 0.0,
            "sum_err_rr": 0.0,
            "sum_abs_rr": 0.0,
            "sum_err_norm": 0.0,
            "sum_abs_norm": 0.0,
        }

    obs_r = obs_rr[mask].astype(
        np.float64,
        copy=False,
    )

    rec_r = rec_rr[mask].astype(
        np.float64,
        copy=False,
    )

    obs_n = obs_norm[mask].astype(
        np.float64,
        copy=False,
    )

    rec_n = rec_norm[mask].astype(
        np.float64,
        copy=False,
    )

    err_rr = rec_r - obs_r
    err_norm = rec_n - obs_n

    return {
        "n": n,

        "sum_obs_rr": float(
            np.sum(
                obs_r,
                dtype=np.float64,
            )
        ),

        "sum_rec_rr": float(
            np.sum(
                rec_r,
                dtype=np.float64,
            )
        ),

        "sum_err_rr": float(
            np.sum(
                err_rr,
                dtype=np.float64,
            )
        ),

        "sum_abs_rr": float(
            np.sum(
                np.abs(err_rr),
                dtype=np.float64,
            )
        ),

        "sum_err_norm": float(
            np.sum(
                err_norm,
                dtype=np.float64,
            )
        ),

        "sum_abs_norm": float(
            np.sum(
                np.abs(err_norm),
                dtype=np.float64,
            )
        ),
    }


def finalize_summary(df, group_cols):

    sums = [
        "n",
        "sum_obs_rr",
        "sum_rec_rr",
        "sum_err_rr",
        "sum_abs_rr",
        "sum_err_norm",
        "sum_abs_norm",
    ]

    out = (
        df
        .groupby(
            group_cols,
            as_index=False,
        )[sums]
        .sum()
    )

    n = out["n"].to_numpy(
        dtype=np.float64
    )

    out["obs_mean_rr"] = (
        out["sum_obs_rr"] / n
    )

    out["rec_mean_rr"] = (
        out["sum_rec_rr"] / n
    )

    out["bias_rr"] = (
        out["sum_err_rr"] / n
    )

    out["mae_rr"] = (
        out["sum_abs_rr"] / n
    )

    out["bias_norm"] = (
        out["sum_err_norm"] / n
    )

    out["mae_norm"] = (
        out["sum_abs_norm"] / n
    )

    return out


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ae",
        required=True,
        choices=["10pct", "full"],
    )

    parser.add_argument(
        "--observation-root",
        required=True,
    )

    parser.add_argument(
        "--belgium-mask",
        required=True,
    )

    parser.add_argument(
        "--mlcast-dir",
        default=(
            "/data/brussel/114/vsc11442/"
            "code/mlcast-ldcast"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Omit for all 1460 cases."
        ),
    )

    args = parser.parse_args()

    mlcast_dir = Path(
        args.mlcast_dir
    )

    if args.ae == "10pct":
        checkpoint = (
            mlcast_dir
            / "results/ldcast/"
            "autoenc_10pct_crop256_b32/"
            "epoch029.ckpt"
        )
    else:
        checkpoint = (
            mlcast_dir
            / "results/ldcast/"
            "autoenc_full_crop256_b32/"
            "epoch002.ckpt"
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required for this diagnostic."
        )

    device = torch.device("cuda")

    print("=" * 80)
    print("LDCast AE 2021 diagnostic")
    print("=" * 80)
    print("AE:", args.ae)
    print("Checkpoint:", checkpoint)
    print("Device:", device)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print()

    belgium = load_belgium_mask(
        args.belgium_mask
    )

    print(
        "Belgium pixels:",
        int(belgium.sum()),
    )

    ae = build_autoencoder(
        mlcast_dir=mlcast_dir,
        checkpoint=checkpoint,
    )

    ae = ae.to(device)
    ae.eval()

    rows = []

    processed = 0
    printed_shape = False

    for month in range(1, 13):

        obs_path = find_month_store(
            args.observation_root,
            month,
        )

        print(
            "\nOpening:",
            obs_path,
        )

        g = zarr.open_group(
            str(obs_path),
            mode="r",
        )

        truth_arr = g["truth"]
        valid_arr = g["valid_mask"]

        t_starts = np.asarray(
            g["t_start"][:],
            dtype=np.int64,
        )

        n_events = int(
            truth_arr.shape[0]
        )

        print(
            f"Month {month:02d}: "
            f"{n_events} events"
        )

        for e in range(n_events):

            if (
                args.max_events is not None
                and processed
                >= args.max_events
            ):
                break

            t_start = int(
                t_starts[e]
            )

            # ------------------------------------------------
            # Corrected observation stores are raw rain rate.
            # Shape: [20, 704, 704]
            # ------------------------------------------------

            truth_rr = np.asarray(
                truth_arr[e],
                dtype=np.float32,
            )

            valid = np.asarray(
                valid_arr[e],
                dtype=bool,
            )

            if truth_rr.shape != (
                TOUT,
                704,
                704,
            ):
                raise ValueError(
                    f"Unexpected truth shape "
                    f"{truth_rr.shape} "
                    f"for t_start={t_start}"
                )

            # Convert raw truth to the exact normalized-dBZ
            # representation seen by the AE.
            truth_norm = (
                rainrate_to_norm_dbz_np(
                    truth_rr
                )
            )

            # Invalid values become model minimum -1,
            # matching full-domain inference handling.
            model_input = np.where(
                valid
                & np.isfinite(
                    truth_norm
                ),
                truth_norm,
                -1.0,
            ).astype(np.float32)

            x = (
                torch
                .from_numpy(model_input)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.inference_mode():

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):

                    rec, latent = (
                        ae_reconstruct(
                            ae,
                            x,
                        )
                    )

            if not printed_shape:

                print(
                    "\nSanity shapes:"
                )

                print(
                    "AE input:",
                    tuple(x.shape),
                )

                print(
                    "AE latent:",
                    tuple(latent.shape),
                )

                print(
                    "AE reconstruction:",
                    tuple(rec.shape),
                )

                printed_shape = True

            rec_norm = (
                rec
                .detach()
                .float()
                .cpu()
                .numpy()[0, 0]
                .astype(np.float32)
            )

            if rec_norm.shape != (
                TOUT,
                704,
                704,
            ):
                raise ValueError(
                    "Unexpected AE reconstruction "
                    f"shape: {rec_norm.shape}"
                )

            rec_rr = (
                norm_to_rainrate_np(
                    rec_norm
                )
            )

            # ------------------------------------------------
            # Lead-wise metrics
            # ------------------------------------------------

            for lead in range(TOUT):

                base_valid = (
                    valid[lead]
                    & np.isfinite(
                        truth_rr[lead]
                    )
                    & np.isfinite(
                        truth_norm[lead]
                    )
                    & np.isfinite(
                        rec_rr[lead]
                    )
                    & np.isfinite(
                        rec_norm[lead]
                    )
                )

                domains = {
                    "belgium":
                        base_valid
                        & belgium,

                    "radclim_valid":
                        base_valid,
                }

                for (
                    domain_name,
                    domain_mask,
                ) in domains.items():

                    comp = metric_components(
                        obs_rr=truth_rr[lead],
                        rec_rr=rec_rr[lead],
                        obs_norm=truth_norm[lead],
                        rec_norm=rec_norm[lead],
                        mask=domain_mask,
                    )

                    rows.append({
                        "ae": args.ae,
                        "checkpoint":
                            str(checkpoint),

                        "month":
                            int(month),

                        "t_start":
                            t_start,

                        "lead_index":
                            int(lead),

                        "lead_min":
                            int(
                                (lead + 1)
                                * 5
                            ),

                        "position_in_block":
                            int(
                                lead % 4
                            ),

                        "latent_block":
                            int(
                                lead // 4
                            ),

                        "domain":
                            domain_name,

                        **comp,
                    })

            processed += 1

            if (
                processed % 25
                == 0
            ):
                print(
                    f"Processed "
                    f"{processed} events"
                )

            del (
                x,
                rec,
                latent,
                truth_rr,
                valid,
                truth_norm,
                model_input,
                rec_norm,
                rec_rr,
            )

            if (
                processed % 10
                == 0
            ):
                torch.cuda.empty_cache()
                gc.collect()

        if (
            args.max_events is not None
            and processed
            >= args.max_events
        ):
            break

    # ========================================================
    # Save raw metric components
    # ========================================================

    df = pd.DataFrame(rows)

    per_event_path = (
        output_dir
        / f"ae_{args.ae}_"
          "2021_per_event_lead.csv"
    )

    df.to_csv(
        per_event_path,
        index=False,
    )

    # ========================================================
    # Pooled by lead
    # ========================================================

    by_lead = finalize_summary(
        df,
        [
            "ae",
            "domain",
            "lead_index",
            "lead_min",
            "position_in_block",
            "latent_block",
        ],
    )

    by_lead_path = (
        output_dir
        / f"ae_{args.ae}_"
          "2021_by_lead.csv"
    )

    by_lead.to_csv(
        by_lead_path,
        index=False,
    )

    # ========================================================
    # Pooled by temporal position within 4-frame block
    # ========================================================

    by_position = finalize_summary(
        df,
        [
            "ae",
            "domain",
            "position_in_block",
        ],
    )

    by_position_path = (
        output_dir
        / f"ae_{args.ae}_"
          "2021_by_position.csv"
    )

    by_position.to_csv(
        by_position_path,
        index=False,
    )

    # ========================================================
    # Pooled by 5 latent blocks
    # ========================================================

    by_block = finalize_summary(
        df,
        [
            "ae",
            "domain",
            "latent_block",
        ],
    )

    by_block_path = (
        output_dir
        / f"ae_{args.ae}_"
          "2021_by_latent_block.csv"
    )

    by_block.to_csv(
        by_block_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(
        "Processed events:",
        processed,
    )

    if args.max_events is None:

        if processed != 1460:
            raise RuntimeError(
                "Expected 1460 events, "
                f"processed {processed}"
            )

        print(
            "All 1460 2021 test "
            "cases processed."
        )

    print(
        "\nBelgium-only "
        "position diagnostic:"
    )

    print(
        by_position[
            by_position["domain"]
            == "belgium"
        ][
            [
                "ae",
                "position_in_block",
                "n",
                "obs_mean_rr",
                "rec_mean_rr",
                "bias_rr",
                "mae_rr",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(per_event_path)
    print(by_lead_path)
    print(by_position_path)
    print(by_block_path)


if __name__ == "__main__":
    main()
