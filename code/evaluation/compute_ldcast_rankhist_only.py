#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from pysteps import verification as pysteps_verification


# ============================================================
# Final thesis rank-histogram definition
# ============================================================

RANK_X_MIN_MM_H = 0.1
RANK_LEAD_TIMES_MIN = [5, 50, 100]

# lead index -> lead time
RANK_LEADS = {
    0: 5,
    9: 50,
    19: 100,
}


# ============================================================
# LDCast artificial zero-dBZ floor
# ============================================================

ZR_A = 200.0
ZR_B = 1.6

ZERO_DBZ_RR_FLOAT32 = np.float32(
    (1.0 / ZR_A) ** (1.0 / ZR_B)
)

ZERO_DBZ_RR_FLOAT16 = np.float32(
    np.float16(ZERO_DBZ_RR_FLOAT32)
)


def remove_ldcast_zero_floor(samples):
    """
    Convert the artificial rain-rate value corresponding to
    clipped 0 dBZ back to true dry = 0 mm/h.
    """

    samples = np.asarray(
        samples,
        dtype=np.float32,
    ).copy()

    finite = np.isfinite(samples)

    floor_mask = finite & (
        np.isclose(
            samples,
            ZERO_DBZ_RR_FLOAT16,
            rtol=0.0,
            atol=1e-7,
        )
        |
        np.isclose(
            samples,
            ZERO_DBZ_RR_FLOAT32,
            rtol=0.0,
            atol=1e-7,
        )
    )

    samples[floor_mask] = 0.0

    return samples


# ============================================================
# Belgium mask
# ============================================================

def load_belgium_mask(path):
    mask = np.load(path)

    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Belgium mask must be 2-D, got {mask.shape}"
        )

    if mask.dtype != bool:
        mask = np.isfinite(mask) & (mask > 0)

    return mask.astype(bool)


def align_mask(mask, target_shape):
    """
    Symmetrically align 700x700 Belgium mask with 704x704
    prediction fields.

    For 700 -> 704 this gives two False pixels on each side.
    """

    target_h, target_w = target_shape

    h, w = mask.shape

    if (h, w) == (target_h, target_w):
        return mask

    dh = target_h - h
    dw = target_w - w

    if dh < 0 or dw < 0:
        raise ValueError(
            f"Mask {mask.shape} is larger than target {target_shape}"
        )

    pad_top = dh // 2
    pad_bottom = dh - pad_top

    pad_left = dw // 2
    pad_right = dw - pad_left

    out = np.pad(
        mask,
        (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        ),
        mode="constant",
        constant_values=False,
    )

    if out.shape != tuple(target_shape):
        raise ValueError(
            f"Aligned mask shape {out.shape} != {target_shape}"
        )

    return out


# ============================================================
# Event alignment
# ============================================================

def observation_indices(
    forecast_t_starts,
    observation_group,
):
    """
    Match prediction events to observations using t_start.
    """

    if "t_start" not in observation_group:
        raise RuntimeError(
            "Observation store has no t_start array."
        )

    obs_t_starts = np.asarray(
        observation_group["t_start"][:]
    ).astype(np.int64)

    lookup = {
        int(t): i
        for i, t in enumerate(obs_t_starts)
    }

    missing = [
        int(t)
        for t in forecast_t_starts
        if int(t) not in lookup
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} forecast events are missing "
            f"from observation store. First: {missing[:10]}"
        )

    return np.asarray(
        [lookup[int(t)] for t in forecast_t_starts],
        dtype=np.int64,
    )


# ============================================================
# Main computation
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction-root",
        required=True,
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
        "--split",
        required=True,
        choices=["10pct", "full"],
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    pred_root = Path(args.prediction_root)
    obs_root = Path(args.observation_root)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    belgium_mask_700 = load_belgium_mask(
        args.belgium_mask
    )

    rank_rows = []

    total_events = 0

    # ========================================================
    # Process all 12 monthly LDCast stores
    # ========================================================

    for month in range(1, 13):

        pred_store = (
            pred_root
            / "ldcast"
            / f"ldcast_{args.split}_2021_month{month:02d}.zarr"
        )

        monthly_obs = (
            obs_root
            / f"observations_2021_month{month:02d}.zarr"
        )

        yearly_obs = (
            obs_root
            / "observations_2021.zarr"
        )

        obs_store = (
            monthly_obs
            if monthly_obs.exists()
            else yearly_obs
        )

        if not pred_store.exists():
            raise FileNotFoundError(
                f"Missing forecast store: {pred_store}"
            )

        if not obs_store.exists():
            raise FileNotFoundError(
                f"Missing observation store: {obs_store}"
            )

        print()
        print("=" * 80)
        print(
            f"LDCast {args.split} | month {month:02d}"
        )
        print("Forecast:", pred_store)
        print("Observation:", obs_store)
        print("=" * 80)

        fg = zarr.open_group(
            str(pred_store),
            mode="r",
        )

        og = zarr.open_group(
            str(obs_store),
            mode="r",
        )

        forecast = fg["forecast"]
        truth = og["truth"]
        valid = og["valid_mask"]

        if forecast.ndim != 5:
            raise RuntimeError(
                f"Forecast shape must be "
                f"[event,member,lead,y,x], got {forecast.shape}"
            )

        n_events = int(forecast.shape[0])
        n_members = int(forecast.shape[1])
        n_leads = int(forecast.shape[2])

        print(
            "Forecast shape:",
            forecast.shape,
        )

        # Final comparison uses 20 members for all ensembles.
        if n_members != 20:
            raise RuntimeError(
                f"LDCast {args.split} has {n_members} members; "
                f"expected 20."
            )

        if n_leads < 20:
            raise RuntimeError(
                f"Only {n_leads} forecast leads found."
            )

        spatial_shape = tuple(
            map(int, forecast.shape[-2:])
        )

        belgium_mask = align_mask(
            belgium_mask_700,
            spatial_shape,
        )

        t_starts = np.asarray(
            fg["t_start"][:]
        ).astype(np.int64)

        obs_indices = observation_indices(
            t_starts,
            og,
        )

        total_events += n_events

        # ====================================================
        # Events
        # ====================================================

        for event_index in range(n_events):

            if (
                event_index == 0
                or (event_index + 1) % 20 == 0
                or event_index + 1 == n_events
            ):
                print(
                    f"Month {month:02d}: "
                    f"event {event_index + 1}/{n_events}"
                )

            obs_event_index = int(
                obs_indices[event_index]
            )

            # Only +5, +50 and +100.
            for lead_index, lead_time_min in RANK_LEADS.items():

                obs = np.asarray(
                    truth[
                        obs_event_index,
                        lead_index,
                    ],
                    dtype=np.float32,
                )

                observation_valid = np.asarray(
                    valid[
                        obs_event_index,
                        lead_index,
                    ],
                    dtype=bool,
                )

                samples = np.asarray(
                    forecast[
                        event_index,
                        :,
                        lead_index,
                    ],
                    dtype=np.float32,
                )

                samples = remove_ldcast_zero_floor(
                    samples
                )

                # --------------------------------------------
                # Final model-independent score mask
                # --------------------------------------------

                score_mask = (
                    belgium_mask
                    & observation_valid
                    & np.isfinite(obs)
                )

                n_score = int(
                    np.sum(score_mask)
                )

                if n_score == 0:
                    raise RuntimeError(
                        f"No valid Belgium pixels: "
                        f"month={month}, "
                        f"event={event_index}, "
                        f"lead={lead_time_min}"
                    )

                # Forecast validity must NOT modify score mask.
                all_members_finite = np.all(
                    np.isfinite(samples),
                    axis=0,
                )

                bad = (
                    score_mask
                    & (~all_members_finite)
                )

                n_bad = int(np.sum(bad))

                if n_bad > 0:
                    raise RuntimeError(
                        f"Non-finite LDCast forecasts inside Belgium: "
                        f"month={month}, "
                        f"event={event_index}, "
                        f"lead={lead_time_min}, "
                        f"bad_pixels={n_bad}"
                    )

                # --------------------------------------------
                # PySTEPS rank histogram
                # --------------------------------------------
                #
                # IMPORTANT:
                # Do NOT condition on obs >= 0.1.
                #
                # X_min is handled internally by PySTEPS.
                # A pixel is excluded only when the observation
                # AND all ensemble members are below X_min.
                # --------------------------------------------

                rank_fc = samples[
                    :,
                    score_mask,
                ]

                rank_obs = obs[
                    score_mask
                ]

                rankhist = (
                    pysteps_verification.rankhist_init(
                        n_members,
                        RANK_X_MIN_MM_H,
                    )
                )

                pysteps_verification.rankhist_accum(
                    rankhist,
                    rank_fc,
                    rank_obs,
                )

                counts = np.asarray(
                    pysteps_verification.rankhist_compute(
                        rankhist,
                        normalize=False,
                    )
                )

                if counts.shape != (n_members + 1,):
                    raise RuntimeError(
                        f"Unexpected rank histogram shape "
                        f"{counts.shape}; expected {(n_members + 1,)}"
                    )

                for rank_bin, count in enumerate(counts):

                    rank_rows.append(
                        {
                            "model": "ldcast",
                            "evaluation_domain": "belgium",
                            "month": month,
                            "t_start": int(
                                t_starts[event_index]
                            ),
                            "lead_index": int(
                                lead_index
                            ),
                            "lead_time_min": int(
                                lead_time_min
                            ),
                            "rank_bin": int(
                                rank_bin
                            ),
                            "count": int(
                                count
                            ),
                            "n_members": int(
                                n_members
                            ),
                            "rank_x_min_mm_h": float(
                                RANK_X_MIN_MM_H
                            ),
                        }
                    )

    # ========================================================
    # Raw per-event counts
    # ========================================================

    raw = pd.DataFrame(rank_rows)

    raw_path = (
        output_dir
        / "rank_histograms.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    print()
    print("Wrote:", raw_path)
    print("Raw rows:", len(raw))

    # ========================================================
    # Pooled final histogram
    # ========================================================

    group_columns = [
        "model",
        "evaluation_domain",
        "n_members",
        "rank_x_min_mm_h",
        "lead_time_min",
        "rank_bin",
    ]

    summary = (
        raw.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            count=("count", "sum")
        )
    )

    totals = summary.groupby(
        [
            "model",
            "evaluation_domain",
            "n_members",
            "rank_x_min_mm_h",
            "lead_time_min",
        ],
        dropna=False,
    )["count"].transform("sum")

    summary["relative_frequency"] = (
        summary["count"] / totals
    )

    summary[
        "expected_relative_frequency"
    ] = (
        1.0
        / (summary["n_members"] + 1)
    )

    summary_path = (
        output_dir
        / "summary_rank_histograms_by_lead.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Final checks
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL CHECK")
    print("=" * 80)

    print(
        "Events processed:",
        total_events,
    )

    print(
        "Lead times:",
        sorted(
            summary["lead_time_min"].unique()
        ),
    )

    print(
        "Members:",
        sorted(
            summary["n_members"].unique()
        ),
    )

    print(
        "X_min:",
        sorted(
            summary[
                "rank_x_min_mm_h"
            ].unique()
        ),
    )

    print(
        "Bins per lead:",
        summary.groupby(
            "lead_time_min"
        )["rank_bin"]
        .nunique()
        .to_dict(),
    )

    print(
        "Expected frequency:",
        sorted(
            summary[
                "expected_relative_frequency"
            ].unique()
        ),
    )

    for lead in RANK_LEAD_TIMES_MIN:

        q = summary[
            summary["lead_time_min"] == lead
        ]

        print(
            f"+{lead} min frequency sum:",
            q["relative_frequency"].sum(),
        )

    if total_events != 1460:
        raise RuntimeError(
            f"Expected 1460 events, processed {total_events}"
        )

    if len(summary) != 63:
        raise RuntimeError(
            f"Expected 63 pooled rows "
            f"(3 leads x 21 bins), got {len(summary)}"
        )

    print()
    print("Wrote:", summary_path)
    print("SUCCESS")


if __name__ == "__main__":
    main()
