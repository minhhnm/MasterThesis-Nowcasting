#!/usr/bin/env python3
"""
Select observation-only qualitative events from the 2021 6-hourly TEST set.

Selection domain
----------------
All event descriptors are calculated over Belgium only using a fixed
geographical mask.

Visualization domain
--------------------
This script does not crop any event. The selected t_start values can later be
visualized over the complete RADCLIM domain.

Selected cases
--------------
1. Intense convection
2. Widespread organized rainfall
3. Rapidly evolving rainfall
4. July 2021 flood event

The first three cases are selected automatically from the 2021 test cases,
excluding the documented 13-16 July flood period.

The flood case is selected independently within 13-16 July 2021 as the
6-hourly event with the largest 95th percentile of 100-minute accumulated
precipitation over Belgium.

No model forecasts are used anywhere in the selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--observation-root",
        required=True,
        help=(
            "Directory containing observations_2021_month01.zarr, "
            "... month12.zarr."
        ),
    )

    ap.add_argument(
        "--belgium-mask",
        required=True,
        help="Belgium geographical mask, normally belgium_mask_700.npy.",
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    ap.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
    )

    ap.add_argument(
        "--selection-separation-steps",
        type=int,
        default=288,
        help=(
            "Minimum separation between generic selected cases in original "
            "5-minute indices. 288 steps = 24 hours."
        ),
    )

    ap.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--flood-start",
        default="2021-07-13 00:00:00",
    )

    ap.add_argument(
        "--flood-end",
        default="2021-07-17 00:00:00",
        help="Exclusive upper bound. Default therefore includes July 13-16.",
    )

    return ap.parse_args()


# ============================================================
# Utilities
# ============================================================

def decode_strings(values) -> list[str]:
    out = []

    for value in values:
        if isinstance(value, (bytes, bytearray)):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))

    return out


def load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npy":
        mask = np.load(path)

    elif path.suffix.lower() == ".npz":
        loaded = np.load(path)

        if len(loaded.files) != 1:
            raise ValueError(
                f"NPZ mask must contain one array; found {loaded.files}"
            )

        mask = loaded[loaded.files[0]]

    else:
        raise ValueError(
            "Belgium mask must currently be .npy or .npz."
        )

    mask = np.asarray(mask).astype(bool)

    if mask.ndim != 2:
        raise ValueError(
            f"Belgium mask must be 2-D, got {mask.shape}"
        )

    return mask


def align_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    Centre-pad the 700x700 Belgium mask to the 704x704 stored grid.

    Also works when the input already matches the target.
    """

    mask = np.asarray(mask, dtype=bool)

    if tuple(mask.shape) == tuple(target_shape):
        return mask

    mh, mw = mask.shape
    th, tw = target_shape

    if mh > th or mw > tw:
        raise ValueError(
            f"Mask shape {mask.shape} is larger than target {target_shape}"
        )

    pad_h = th - mh
    pad_w = tw - mw

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    aligned = np.pad(
        mask,
        (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        ),
        mode="constant",
        constant_values=False,
    )

    if tuple(aligned.shape) != tuple(target_shape):
        raise RuntimeError(
            f"Aligned mask is {aligned.shape}, expected {target_shape}"
        )

    print(
        "Mask alignment:",
        mask.shape,
        "->",
        aligned.shape,
        "padding=",
        (pad_top, pad_bottom, pad_left, pad_right),
    )

    return aligned


def safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values)

    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return np.nan

    return float(np.mean(finite))


def fraction_above_by_lead(
    future: np.ndarray,
    observation_valid: np.ndarray,
    domain_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Fraction of valid Belgian pixels >= threshold for each forecast lead.

    Important:
    outside-Belgium pixels are excluded from the denominator rather than
    treated as non-rain pixels.
    """

    finite = np.isfinite(future)

    valid = (
        observation_valid
        & finite
        & domain_mask[None, :, :]
    )

    numerator = np.sum(
        valid & (future >= threshold),
        axis=(1, 2),
    ).astype(np.float64)

    denominator = np.sum(
        valid,
        axis=(1, 2),
    ).astype(np.float64)

    fraction = np.full(
        len(future),
        np.nan,
        dtype=np.float64,
    )

    good = denominator > 0

    fraction[good] = (
        numerator[good]
        / denominator[good]
    )

    return fraction


def mean_rain_by_lead(
    future: np.ndarray,
    observation_valid: np.ndarray,
    domain_mask: np.ndarray,
) -> np.ndarray:

    finite = np.isfinite(future)

    valid = (
        observation_valid
        & finite
        & domain_mask[None, :, :]
    )

    result = np.full(
        future.shape[0],
        np.nan,
        dtype=np.float64,
    )

    for lead in range(future.shape[0]):
        values = future[lead][valid[lead]]

        if values.size:
            result[lead] = float(
                np.mean(values, dtype=np.float64)
            )

    return result


def weighted_centroid(
    field: np.ndarray,
    observation_valid: np.ndarray,
    domain_mask: np.ndarray,
    threshold: float = 1.0,
) -> tuple[float, float]:

    field = np.asarray(
        field,
        dtype=np.float32,
    )

    valid = (
        domain_mask
        & observation_valid
        & np.isfinite(field)
        & (field >= threshold)
    )

    if not valid.any():
        return np.nan, np.nan

    weights = np.where(
        valid,
        field,
        0.0,
    ).astype(np.float64)

    total = float(weights.sum())

    if total <= 0:
        return np.nan, np.nan

    y, x = np.indices(field.shape)

    cy = float(
        (weights * y).sum()
        / total
    )

    cx = float(
        (weights * x).sum()
        / total
    )

    return cy, cx


# ============================================================
# Observation descriptors
# ============================================================

def event_descriptors(
    future: np.ndarray,
    observation_valid: np.ndarray,
    domain_mask: np.ndarray,
    past: np.ndarray | None = None,
) -> dict[str, float]:

    future = np.asarray(
        future,
        dtype=np.float32,
    )

    observation_valid = np.asarray(
        observation_valid,
        dtype=bool,
    )

    if future.ndim != 3:
        raise ValueError(
            f"Expected future [lead,y,x], got {future.shape}"
        )

    if observation_valid.shape != future.shape:
        raise ValueError(
            "valid_mask and truth shapes differ: "
            f"{observation_valid.shape} vs {future.shape}"
        )

    if domain_mask.shape != future.shape[-2:]:
        raise ValueError(
            f"Domain mask {domain_mask.shape} does not match "
            f"future {future.shape[-2:]}"
        )

    finite_mask = np.isfinite(future)

    score_valid = (
        observation_valid
        & finite_mask
        & domain_mask[None, :, :]
    )

    finite_values = future[score_valid]

    if finite_values.size == 0:
        raise ValueError(
            "No finite observation pixels inside Belgium."
        )

    # --------------------------------------------------------
    # Rainfall coverage
    # --------------------------------------------------------

    area01 = fraction_above_by_lead(
        future,
        observation_valid,
        domain_mask,
        0.1,
    )

    area1 = fraction_above_by_lead(
        future,
        observation_valid,
        domain_mask,
        1.0,
    )

    area5 = fraction_above_by_lead(
        future,
        observation_valid,
        domain_mask,
        5.0,
    )

    area10 = fraction_above_by_lead(
        future,
        observation_valid,
        domain_mask,
        10.0,
    )

    mean_by_lead = mean_rain_by_lead(
        future,
        observation_valid,
        domain_mask,
    )

    # --------------------------------------------------------
    # Structural change between +5 and +100 min
    # --------------------------------------------------------

    first = future[0]
    last = future[-1]

    pair_valid = (
        domain_mask
        & observation_valid[0]
        & observation_valid[-1]
        & np.isfinite(first)
        & np.isfinite(last)
    )

    if pair_valid.any():
        structural_change = float(
            np.mean(
                np.abs(
                    last[pair_valid]
                    - first[pair_valid]
                ),
                dtype=np.float64,
            )
        )
    else:
        structural_change = np.nan

    # --------------------------------------------------------
    # Movement of >1 mm/h rainfall centroid
    # --------------------------------------------------------

    cy0, cx0 = weighted_centroid(
        first,
        observation_valid[0],
        domain_mask,
        threshold=1.0,
    )

    cy1, cx1 = weighted_centroid(
        last,
        observation_valid[-1],
        domain_mask,
        threshold=1.0,
    )

    if np.isfinite(
        [cy0, cx0, cy1, cx1]
    ).all():
        centroid_displacement = float(
            np.hypot(
                cy1 - cy0,
                cx1 - cx0,
            )
        )
    else:
        centroid_displacement = np.nan

    # --------------------------------------------------------
    # Last input -> final forecast-time observation change
    # --------------------------------------------------------

    input_to_end_change = np.nan

    if past is not None:
        past = np.asarray(
            past,
            dtype=np.float32,
        )

        last_input = past[-1]

        pair_input = (
            domain_mask
            & np.isfinite(last_input)
            & observation_valid[-1]
            & np.isfinite(last)
        )

        if pair_input.any():
            input_to_end_change = float(
                np.mean(
                    np.abs(
                        last[pair_input]
                        - last_input[pair_input]
                    ),
                    dtype=np.float64,
                )
            )

    # --------------------------------------------------------
    # 100-minute accumulated precipitation
    #
    # R [mm/h] * 5/60 h for each of 20 leads.
    #
    # For flood ranking we require a pixel to be valid at all
    # twenty forecast times so that missing data cannot reduce
    # the apparent accumulation.
    # --------------------------------------------------------

    all_leads_valid = (
        domain_mask
        & np.all(
            observation_valid
            & finite_mask,
            axis=0,
        )
    )

    if all_leads_valid.any():

        accumulated = (
            np.sum(
                future[:, all_leads_valid],
                axis=0,
                dtype=np.float64,
            )
            * (5.0 / 60.0)
        )

        mean_accumulated = float(
            np.mean(accumulated)
        )

        p95_accumulated = float(
            np.percentile(
                accumulated,
                95,
            )
        )

        p99_accumulated = float(
            np.percentile(
                accumulated,
                99,
            )
        )

        max_accumulated = float(
            np.max(accumulated)
        )

    else:

        mean_accumulated = np.nan
        p95_accumulated = np.nan
        p99_accumulated = np.nan
        max_accumulated = np.nan

    valid_counts = np.sum(
        score_valid,
        axis=(1, 2),
    )

    domain_pixels = int(
        np.sum(domain_mask)
    )

    return {

        "belgium_domain_pixels":
            domain_pixels,

        "mean_valid_belgium_pixels":
            float(np.mean(valid_counts)),

        "mean_future_rain_mm_h":
            float(
                np.mean(
                    finite_values,
                    dtype=np.float64,
                )
            ),

        "p95_future_rain_mm_h":
            float(
                np.percentile(
                    finite_values,
                    95,
                )
            ),

        "p99_future_rain_mm_h":
            float(
                np.percentile(
                    finite_values,
                    99,
                )
            ),

        "maximum_future_rain_mm_h":
            float(
                np.max(
                    finite_values
                )
            ),

        "mean_fraction_above_0.1":
            safe_nanmean(area01),

        "mean_fraction_above_1":
            safe_nanmean(area1),

        "mean_fraction_above_5":
            safe_nanmean(area5),

        "mean_fraction_above_10":
            safe_nanmean(area10),

        "maximum_fraction_above_1":
            float(np.nanmax(area1)),

        "maximum_fraction_above_10":
            float(np.nanmax(area10)),

        "absolute_area1_change":
            (
                float(
                    abs(
                        area1[-1]
                        - area1[0]
                    )
                )
                if np.isfinite(
                    [area1[0], area1[-1]]
                ).all()
                else np.nan
            ),

        "area1_variability":
            (
                float(np.nanstd(area1))
                if np.isfinite(area1).any()
                else np.nan
            ),

        "absolute_mean_rain_change":
            (
                float(
                    abs(
                        mean_by_lead[-1]
                        - mean_by_lead[0]
                    )
                )
                if np.isfinite(
                    [
                        mean_by_lead[0],
                        mean_by_lead[-1],
                    ]
                ).all()
                else np.nan
            ),

        "structural_change_mm_h":
            structural_change,

        "input_to_end_change_mm_h":
            input_to_end_change,

        "centroid_displacement_px":
            centroid_displacement,

        "mean_accumulated_100min_mm":
            mean_accumulated,

        "p95_accumulated_100min_mm":
            p95_accumulated,

        "p99_accumulated_100min_mm":
            p99_accumulated,

        "maximum_accumulated_100min_mm":
            max_accumulated,
    }


# ============================================================
# Robust ranking
# ============================================================

def robust_zscore(
    series: pd.Series,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    median = values.median()

    filled = values.fillna(
        median
    )

    median = filled.median()

    mad = (
        filled - median
    ).abs().median()

    if (
        not np.isfinite(mad)
        or mad == 0
    ):

        std = filled.std()

        if (
            not np.isfinite(std)
            or std == 0
        ):
            return pd.Series(
                np.zeros(
                    len(filled)
                ),
                index=filled.index,
            )

        return (
            filled
            - filled.mean()
        ) / std

    return (
        0.67448975
        * (filled - median)
        / mad
    )


def add_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    numeric_cols = [
        "p99_future_rain_mm_h",
        "mean_fraction_above_10",
        "mean_fraction_above_1",
        "mean_future_rain_mm_h",
        "structural_change_mm_h",
        "absolute_area1_change",
        "area1_variability",
        "centroid_displacement_px",
    ]

    for col in numeric_cols:
        df[f"z_{col}"] = robust_zscore(
            df[col]
        )

    # Same basic criteria as the previous validation-event selector.

    df["intense_score"] = (
        df[
            "z_p99_future_rain_mm_h"
        ]
        + 0.60
        * df[
            "z_mean_fraction_above_10"
        ]
    )

    df["widespread_score"] = (
        df[
            "z_mean_fraction_above_1"
        ]
        + 0.35
        * df[
            "z_mean_future_rain_mm_h"
        ]
    )

    df["evolving_score"] = (
        df[
            "z_structural_change_mm_h"
        ]
        + 0.50
        * df[
            "z_absolute_area1_change"
        ]
        + 0.35
        * df[
            "z_area1_variability"
        ]
        + 0.25
        * df[
            "z_centroid_displacement_px"
        ]
    )

    for score in [
        "intense_score",
        "widespread_score",
        "evolving_score",
    ]:

        rank_name = (
            "rank_"
            + score.removesuffix(
                "_score"
            )
        )

        df[rank_name] = (
            df[score]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

    return df


def choose_nonoverlapping(
    df: pd.DataFrame,
    score_col: str,
    chosen_t: list[int],
    minimum_separation: int,
) -> pd.Series:

    ordered = df.sort_values(
        [
            score_col,
            "t_start",
        ],
        ascending=[
            False,
            True,
        ],
    )

    for _, row in ordered.iterrows():

        t_start = int(
            row["t_start"]
        )

        if all(
            abs(
                t_start
                - other
            )
            >= minimum_separation
            for other in chosen_t
        ):
            return row

    raise RuntimeError(
        "Could not find a separated "
        f"candidate for {score_col}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    observation_root = Path(
        args.observation_root
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    static_mask = load_mask(
        args.belgium_mask
    )

    rows: list[
        dict[str, object]
    ] = []

    aligned_mask = None

    # --------------------------------------------------------
    # Read the exact monthly observation stores used by
    # quantitative evaluation.
    # --------------------------------------------------------

    for month in args.months:

        store = (
            observation_root
            / (
                "observations_2021_"
                f"month{month:02d}.zarr"
            )
        )

        if not store.exists():
            raise FileNotFoundError(
                f"Missing observation store: {store}"
            )

        print(
            "\nOpening:",
            store,
        )

        group = zarr.open_group(
            str(store),
            mode="r",
        )

        for required in [
            "truth",
            "valid_mask",
            "t_start",
            "start_time",
        ]:
            if required not in group:
                raise KeyError(
                    f"{required!r} missing from {store}"
                )

        truth = group["truth"]
        valid = group["valid_mask"]
        t_starts = np.asarray(
            group["t_start"][:]
        ).astype(np.int64)

        start_times = decode_strings(
            group["start_time"][:]
        )

        past_array = (
            group["past"]
            if "past" in group
            else None
        )

        if truth.shape != valid.shape:
            raise ValueError(
                f"{store}: truth {truth.shape} "
                f"!= valid {valid.shape}"
            )

        if len(t_starts) != truth.shape[0]:
            raise ValueError(
                f"{store}: t_start length mismatch"
            )

        if len(start_times) != truth.shape[0]:
            raise ValueError(
                f"{store}: start_time length mismatch"
            )

        spatial_shape = tuple(
            map(
                int,
                truth.shape[-2:],
            )
        )

        if aligned_mask is None:
            aligned_mask = align_mask(
                static_mask,
                spatial_shape,
            )

            print(
                "Belgium pixels:",
                int(
                    aligned_mask.sum()
                ),
            )

        elif (
            aligned_mask.shape
            != spatial_shape
        ):
            raise ValueError(
                "Observation stores use inconsistent "
                f"spatial shapes: {spatial_shape}"
            )

        n_events = int(
            truth.shape[0]
        )

        print(
            f"Month {month:02d}: "
            f"{n_events} events"
        )

        for event_index in range(
            n_events
        ):

            if (
                event_index == 0
                or (
                    event_index + 1
                ) % 25 == 0
                or (
                    event_index + 1
                    == n_events
                )
            ):
                print(
                    f"  [{event_index + 1}/"
                    f"{n_events}] "
                    f"t_start="
                    f"{int(t_starts[event_index])}",
                    flush=True,
                )

            future = np.asarray(
                truth[event_index],
                dtype=np.float32,
            )

            valid_event = np.asarray(
                valid[event_index],
                dtype=bool,
            )

            past = (
                np.asarray(
                    past_array[event_index],
                    dtype=np.float32,
                )
                if past_array is not None
                else None
            )

            descriptors = (
                event_descriptors(
                    future=future,
                    observation_valid=valid_event,
                    domain_mask=aligned_mask,
                    past=past,
                )
            )

            rows.append(
                {
                    "month": int(month),
                    "store_event_index":
                        int(event_index),
                    "t_start":
                        int(
                            t_starts[
                                event_index
                            ]
                        ),
                    "start_time":
                        start_times[
                            event_index
                        ],
                    **descriptors,
                }
            )

    if not rows:
        raise RuntimeError(
            "No event descriptors generated."
        )

    descriptors = pd.DataFrame(
        rows
    )

    if (
        descriptors["t_start"]
        .duplicated()
        .any()
    ):
        duplicates = (
            descriptors.loc[
                descriptors[
                    "t_start"
                ].duplicated(
                    keep=False
                ),
                "t_start",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate t_start values: "
            f"{duplicates[:20]}"
        )

    descriptors[
        "timestamp"
    ] = pd.to_datetime(
        descriptors[
            "start_time"
        ],
        errors="coerce",
    )

    if (
        descriptors[
            "timestamp"
        ]
        .isna()
        .any()
    ):
        bad = descriptors.loc[
            descriptors[
                "timestamp"
            ].isna(),
            [
                "t_start",
                "start_time",
            ],
        ]

        raise ValueError(
            "Could not parse some event times:\n"
            + bad.head(
                10
            ).to_string(
                index=False
            )
        )

    descriptors = add_scores(
        descriptors
    )

    descriptors = (
        descriptors
        .sort_values(
            "t_start"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Documented July flood interval
    # --------------------------------------------------------

    flood_start = pd.Timestamp(
        args.flood_start
    )

    flood_end = pd.Timestamp(
        args.flood_end
    )

    flood_mask = (
        (
            descriptors[
                "timestamp"
            ]
            >= flood_start
        )
        & (
            descriptors[
                "timestamp"
            ]
            < flood_end
        )
    )

    flood_candidates = (
        descriptors.loc[
            flood_mask
        ]
        .copy()
    )

    if flood_candidates.empty:
        raise RuntimeError(
            "No 2021 test events found "
            "inside flood interval "
            f"{flood_start} -> {flood_end}"
        )

    # Generic event selection explicitly excludes the flood
    # interval so that we obtain four distinct case studies.

    generic_pool = (
        descriptors.loc[
            ~flood_mask
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Top generic candidates
    # --------------------------------------------------------

    top_tables = []

    category_specs = [
        (
            "intense convection",
            "intense_score",
        ),
        (
            "widespread organized rainfall",
            "widespread_score",
        ),
        (
            "rapidly evolving rainfall",
            "evolving_score",
        ),
    ]

    for (
        category,
        score_col,
    ) in category_specs:

        top = (
            generic_pool
            .sort_values(
                [
                    score_col,
                    "t_start",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(
                args.top_k
            )
            .copy()
        )

        top.insert(
            0,
            "category",
            category,
        )

        top.insert(
            1,
            "category_score",
            top[
                score_col
            ],
        )

        top_tables.append(
            top
        )

    top_candidates = pd.concat(
        top_tables,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Automatically select three generic cases
    # --------------------------------------------------------

    selected_rows = []
    chosen_t: list[int] = []

    for (
        category,
        score_col,
    ) in category_specs:

        row = choose_nonoverlapping(
            generic_pool,
            score_col=score_col,
            chosen_t=chosen_t,
            minimum_separation=(
                args.selection_separation_steps
            ),
        )

        chosen_t.append(
            int(
                row["t_start"]
            )
        )

        out = row.to_dict()

        out[
            "category"
        ] = category

        out[
            "selection_metric"
        ] = score_col

        out[
            "selection_score"
        ] = float(
            row[
                score_col
            ]
        )

        selected_rows.append(
            out
        )

    # --------------------------------------------------------
    # Flood ranking
    #
    # Primary criterion:
    # p95 of accumulated 100-minute rainfall over Belgian
    # pixels.
    #
    # Secondary tie breakers:
    # p99 accumulation, then mean accumulation.
    # --------------------------------------------------------

    flood_candidates = (
        flood_candidates
        .sort_values(
            [
                "p95_accumulated_100min_mm",
                "p99_accumulated_100min_mm",
                "mean_accumulated_100min_mm",
                "t_start",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    flood_candidates[
        "flood_rank"
    ] = (
        np.arange(
            len(
                flood_candidates
            )
        )
        + 1
    )

    flood_row = (
        flood_candidates
        .iloc[0]
    )

    flood_out = (
        flood_row.to_dict()
    )

    flood_out[
        "category"
    ] = (
        "July 2021 flood event"
    )

    flood_out[
        "selection_metric"
    ] = (
        "p95_accumulated_100min_mm"
    )

    flood_out[
        "selection_score"
    ] = float(
        flood_row[
            "p95_accumulated_100min_mm"
        ]
    )

    selected_rows.append(
        flood_out
    )

    selected = pd.DataFrame(
        selected_rows
    )

    # --------------------------------------------------------
    # Final ordering
    # --------------------------------------------------------

    category_order = [
        "intense convection",
        "widespread organized rainfall",
        "rapidly evolving rainfall",
        "July 2021 flood event",
    ]

    selected[
        "_category_order"
    ] = selected[
        "category"
    ].map(
        {
            category: i
            for i, category
            in enumerate(
                category_order
            )
        }
    )

    selected = (
        selected
        .sort_values(
            "_category_order"
        )
        .drop(
            columns=[
                "_category_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        selected[
            "t_start"
        ]
        .nunique()
        != 4
    ):
        raise RuntimeError(
            "Final selection does not contain "
            "four unique events."
        )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    descriptor_file = (
        output_dir
        / (
            "qualitative_test_"
            "event_descriptors_"
            "belgium.csv"
        )
    )

    top_file = (
        output_dir
        / (
            "qualitative_test_"
            "top_candidates_"
            "belgium.csv"
        )
    )

    flood_file = (
        output_dir
        / (
            "qualitative_flood_"
            "candidates_belgium.csv"
        )
    )

    selected_file = (
        output_dir
        / (
            "qualitative_test_"
            "selected_events.csv"
        )
    )

    descriptors.to_csv(
        descriptor_file,
        index=False,
    )

    top_candidates.to_csv(
        top_file,
        index=False,
    )

    flood_candidates.to_csv(
        flood_file,
        index=False,
    )

    selected.to_csv(
        selected_file,
        index=False,
    )

    # --------------------------------------------------------
    # Human-readable output
    # --------------------------------------------------------

    display_cols = [
        "category",
        "t_start",
        "start_time",
        "selection_metric",
        "selection_score",
        "p99_future_rain_mm_h",
        "mean_fraction_above_1",
        "mean_fraction_above_10",
        "structural_change_mm_h",
        "centroid_displacement_px",
        "p95_accumulated_100min_mm",
        "p99_accumulated_100min_mm",
    ]

    print(
        "\n"
        "=========================================="
    )

    print(
        "FINAL QUALITATIVE TEST EVENT SELECTION"
    )

    print(
        "==========================================\n"
    )

    print(
        selected[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nTest events screened:",
        len(
            descriptors
        ),
    )

    print(
        "Unique t_start:",
        descriptors[
            "t_start"
        ].nunique(),
    )

    print(
        "Belgium mask pixels:",
        int(
            aligned_mask.sum()
        ),
    )

    print(
        "Flood candidates:",
        len(
            flood_candidates
        ),
    )

    print(
        "\nTop flood candidates:\n"
    )

    print(
        flood_candidates[
            [
                "flood_rank",
                "t_start",
                "start_time",
                "p95_accumulated_100min_mm",
                "p99_accumulated_100min_mm",
                "maximum_accumulated_100min_mm",
                "p99_future_rain_mm_h",
                "mean_fraction_above_1",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nWrote:"
    )

    for file in [
        descriptor_file,
        top_file,
        flood_file,
        selected_file,
    ]:
        print(
            " ",
            file,
        )


if __name__ == "__main__":
    main()