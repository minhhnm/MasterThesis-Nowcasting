#!/usr/bin/env python3
"""Create a fixed Belgium mask aligned with the RADCLIM grid.

The output is a 2-D boolean NumPy array:
    True  = grid-cell centre lies inside or on the Belgian boundary
    False = outside Belgium

Recommended use:
    python make_belgium_mask.py \
      --zarr /path/to/RADCLIMrates_f16.zarr \
      --variable precip_intensity_EDK \
      --countries /path/to/ne_10m_admin_0_countries.shp \
      --output /data/brussel/114/vsc11442/belgium_mask_700.npy

If the RADCLIM store has projected x/y coordinates, also pass:
      --grid-crs EPSG:xxxx

The script prefers actual lon/lat or x/y coordinates stored in the Zarr.
A fallback geographical extent can be supplied explicitly, but this should
only be used after confirming that it matches the RADCLIM georeferencing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


COUNTRY_NAME_FIELDS = (
    "ADMIN",
    "NAME",
    "NAME_EN",
    "SOVEREIGNT",
    "FORMAL_EN",
    "CNTR_NAME",
)
COUNTRY_CODE_FIELDS = (
    "ISO_A3",
    "ADM0_A3",
    "SOV_A3",
    "CNTR_ID",
)


def _is_index_coordinate(values: np.ndarray) -> bool:
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        return False
    return np.allclose(values, np.arange(values.size), atol=1e-8)


def _find_belgium(countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    matches = np.zeros(len(countries), dtype=bool)

    for field in COUNTRY_NAME_FIELDS:
        if field in countries.columns:
            values = countries[field].astype(str).str.strip().str.casefold()
            matches |= values.eq("belgium").to_numpy()

    for field in COUNTRY_CODE_FIELDS:
        if field in countries.columns:
            values = countries[field].astype(str).str.strip().str.upper()
            matches |= values.isin(["BEL", "BE"]).to_numpy()

    belgium = countries.loc[matches].copy()
    if belgium.empty:
        raise ValueError(
            "Belgium was not found in the country-boundary file. "
            f"Available columns are: {list(countries.columns)}"
        )

    # Dissolve multipart records into one geometry.
    geometry = belgium.geometry.union_all()
    return gpd.GeoDataFrame(
        {"country": ["Belgium"]},
        geometry=[geometry],
        crs=countries.crs,
    )


def _coord_from_candidates(ds: xr.Dataset, candidates: tuple[str, ...]):
    for name in candidates:
        if name in ds.coords:
            return name, ds.coords[name]
        if name in ds.variables:
            return name, ds[name]
    return None, None


def _get_grid(
    ds: xr.Dataset,
    variable: str,
    grid_crs: str | None,
    fallback_extent: tuple[float, float, float, float] | None,
):
    """Return xx, yy, CRS and spatial dimension names."""
    da = ds[variable]
    if da.ndim < 2:
        raise ValueError(f"{variable} has fewer than two dimensions: {da.dims}")

    y_dim, x_dim = da.dims[-2], da.dims[-1]
    height, width = da.sizes[y_dim], da.sizes[x_dim]

    lon_name, lon = _coord_from_candidates(
        ds,
        ("lon", "longitude", "LONGITUDE", "xlon"),
    )
    lat_name, lat = _coord_from_candidates(
        ds,
        ("lat", "latitude", "LATITUDE", "xlat"),
    )

    if lon is not None and lat is not None:
        lon_values = np.asarray(lon.values)
        lat_values = np.asarray(lat.values)

        if lon_values.ndim == 1 and lat_values.ndim == 1:
            if lon_values.size == width and lat_values.size == height:
                xx, yy = np.meshgrid(lon_values, lat_values)
                return xx, yy, "EPSG:4326", y_dim, x_dim

        if (
            lon_values.ndim == 2
            and lat_values.ndim == 2
            and lon_values.shape == (height, width)
            and lat_values.shape == (height, width)
        ):
            return lon_values, lat_values, "EPSG:4326", y_dim, x_dim

    # Try coordinates named after the spatial dimensions.
    if x_dim in ds.coords and y_dim in ds.coords:
        x_values = np.asarray(ds.coords[x_dim].values)
        y_values = np.asarray(ds.coords[y_dim].values)

        if (
            x_values.ndim == 1
            and y_values.ndim == 1
            and x_values.size == width
            and y_values.size == height
            and not (
                _is_index_coordinate(x_values)
                and _is_index_coordinate(y_values)
            )
        ):
            if not grid_crs:
                raise ValueError(
                    f"The Zarr contains coordinates '{x_dim}' and '{y_dim}', "
                    "but their CRS could not be inferred. Pass --grid-crs, "
                    "for example --grid-crs EPSG:31370."
                )
            xx, yy = np.meshgrid(x_values, y_values)
            return xx, yy, grid_crs, y_dim, x_dim

    if fallback_extent is None:
        raise ValueError(
            "No usable geographical grid coordinates were found in the Zarr. "
            "Inspect the dataset coordinates/attributes and either pass the "
            "correct --grid-crs or provide --fallback-extent "
            "LON_MIN LON_MAX LAT_MIN LAT_MAX."
        )

    lon_min, lon_max, lat_min, lat_max = fallback_extent

    # Pixel centres, not domain edges.
    dx = (lon_max - lon_min) / width
    dy = (lat_max - lat_min) / height
    x_centres = lon_min + (np.arange(width) + 0.5) * dx

    # The stored RADCLIM rows used in the figures run from north to south.
    y_centres = lat_max - (np.arange(height) + 0.5) * dy
    xx, yy = np.meshgrid(x_centres, y_centres)

    return xx, yy, "EPSG:4326", y_dim, x_dim


def _polygon_mask(geometry, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Use vectorized Shapely operations where available."""
    try:
        from shapely import intersects_xy

        return np.asarray(intersects_xy(geometry, xx, yy), dtype=bool)
    except ImportError:
        # Fallback for older Shapely versions.
        points = gpd.GeoSeries(
            gpd.points_from_xy(xx.ravel(), yy.ravel()),
            crs=None,
        )
        mask = points.intersects(geometry).to_numpy()
        return mask.reshape(xx.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True)
    parser.add_argument(
        "--variable",
        default="precip_intensity_EDK",
    )
    parser.add_argument(
        "--countries",
        required=True,
        help=(
            "Country polygon file, e.g. Natural Earth "
            "ne_10m_admin_0_countries.shp, or the exact boundary file "
            "already used for the thesis maps."
        ),
    )
    parser.add_argument(
        "--grid-crs",
        default=None,
        help=(
            "CRS of projected x/y grid coordinates, for example "
            "EPSG:31370. Not needed for lon/lat coordinates."
        ),
    )
    parser.add_argument(
        "--fallback-extent",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=None,
        help=(
            "Fallback only when the Zarr has no geographical coordinates. "
            "Use confirmed domain edges."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--preview",
        default=None,
        help="Optional PNG path for a visual mask check.",
    )
    args = parser.parse_args()

    ds = xr.open_zarr(args.zarr, consolidated=None)
    if args.variable not in ds:
        raise KeyError(
            f"Variable '{args.variable}' not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    xx, yy, grid_crs, y_dim, x_dim = _get_grid(
        ds=ds,
        variable=args.variable,
        grid_crs=args.grid_crs,
        fallback_extent=(
            tuple(args.fallback_extent)
            if args.fallback_extent is not None
            else None
        ),
    )

    countries = gpd.read_file(args.countries)
    if countries.crs is None:
        raise ValueError(
            "The country-boundary file has no CRS. Assign the correct CRS "
            "before using it."
        )

    belgium = _find_belgium(countries).to_crs(grid_crs)
    belgium_geometry = belgium.geometry.iloc[0]

    mask = _polygon_mask(belgium_geometry, xx, yy)

    expected_shape = (
        ds[args.variable].sizes[y_dim],
        ds[args.variable].sizes[x_dim],
    )
    if mask.shape != expected_shape:
        raise RuntimeError(
            f"Generated mask shape {mask.shape}, expected {expected_shape}."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, mask.astype(bool))

    np.savez_compressed(
        output.with_suffix(".npz"),
        belgium_mask=mask.astype(bool),
        grid_crs=np.array(str(grid_crs)),
        x_coordinate=xx.astype(np.float64),
        y_coordinate=yy.astype(np.float64),
    )

    print("Saved:", output)
    print("Also saved:", output.with_suffix(".npz"))
    print("Mask shape:", mask.shape)
    print("Belgium pixels:", int(mask.sum()))
    print("Belgium fraction of full grid:", float(mask.mean()))
    print("Grid CRS:", grid_crs)

    if args.preview:
        preview = Path(args.preview)
        preview.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(mask, origin="upper", interpolation="nearest")
        ax.set_title("Belgium mask on the RADCLIM grid")
        ax.set_xlabel(x_dim)
        ax.set_ylabel(y_dim)
        fig.tight_layout()
        fig.savefig(preview, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved preview:", preview)


if __name__ == "__main__":
    main()
