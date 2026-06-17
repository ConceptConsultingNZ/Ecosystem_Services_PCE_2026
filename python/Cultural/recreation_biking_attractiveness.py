#!/usr/bin/env python3
"""
Create a mountain biking attractiveness raster by combining:

1) A land-class-based biking supply raster
   - Uses one or two land classification rasters, such as LCDB6 land cover and NZLUM land use/management.
   - Each raster has a lookup CSV assigning a biking supply score to each class.
   - If both layers are enabled, each is mapped to supply and the maximum score is used.

2) A mountain biking park / trailhead proximity surface
   - Uses point features for recognised MTB parks or trail networks.
   - Around each point, a Gaussian distance-decay kernel is created.
   - The kernel is multiplied by the land-based supply raster so high attractiveness only occurs
     where land cover or land use is suitable.
   - Contributions from multiple parks are combined using either pixelwise maximum or sum.

Outputs:
    mountain_biking_supply.tif
        Intermediate land-based supply raster.

    mountain_biking_attractiveness.tif
        Final MTB attractiveness raster, constrained by both recognised MTB sites and land suitability.

Notes:
    - Distance parameters are in metres, so rasters should use a projected CRS such as NZTM.
    - If both land classification layers are enabled, the output grid follows raster 1.
    - If only layer 2 is enabled, the output grid follows raster 2.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import reproject
from rasterio.windows import Window


# =========================================================
# CONFIGURATION
# =========================================================

# Toggle land classification layers
USE_LAYER1 = True
USE_LAYER2 = False

# Land classification rasters
RASTER1_PATH = Path(r"<PROJECT_DIRECTORY>\Common\lcdb6_expanded.tif")
RASTER2_PATH = Path(r"D:\Data\LRIS\lris-new-zealand-land-use-management-version-03-nzlum-v03-FGDB\nzlum.tif")

# Working directory
WORKING_DIR = Path(r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate")

# Lookup tables
CSV1_PATH = WORKING_DIR / "lcdb6_attractiveness.csv"
CSV2_PATH = WORKING_DIR / "nzlum_biking_supply.csv"

CLASS_COL_1 = "class"
CLASS_COL_2 = "class"
SUPPLY_COL_1 = "mountain_biking"
SUPPLY_COL_2 = "biking_supply"

# MTB parks / trail network points
PARKS_GPKG = WORKING_DIR / "Recreation.gpkg"
PARKS_LAYER = "mountain_bike_parks"

# Outputs
SUPPLY_OUTPUT = WORKING_DIR / "mountain_biking_supply.tif"
ATTRACTIVENESS_OUTPUT = WORKING_DIR / "mountain_biking_attractiveness.tif"

# Supply raster nodata
SUPPLY_NODATA = -9999.0

# Park proximity kernel
RADIUS_M = 2000.0
SIGMA_M = 1000.0
MIN_SUPPLY = 0.0
MAX_OUTPUT = 1.0
COMBINE_MODE = "max"  # "max" or "sum"

# GeoTIFF settings
OUTPUT_DTYPE = "float32"
COMPRESS = "DEFLATE"
PREDICTOR_FLOAT = 3
TILED = True
BLOCKXSIZE = 512
BLOCKYSIZE = 512


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class RasterTemplate:
    crs: object
    transform: object
    width: int
    height: int
    nodata: Optional[float]
    profile: dict


# =========================================================
# LAND SUPPLY FUNCTIONS
# =========================================================

def build_lookup(csv_path: Path, class_col: str, supply_col: str) -> dict[int, float]:
    """Read a class-to-supply lookup CSV and return {class_code: supply_score}."""
    df = pd.read_csv(csv_path)
    df = df[[class_col, supply_col]].dropna()

    df[class_col] = pd.to_numeric(df[class_col], errors="coerce")
    df[supply_col] = pd.to_numeric(df[supply_col], errors="coerce")
    df = df.dropna()

    # If duplicate classes exist, keep the maximum supply score.
    df = df.groupby(class_col, as_index=False)[supply_col].max()

    return {int(k): float(v) for k, v in zip(df[class_col], df[supply_col])}


def map_classes_to_supply(
    class_arr: np.ndarray,
    lookup: dict[int, float],
    nodata: Optional[float],
) -> np.ndarray:
    """Map integer class codes to floating-point biking supply scores."""
    out = np.full(class_arr.shape, np.nan, dtype=np.float32)

    for val in np.unique(class_arr):
        if nodata is not None and val == nodata:
            continue

        try:
            key = int(val)
        except (ValueError, TypeError):
            continue

        if key in lookup:
            out[class_arr == val] = np.float32(lookup[key])

    return out


def aligned_read_raster_to_reference(
    src: rasterio.io.DatasetReader,
    ref_profile: dict,
) -> np.ndarray:
    """Read a raster and align it to a reference grid if needed."""
    ref_crs = ref_profile["crs"]
    ref_transform = ref_profile["transform"]
    ref_height = ref_profile["height"]
    ref_width = ref_profile["width"]

    arr = src.read(1)

    if (
        src.crs == ref_crs
        and src.transform == ref_transform
        and src.width == ref_width
        and src.height == ref_height
    ):
        return arr

    dst = np.empty((ref_height, ref_width), dtype=arr.dtype)

    reproject(
        source=arr,
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
        src_nodata=src.nodata,
        dst_nodata=src.nodata,
    )

    return dst


def create_biking_supply_raster() -> Path:
    """
    Create a land-based biking supply raster from one or two land classification rasters.
    If two layers are enabled, the maximum supply score across layers is used.
    """
    if not USE_LAYER1 and not USE_LAYER2:
        raise ValueError("Both USE_LAYER1 and USE_LAYER2 are False. Nothing to do.")

    lookup1 = build_lookup(CSV1_PATH, CLASS_COL_1, SUPPLY_COL_1) if USE_LAYER1 else None
    lookup2 = build_lookup(CSV2_PATH, CLASS_COL_2, SUPPLY_COL_2) if USE_LAYER2 else None

    ref_path = RASTER1_PATH if USE_LAYER1 else RASTER2_PATH

    with rasterio.open(ref_path) as ref_src:
        ref_profile = ref_src.profile.copy()

    supply_layers: list[np.ndarray] = []

    if USE_LAYER1:
        with rasterio.open(RASTER1_PATH) as src1:
            class1 = aligned_read_raster_to_reference(src1, ref_profile)
            nodata1 = src1.nodata

        supply1 = map_classes_to_supply(class1, lookup1, nodata1)
        if nodata1 is not None:
            supply1 = np.where(class1 == nodata1, np.nan, supply1)

        supply_layers.append(supply1)

    if USE_LAYER2:
        with rasterio.open(RASTER2_PATH) as src2:
            class2 = aligned_read_raster_to_reference(src2, ref_profile)
            nodata2 = src2.nodata

        supply2 = map_classes_to_supply(class2, lookup2, nodata2)
        if nodata2 is not None:
            supply2 = np.where(class2 == nodata2, np.nan, supply2)

        supply_layers.append(supply2)

    if len(supply_layers) == 1:
        supply_max = supply_layers[0]
    else:
        supply_max = np.fmax.reduce(supply_layers)

    out_arr = np.where(
        np.isnan(supply_max),
        np.float32(SUPPLY_NODATA),
        supply_max,
    ).astype(np.float32)

    out_profile = ref_profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.float32(SUPPLY_NODATA),
        compress=COMPRESS,
        predictor=PREDICTOR_FLOAT,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    out_profile.pop("photometric", None)

    with rasterio.open(SUPPLY_OUTPUT, "w", **out_profile) as dst:
        dst.write(out_arr, 1)

    print(f"Wrote supply raster: {SUPPLY_OUTPUT}")
    return SUPPLY_OUTPUT


# =========================================================
# ATTRACTIVENESS FUNCTIONS
# =========================================================

def read_template(path: Path) -> RasterTemplate:
    """Read raster metadata used as the output template."""
    with rasterio.open(path) as src:
        return RasterTemplate(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            nodata=src.nodata,
            profile=src.profile.copy(),
        )


def pixel_sizes(transform) -> tuple[float, float]:
    """Return x and y pixel sizes for a north-up raster."""
    return abs(transform.a), abs(transform.e)


def read_park_points(path: Path, layer: Optional[str], target_crs) -> gpd.GeoDataFrame:
    """Read MTB park points and reproject to the raster CRS if needed."""
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)

    if gdf.empty:
        raise ValueError("Parks layer is empty.")

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()

    if gdf.empty:
        raise ValueError("No Point or MultiPoint geometries found in parks layer.")

    if gdf.crs is None:
        raise ValueError("Parks layer has no CRS. Define it before running this script.")

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()

    return gdf


def gaussian_kernel(dist_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """Gaussian kernel with peak value 1 at the park point."""
    if sigma_m <= 0:
        raise ValueError("SIGMA_M must be greater than 0.")

    return np.exp(-(dist_m ** 2) / (2.0 * sigma_m ** 2))


def build_distance_grid(px_x_m: float, px_y_m: float, half_w: int, half_h: int) -> np.ndarray:
    """Build a distance grid in metres centred on a park point."""
    xs = (np.arange(-half_w, half_w + 1) * px_x_m).astype("float32")
    ys = (np.arange(-half_h, half_h + 1) * px_y_m).astype("float32")
    xx, yy = np.meshgrid(xs, ys)

    return np.sqrt(xx * xx + yy * yy)


def create_biking_attractiveness_raster(supply_raster: Path) -> Path:
    """
    Create an attractiveness surface around MTB park points, constrained by the
    land-based supply raster.
    """
    template = read_template(supply_raster)
    parks = read_park_points(PARKS_GPKG, PARKS_LAYER, template.crs)

    with rasterio.open(supply_raster) as supply_src:
        if supply_src.count != 1:
            raise ValueError("Supply raster must be a single-band raster.")

        out = np.zeros((template.height, template.width), dtype=np.float32)

        px_x_m, px_y_m = pixel_sizes(template.transform)
        if px_x_m == 0 or px_y_m == 0:
            raise ValueError("Invalid raster transform; pixel size is zero.")

        if template.crs and template.crs.is_geographic:
            warnings.warn(
                "Raster CRS is geographic. RADIUS_M and SIGMA_M are in metres, "
                "but pixel size is in degrees. Reproject to a projected CRS such as NZTM.",
                RuntimeWarning,
            )

        half_w = int(math.ceil(RADIUS_M / px_x_m))
        half_h = int(math.ceil(RADIUS_M / px_y_m))

        dist_grid = build_distance_grid(px_x_m, px_y_m, half_w, half_h)
        within = dist_grid <= RADIUS_M

        base_kernel = gaussian_kernel(dist_grid, SIGMA_M).astype(np.float32)
        base_kernel[~within] = 0.0

        for geom in parks.geometry:
            x, y = float(geom.x), float(geom.y)
            r, c = rowcol(template.transform, x, y)

            r0 = max(0, r - half_h)
            r1 = min(template.height, r + half_h + 1)
            c0 = max(0, c - half_w)
            c1 = min(template.width, c + half_w + 1)

            if r0 >= r1 or c0 >= c1:
                continue

            win = Window(col_off=c0, row_off=r0, width=(c1 - c0), height=(r1 - r0))

            supply = supply_src.read(
                1,
                window=win,
                out_dtype="float32",
                resampling=Resampling.nearest,
            )

            if supply_src.nodata is not None:
                supply = np.where(supply == supply_src.nodata, 0.0, supply)

            supply = np.nan_to_num(supply, nan=0.0, posinf=0.0, neginf=0.0)
            supply = np.clip(supply, MIN_SUPPLY, 1.0, out=supply)

            kr0 = r0 - (r - half_h)
            kr1 = kr0 + (r1 - r0)
            kc0 = c0 - (c - half_w)
            kc1 = kc0 + (c1 - c0)

            kern = base_kernel[kr0:kr1, kc0:kc1]
            contrib = kern * supply

            if COMBINE_MODE.lower() == "sum":
                out[r0:r1, c0:c1] += contrib
            elif COMBINE_MODE.lower() == "max":
                out[r0:r1, c0:c1] = np.maximum(out[r0:r1, c0:c1], contrib)
            else:
                raise ValueError("COMBINE_MODE must be either 'max' or 'sum'.")

        out = np.clip(out, 0.0, MAX_OUTPUT).astype(np.float32)

    out_profile = template.profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype=OUTPUT_DTYPE,
        nodata=0.0,
        compress=COMPRESS,
        predictor=PREDICTOR_FLOAT,
        tiled=TILED,
        blockxsize=BLOCKXSIZE if TILED else None,
        blockysize=BLOCKYSIZE if TILED else None,
        BIGTIFF="IF_SAFER",
    )

    out_profile.pop("photometric", None)

    with rasterio.open(ATTRACTIVENESS_OUTPUT, "w", **out_profile) as dst:
        dst.write(out, 1)

    print(f"Wrote attractiveness raster: {ATTRACTIVENESS_OUTPUT}")
    return ATTRACTIVENESS_OUTPUT


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    supply_raster = create_biking_supply_raster()
    create_biking_attractiveness_raster(supply_raster)
    print("Mountain biking supply and attractiveness rasters created successfully.")


if __name__ == "__main__":
    main()