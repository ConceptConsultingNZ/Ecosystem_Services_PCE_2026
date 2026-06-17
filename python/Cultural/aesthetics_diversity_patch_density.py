from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from skimage.measure import label as cc_label
import time

"""
Calculate landscape diversity (Shannon Diversity Index, SHDI) and patchiness (Patch Density, PD)
from a categorical landcover raster, aggregated to non-overlapping analysis windows.

Why non-overlapping windows?
- A full moving-window implementation over a national 1 ha raster would be computationally brutal.
- This script produces a coarser “analysis grid” (one output cell per window), which is typically
  what we want for national ES layers anyway.

Metrics
- SHDI (diversity): -sum(p_i * ln(p_i)) over landcover classes in the window.
- Patch Density (patchiness): number_of_patches / window_area_km2, where patches are contiguous
  (8-neighbour connectivity) regions of the same landcover code within the window.
- Patchiness score (U-shaped): rewards moderate patch density and penalises very low (uniform)
  and very high (fragmented) patch density.

Outputs
- writes 100 m versions of the outputs by warping/resampling the
coarse analysis-grid rasters back onto the source grid (aligned to the landcover raster).
This avoids a moving window (expensive) and instead replicates each coarse window value
across its footprint at 100 m resolution.
  1) shdi.tif: SHDI values
  2) pd_patches_per_km2.tif: patch density (patches per km²)
  3) pd_score_0_100.tif: U-shaped patchiness score from 0 to 100

Dependencies
- rasterio
- numpy
- scikit-image (for connected components)

Notes
- Treat SHDI and PD as scale-dependent. Window size is part of the model definition.
- If you want a moving-window surface at native resolution, you can adapt this script,
  but expect large runtimes and memory pressure at national scale.
"""
"""
This version now also writes 100 m versions of the outputs by warping/resampling the
coarse analysis-grid rasters back onto the source grid (aligned to the landcover raster).
This avoids a moving window (expensive) and instead replicates each coarse window value
across its footprint at 100 m resolution.
"""

# ----------------------------
# Filepaths
# ----------------------------
LANDCOVER_RASTER = r"D:\Data\LRIS\lris-lcdb-v60-land-cover-database-version-60-mainland-new-zealand\lcdb6_expanded.tif"

OUTPUT_DIR = r"D:\tmp"

# Coarse (analysis-grid) outputs
OUT_SHDI_TIF = os.path.join(OUTPUT_DIR, "shdi_coarse.tif")
OUT_PD_TIF = os.path.join(OUTPUT_DIR, "pd_patches_per_km2_coarse.tif")
OUT_PD_SCORE_TIF = os.path.join(OUTPUT_DIR, "pd_score_0_100_coarse.tif")

# 100 m outputs (aligned to source raster)
OUT_SHDI_100M_TIF = os.path.join(OUTPUT_DIR, "shdi_100m.tif")
OUT_PD_100M_TIF = None #os.path.join(OUTPUT_DIR, "pd_patches_per_km2_100m.tif")
OUT_PD_SCORE_100M_TIF = os.path.join(OUTPUT_DIR, "pd_score_100m.tif")

# ----------------------------
# Analysis settings
# ----------------------------
NODATA_VALUE: Optional[int] = None
CONNECTIVITY_8 = True

# Parameters for the moving window and evaluation size
# Runtime is proportional to (1/g^2) x w^2. With values (500,3000) it took 1:11 hours to run.
GRID_CELL_SIZE_M = 500      # output grid spacing (coarse grid cell size)
EVAL_WINDOW_SIZE_M = 3000   # neighbourhood size used to compute metrics for each output cell

PD_OPT = 8.0
PD_ZERO_LOW = 0.5
PD_ZERO_HIGH = 30.0

MIN_VALID_FRAC = 0.50

COMPRESS = "DEFLATE"
PREDICTOR = 2
FLOAT_NODATA = -9999.0


@dataclass(frozen=True)
class GridSpec:
    out_height: int
    out_width: int
    out_transform: Affine
    grid_px: int          # pixel step for output cells
    eval_px: int          # pixel size of evaluation window
    px_size_m: float


def build_grid_spec(
    src: rasterio.io.DatasetReader,
    grid_cell_size_m: float,
    eval_window_size_m: float,
) -> GridSpec:
    px_size_x = abs(src.transform.a)
    px_size_y = abs(src.transform.e)

    if not np.isclose(px_size_x, px_size_y):
        raise ValueError(f"Pixel sizes differ (x={px_size_x}, y={px_size_y}). Assumes square pixels.")

    px_size_m = float(px_size_x)

    grid_px = int(round(grid_cell_size_m / px_size_m))
    if grid_px < 1:
        raise ValueError("GRID_CELL_SIZE_M is smaller than one pixel.")

    eval_px = int(round(eval_window_size_m / px_size_m))
    if eval_px < 1:
        raise ValueError("EVAL_WINDOW_SIZE_M is smaller than one pixel.")

    out_width = int(math.ceil(src.width / grid_px))
    out_height = int(math.ceil(src.height / grid_px))

    # Output pixel represents one output cell at GRID_CELL_SIZE_M
    out_transform = src.transform * Affine.scale(grid_px, grid_px)

    return GridSpec(
        out_height=out_height,
        out_width=out_width,
        out_transform=out_transform,
        grid_px=grid_px,
        eval_px=eval_px,
        px_size_m=px_size_m,
    )


def shannon_diversity(window_data: np.ndarray, nodata: Optional[int]) -> float:
    if nodata is not None:
        vals = window_data[window_data != nodata]
    else:
        vals = window_data

    if vals.size == 0:
        return float("nan")

    _, counts = np.unique(vals, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log(p)).sum())


def patch_density(window_data: np.ndarray, nodata: Optional[int], win_area_km2: float, connectivity_8: bool) -> float:
    if nodata is not None:
        valid = (window_data != nodata)
        if not valid.any():
            return float("nan")
        data = window_data.copy()
    else:
        data = window_data.copy()

    connectivity = 2 if connectivity_8 else 1
    total_patches = 0

    classes = np.unique(data[data != nodata]) if nodata is not None else np.unique(data)
    for c in classes:
        mask = (data == c)
        if nodata is not None:
            mask &= (data != nodata)
        labeled = cc_label(mask, connectivity=connectivity)
        total_patches += int(labeled.max())

    return float(total_patches / win_area_km2)


def patchiness_u_score(pd: float, pd_opt: float, pd_zero_low: float, pd_zero_high: float) -> float:
    if not np.isfinite(pd):
        return float("nan")

    if pd <= pd_zero_low or pd >= pd_zero_high:
        return 0.0

    if pd < pd_opt:
        denom = (pd_opt - pd_zero_low)
        if denom <= 0:
            return 0.0
        x = (pd_opt - pd) / denom
    else:
        denom = (pd_zero_high - pd_opt)
        if denom <= 0:
            return 0.0
        x = (pd - pd_opt) / denom

    score = 100.0 * (1.0 - x * x)
    return float(max(0.0, min(100.0, score)))


def write_float_tif(path: str, arr: np.ndarray, transform: Affine, crs) -> None:
    out = arr.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = FLOAT_NODATA

    profile = {
        "driver": "GTiff",
        "height": out.shape[0],
        "width": out.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": FLOAT_NODATA,
        "compress": COMPRESS,
        "predictor": PREDICTOR,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def warp_to_template_grid(
    src_path: str,
    dst_path: str,
    template: rasterio.io.DatasetReader,
    resampling: Resampling = Resampling.nearest,
) -> None:
    with rasterio.open(src_path) as src:
        dst_profile = template.profile.copy()
        dst_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=FLOAT_NODATA,
            compress=COMPRESS,
            predictor=PREDICTOR,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=template.transform,
                dst_crs=template.crs,
                dst_nodata=FLOAT_NODATA,
                resampling=resampling,
            )


def main() -> None:
    print("Opening landcover raster...")
    with rasterio.open(LANDCOVER_RASTER) as src:
        nodata = src.nodata if NODATA_VALUE is None else NODATA_VALUE
        grid = build_grid_spec(src, GRID_CELL_SIZE_M, EVAL_WINDOW_SIZE_M)

        print(f"Raster size: {src.width} x {src.height} pixels")
        print(f"Pixel size: {grid.px_size_m:.2f} m")
        print(f"Output grid cell size: {grid.grid_px} px ({GRID_CELL_SIZE_M} m)")
        print(f"Evaluation window size: {grid.eval_px} px ({EVAL_WINDOW_SIZE_M} m)")
        print(f"Output grid: {grid.out_width} x {grid.out_height} cells")
        print(f"Total output cells to process: {grid.out_width * grid.out_height}")
        print("Starting processing...\n")

        shdi_out = np.full((grid.out_height, grid.out_width), np.nan, dtype=np.float32)
        pd_out = np.full((grid.out_height, grid.out_width), np.nan, dtype=np.float32)
        pd_score_out = np.full((grid.out_height, grid.out_width), np.nan, dtype=np.float32)

        pixel_area_m2 = grid.px_size_m * grid.px_size_m
        total_cells = grid.out_width * grid.out_height
        processed = 0

        half = grid.eval_px // 2

        for oy in range(grid.out_height):
            for ox in range(grid.out_width):
                # Centre of this output cell in source pixel coordinates
                centre_row = oy * grid.grid_px + (grid.grid_px // 2)
                centre_col = ox * grid.grid_px + (grid.grid_px // 2)

                # Evaluation window bounds (clamped at raster edges)
                row0 = max(0, centre_row - half)
                col0 = max(0, centre_col - half)
                row1 = min(src.height, row0 + grid.eval_px)
                col1 = min(src.width, col0 + grid.eval_px)

                h = row1 - row0
                w = col1 - col0
                if h <= 0 or w <= 0:
                    processed += 1
                    continue

                win = Window(col0, row0, w, h)
                data = src.read(1, window=win)

                if nodata is not None:
                    valid_mask = (data != nodata)
                    valid_count = int(np.count_nonzero(valid_mask))
                else:
                    valid_count = int(data.size)

                if valid_count == 0:
                    processed += 1
                    continue

                total_count = int(data.size)
                valid_frac = valid_count / total_count
                if valid_frac < MIN_VALID_FRAC:
                    processed += 1
                    continue

                area_km2 = (valid_count * pixel_area_m2) / 1_000_000.0
                if area_km2 <= 0:
                    processed += 1
                    continue

                shdi = shannon_diversity(data, nodata)
                pd = patch_density(data, nodata, area_km2, CONNECTIVITY_8)
                score = patchiness_u_score(pd, PD_OPT, PD_ZERO_LOW, PD_ZERO_HIGH)

                shdi_out[oy, ox] = shdi
                pd_out[oy, ox] = pd
                pd_score_out[oy, ox] = score

                processed += 1

            if (oy + 1) % 10 == 0 or (oy + 1) == grid.out_height:
                pct = 100.0 * processed / total_cells
                print(f"Processed {processed:,} / {total_cells:,} cells ({pct:5.1f}%)")

        print("\nWriting coarse rasters...")
        write_float_tif(OUT_SHDI_TIF, shdi_out, grid.out_transform, src.crs)
        write_float_tif(OUT_PD_TIF, pd_out, grid.out_transform, src.crs)
        write_float_tif(OUT_PD_SCORE_TIF, pd_score_out, grid.out_transform, src.crs)

        print("\nWarping coarse rasters to 100 m grid (aligned to source)...")

        if OUT_SHDI_100M_TIF is not None:
            warp_to_template_grid(OUT_SHDI_TIF, OUT_SHDI_100M_TIF, src, resampling=Resampling.nearest)

        if OUT_PD_100M_TIF is not None:
            warp_to_template_grid(OUT_PD_TIF, OUT_PD_100M_TIF, src, resampling=Resampling.nearest)

        if OUT_PD_SCORE_100M_TIF is not None:
            warp_to_template_grid(OUT_PD_SCORE_TIF, OUT_PD_SCORE_100M_TIF, src, resampling=Resampling.nearest)

    print("Done: coarse and 100 m aligned SHDI, PD, and score rasters written successfully.")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"Total runtime: {hours:d}h {minutes:02d}m {seconds:05.2f}s")