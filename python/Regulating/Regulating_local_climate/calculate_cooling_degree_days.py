"""
Calculate Cooling Degree Days (CDD) from WorldClim GeoTIFFs and output to a fixed NZTM2000 grid.

Assumptions / notes:
- Computes monthly mean as (tmax + tmin) / 2.
- Monthly CDD is approximated as: max(0, Tmean_month - BASE_TEMP) * days_in_month
- Outputs annual CDD for a chosen year (sum of 12 monthly CDD rasters).
- Reprojects/resamples to EPSG:2193 on a fixed 100 m grid with target extent and nodata -9999.

Dependencies:
  pip install rasterio numpy

Example inputs:
  D:\Data\WorldClim\wc2.1_cruts4.09_2.5m_tmax_2020-2024\wc2.1_cruts4.09_2.5m_tmax_2024-01.tif
  (same pattern for tmin and/or tavg)

"""

import os
import time
import logging
import calendar
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import from_origin


# ============================================================
# ---------------------- CONSTANTS ----------------------
# ============================================================

YEAR = 2024
BASE_TEMP = 18.0  # Cooling degree base temperature (°C)

TMAX_DIR = r"D:\Data\WorldClim\wc2.1_cruts4.09_2.5m_tmax_2020-2024"
TMIN_DIR = r"D:\Data\WorldClim\wc2.1_cruts4.09_2.5m_tmin_2020-2024"

# Filename pattern pieces
FILE_PREFIX = "wc2.1_cruts4.09_2.5m_"
TMAX_TAG = "tmax"
TMIN_TAG = "tmin"

OUTPUT_FILE = r"<PROJECT_DIRECTORY>\Regulating\Local climate\Intermediate\cooling_degree_days.tif"

# Target grid parameters
DST_CRS = "EPSG:2193"
DST_RES = 100.0
DST_BOUNDS = (715100.0, 3728500.0, 2893500.0, 7142400.0)
DST_NODATA = -9999.0

# ------------------------------------------------------------
# ---------------------- LOGGING SETUP -----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


# ---------------------- FUNCTIONS ---------------------------

def build_path(base_dir, var_tag, year, month):
    month_str = f"{month:02d}"
    filename = f"{FILE_PREFIX}{var_tag}_{year}-{month_str}.tif"
    return Path(base_dir) / filename


def create_destination_grid():
    xmin, ymin, xmax, ymax = DST_BOUNDS
    width = int((xmax - xmin) / DST_RES)
    height = int((ymax - ymin) / DST_RES)

    transform = from_origin(xmin, ymax, DST_RES, DST_RES)

    logger.info(f"Destination grid: {width} x {height} pixels")
    logger.info(f"Resolution: {DST_RES} m")
    logger.info(f"CRS: {DST_CRS}")

    return width, height, transform


def reproject_to_grid(src_path, dst_shape, dst_transform):

    logger.info(f"  Reprojecting: {src_path.name}")

    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_nodata = src.nodata

        dst = np.full(dst_shape, DST_NODATA, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            dst_nodata=DST_NODATA,
            resampling=Resampling.bilinear
        )

    return dst


# ---------------------- MAIN ---------------------------

def main():

    start_time = time.time()
    logger.info("Starting annual CDD calculation")
    logger.info(f"Year: {YEAR}")
    logger.info(f"Base temperature: {BASE_TEMP} °C")

    width, height, dst_transform = create_destination_grid()
    dst_shape = (height, width)

    annual_cdd = np.zeros(dst_shape, dtype=np.float32)
    valid_mask = np.zeros(dst_shape, dtype=bool)

    for month in range(1, 13):

        month_start = time.time()
        logger.info(f"Processing month {month:02d}/12")

        days = calendar.monthrange(YEAR, month)[1]

        tmax_path = build_path(TMAX_DIR, TMAX_TAG, YEAR, month)
        tmin_path = build_path(TMIN_DIR, TMIN_TAG, YEAR, month)

        if not tmax_path.exists():
            raise FileNotFoundError(f"Missing: {tmax_path}")
        if not tmin_path.exists():
            raise FileNotFoundError(f"Missing: {tmin_path}")

        tmax = reproject_to_grid(tmax_path, dst_shape, dst_transform)
        tmin = reproject_to_grid(tmin_path, dst_shape, dst_transform)

        logger.info("  Calculating monthly mean temperature")

        tmean = np.full(dst_shape, DST_NODATA, dtype=np.float32)
        ok = (tmax != DST_NODATA) & (tmin != DST_NODATA)
        tmean[ok] = (tmax[ok] + tmin[ok]) * 0.5

        logger.info("  Calculating monthly CDD")

        diff = np.zeros(dst_shape, dtype=np.float32)
        diff[ok] = tmean[ok] - BASE_TEMP
        diff = np.where(diff > 0, diff, 0.0)

        monthly_cdd = diff * days

        annual_cdd[ok] += monthly_cdd[ok]
        valid_mask[ok] = True

        month_elapsed = time.time() - month_start
        logger.info(f"  Month {month:02d} complete in {month_elapsed:.2f} seconds")

    logger.info("Finalising output raster")

    output = np.full(dst_shape, DST_NODATA, dtype=np.float32)
    output[valid_mask] = annual_cdd[valid_mask]

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": DST_CRS,
        "transform": dst_transform,
        "nodata": DST_NODATA,
        "compress": "DEFLATE",
        "predictor": 2,
        "tiled": True
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:
        dst.write(output, 1)

    total_time = time.time() - start_time
    logger.info("Annual CDD calculation complete")
    logger.info(f"Output written to: {OUTPUT_FILE}")
    logger.info(f"Total runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()