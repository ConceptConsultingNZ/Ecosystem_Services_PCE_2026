"""
Mask an avoided erosion raster using LCDB land-cover classes.

Pixels are retained only where the LCDB class is marked with include = 1
in a CSV lookup table. All other pixels are set to NoData in the output.
"""

import rasterio
import numpy as np
import pandas as pd
import os
from rasterio.warp import reproject, Resampling

from ES.Cultural.Recreation_create_biking_extent import WORKING_DIR

# =========================
# FILE PATH CONSTANTS
# =========================
WORKING_DIR = r"<PROJECT_DIRECTORY>\Regulating\Erosion regulation\Intermediate"
INPUT_AVOIDED_EROSION = r"D:\InVEST\SDR 20260304\avoided_erosion.tif"
INPUT_LCDB = r"<PROJECT_DIRECTORY>\Common\lcdb6_expanded.tif"
CLASS_CSV = os.path.join(WORKING_DIR,"lcdb_mask.csv")
OUTPUT_MASKED = os.path.join(WORKING_DIR,"avoided_erosion_masked.tif")

# Optional: keep an intermediate warped raster for debugging
WRITE_WARPED_INTERMEDIATE = False
OUTPUT_WARPED = os.path.join(WORKING_DIR,"avoided_erosion_warped_to_lcdb.tif")

# =========================
# READ CLASS MASK CSV
# =========================
df = pd.read_csv(CLASS_CSV)

class_col = df.columns[0]
include_col = df.columns[1]

included_classes = df.loc[df[include_col] == 1, class_col].astype(int).values

# =========================
# OPEN LCDB (TARGET GRID)
# =========================
with rasterio.open(INPUT_LCDB) as lcdb_src:
    lcdb = lcdb_src.read(1)
    lcdb_profile = lcdb_src.profile.copy()
    target_crs = lcdb_src.crs
    target_transform = lcdb_src.transform
    target_height = lcdb_src.height
    target_width = lcdb_src.width

# =========================
# OPEN AVOIDED EROSION, WARP IF NEEDED
# =========================
with rasterio.open(INPUT_AVOIDED_EROSION) as erosion_src:
    erosion_profile = erosion_src.profile.copy()

    grids_match = (
        erosion_src.crs == target_crs
        and erosion_src.transform == target_transform
        and erosion_src.width == target_width
        and erosion_src.height == target_height
    )

    if grids_match:
        erosion = erosion_src.read(1)
        warped_profile = erosion_profile
    else:
        # Allocate destination array on LCDB grid
        erosion = np.full((target_height, target_width), np.nan, dtype=np.float32)

        src_nodata = erosion_src.nodata
        # Use nearest only for categorical data; avoided erosion is continuous, so bilinear
        reproject(
            source=rasterio.band(erosion_src, 1),
            destination=erosion,
            src_transform=erosion_src.transform,
            src_crs=erosion_src.crs,
            src_nodata=src_nodata,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        warped_profile = lcdb_profile.copy()
        warped_profile.update(dtype=rasterio.float32, nodata=np.nan, count=1)

        if WRITE_WARPED_INTERMEDIATE:
            os.makedirs(os.path.dirname(OUTPUT_WARPED), exist_ok=True)
            with rasterio.open(OUTPUT_WARPED, "w", **warped_profile) as tmpdst:
                tmpdst.write(erosion.astype(np.float32), 1)

# =========================
# BUILD MASK & APPLY
# =========================
mask = np.isin(lcdb, included_classes)

masked_erosion = np.where(mask, erosion, np.nan).astype(np.float32)

# =========================
# WRITE OUTPUT
# =========================
out_profile = lcdb_profile.copy()
out_profile.update(dtype=rasterio.float32, nodata=np.nan, count=1)

os.makedirs(os.path.dirname(OUTPUT_MASKED), exist_ok=True)
with rasterio.open(OUTPUT_MASKED, "w", **out_profile) as dst:
    dst.write(masked_erosion, 1)

print("Masked raster written to:", OUTPUT_MASKED)