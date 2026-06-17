"""
Create ecosystem service value rasters from a seafloor community classification raster and SCC value table.

This script reads a categorical seafloor community classification raster and a CSV lookup table
containing SCC class IDs and associated fish provisioning values. For each value column in the CSV,
it creates a single-band GeoTIFF with the same extent, resolution, geotransform, and projection as
the input classification raster.

The first value column is written as Float32, while all subsequent value columns are rounded to the
nearest integer and written as Int16. Output rasters use -9999 as nodata and are compressed using
DEFLATE with tiled GeoTIFF output.

Inputs:
class_raster: Path to the SCC categorical raster.
csv_path: Path to the CSV lookup table. The first column must contain SCC class IDs; all remaining
columns are treated as values to be mapped.
out_dir: Directory where output rasters will be written.
PREFIX: Prefix added to each output raster filename.

Outputs:
One GeoTIFF per value column in the CSV, named using the configured prefix and a cleaned version
of the source column name.

Raises:
RuntimeError: If the class raster cannot be opened, contains only nodata, or if the CSV does not
contain at least one SCC column and one value column.
"""

import os
import re
import numpy as np
import pandas as pd
from osgeo import gdal

# -----------------------------
# Inputs
# -----------------------------
class_raster = r"D:\Data\DOC\New Zealand Seafloor Community Classification\SCC_GF75_100m_EPSG2193_Int16.tif"
#csv headings: SCC,flow_value,flow,supply
csv_path = r"<PROJECT_DIRECTORY>\Provisioning\Fish\scc_values.csv"

out_dir = r"<PROJECT_DIRECTORY>\Provisioning\Fish\Output"
PREFIX = "fish_"

OUT_NODATA = -9999

# First value column: Float32
FIRST_COL_GDAL_TYPE = gdal.GDT_Float32
FIRST_COL_NP_DTYPE = np.float32

# Other value columns: Int16 (0-100)
OTHER_COL_GDAL_TYPE = gdal.GDT_Int16
OTHER_COL_NP_DTYPE = np.int16

# -----------------------------
# Helpers
# -----------------------------
def safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "value"

def write_like_template(template_ds, out_path, array, nodata, gdal_type):
    driver = gdal.GetDriverByName("GTiff")
    ysize, xsize = array.shape

    is_float = gdal_type in (gdal.GDT_Float32, gdal.GDT_Float64)
    predictor = "PREDICTOR=3" if is_float else "PREDICTOR=2"

    out_ds = driver.Create(
        out_path,
        xsize,
        ysize,
        1,
        gdal_type,
        options=[
            "COMPRESS=DEFLATE",
            predictor,
            "ZLEVEL=9",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
        ],
    )
    out_ds.SetGeoTransform(template_ds.GetGeoTransform())
    out_ds.SetProjection(template_ds.GetProjection())

    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata) if is_float else int(nodata))
    band.WriteArray(array)
    band.FlushCache()
    out_ds = None

def build_lut_float(df, scc_col, val_col, lut_size, nodata):
    scc_ids = pd.to_numeric(df[scc_col], errors="raise").astype(int).to_numpy()
    vals = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=np.float64)

    lut = np.full(lut_size, float(nodata), dtype=np.float32)
    good = ~np.isnan(vals)
    lut[scc_ids[good]] = vals[good].astype(np.float32)
    return lut

def build_lut_int(df, scc_col, val_col, lut_size, nodata):
    scc_ids = pd.to_numeric(df[scc_col], errors="raise").astype(int).to_numpy()
    vals = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=np.float64)

    lut = np.full(lut_size, int(nodata), dtype=np.int64)
    good = ~np.isnan(vals)
    lut[scc_ids[good]] = np.rint(vals[good]).astype(np.int64)  # keep your rounding choice
    return lut

# -----------------------------
# Load class raster
# -----------------------------
ds = gdal.Open(class_raster, gdal.GA_ReadOnly)
if ds is None:
    raise RuntimeError(f"Could not open class raster: {class_raster}")

band = ds.GetRasterBand(1)
classes = band.ReadAsArray()
class_nodata = band.GetNoDataValue()
print("Class raster nodata:", class_nodata)

valid_mask = np.ones_like(classes, dtype=bool) if class_nodata is None else (classes != class_nodata)
if not np.any(valid_mask):
    raise RuntimeError("Class raster appears to be entirely nodata.")

max_class_in_raster = int(np.max(classes[valid_mask]))

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_path)
if df.shape[1] < 2:
    raise RuntimeError("CSV must have SCC plus at least one value column.")

scc_col = df.columns[0]
value_cols = list(df.columns[1:])

df[scc_col] = pd.to_numeric(df[scc_col], errors="raise").astype(int)

lut_size = max(max_class_in_raster, int(df[scc_col].max())) + 1
print("Value columns:", value_cols)

# -----------------------------
# Create rasters
# -----------------------------
os.makedirs(out_dir, exist_ok=True)

for i, col in enumerate(value_cols):
    col_clean = safe_name(col)
    out_path = os.path.join(out_dir, f"{PREFIX}{col_clean}.tif")

    if i == 0:
        lut = build_lut_float(df, scc_col, col, lut_size, OUT_NODATA)
        out = np.full(classes.shape, float(OUT_NODATA), dtype=np.float32)
        out[valid_mask] = lut[classes[valid_mask].astype(int)]
        write_like_template(ds, out_path, out.astype(FIRST_COL_NP_DTYPE, copy=False), OUT_NODATA, FIRST_COL_GDAL_TYPE)
        print("Wrote (Float32):", out_path)
    else:
        lut = build_lut_int(df, scc_col, col, lut_size, OUT_NODATA)
        out_int64 = np.full(classes.shape, int(OUT_NODATA), dtype=np.int64)
        out_int64[valid_mask] = lut[classes[valid_mask].astype(int)]
        out_clipped = np.clip(out_int64, -32768, 32767).astype(OTHER_COL_NP_DTYPE, copy=False)
        write_like_template(ds, out_path, out_clipped, OUT_NODATA, OTHER_COL_GDAL_TYPE)
        print("Wrote (Int16):", out_path)

ds = None
print("Done.")
