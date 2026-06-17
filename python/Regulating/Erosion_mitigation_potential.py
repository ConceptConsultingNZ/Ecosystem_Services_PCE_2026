"""
Create a tree-mitigability raster from NZLRI erosion features.

Method
------
For each polygon:
    score = sum(severity_i * mitigation_i)
    total_severity = sum(severity_i)
    mitigability = score / total_severity

This yields a raster from 0 to 1, where:
    0 = erosion types in that polygon are not very responsive to trees
    1 = erosion types in that polygon are highly responsive to trees

Notes
-----
- NZLRI severities are treated as weights, not physical shares of tonnes.
- Non-numeric severity values (urban, water, etc.) are ignored.
- Polygons with no valid erosion severities get NODATA.
- Output is aligned to a template raster.

Requirements
------------
pip install geopandas pandas rasterio numpy pyogrio
"""

from pathlib import Path
import time
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
import rasterio
from rasterio.features import rasterize


# =========================
# Constants / configuration
# =========================
WORKING_DIR = r"<PROJECT_DIRECTORY>\Regulating\Erosion regulation\Intermediate"

# Inputs
NZLRI_VECTOR = Path(r"D:\Data\LRIS\lris-nzlri-erosion-type-and-severity-FGDB\nzlri-erosion-type-and-severity.gdb")          # feature layer with ERO1S/ERO1T ...
LAYER_NAME = "NZLRI_Erosion_Type_and_Severity"                       # leave None for shapefile / first layer
MITIGATION_CSV = os.path.join(WORKING_DIR,"mitigation_by_erosion_type.csv")    # CSV with columns: Erosion_type, Mitigation
TEMPLATE_RASTER = Path(r"D:\Data\LRIS\lris-nzeem-erosion-rates\nzeem-erosion-rates_100m.tif")

# Outputs
OUT_MITIGABILITY = os.path.join(WORKING_DIR,"mitigability_100m.tif")
OUT_SCORE = os.path.join(WORKING_DIR,"mitigability_score_100m.tif")            # optional diagnostic
OUT_TOTAL_SEVERITY = os.path.join(WORKING_DIR,"mitigability_total_severity_100m.tif")  # optional diagnostic

# Erosion field pairs in the NZLRI vector
EROSION_FIELD_PAIRS = [
    ("ERO1S", "ERO1T"),
    ("ERO2S", "ERO2T"),
    ("ERO3S", "ERO3T"),
    ("ERO4S", "ERO4T"),
]

# Raster settings
NODATA = 0
ALL_TOUCHED = False
GDAL_COMPRESS = "DEFLATE"
GDAL_PREDICTOR = 3
GDAL_ZLEVEL = 9

# If True, output mitigability as percent 0-100 instead of fraction 0-1
OUTPUT_PERCENT = False

# Output as 0-100 instead of 0-1
OUTPUT_PERCENT = False

# Logging / progress
LOG_LEVEL = logging.INFO
LOG_EVERY = 5000


# =========================
# Logging setup
# =========================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# Helper functions
# =========================

def parse_mitigation_value(x):
    """
    Convert values like '35%' or 0.35 or 35 into a fraction (0-1).
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        x = x.strip()
        if x.endswith("%"):
            return float(x[:-1]) / 100.0
        x = float(x)

    x = float(x)

    if x > 1.0:
        return x / 100.0
    return x

def parse_severity_value(x):
    """
    Convert severity to float if numeric, else NaN.
    NZLRI may contain non-numeric codes for urban/water/etc.
    """
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def build_mitigation_lookup(csv_path):
    df = pd.read_csv(csv_path)
    required = {"Erosion_type", "Mitigation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["Erosion_type"] = df["Erosion_type"].astype(str).str.strip()
    df["Mitigation_frac"] = df["Mitigation"].apply(parse_mitigation_value)

    if df["Mitigation_frac"].isna().any():
        bad = df.loc[df["Mitigation_frac"].isna(), "Erosion_type"].tolist()
        raise ValueError(f"Could not parse mitigation values for: {bad}")

    lookup = dict(zip(df["Erosion_type"], df["Mitigation_frac"]))

    for k, v in lookup.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Mitigation for '{k}' is outside 0-1: {v}")

    return lookup

def compute_polygon_mitigability_from_tuple(row, mitigation_lookup):
    """
    row is an itertuples() result
    Returns:
        score, total_severity, mitigability
    """
    score = 0.0
    total = 0.0

    for sev_field, type_field in EROSION_FIELD_PAIRS:
        sev = parse_severity_value(getattr(row, sev_field, None))
        etype = getattr(row, type_field, None)

        if pd.isna(sev):
            continue
        if pd.isna(etype):
            continue

        etype = str(etype).strip()
        if etype == "":
            continue

        mit = mitigation_lookup.get(etype)
        if mit is None:
            continue

        score += sev * mit
        total += sev

    if total == 0:
        return np.nan, np.nan, np.nan

    mitigability = score / total
    return score, total, mitigability

def rasterize_column(gdf, column, template_meta, out_path):
    logger.info("Rasterising column '%s' to %s", column, out_path)

    with rasterio.open(TEMPLATE_RASTER) as src:
        transform = src.transform
        height = src.height
        width = src.width

    shapes = []
    kept = 0
    skipped = 0

    for geom, value in zip(gdf.geometry, gdf[column]):
        if geom is None or geom.is_empty or pd.isna(value):
            skipped += 1
            continue
        shapes.append((geom, value))
        kept += 1

    logger.info("Rasterising %s features for '%s' (%s skipped)", kept, column, skipped)

    arr = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=NODATA,
        dtype="float32",
        all_touched=ALL_TOUCHED,
    )

    out_meta = template_meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "nodata": NODATA,
            "compress": GDAL_COMPRESS,
            "predictor": GDAL_PREDICTOR,
            "zlevel": GDAL_ZLEVEL,
        }
    )

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(arr.astype("float32"), 1)

    logger.info("Finished writing %s", out_path)


# =========================
# Main
# =========================

def main():
    t0 = time.time()
    logger.info("Starting mitigability raster creation")

    mitigation_lookup = build_mitigation_lookup(MITIGATION_CSV)
    logger.info("Loaded %s mitigation coefficients", len(mitigation_lookup))

    logger.info("Reading NZLRI vector: %s", NZLRI_VECTOR)
    read_start = time.time()

    # If pyogrio is slow because of multipart polygon organisation,
    # try engine="fiona" here as a fallback.
    if LAYER_NAME:
        gdf = gpd.read_file(NZLRI_VECTOR, layer=LAYER_NAME)
    else:
        gdf = gpd.read_file(NZLRI_VECTOR)

    logger.info("Finished reading vector in %.1f seconds", time.time() - read_start)
    logger.info("Read %s features", len(gdf))

    gdf = gdf[~gdf.geometry.isna()].copy()
    logger.info("Null geometry: %s", gdf.geometry.isna().sum())
    logger.info("Empty geometry: %s", gdf.geometry.is_empty.sum())
    logger.info("NaN mitigability: %s", gdf["mitigability"].isna().sum())
    logger.info("Retained %s features with non-null geometry", len(gdf))

    with rasterio.open(TEMPLATE_RASTER) as tmpl:
        template_meta = tmpl.meta.copy()
        template_crs = tmpl.crs

    if gdf.crs != template_crs:
        logger.info("Reprojecting features from %s to %s", gdf.crs, template_crs)
        reproj_start = time.time()
        gdf = gdf.to_crs(template_crs)
        logger.info("Finished reprojection in %.1f seconds", time.time() - reproj_start)

    logger.info("Computing mitigability fields")
    calc_start = time.time()

    scores = np.full(len(gdf), np.nan, dtype="float32")
    totals = np.full(len(gdf), np.nan, dtype="float32")
    mitigabilities = np.full(len(gdf), np.nan, dtype="float32")

    for i, row in enumerate(gdf.itertuples(index=False), start=0):
        score, total, mitigability = compute_polygon_mitigability_from_tuple(row, mitigation_lookup)
        scores[i] = score
        totals[i] = total
        mitigabilities[i] = mitigability

        if (i + 1) % LOG_EVERY == 0:
            elapsed = time.time() - calc_start
            logger.info(
                "Processed %s / %s features (%.1f%%) in %.1f s",
                i + 1,
                len(gdf),
                100.0 * (i + 1) / len(gdf),
                elapsed,
            )

    gdf["mit_score"] = scores
    gdf["mit_total_sev"] = totals
    gdf["mitigability"] = mitigabilities

    if OUTPUT_PERCENT:
        gdf["mitigability"] = gdf["mitigability"] * 100.0

    logger.info("Finished mitigability calculation in %.1f seconds", time.time() - calc_start)

    rasterize_column(gdf, "mitigability", template_meta, OUT_MITIGABILITY)
    rasterize_column(gdf, "mit_score", template_meta, OUT_SCORE)
    rasterize_column(gdf, "mit_total_sev", template_meta, OUT_TOTAL_SEVERITY)

    logger.info("All done in %.1f seconds", time.time() - t0)
    logger.info("Mitigability raster: %s", OUT_MITIGABILITY)
    logger.info("Score raster: %s", OUT_SCORE)
    logger.info("Total severity raster: %s", OUT_TOTAL_SEVERITY)


if __name__ == "__main__":
    main()