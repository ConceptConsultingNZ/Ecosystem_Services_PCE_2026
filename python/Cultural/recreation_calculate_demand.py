"""
Compute recreation demand rasters by region for a specified list of activity columns.

For each raster cell and each chosen activity column:
    demand_activity = population_cell_value * rate_for_region(activity)

The activity rates are looked up from a CSV using the region code raster.

Outputs:
    recreation_demand_[column].tif  (Float32, aligned to population raster)
"""

import os
import numpy as np
import pandas as pd
import rasterio

# ==============================
# --------- CONSTANTS ----------
# ==============================
WORKING_DIR = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate"
OUTPUT_DIR = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate"

POPULATION_RASTER = r"<PROJECT_DIRECTORY>\Common\pop_adults.tif"
REGION_RASTER = r"D:\Data\Stats NZ geography\region.tif"
REGION_LOOKUP_CSV = os.path.join(WORKING_DIR, "values_per_capita.csv")

REGION_CODE_FIELD = "region_code"  # column in CSV matching region raster values

# List the CSV fields you want rasters for (exact column names).
# Example:
# ACTIVITY_FIELDS = ["mountain_biking", "short_walk","day_tramp","overnight_tramp","freshwater","freshwater_fishing",marine_boating"]
ACTIVITY_FIELDS = ["short_walk"]

ALLOW_MISSING_REGIONS = True  # if False, error if region raster has codes not in CSV

OUTPUT_PREFIX = "demand_"
OUTPUT_DTYPE = "float32"
NODATA_VALUE = -9999

# ==============================
# --------- SCRIPT -------------
# ==============================


def _check_alignment(pop_src: rasterio.DatasetReader, region_src: rasterio.DatasetReader) -> None:
    if (
        pop_src.width != region_src.width
        or pop_src.height != region_src.height
        or pop_src.transform != region_src.transform
        or pop_src.crs != region_src.crs
    ):
        raise ValueError("Population and region rasters must be aligned (same shape, transform, CRS).")


def _safe_name(col: str) -> str:
    return str(col).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(REGION_LOOKUP_CSV)

    if REGION_CODE_FIELD not in df.columns:
        raise ValueError(f"CSV must contain column '{REGION_CODE_FIELD}'")

    # Validate requested activity fields exist
    missing_cols = [c for c in ACTIVITY_FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Requested fields not found in CSV: {missing_cols}. CSV columns: {list(df.columns)}")

    # Ensure region codes are unique in the CSV
    if df[REGION_CODE_FIELD].duplicated().any():
        dupes = df.loc[df[REGION_CODE_FIELD].duplicated(), REGION_CODE_FIELD].tolist()
        raise ValueError(f"Duplicate region codes found in CSV: {dupes[:20]} (showing up to 20)")

    # Open rasters
    with rasterio.open(POPULATION_RASTER) as pop_src, rasterio.open(REGION_RASTER) as region_src:
        _check_alignment(pop_src, region_src)

        pop = pop_src.read(1)
        region = region_src.read(1)

        pop_nodata = pop_src.nodata
        valid_pop_mask = np.ones(pop.shape, dtype=bool)
        if pop_nodata is not None:
            valid_pop_mask = pop != pop_nodata

        unique_regions = np.unique(region)

        # Check missing regions
        csv_regions = set(df[REGION_CODE_FIELD].tolist())
        raster_regions = set(unique_regions.tolist())
        missing_regions = sorted(list(raster_regions - csv_regions))
        if missing_regions and not ALLOW_MISSING_REGIONS:
            raise ValueError(
                f"Region codes present in region raster but missing from CSV: {missing_regions[:50]} (showing up to 50)"
            )

        # Output profile aligned to population raster
        profile = pop_src.profile.copy()
        profile.update(
            dtype=OUTPUT_DTYPE,
            count=1,
            nodata=NODATA_VALUE,
            COMPRESS="DEFLATE",
            PREDICTOR=3,
            TILED=True,
            BLOCKXSIZE=512,
            BLOCKYSIZE=512,
        )

        # Precompute region->index mask mapping once (speeds up multiple outputs)
        region_masks = {r: (region == r) for r in unique_regions.tolist()}

        for col in ACTIVITY_FIELDS:
            lookup = dict(zip(df[REGION_CODE_FIELD], df[col]))

            rate = np.zeros(region.shape, dtype=np.float32)
            for r, mask in region_masks.items():
                rate_val = float(lookup.get(r, 0.0))  # default to 0 if missing
                rate[mask] = rate_val

            demand = np.full(pop.shape, np.float32(NODATA_VALUE), dtype=np.float32)
            demand[valid_pop_mask] = pop[valid_pop_mask].astype(np.float32) * rate[valid_pop_mask]

            out_name = f"{OUTPUT_PREFIX}{_safe_name(col)}.tif"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(demand.astype(np.float32), 1)

            print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()