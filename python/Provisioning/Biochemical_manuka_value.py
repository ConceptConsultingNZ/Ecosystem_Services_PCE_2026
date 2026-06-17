import os
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

"""
    Allocate region-level values to raster cells using a supply raster as weights.
    For each region r: cell_value_i = region_value_r * (weight_i / sum_weights_r)

    Inputs
    ------
    supply_raster_path:     Path to supply raster (weights). Typically Int16 0–100 with NoData.
    region_shapefile_path:        Path to region polygon shapefile containing 'value_field'.
    value_field:        Field name with region-level total to allocate (e.g., "mono_honey").
    out_value_raster_path:        Output path for allocated value raster (Int16 if safe, else Int32).
    out_norm_raster_path:        Output path for normalised raster (0–100 Int16).
    nodata_out:        NoData value for outputs.

    Outputs
    -------
    Writes:
      1) Allocated value raster (integer)
      2) Normalised raster 0–100 (Int16)

    """

# =========================
# User-defined paths
# =========================

INTERMEDIATE_DIR = r"<PROJECT_DIRECTORY>\Provisioning\Biochemical\Intermediate"
OUTPUT_DIR = r"<PROJECT_DIRECTORY>\Provisioning\Biochemical\Output"

SUPPLY_RASTER = os.path.join(OUTPUT_DIR, "biochemical_supply_raw.tif")
REGION_SHP = os.path.join(INTERMEDIATE_DIR, "region_values.shp")

REGION_ID_RASTER = os.path.join(INTERMEDIATE_DIR, "region_values.tif")

OUT_VALUE_RASTER = os.path.join(OUTPUT_DIR, "biochemical_flow_value.tif")
OUT_NORM_RASTER = os.path.join(OUTPUT_DIR, "biochemical_flow.tif")

VALUE_FIELD = "mono_honey"
NODATA_OUT = -9999


# =========================
# Allocation function
# =========================

def allocate_region_values_to_cells():

    with rasterio.open(SUPPLY_RASTER) as src:
        supply = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        supply_nodata = src.nodata

    if supply_nodata is None:
        supply_nodata = NODATA_OUT

    supply_valid = (supply != supply_nodata) & np.isfinite(supply)
    weights = np.where(supply_valid, supply, 0.0).astype(np.float32)

    # =========================
    # Rasterise regions (or reuse)
    # =========================

    if os.path.exists(REGION_ID_RASTER):
        print("Region ID raster exists. Loading...")
        with rasterio.open(REGION_ID_RASTER) as rds:
            rid = rds.read(1)
    else:
        print("Rasterising region polygons...")
        gdf = gpd.read_file(REGION_SHP)

        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)

        if VALUE_FIELD not in gdf.columns:
            raise ValueError(f"{VALUE_FIELD} not found in shapefile.")

        gdf = gdf.reset_index(drop=True)
        gdf["__rid__"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

        shapes = list(zip(gdf.geometry, gdf["__rid__"].astype(int)))

        rid = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.int32,
            all_touched=False
        )

        region_profile = profile.copy()
        region_profile.update(dtype=rasterio.int32, nodata=0, count=1)

        with rasterio.open(REGION_ID_RASTER, "w", **region_profile) as dst:
            dst.write(rid, 1)

        print("Region ID raster saved to intermediate folder.")

    # =========================
    # Allocation
    # =========================

    gdf = gpd.read_file(REGION_SHP)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    gdf = gdf.reset_index(drop=True)
    gdf["__rid__"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    region_values = gdf[VALUE_FIELD].astype(float).fillna(0.0).to_numpy()

    alloc_mask = (rid > 0) & supply_valid & (weights > 0)

    max_rid = int(gdf["__rid__"].max())

    rid_flat = rid[alloc_mask].ravel()
    w_flat = weights[alloc_mask].astype(np.float64).ravel()

    sum_w = np.bincount(rid_flat, weights=w_flat, minlength=max_rid + 1)

    region_total = np.zeros(max_rid + 1, dtype=np.float64)
    region_total[1:max_rid + 1] = region_values

    alloc_arr = np.full((height, width), np.nan, dtype=np.float64)

    denom = sum_w[rid_flat]
    safe = denom > 0

    alloc_vals = np.zeros_like(w_flat, dtype=np.float64)
    alloc_vals[safe] = region_total[rid_flat[safe]] * (w_flat[safe] / denom[safe])

    alloc_arr[alloc_mask] = alloc_vals

    # =========================
    # Write outputs
    # =========================

    max_cell_value = float(np.nanmax(alloc_arr)) if np.any(np.isfinite(alloc_arr)) else 0.0

    int16_max = np.iinfo(np.int16).max
    use_int16 = max_cell_value <= int16_max

    value_dtype = np.int16 if use_int16 else np.int32
    value_int = np.full((height, width), NODATA_OUT, dtype=value_dtype)

    finite = np.isfinite(alloc_arr)
    if np.any(finite):
        rounded = np.rint(alloc_arr[finite])
        if use_int16:
            rounded = np.clip(rounded, 0, int16_max)
        value_int[finite] = rounded.astype(value_dtype)

    norm_int = np.full((height, width), NODATA_OUT, dtype=np.int16)
    if max_cell_value > 0 and np.any(finite):
        norm = (alloc_arr / max_cell_value) * 100.0
        norm = np.clip(np.rint(norm), 0, 100)
        norm_int[finite] = norm[finite].astype(np.int16)

    profile.update(compress="DEFLATE", predictor=2, tiled=True)

    prof_val = profile.copy()
    prof_val.update(dtype=value_dtype, nodata=NODATA_OUT, count=1)

    with rasterio.open(OUT_VALUE_RASTER, "w", **prof_val) as dst:
        dst.write(value_int, 1)

    prof_norm = profile.copy()
    prof_norm.update(dtype=rasterio.int16, nodata=NODATA_OUT, count=1)

    with rasterio.open(OUT_NORM_RASTER, "w", **prof_norm) as dst:
        dst.write(norm_int, 1)

    print("Allocation complete.")
    print("Outputs written:")
    print(OUT_VALUE_RASTER)
    print(OUT_NORM_RASTER)


# =========================
# Run
# =========================

if __name__ == "__main__":
    allocate_region_values_to_cells()