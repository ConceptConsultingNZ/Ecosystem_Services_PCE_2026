"""
PURPOSE
Create 100 m (EPSG:2193) compressed rasters for air-quality ES:
1) SUPPLY = Hansen tree cover (%)
2) DEMAND = SA1 demand index (0–1) rasterised to the same grid
3) FLOW   = SUPPLY × DEMAND
Also computes mean tree cover per SA1 polygon (tc_mean) from the supply raster (zonal mean).
"""

import os
import warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

# -----------------------------------------------------------------------------
# Housekeeping
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message="GeoSeries.notna", category=UserWarning)

print("Starting ES supply, demand, flow rasters (EPSG:2193, 100 m, compressed) + SA1 tc_mean...")

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
WORK_DIR = r"<PROJECT_DIRECTORY>\Regulating\air quality"

SUPPLY_RASTER = r"D:\Data\Tree cover\Hansen_treecover_compressed.tif"  # 0–100
GPKG_PATH = os.path.join(WORK_DIR, "air_quality.gpkg")

DEMAND_LAYER = "SA12023_population_PM25"
DEMAND_FIELD = "demand_index"  # 0–1

# Optional ID field (only needed if you later want to join/export reliably)
SA1_ID_FIELD = "SA12023_CODE"  # change if needed

# Target grid
TARGET_CRS = "EPSG:2193"
TARGET_RES = 100  # metres

# Outputs
SUPPLY_RASTER_OUT = os.path.join(WORK_DIR, "air_quality_supply_treecover_100m_2193.tif")
DEMAND_RASTER_OUT = os.path.join(WORK_DIR, "air_quality_demand_100m_2193.tif")
FLOW_RASTER_OUT   = os.path.join(WORK_DIR, "air_quality_flow_supply_x_demand_100m_2193.tif")

# Optional: write polygons with tc_mean back to GPKG
WRITE_TC_MEAN_TO_GPKG = False
OUT_LAYER_TC = f"{DEMAND_LAYER}_with_tc_mean"  # only used if WRITE_TC_MEAN_TO_GPKG=True

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def compressed_gtiff_profile(base_profile, dtype, nodata):
    """Create a compressed GeoTIFF profile (do not inherit VRT driver)."""
    p = base_profile.copy()
    p.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="lzw",
        predictor=2,
        tiled=True,
        BIGTIFF="YES",
        interleave="band"
    )
    return p


def mean_raster_value_in_geom(src, geom, nodata):
    """
    Fast-ish zonal mean for a single polygon:
    reads only the bounding window then masks by geometry.
    """
    minx, miny, maxx, maxy = geom.bounds
    w = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

    if w.width <= 0 or w.height <= 0:
        return np.nan

    data = src.read(1, window=w).astype("float32")
    if data.size == 0:
        return np.nan

    win_transform = rasterio.windows.transform(w, src.transform)
    inside = geometry_mask([geom], out_shape=data.shape, transform=win_transform, invert=True)

    # keep only inside-geom pixels
    data = np.where(inside, data, np.nan)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    m = np.nanmean(data)
    return float(m) if np.isfinite(m) else np.nan


# -----------------------------------------------------------------------------
# Step 1: Read demand polygons
# -----------------------------------------------------------------------------
print("\n[1/6] Reading demand polygons...")
gdf = gpd.read_file(GPKG_PATH, layer=DEMAND_LAYER)

if DEMAND_FIELD not in gdf.columns:
    raise ValueError(f"Demand field '{DEMAND_FIELD}' not found in layer '{DEMAND_LAYER}'.")

if SA1_ID_FIELD not in gdf.columns:
    print(f"  Note: SA1 ID field '{SA1_ID_FIELD}' not found. Continuing anyway (not required for rasterisation).")

# Clip to [0,1]
gdf[DEMAND_FIELD] = gdf[DEMAND_FIELD].clip(0, 1)

# Drop bad geometries
valid_geom = gdf.geometry.notna() & (~gdf.geometry.is_empty)
dropped = int((~valid_geom).sum())
if dropped:
    print(f"  Dropping {dropped:,} features with None/empty geometry")
gdf = gdf[valid_geom].copy()

print(f"  Loaded {len(gdf):,} polygons")
print("  Demand summary:")
print(gdf[DEMAND_FIELD].describe())

# -----------------------------------------------------------------------------
# Step 2: Compute SA1 mean treecover (tc_mean) from supply raster
# -----------------------------------------------------------------------------
print("\n[2/6] Calculating SA1 mean tree cover (tc_mean) from the supply raster...")
with rasterio.open(SUPPLY_RASTER) as src_supply_native:
    supply_native_crs = src_supply_native.crs
    supply_native_nodata = src_supply_native.nodata

    if gdf.crs != supply_native_crs:
        print("  Reprojecting polygons to supply raster CRS for zonal mean...")
        gdf_tc = gdf.to_crs(supply_native_crs)
    else:
        gdf_tc = gdf

    tc_means = np.full(len(gdf_tc), np.nan, dtype="float32")

    for i, geom in enumerate(gdf_tc.geometry, start=1):
        tc_means[i - 1] = mean_raster_value_in_geom(src_supply_native, geom, supply_native_nodata)
        if i % 500 == 0 or i == len(gdf_tc):
            print(f"  Processed {i:,} / {len(gdf_tc):,} polygons")

# Attach back to original gdf (row order preserved)
gdf["tc_mean"] = tc_means

print("  tc_mean summary:")
print(gdf["tc_mean"].describe())

if WRITE_TC_MEAN_TO_GPKG:
    print(f"  Writing polygons with tc_mean to GeoPackage layer: {OUT_LAYER_TC}")
    gdf.to_file(GPKG_PATH, layer=OUT_LAYER_TC, driver="GPKG")

# -----------------------------------------------------------------------------
# Step 3: Define the 100 m EPSG:2193 target grid via WarpedVRT of the supply raster
# -----------------------------------------------------------------------------
print("\n[3/6] Building target grid (100 m, EPSG:2193) from supply raster...")
OUT_NODATA = -9999.0

with rasterio.open(SUPPLY_RASTER) as src_supply:
    with WarpedVRT(
        src_supply,
        crs=TARGET_CRS,
        resolution=TARGET_RES,
        resampling=Resampling.average
    ) as vrt:

        print(f"  Target size: {vrt.height:,} rows × {vrt.width:,} cols")

        # Base target profile (force GTiff later)
        base_target_profile = vrt.profile.copy()
        base_target_profile.update(
            crs=vrt.crs,
            transform=vrt.transform,
            width=vrt.width,
            height=vrt.height
        )

        supply_profile = compressed_gtiff_profile(base_target_profile, dtype="float32", nodata=OUT_NODATA)
        demand_profile = compressed_gtiff_profile(base_target_profile, dtype="float32", nodata=OUT_NODATA)
        flow_profile   = compressed_gtiff_profile(base_target_profile, dtype="float32", nodata=OUT_NODATA)

        # -----------------------------------------------------------------------------
        # Step 4: Write SUPPLY (fast: read/resample via VRT, write tiled GeoTIFF)
        # -----------------------------------------------------------------------------
        print("\n[4/6] Writing SUPPLY raster (tree cover) to 100 m grid (windowed)...")
        with rasterio.open(SUPPLY_RASTER_OUT, "w", **supply_profile) as dst_supply:
            windows = list(dst_supply.block_windows(1))
            nwin = len(windows)
            print(f"  Total windows: {nwin:,}")

            for k, (ji, window) in enumerate(windows, start=1):
                block = vrt.read(1, window=window).astype("float32")

                # Standardise nodata (WarpedVRT may use src nodata; convert to OUT_NODATA)
                if src_supply.nodata is not None:
                    block = np.where(block == src_supply.nodata, OUT_NODATA, block)

                dst_supply.write(block, 1, window=window)

                if k % 50 == 0 or k == nwin:
                    print(f"  Supply: {k:,}/{nwin:,} ({100*k/nwin:.1f}%)")

        # -----------------------------------------------------------------------------
        # Step 5: Write DEMAND by rasterising ONCE across the full grid (fast like QGIS)
        # -----------------------------------------------------------------------------
        print("\n[5/6] Writing DEMAND raster (rasterise once on full grid)...")

        # Reproject polygons to target CRS for rasterisation
        if gdf.crs != vrt.crs:
            gdf_2193 = gdf.to_crs(vrt.crs)
        else:
            gdf_2193 = gdf

        shapes = list(zip(gdf_2193.geometry, gdf_2193[DEMAND_FIELD].astype("float32")))

        # Rasterise once (this is why QGIS feels instant)
        demand_full = rasterize(
            shapes=shapes,
            out_shape=(vrt.height, vrt.width),
            transform=vrt.transform,
            fill=0.0,
            dtype="float32",
            all_touched=False  # generally faster and more defensible for polygon assignment
        )

        with rasterio.open(DEMAND_RASTER_OUT, "w", **demand_profile) as dst_demand:
            dst_demand.write(demand_full, 1)

        # Free RAM early
        del demand_full

        # -----------------------------------------------------------------------------
        # Step 6: Write FLOW = SUPPLY × DEMAND (windowed, low RAM)
        # -----------------------------------------------------------------------------
        print("\n[6/6] Writing FLOW raster = SUPPLY × DEMAND (windowed)...")
        with rasterio.open(SUPPLY_RASTER_OUT) as src_supply_100m, rasterio.open(DEMAND_RASTER_OUT) as src_demand_100m:
            with rasterio.open(FLOW_RASTER_OUT, "w", **flow_profile) as dst_flow:

                windows = list(dst_flow.block_windows(1))
                nwin = len(windows)
                print(f"  Total windows: {nwin:,}")

                for k, (ji, window) in enumerate(windows, start=1):
                    supply_block = src_supply_100m.read(1, window=window).astype("float32")
                    demand_block = src_demand_100m.read(1, window=window).astype("float32")

                    supply_mask = (supply_block == OUT_NODATA)
                    flow_block = supply_block * demand_block
                    flow_block = np.where(supply_mask, OUT_NODATA, flow_block)

                    dst_flow.write(flow_block.astype("float32"), 1, window=window)

                    if k % 50 == 0 or k == nwin:
                        print(f"  Flow: {k:,}/{nwin:,} ({100*k/nwin:.1f}%)")

print("\nDone.")
print(f"Supply raster: {SUPPLY_RASTER_OUT}")
print(f"Demand raster: {DEMAND_RASTER_OUT}")
print(f"Flow raster:   {FLOW_RASTER_OUT}")
print("Note: tc_mean exists in-memory in the GeoDataFrame; set WRITE_TC_MEAN_TO_GPKG=True to persist it.")
