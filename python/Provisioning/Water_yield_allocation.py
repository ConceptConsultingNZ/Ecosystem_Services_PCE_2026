import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

"""
Spatially allocate catchment-level water volume and economic value to a raster grid
using water yield as proportional weights, with optional capping for small catchments.

Outputs (all on the same grid as the water yield raster):
1) water_flow_m3.tif          Float32  Allocated water volume (m³)
2) water_flow_value.tif       Float32  Allocated water value (currency units)
3) water_flow.tif             Int16    Allocated value normalised to 0–100
4) water_supply.tif           Int16    Water yield (supply) normalised to 0–100

Capping rule:
- For catchments where area_ha < 5000 and use_yield_ratio > 2, total_volume_m3 is capped so that
  capped_ratio = total_volume_m3 / yield_sum <= 2. Value is scaled by the same factor.

Nodata propagation:
- Wherever water_yield is nodata, all outputs are nodata.

Compression:
- GeoTIFF outputs use DEFLATE compression, predictor=2, zlevel=9.
"""
# ----------------------------
# Inputs
# ----------------------------
water_yield_tif = r"<PROJECT_DIRECTORY>\Provisioning\Water yield\Output\water_supply_m3.tif"
catchments_gpkg = r"<PROJECT_DIRECTORY>\Provisioning\Water yield\Intermediate\catchments_water_value.gpkg"
catchments_layer = "catchment_water"

out_dir = r"<PROJECT_DIRECTORY>\Provisioning\Water yield\Output"
out_volume_tif = os.path.join(out_dir, "water_flow_m3.tif")
out_value_tif = os.path.join(out_dir, "water_flow_value.tif")
out_normalised_tif = os.path.join(out_dir, "water_flow.tif")
out_supply_norm_tif = os.path.join(out_dir, "water_supply.tif")

VALUE_FIELD = "total_value"
VOLUME_FIELD = "total_volume"

AREA_HA_FIELD = "area_ha"
RATIO_FIELD = "use_yield_ratio"
YIELD_FIELD = "water_yield"

CAP_AREA_HA_MAX = 5000.0
CAP_RATIO_MAX = 5.0

# ----------------------------
# Read water yield raster (weights)
# ----------------------------
with rasterio.open(water_yield_tif) as src:
    w = src.read(1).astype("float64")
    profile = src.profile.copy()
    transform = src.transform
    raster_crs = src.crs
    nodata = src.nodata
    height, width = src.height, src.width

if nodata is not None:
    w_valid = np.isfinite(w) & (w != nodata)
else:
    w_valid = np.isfinite(w)

w_valid &= (w >= 0)
w0 = np.where(w_valid, w, 0.0)

# ----------------------------
# Create normalised SUPPLY layer (0–100) from water yield
# ----------------------------
supply_norm = np.full((height, width), np.nan, dtype="float64")
if np.any(w_valid):
    smin = np.nanmin(w[w_valid])
    smax = np.nanmax(w[w_valid])
    if smax > smin:
        supply_norm[w_valid] = ((w[w_valid] - smin) / (smax - smin)) * 100.0
    else:
        supply_norm[w_valid] = 0.0

# ----------------------------
# Read catchments and align CRS
# ----------------------------
gdf = gpd.read_file(catchments_gpkg, layer=catchments_layer)

if gdf.crs is None:
    raise ValueError("Catchments layer has no CRS. Please define it before running.")
if raster_crs is None:
    raise ValueError("Water yield raster has no CRS. Please define it before running.")
if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)

required_cols = [VALUE_FIELD, VOLUME_FIELD, AREA_HA_FIELD, RATIO_FIELD, YIELD_FIELD]
missing_cols = [c for c in required_cols if c not in gdf.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in catchments layer: {missing_cols}")

gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)].copy()
gdf = gdf.reset_index(drop=True)
gdf["zone_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

gdf["total_volume_m3"] = gdf[VOLUME_FIELD].astype("float64")
gdf["total_value"] = gdf[VALUE_FIELD].astype("float64")

# ----------------------------
# Cap totals for small problematic catchments
# ----------------------------
area_ha = gdf[AREA_HA_FIELD].astype("float64").values
ratio = gdf[RATIO_FIELD].astype("float64").values
yield_sum = gdf[YIELD_FIELD].astype("float64").values

orig_vol = gdf["total_volume_m3"].values
orig_val = gdf["total_value"].values

cap_mask = (
    (area_ha < CAP_AREA_HA_MAX) &
    np.isfinite(ratio) & np.isfinite(yield_sum) &
    (yield_sum > 0) &
    (ratio > CAP_RATIO_MAX)
)

cap_vol = CAP_RATIO_MAX * yield_sum
scale = np.ones_like(orig_vol, dtype="float64")
scale[cap_mask] = np.minimum(1.0, cap_vol[cap_mask] / orig_vol[cap_mask])

gdf.loc[cap_mask, "total_volume_m3"] = orig_vol[cap_mask] * scale[cap_mask]
gdf.loc[cap_mask, "total_value"] = orig_val[cap_mask] * scale[cap_mask]

if np.any(cap_mask):
    print(f"Capped {int(np.sum(cap_mask))} catchments (area_ha < {CAP_AREA_HA_MAX} and {RATIO_FIELD} > {CAP_RATIO_MAX}).")
    print(f"Mean scale factor among capped catchments: {np.mean(scale[cap_mask]):.3f}")

# ----------------------------
# Rasterize catchments
# ----------------------------
shapes = ((geom, int(zid)) for geom, zid in zip(gdf.geometry, gdf["zone_id"]))
zones = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="int32",
    all_touched=False
)

zone_mask = zones > 0
valid_mask = zone_mask & w_valid

# ----------------------------
# Aggregate weights by zone
# ----------------------------
zone_ids = zones[valid_mask]
weights = w0[valid_mask]

n_zones = int(gdf["zone_id"].max())

sum_w = np.bincount(zone_ids, weights=weights, minlength=n_zones + 1).astype("float64")
cnt_w = np.bincount(zone_ids, minlength=n_zones + 1).astype("int64")
zero_sum = (sum_w == 0) & (cnt_w > 0)

tot_vol = np.zeros(n_zones + 1, dtype="float64")
tot_val = np.zeros(n_zones + 1, dtype="float64")
tot_vol[gdf["zone_id"].values] = gdf["total_volume_m3"].values
tot_val[gdf["zone_id"].values] = gdf["total_value"].values

# ----------------------------
# Allocate per-pixel volume and value
# ----------------------------
alloc_vol = np.full((height, width), np.nan, dtype="float64")
alloc_val = np.full((height, width), np.nan, dtype="float64")

weighted_ok = valid_mask & (sum_w[zones] > 0)
alloc_vol[weighted_ok] = (w0[weighted_ok] / sum_w[zones[weighted_ok]]) * tot_vol[zones[weighted_ok]]
alloc_val[weighted_ok] = (w0[weighted_ok] / sum_w[zones[weighted_ok]]) * tot_val[zones[weighted_ok]]

uniform_ok = valid_mask & zero_sum[zones]
alloc_vol[uniform_ok] = tot_vol[zones[uniform_ok]] / cnt_w[zones[uniform_ok]]
alloc_val[uniform_ok] = tot_val[zones[uniform_ok]] / cnt_w[zones[uniform_ok]]

alloc_vol[~w_valid] = np.nan
alloc_val[~w_valid] = np.nan

# ----------------------------
# Normalised FLOW layer (0–100) from alloc_val
# ----------------------------
flow_norm = np.full((height, width), np.nan, dtype="float64")
valid_flow = np.isfinite(alloc_val)

if np.any(valid_flow):
    vmin = np.nanmin(alloc_val[valid_flow])
    vmax = np.nanmax(alloc_val[valid_flow])
    if vmax > vmin:
        flow_norm[valid_flow] = ((alloc_val[valid_flow] - vmin) / (vmax - vmin)) * 100.0
    else:
        flow_norm[valid_flow] = 0.0

flow_norm[~w_valid] = np.nan

# ----------------------------
# Write outputs with compression
# ----------------------------

INT16_MAX = np.iinfo(np.int16).max
INT16_MIN = np.iinfo(np.int16).min

float_profile = profile.copy()
float_profile.update(
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="DEFLATE",
    predictor=2,
    zlevel=9,
    tiled=True,
    blockxsize=256,
    blockysize=256
)

int_profile = profile.copy()
int_profile.update(
    dtype="int16",
    count=1,
    nodata=-32768,
    compress="DEFLATE",
    predictor=2,
    zlevel=9,
    tiled=True,
    blockxsize=256,
    blockysize=256
)

def write_float(path, arr):
    out = np.where(np.isfinite(arr), arr, float_profile["nodata"]).astype("float32")
    with rasterio.open(path, "w", **float_profile) as dst:
        dst.write(out, 1)

def write_int(path, arr):
    out = np.where(np.isfinite(arr), np.round(arr), int_profile["nodata"]).astype("int16")
    with rasterio.open(path, "w", **int_profile) as dst:
        dst.write(out, 1)

# Always write volume as float
write_float(out_volume_tif, alloc_vol)

# Conditionally write value as Int16 if safe
valid_val = np.isfinite(alloc_val)
if np.any(valid_val):
    vmax_val = np.nanmax(alloc_val[valid_val])
    vmin_val = np.nanmin(alloc_val[valid_val])
else:
    vmax_val = 0
    vmin_val = 0

if vmin_val >= INT16_MIN and vmax_val <= INT16_MAX:
    print("Writing water_flow_value as Int16")
    write_int(out_value_tif, alloc_val)
else:
    print("Writing water_flow_value as Float32 (exceeds Int16 range)")
    write_float(out_value_tif, alloc_val)

# Normalised layers are always Int16
write_int(out_normalised_tif, flow_norm)
write_int(out_supply_norm_tif, supply_norm)


# ----------------------------
# Sanity checks vs (possibly capped) totals
# ----------------------------
finite = np.isfinite(alloc_vol) & (zones > 0)
chk_zone = zones[finite]

sum_alloc_vol = np.bincount(chk_zone, weights=alloc_vol[finite], minlength=n_zones + 1)
sum_alloc_val = np.bincount(chk_zone, weights=alloc_val[finite], minlength=n_zones + 1)

vol_diff = sum_alloc_vol - tot_vol
val_diff = sum_alloc_val - tot_val

print("Wrote:")
print("  Volume          :", out_volume_tif)
print("  Value           :", out_value_tif)
print("  Flow normalised :", out_normalised_tif)
print("  Supply (norm)   :", out_supply_norm_tif)
print("")
print("Sanity check vs (possibly capped) catchment totals:")
print("  Max abs volume diff (m3):", np.nanmax(np.abs(vol_diff[1:])))
print("  Max abs value  diff     :", np.nanmax(np.abs(val_diff[1:])))
