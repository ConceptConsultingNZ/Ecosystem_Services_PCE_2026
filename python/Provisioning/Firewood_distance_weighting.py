"""
Firewood provisioning model with island-constrained proximity weighting and capped allocation.

This script estimates the spatial allocation of firewood supply to meet demand
across New Zealand using a distance-decay approach. Tree cover is used as a proxy
for potential supply, while demand is represented as delivered energy demand per hectare.
A burner efficiency factor is applied to convert delivered energy demand to useful heat.
Demand is then allocated only to tree supply located on the same island feature as defined
by an input islands polygon layer.

Key steps:
1. Load demand (GJ/ha) and tree cover (%) rasters on a common grid.
2. Apply a burner efficiency factor to convert delivered energy demand to useful heat.
3. Rasterise the islands polygon layer using a unique ID for each feature.
4. For each island feature:
   - calculate a proximity weight surface from demand on that island only
   - define a supply attractiveness surface from tree cover and proximity
   - allocate island demand across supply cells on that island only,
     subject to per-cell capacity constraints
5. Output rasters for proximity weights, weighted supply index, and allocated flow (GJ/cell).

Assumptions:
- Tree cover represents potential firewood supply.
- Demand influences nearby supply via an exponential distance-decay function.
- Allocation is constrained to the same island feature.
- Maximum sustainable harvest is capped (e.g. 1 m3/ha/year).
- Energy density of firewood is fixed (e.g. 8 GJ/m3).
- Burner efficiency (e.g. 0.7) converts delivered energy to useful heat.

Outputs:
- Proximity weight raster (0–1)
- Weighted supply index raster
- Allocated firewood flow raster (GJ per cell per year)

Dependencies:
- numpy, rasterio, scipy, geopandas

Notes:
- Allocation may not fully meet demand on an island if total capacity on that island
  is below total demand.
- Distance-decay parameters (lambda, truncation radius) strongly influence spatial results.
- The islands layer should contain one polygon feature per island or island group to be
  treated as a separate allocation region.
"""

import os
import time

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window
from scipy.signal import fftconvolve


# -----------------------------
# INPUT FILES
# -----------------------------
ISLANDS_FILE = r"D:\Data\LINZ\lds-nz-coastlines-and-islands-polygons-topo-150k\nz-coastlines-and-islands-polygons-topo-150k.gpkg"
ISLANDS_LAYER = "nz_coastlines_and_islands_polygons_topo_150k" # required only if the GeoPackage has multiple layers
ISLAND_ID_FIELD = "feature_id"  # e.g. "island_id"; use None to assign sequential IDs from feature order

TREECOVER_TIF = r"<PROJECT_DIRECTORY>\Provisioning\Firewood\Output\firewood_supply.tif"   # 0-100
DEMAND_TIF = r"<PROJECT_DIRECTORY>\Provisioning\Firewood\Output\firewood_demand_GJ.tif"   # delivered GJ per hectare

OUT_DIR = r"<PROJECT_DIRECTORY>\Provisioning\Firewood\Intermediate"
OUT_WEIGHT_TIF = os.path.join(OUT_DIR, "firewood_proximity_weight_nz.tif")
OUT_WEIGHTED_SUPPLY_TIF = os.path.join(OUT_DIR, "firewood_weighted_supply_index_nz.tif")
OUT_ALLOC_FLOW_TIF = os.path.join(OUT_DIR, "allocated_flow_capped_gj_per_cell.tif")


# -----------------------------
# PARAMETERS
# -----------------------------
# Half of influence within 30 km implies lambda ≈ 17.9 km for 2D exponential kernel
LAMBDA_KM = 17.9
R_MAX_KM = 90.0  # truncate ~5*lambda

# Cap: max 1 m3 per hectare per year, scaled by treecover fraction
CAP_M3_PER_HA = 1.0

# Energy density
GJ_PER_M3 = 8.0

# Burner efficiency to convert delivered energy demand to useful heat
BURNER_EFFICIENCY = 0.7

NODATA_OUT = -9999.0
MAX_ALLOC_ITERS = 100


# -----------------------------
# HELPERS
# -----------------------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_vector(path: str, layer: str | None = None) -> gpd.GeoDataFrame:
    """
    Read a vector dataset. Supports shapefiles, GeoPackages, GeoJSON, etc.
    If a GeoPackage has multiple layers, pass the required layer name.
    """
    if layer is None:
        return gpd.read_file(path)
    return gpd.read_file(path, layer=layer)


def rasterize_feature_ids(
    vector_path: str,
    out_shape,
    out_transform,
    out_crs,
    id_field: str | None = None,
    layer: str | None = None
) -> np.ndarray:
    """
    Rasterise polygon features using a unique integer ID per feature.

    If id_field is None, sequential IDs starting from 1 are assigned based on feature order.
    Background pixels are 0.

    Parameters
    ----------
    vector_path : str
        Path to polygon dataset (.shp, .gpkg, etc.).
    out_shape : tuple[int, int]
        Output raster shape.
    out_transform : affine.Affine
        Output raster transform.
    out_crs : any
        Output raster CRS.
    id_field : str or None
        Field containing unique positive integer IDs. If None, IDs are assigned sequentially.
    layer : str or None
        Layer name for multi-layer sources such as GeoPackages.
    """
    log(f"Reading islands layer: {vector_path}" + (f" | layer={layer}" if layer else ""))
    gdf = read_vector(vector_path, layer=layer)

    if gdf.empty:
        raise ValueError("Islands layer contains no features.")
    if gdf.crs is None:
        raise ValueError(f"{vector_path} has no CRS. Assign CRS before use.")
    if gdf.crs != out_crs:
        log("Reprojecting islands layer to raster CRS")
        gdf = gdf.to_crs(out_crs)

    gdf = gdf.loc[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("Islands layer has no valid geometries after filtering null/empty features.")

    if id_field is None:
        gdf = gdf.reset_index(drop=True)
        gdf["island_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
        id_field = "island_id"
    else:
        if id_field not in gdf.columns:
            raise ValueError(f"Field '{id_field}' not found in islands layer.")
        if gdf[id_field].isnull().any():
            raise ValueError(f"Field '{id_field}' contains null values.")
        if not np.issubdtype(gdf[id_field].dtype, np.integer):
            raise ValueError(f"Field '{id_field}' must contain integer values.")
        if (gdf[id_field] <= 0).any():
            raise ValueError(f"Field '{id_field}' must contain positive integers only.")

    if gdf[id_field].duplicated().any():
        raise ValueError(f"Field '{id_field}' contains duplicate values. IDs must be unique.")

    geoms = [(geom, int(val)) for geom, val in zip(gdf.geometry, gdf[id_field])]

    island_ids = features.rasterize(
        geoms,
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        dtype="int32",
        all_touched=False
    )

    unique_ids = np.unique(island_ids)
    unique_ids = unique_ids[unique_ids != 0]
    log(f"Rasterised {len(unique_ids):,} island features with pixels on the analysis grid")

    return island_ids


def tight_window(mask: np.ndarray) -> Window:
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("Mask has no True pixels; cannot build a tight window.")
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    return Window(
        col_off=int(c0),
        row_off=int(r0),
        width=int(c1 - c0 + 1),
        height=int(r1 - r0 + 1)
    )


def make_exp_kernel(pixel_size_m: float, lambda_m: float, r_max_m: float) -> np.ndarray:
    """
    Exponential distance-decay kernel w(d)=exp(-d/lambda), truncated at r_max.
    Unnormalised kernel: influence scales with nearby demand magnitude.
    """
    radius_px = int(np.ceil(r_max_m / pixel_size_m))
    log(f"Building kernel: radius {radius_px} px -> size {2 * radius_px + 1} x {2 * radius_px + 1}")
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    dist_m = np.sqrt(x * x + y * y) * pixel_size_m
    k = np.exp(-dist_m / lambda_m)
    k[dist_m > r_max_m] = 0.0
    log(f"Kernel nonzero cells: {int((k > 0).sum()):,}")
    return k.astype(np.float32)


def proximity_weight_from_demand(demand_sub: np.ndarray, kernel: np.ndarray, label: str) -> np.ndarray:
    """
    influence = demand ⊗ kernel, then scaled to 0–1 using p99 to reduce outlier dominance.
    demand_sub should be float array with NaN for outside-region / nodata.
    """
    log(f"{label}: Convolving demand with kernel (FFT)")
    demand0 = np.nan_to_num(demand_sub, nan=0.0).astype(np.float32)
    influence = fftconvolve(demand0, kernel, mode="same").astype(np.float32)

    if not np.isfinite(influence).any():
        log(f"{label}: Influence surface degenerate; returning zeros")
        return np.zeros_like(influence, dtype=np.float32)

    p99 = np.nanpercentile(influence, 99)
    den = p99 if p99 > 0 else np.nanmax(influence)
    if not np.isfinite(den) or den <= 0:
        log(f"{label}: Influence surface degenerate; returning zeros")
        return np.zeros_like(influence, dtype=np.float32)

    w = np.clip(influence / den, 0.0, 1.0).astype(np.float32)
    log(
        f"{label}: Weight stats (min/mean/max) = "
        f"{float(np.nanmin(w)):.4f} / {float(np.nanmean(w)):.4f} / {float(np.nanmax(w)):.4f}"
    )
    return w


def write_raster(path: str, arr: np.ndarray, profile: dict, nodata: float):
    prof = profile.copy()
    prof.update(dtype="float32", nodata=nodata, compress="deflate")
    out = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    log(f"Writing: {path}")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def allocate_with_capacity(
    total_flow_gj: float,
    attractiveness: np.ndarray,
    capacity_gj: np.ndarray,
    mask: np.ndarray,
    max_iter: int = 50
) -> np.ndarray:
    """
    Allocate total_flow_gj across cells proportional to attractiveness, subject to per-cell capacity caps.
    Iteratively redistributes overflow from capped cells.

    attractiveness: e.g., tree_frac * proximity_weight
    capacity_gj: per-cell cap in GJ/year
    mask: valid supply cells
    """
    flow = np.zeros_like(attractiveness, dtype=np.float32)

    valid = (
        mask
        & np.isfinite(attractiveness)
        & (attractiveness > 0)
        & np.isfinite(capacity_gj)
        & (capacity_gj > 0)
    )
    remaining = float(total_flow_gj)

    log(f"Allocation: valid supply cells = {int(valid.sum()):,}")
    log(f"Allocation: total_flow_gj target = {remaining:,.2f}")

    for it in range(1, max_iter + 1):
        if remaining <= 1e-6:
            log(f"Allocation converged at iteration {it - 1}.")
            break

        A = np.where(valid, attractiveness, 0.0)
        A_sum = float(np.sum(A))
        if A_sum <= 0:
            log("Allocation stopped: no remaining attractiveness.")
            break

        add = remaining * (A / A_sum)

        cap_left = np.where(valid, capacity_gj - flow, 0.0)
        accepted = np.minimum(add, cap_left)

        flow += accepted.astype(np.float32)

        overflow = float(np.sum(add - accepted))
        newly_capped = int(np.sum(valid & ((capacity_gj - flow) <= 1e-6)))

        allocated_total = float(np.sum(flow))
        log(
            f"Iter {it:03d}: allocated={allocated_total:,.2f} GJ, "
            f"overflow={overflow:,.2f} GJ, newly_capped={newly_capped:,}"
        )

        remaining = overflow
        valid = valid & ((capacity_gj - flow) > 1e-6)

        if overflow <= 1e-6:
            log(f"Allocation converged at iteration {it}.")
            break

    if remaining > 1e-3:
        log(f"WARNING: Unallocated flow remaining after {max_iter} iterations: {remaining:,.2f} GJ")
        log("This indicates total capacity < total demand (given your cap settings).")

    return flow


# -----------------------------
# MAIN
# -----------------------------
log("Starting proximity weighting + capped allocation (island constrained)")

# Read demand as the reference grid
log(f"Reading demand raster: {DEMAND_TIF}")
with rasterio.open(DEMAND_TIF) as dem_src:
    demand = dem_src.read(1).astype(np.float32)
    dem_profile = dem_src.profile
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs
    dem_nodata = dem_src.nodata
    shape = demand.shape

log(f"Demand raster shape: {shape[0]} x {shape[1]}")
if dem_nodata is not None:
    demand = np.where(demand == dem_nodata, np.nan, demand)

# Apply burner efficiency scaling: delivered energy -> useful heat
demand = demand * BURNER_EFFICIENCY

# Read treecover and ensure same grid
log(f"Reading treecover raster: {TREECOVER_TIF}")
with rasterio.open(TREECOVER_TIF) as tree_src:
    tree = tree_src.read(1).astype(np.float32)
    if tree_src.transform != dem_transform or tree_src.crs != dem_crs or tree.shape != shape:
        raise ValueError("Treecover raster does not match demand raster grid (shape/transform/CRS). Resample first.")
    tree_nodata = tree_src.nodata

if tree_nodata is not None:
    tree = np.where(tree == tree_nodata, np.nan, tree)

# Pixel size and cell area
pixel_size_m = abs(dem_transform.a)
cell_area_ha = (abs(dem_transform.a) * abs(dem_transform.e)) / 10000.0
log(f"Pixel size: {pixel_size_m:.2f} m; Cell area: {cell_area_ha:.4f} ha")

# Rasterise islands to unique IDs on the demand grid
island_id_raster = rasterize_feature_ids(
    ISLANDS_FILE,
    shape,
    dem_transform,
    dem_crs,
    id_field=ISLAND_ID_FIELD,
    layer=ISLANDS_LAYER
)
island_ids = np.unique(island_id_raster)
island_ids = island_ids[island_ids != 0]

if island_ids.size == 0:
    raise ValueError("No island features intersect the raster grid.")

# Kernel (in metres)
log(f"Kernel parameters: lambda={LAMBDA_KM} km, r_max={R_MAX_KM} km")
kernel = make_exp_kernel(
    pixel_size_m=pixel_size_m,
    lambda_m=LAMBDA_KM * 1000.0,
    r_max_m=R_MAX_KM * 1000.0
)

# Precompute tree fraction and capacity
tree_frac = np.clip(tree / 100.0, 0.0, 1.0).astype(np.float32)
capacity_gj = (CAP_M3_PER_HA * cell_area_ha * tree_frac * GJ_PER_M3).astype(np.float32)

# National outputs
weight_nz = np.full(shape, np.nan, dtype=np.float32)
weighted_supply = np.full(shape, np.nan, dtype=np.float32)
flow_alloc_capped = np.zeros(shape, dtype=np.float32)

# Optional total checks
total_demand_nz = float(np.nansum(demand) * cell_area_ha)
log(f"Total national demand after efficiency scaling = {total_demand_nz:,.2f} GJ")

# Process each island separately
for i, island_id in enumerate(island_ids, start=1):
    label = f"Island {int(island_id)} ({i}/{len(island_ids)})"
    island_mask = island_id_raster == island_id

    if not island_mask.any():
        continue

    log(f"{label}: computing tight window")
    win = tight_window(island_mask)
    r0, c0 = int(win.row_off), int(win.col_off)
    r1, c1 = r0 + int(win.height), c0 + int(win.width)
    log(f"{label}: window rows {r0}-{r1 - 1}, cols {c0}-{c1 - 1} ({int(win.height)} x {int(win.width)})")

    island_submask = island_mask[r0:r1, c0:c1]

    demand_sub = demand[r0:r1, c0:c1].copy()
    demand_sub[~island_submask] = np.nan

    # Skip if no finite demand and no finite tree on this island
    island_demand_gj = float(np.nansum(demand_sub) * cell_area_ha)
    island_tree_sub = tree_frac[r0:r1, c0:c1]
    has_supply = bool(np.any(island_submask & np.isfinite(island_tree_sub) & (island_tree_sub > 0)))

    if island_demand_gj <= 0 and not has_supply:
        log(f"{label}: no demand and no supply, skipping")
        continue

    # Proximity weight from demand on this island only
    w_sub = proximity_weight_from_demand(demand_sub, kernel, label=label)

    target = weight_nz[r0:r1, c0:c1]
    target[island_submask] = w_sub[island_submask]
    weight_nz[r0:r1, c0:c1] = target

    # Weighted supply index on this island
    weighted_sub = (tree_frac[r0:r1, c0:c1] * w_sub).astype(np.float32)
    target = weighted_supply[r0:r1, c0:c1]
    target[island_submask] = weighted_sub[island_submask]
    weighted_supply[r0:r1, c0:c1] = target

    # Allocation constrained to this island only
    attractiveness_sub = weighted_sub
    capacity_sub = capacity_gj[r0:r1, c0:c1]
    supply_mask_sub = (
        island_submask
        & np.isfinite(tree_frac[r0:r1, c0:c1])
        & (tree_frac[r0:r1, c0:c1] > 0)
        & np.isfinite(w_sub)
    )

    island_capacity_gj = float(np.nansum(np.where(supply_mask_sub, capacity_sub, 0.0)))
    log(f"{label}: total demand = {island_demand_gj:,.2f} GJ")
    log(f"{label}: total capacity under cap = {island_capacity_gj:,.2f} GJ")

    if island_capacity_gj + 1e-6 < island_demand_gj:
        log(f"{label}: WARNING: capacity is less than demand. Some flow will remain unallocated.")

    if island_demand_gj > 0 and island_capacity_gj > 0:
        flow_sub = allocate_with_capacity(
            total_flow_gj=island_demand_gj,
            attractiveness=attractiveness_sub,
            capacity_gj=capacity_sub,
            mask=supply_mask_sub,
            max_iter=MAX_ALLOC_ITERS
        )
        flow_alloc_capped[r0:r1, c0:c1] += np.where(island_submask, flow_sub, 0.0).astype(np.float32)
    else:
        log(f"{label}: skipping allocation because demand or capacity is zero")

# Write outputs
write_raster(OUT_WEIGHT_TIF, weight_nz, dem_profile, NODATA_OUT)
write_raster(OUT_WEIGHTED_SUPPLY_TIF, weighted_supply, dem_profile, NODATA_OUT)
write_raster(OUT_ALLOC_FLOW_TIF, flow_alloc_capped, dem_profile, NODATA_OUT)

log(f"Allocation complete. Total allocated = {float(np.nansum(flow_alloc_capped)):,.2f} GJ")
log("All done.")