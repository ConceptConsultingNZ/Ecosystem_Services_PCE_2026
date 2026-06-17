import os
import time
import uuid
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
"""
Local Climate Regulation – Cooling Energy Avoidance Model

This script estimates the annual value of local climate regulation provided by
existing urban tree canopy through avoided residential space cooling energy.

Conceptual framework
--------------------
Observed national cooling energy demand (EEUD 2023 total) already reflects
current tree canopy conditions. Therefore, avoided energy must be estimated
relative to a counterfactual “no-canopy” scenario.

Let:
    r = MAX_REDUCTION * (treecover / 100) 

where:
    MAX_REDUCTION = maximum proportional reduction in cooling demand under
                    100% canopy and 100% impervious cover (default = 0.3)

Observed cooling energy in each cell (E_obs) is treated as:
    E_obs = (1 - r) * E_no_canopy

Therefore, avoided energy attributable to existing canopy is:
    E_saved = E_no_canopy - E_obs
            = E_obs * (r / (1 - r))

This avoids double counting because it treats EEUD total cooling energy as
post-canopy observed energy.

Spatial allocation
------------------
The national cooling energy total (TOTAL_SPACE_COOLING_ENERGY_GWH) is allocated
across 100 m grid cells proportional to:

    households * cooling_degree_days * impervious

This assumes:
- More households → higher cooling demand
- Higher CDD → greater climatic cooling requirement
- Higher imperviousness → greater urban heat amplification

Outputs
-------
FLOW (local_climate_flow.tif)
    Annual kWh avoided per 100 m cell due to existing canopy.

FLOW_VALUE (local_climate_flow_value.tif)
    Annual monetary value ($) of avoided cooling energy:
        FLOW * ELECTRICITY_PRICE

FLOW_SCORE (local_climate_flow_score.tif)
    Min–max normalised (0–100) score of FLOW_VALUE across the study area.

Key assumptions
---------------
- Cooling reduction scales linearly with canopy and impervious cover.
- Maximum reduction under full canopy and full imperviousness is 30%.
- Only cooling energy is modelled (no winter heating disbenefits).
- EEUD national cooling total is treated as observed post-canopy energy.

Units
-----
- Energy in kWh
- Value in NZD
- Grid resolution matches TREECOVER raster (100 m cells)

"""

# ============================================================
# ---------------------- USER CONSTANTS ----------------------
# ============================================================

WORK_DIR = r"<PROJECT_DIRECTORY>\Regulating\Local climate\Intermediate"
OUT_DIR = r"<PROJECT_DIRECTORY>\Regulating\Local climate\Output"

TREECOVER = r"<PROJECT_DIRECTORY>\Regulating\Local climate\Output\local_climate_supply.tif"
HOUSEHOLDS = os.path.join(WORK_DIR, "households.tif")
IMPERVIOUS = os.path.join(WORK_DIR, "SA1_impervious.tif")
COOLING_DEGREE_DAYS = os.path.join(WORK_DIR, "cooling_degree_days.tif")

FLOW = os.path.join(OUT_DIR, "local_climate_flow_v2.tif")
FLOW_VALUE = os.path.join(OUT_DIR, "local_climate_flow_value_v3.tif")
#FLOW_SCORE = os.path.join(OUT_DIR, "local_climate_flow_score.tif")

DELTA_IMPERVIOUS = 0.10  # Additive driveway adjustment to account for surfaces that aren't roads or buildings (e.g. 0.05–0.15 sensitivity)

MAX_REDUCTION = 0.3
ELECTRICITY_PRICE = 0.3
TOTAL_SPACE_COOLING_ENERGY_GWH = 1028
NODATA_OUT = -9999.0

USE_IMPERVIOUS_IN_ALLOCATION = True
R_CAP = 0.95


# ============================================================
# ---------------------- LOGGING SETUP -----------------------
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================

def _read_as_ref(path, ref_transform, ref_crs, ref_shape, resampling):
    with rasterio.open(path) as src:
        # Check if grid already matches reference
        same_crs = src.crs == ref_crs
        same_transform = src.transform == ref_transform
        same_shape = (src.height, src.width) == ref_shape

        if same_crs and same_transform and same_shape:
            logger.info(f"Reading (no reprojection needed): {os.path.basename(path)}")
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            return data

        logger.info(f"Reprojecting: {os.path.basename(path)}")

        src_data = src.read(1).astype(np.float32)
        out = np.full(ref_shape, np.nan, dtype=np.float32)

        reproject(
            source=src_data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

        return out


def main():
    start_time = time.time()
    logger.info("Starting local climate regulation valuation")
    logger.info(f"Total national cooling energy (GWh): {TOTAL_SPACE_COOLING_ENERGY_GWH}")

    os.makedirs(os.path.dirname(FLOW), exist_ok=True)

    # --------------------------------------------------------
    # Reference grid
    # --------------------------------------------------------
    logger.info("Loading treecover reference raster")

    with rasterio.open(TREECOVER) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

        tree = ref.read(1).astype(np.float32)
        tree_nodata = ref.nodata

    logger.info(f"Grid size: {ref.width} x {ref.height}")

    # --------------------------------------------------------
    # Load drivers
    # --------------------------------------------------------
    hh = _read_as_ref(HOUSEHOLDS, ref_transform, ref_crs, ref_shape, Resampling.nearest)
    imp = _read_as_ref(IMPERVIOUS, ref_transform, ref_crs, ref_shape, Resampling.bilinear)
    cdd = _read_as_ref(COOLING_DEGREE_DAYS, ref_transform, ref_crs, ref_shape, Resampling.bilinear)

    logger.info("Cleaning and normalising inputs")

    if tree_nodata is not None:
        tree = np.where(tree == tree_nodata, np.nan, tree)

    tree = np.clip(tree, 0, 100)

    imp_max = np.nanmax(imp)
    if imp_max > 1.5:
        imp = imp / 100.0
    imp = np.clip(imp, 0, 1)

    # --------------------------------------------------------
    # Driveway adjustment (applied only where households > 0)
    # --------------------------------------------------------
    logger.info(f"Applying impervious adjustment (delta = {DELTA_IMPERVIOUS})")
    imp_adj = imp.copy()

    has_households = np.isfinite(hh) & (hh > 0)
    imp_adj[has_households] = np.clip(
        imp_adj[has_households] + DELTA_IMPERVIOUS,
        0.0,
        1.0
    )

    hh = np.where(np.isfinite(hh), np.maximum(hh, 0), np.nan)
    cdd = np.where(np.isfinite(cdd), np.maximum(cdd, 0), np.nan)

    valid = np.isfinite(tree) & np.isfinite(hh) & np.isfinite(imp) & np.isfinite(cdd)
    logger.info(f"Valid cells: {np.sum(valid):,}")

    # --------------------------------------------------------
    # Allocate observed cooling energy
    # --------------------------------------------------------
    total_energy_kwh = TOTAL_SPACE_COOLING_ENERGY_GWH * 1_000_000.0

    logger.info("Allocating observed cooling energy spatially")

    weight = np.zeros(ref_shape, dtype=np.float32)

    if USE_IMPERVIOUS_IN_ALLOCATION:
        weight[valid] = hh[valid] * cdd[valid] * imp_adj[valid]
    else:
        weight[valid] = hh[valid] * cdd[valid]

    weight_sum = float(np.nansum(weight))
    logger.info(f"Weight sum: {weight_sum:,.2f}")

    if weight_sum <= 0:
        raise ValueError("Allocation weights sum to zero.")

    e_obs = np.zeros(ref_shape, dtype=np.float32)
    e_obs[valid] = total_energy_kwh * (weight[valid] / weight_sum)

    logger.info(f"Total allocated energy (kWh): {np.nansum(e_obs):,.0f}")

    # --------------------------------------------------------
    # Existing canopy reduction fraction
    # --------------------------------------------------------
    logger.info("Calculating reduction fractions")

    r = np.zeros(ref_shape, dtype=np.float32)

    # Previous approach - make the cooling dependent on impervious area
    # r[valid] = MAX_REDUCTION * (tree[valid] / 100.0) * imp[valid]

    #New approach - the cooling is only dependent on treecover
    r[valid] = MAX_REDUCTION * (tree[valid] / 100.0)

    r = np.clip(r, 0.0, MAX_REDUCTION)
    r_safe = np.clip(r, 0.0, R_CAP)

    # --------------------------------------------------------
    # Flow = avoided kWh due to existing canopy
    # --------------------------------------------------------
    logger.info("Calculating avoided cooling energy")

    flow_kwh = np.full(ref_shape, np.nan, dtype=np.float32)

    denom = (1.0 - r_safe)
    ok = valid & (denom > 0)

    flow_kwh[ok] = e_obs[ok] * (r_safe[ok] / denom[ok])

    logger.info(f"Total avoided energy (kWh): {np.nansum(flow_kwh):,.0f}")

    flow_value = np.full(ref_shape, np.nan, dtype=np.float32)
    flow_value[ok] = flow_kwh[ok] * ELECTRICITY_PRICE

    logger.info(f"Total avoided value ($): {np.nansum(flow_value):,.0f}")

    # --------------------------------------------------------
    # Normalised score (not used)
    # --------------------------------------------------------
   # logger.info("Normalising to 0–100 score")

    #score = np.full(ref_shape, np.nan, dtype=np.float32)
    #if np.any(ok):
    #    v = flow_value[ok]
    #    vmin = float(np.nanmin(v))
    #    vmax = float(np.nanmax(v))
    #    if vmax > vmin:
    #        score[ok] = (v - vmin) / (vmax - vmin) * 100.0
    #    else:
    #        score[ok] = 0.0

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    logger.info("Writing output rasters")

    out_profile = ref_profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=NODATA_OUT,
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    def write_raster(path, arr):
        out = np.where(np.isfinite(arr), arr, NODATA_OUT).astype(np.float32)

        out_dir = os.path.dirname(path)
        os.makedirs(out_dir, exist_ok=True)

        # Write to a unique temp file first
        tmp_path = os.path.join(out_dir, f".tmp_{uuid.uuid4().hex}_{os.path.basename(path)}")

        # If the destination exists, try to remove it first (handles corrupted TIFFs)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logger.warning(f"Could not remove existing file (may be locked): {path} ({e})")

        with rasterio.open(tmp_path, "w", **out_profile) as dst:
            dst.write(out, 1)

        # Replace destination atomically where possible
        try:
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning(f"os.replace failed, trying fallback copy: {e}")
            # Fallback: remove and rename
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)

    write_raster(FLOW, flow_kwh)
    write_raster(FLOW_VALUE, flow_value)
    #write_raster(FLOW_SCORE, score)

    total_time = time.time() - start_time
    logger.info("Finished successfully")
    logger.info(f"Total runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()