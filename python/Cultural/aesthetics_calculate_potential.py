"""
Compute a national-scale Aesthetic Supply raster representing the scenic
experience a person would have if they stood in each cell.

Components:
1) Structural score (convolved):
   S = Base + (TreeCover/100 * TreeModifier), clipped [0, 100]
   Convolved with Gaussian kernel because viewers see surrounding landcover.

2) Topographic drama (convolved):
   T = min(Slope/30, 1) * 100
   Convolved with Gaussian kernel because viewers see surrounding terrain.

3) Complexity (not convolved -- already neighbourhood-representative):
   SHDI_norm = 100 * sh / shdi_max, clipped [0, 100]
   C = 0.5 * SHDI_norm + 0.5 * PatchScore

4) Water proximity (not convolved -- already a distance-based measure):
   W = 100 * exp(-WaterDist_m / WaterDecay_m), clipped [0, 100]

Overall aesthetic supply:
   A = wS*S_conv + wT*T_conv + wC*C + wW*W, clipped [0, 100]

Gaussian kernel:
   sigma=300m, truncated at 1000m. Represents the spatial scale over which
   surrounding landcover and terrain contribute to the viewer's experience.

Outputs:
   aesthetic_supply.tif       (0-100 index, float32)

Notes:
   - Convolution runs in-memory on the full raster (~600MB per array for NZ
     at 100m resolution). Ensure sufficient RAM (~4GB free recommended).
   - Nodata cells are excluded from the convolution denominator so border
     and ocean cells do not dilute adjacent land cell scores.
   - Slope nodata is treated as 0 degrees (flat).
   - Water distance nodata is treated as no bonus (W=0).
   - All rasters must be aligned (same CRS, transform, width/height).

Dependencies:
   rasterio, numpy, pandas, scipy
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import convolve


# ----------------------------
# Paths
# ----------------------------
INPUT_DIR  = r"<PROJECT_DIRECTORY>\Cultural\Aesthetics\Intermediate"
OUTPUT_DIR = r"<PROJECT_DIRECTORY>\Cultural\Aesthetics\Intermediate"

LANDCOVER_RASTER     = r"<PROJECT_DIRECTORY>\Common\lcdb6_expanded.tif"
TREECOVER_RASTER     = r"<PROJECT_DIRECTORY>\Common\Treecover.tif"
SHDI_RASTER          = os.path.join(INPUT_DIR, "shdi_100m.tif")
PATCHSCORE_RASTER    = os.path.join(INPUT_DIR, "pd_score_100m.tif")
SLOPE_DEG_RASTER     = os.path.join(INPUT_DIR, "slope_degrees.tif")
WATERDIST_RASTER     = os.path.join(INPUT_DIR, "water_proximity.tif")
LANDCOVER_LOOKUP_CSV = os.path.join(INPUT_DIR, "landcover_scenic_lookup.csv")

OUT_AESTHETIC_TIF    = os.path.join(OUTPUT_DIR, "aesthetic_supply.tif")


# ----------------------------
# Model settings
# ----------------------------
PIXEL_SIZE_M   = 100.0   # metres per pixel

# Gaussian kernel
KERNEL_SIGMA_M  = 300.0  # standard deviation (metres)
KERNEL_RADIUS_M = 1000.0 # truncation radius (metres)

# Topographic drama
SLOPE_REF_DEG  = 30.0

# Water proximity decay
WATER_DECAY_M  = 1000.0

# SHDI normalisation
SHDI_FIXED_MAX = 2.3108  # set to None to calculate from data

# Component weights (must sum to 1.0)
W_STRUCTURE  = 0.40
W_TOPO       = 0.20
W_COMPLEXITY = 0.25
W_WATER      = 0.15

OUT_NODATA   = -9999.0
LOG_LEVEL    = logging.INFO
PROGRESS_EVERY_N_BLOCKS = 500


# ----------------------------
# Helpers
# ----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def _is_nan_nodata(nodata_val: Optional[float]) -> bool:
    return nodata_val is not None and isinstance(nodata_val, float) and np.isnan(nodata_val)


def mask_nodata(arr: np.ndarray, nodata_val: Optional[float]) -> np.ndarray:
    if nodata_val is None:
        return np.zeros(arr.shape, dtype=bool)
    if _is_nan_nodata(nodata_val):
        return np.isnan(arr)
    return arr == nodata_val


def ensure_same_grid(
    ref: rasterio.io.DatasetReader,
    other: rasterio.io.DatasetReader,
    name: str
) -> None:
    if ref.crs != other.crs:
        raise ValueError(f"CRS mismatch for {name}: {ref.crs} vs {other.crs}")
    if ref.transform != other.transform:
        raise ValueError(f"Transform mismatch for {name}. Rasters must be aligned.")
    if ref.width != other.width or ref.height != other.height:
        raise ValueError(
            f"Shape mismatch for {name}: "
            f"{ref.width}x{ref.height} vs {other.width}x{other.height}"
        )


def make_gaussian_kernel(sigma_m: float, radius_m: float, pixel_m: float) -> np.ndarray:
    """
    Build a normalised 2-D Gaussian kernel.
    Used rather than exponential decay because aesthetic experience reflects
    passive visual exposure within a spatial neighbourhood rather than a
    travel-cost process.
    """
    sigma_px  = sigma_m  / pixel_m
    radius_px = int(np.ceil(radius_m / pixel_m))

    y, x  = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    dist2 = (x ** 2 + y ** 2).astype(np.float64)
    kernel = np.exp(-dist2 / (2.0 * sigma_px ** 2))
    kernel[np.sqrt(dist2) > radius_px] = 0.0
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def gaussian_convolve(data: np.ndarray, invalid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve data with kernel, excluding nodata cells from the denominator
    so that ocean/border cells do not dilute adjacent land cell scores.
    """
    filled = data.copy()
    filled[invalid] = 0.0
    valid_weight = (~invalid).astype(np.float32)

    conv   = convolve(filled,        kernel, mode="constant", cval=0.0)
    weight = convolve(valid_weight,  kernel, mode="constant", cval=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight > 1e-6, conv / weight, 0.0)

    return result.astype(np.float32)


def load_landcover_lookup(csv_path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    df   = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    class_col = cols.get("class") or cols.get("class_2023") or cols.get("lc_class") or cols.get("code")
    base_col  = cols.get("base")  or cols.get("base_score")  or cols.get("scenic_base")
    mod_col   = cols.get("tree_modifier") or cols.get("treemodifier") or cols.get("modifier") or cols.get("tree_mod")

    if not class_col or not base_col or not mod_col:
        raise ValueError(f"CSV must contain columns for class/base/tree_modifier. Found: {list(df.columns)}")

    df[class_col] = df[class_col].astype(int)
    base_map = dict(zip(df[class_col].values, df[base_col].astype(float).values))
    mod_map  = dict(zip(df[class_col].values, df[mod_col].astype(float).values))
    return base_map, mod_map


def map_lookup_block(
    lc_block: np.ndarray,
    base_map: Dict[int, float],
    mod_map:  Dict[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
    base = np.zeros(lc_block.shape, dtype=np.float32)
    mod  = np.zeros(lc_block.shape, dtype=np.float32)
    for code in np.unique(lc_block):
        mask     = (lc_block == code)
        base[mask] = base_map.get(int(code), 0.0)
        mod[mask]  = mod_map.get(int(code),  0.0)
    return base, mod


def compute_shdi_max(shdi_ds: rasterio.io.DatasetReader) -> float:
    nodata       = shdi_ds.nodata
    total_blocks = sum(1 for _ in shdi_ds.block_windows(1))
    logging.info(f"Computing SHDI maximum across {total_blocks:,} blocks...")
    shdi_max = -np.inf
    for i, (_, win) in enumerate(shdi_ds.block_windows(1), start=1):
        arr  = shdi_ds.read(1, window=win)
        vals = arr[~mask_nodata(arr, nodata)]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            shdi_max = max(shdi_max, float(vals.max()))
        if i % PROGRESS_EVERY_N_BLOCKS == 0 or i == total_blocks:
            logging.info(f"  SHDI max progress: {i:,}/{total_blocks:,} blocks")
    if not np.isfinite(shdi_max) or shdi_max <= 0:
        raise ValueError("SHDI maximum could not be determined. Check SHDI raster.")
    logging.info(f"SHDI maximum: {shdi_max:.6f}")
    return shdi_max


def output_profile(ref_ds: rasterio.io.DatasetReader) -> dict:
    return {
        "driver":     "GTiff",
        "width":      ref_ds.width,
        "height":     ref_ds.height,
        "count":      1,
        "dtype":      "float32",
        "crs":        ref_ds.crs,
        "transform":  ref_ds.transform,
        "nodata":     OUT_NODATA,
        "compress":   "DEFLATE",
        "predictor":  3,
        "zlevel":     9,
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def delete_if_exists(path: str) -> None:
    if os.path.exists(path):
        logging.warning(f"Deleting existing output: {path}")
        os.remove(path)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    setup_logging()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("Loading landcover lookup CSV...")
    base_map, mod_map = load_landcover_lookup(LANDCOVER_LOOKUP_CSV)
    logging.info(f"Lookup loaded: {len(base_map)} classes.")

    kernel = make_gaussian_kernel(KERNEL_SIGMA_M, KERNEL_RADIUS_M, PIXEL_SIZE_M)
    logging.info(
        f"Gaussian kernel: sigma={KERNEL_SIGMA_M}m, "
        f"radius={KERNEL_RADIUS_M}m, "
        f"size={kernel.shape[0]}x{kernel.shape[1]} cells"
    )

    logging.info("Opening rasters...")
    with rasterio.open(LANDCOVER_RASTER)  as lc_ds, \
         rasterio.open(TREECOVER_RASTER)  as tc_ds, \
         rasterio.open(SHDI_RASTER)       as shdi_ds, \
         rasterio.open(PATCHSCORE_RASTER) as ps_ds, \
         rasterio.open(SLOPE_DEG_RASTER)  as sl_ds, \
         rasterio.open(WATERDIST_RASTER)  as wd_ds:

        ensure_same_grid(lc_ds, tc_ds,   "treecover")
        ensure_same_grid(lc_ds, shdi_ds, "shdi")
        ensure_same_grid(lc_ds, ps_ds,   "patchscore")
        ensure_same_grid(lc_ds, sl_ds,   "slope")
        ensure_same_grid(lc_ds, wd_ds,   "water distance")

        logging.info(f"Raster grid: {lc_ds.width} x {lc_ds.height} pixels")

        shdi_max = SHDI_FIXED_MAX if SHDI_FIXED_MAX is not None else compute_shdi_max(shdi_ds)

        prof = output_profile(lc_ds)

        # ---- Read all rasters into memory ----
        logging.info("Reading rasters into memory...")

        lc  = lc_ds.read(1)
        lc_nd = mask_nodata(lc, lc_ds.nodata)

        tc  = tc_ds.read(1).astype(np.float32)
        tc_nd = mask_nodata(tc, tc_ds.nodata) | ~np.isfinite(tc)

        sh  = shdi_ds.read(1).astype(np.float32)
        sh_nd = mask_nodata(sh, shdi_ds.nodata) | ~np.isfinite(sh)

        ps  = ps_ds.read(1).astype(np.float32)
        ps_nd = mask_nodata(ps, ps_ds.nodata) | ~np.isfinite(ps)

        sl  = sl_ds.read(1).astype(np.float32)
        sl_nd = mask_nodata(sl, sl_ds.nodata)
        sl[sl_nd] = 0.0  # treat nodata slope as flat

        wd  = wd_ds.read(1).astype(np.float32)
        wd_nd = mask_nodata(wd, wd_ds.nodata)

        # ---- Structural score (cell-level, then convolve) ----
        logging.info("Computing structural score...")
        total_blocks = sum(1 for _ in lc_ds.block_windows(1))
        structure = np.zeros(lc.shape, dtype=np.float32)
        for i, (_, win) in enumerate(lc_ds.block_windows(1), start=1):
            sl_ = win.toslices()
            base, mod = map_lookup_block(lc[sl_], base_map, mod_map)
            S = base + (tc[sl_] / 100.0) * mod
            structure[sl_] = np.clip(S, 0.0, 100.0)
            if i % PROGRESS_EVERY_N_BLOCKS == 0 or i == total_blocks:
                logging.info(f"  Structure progress: {i:,}/{total_blocks:,} blocks")

        invalid_structure = lc_nd | tc_nd
        logging.info("Convolving structural score...")
        structure_conv = gaussian_convolve(structure, invalid_structure, kernel)

        # ---- Topographic drama (cell-level, then convolve) ----
        logging.info("Computing and convolving topographic drama...")
        topo = np.clip(sl / float(SLOPE_REF_DEG), 0.0, 1.0) * 100.0
        topo_conv = gaussian_convolve(topo, sl_nd, kernel)

        # ---- Complexity (not convolved -- already neighbourhood-representative) ----
        logging.info("Computing complexity score...")
        shdi_norm = np.clip((sh / shdi_max) * 100.0, 0.0, 100.0)
        complexity = np.clip(0.5 * shdi_norm + 0.5 * ps, 0.0, 100.0)

        # ---- Water proximity (not convolved -- already distance-based) ----
        logging.info("Computing water proximity score...")
        wd_clean = np.where(wd_nd, np.inf, np.maximum(wd, 0.0))
        water = np.clip(100.0 * np.exp(-wd_clean / WATER_DECAY_M), 0.0, 100.0).astype(np.float32)

        # ---- Combined supply ----
        logging.info("Computing final aesthetic supply...")
        supply = (
            W_STRUCTURE  * structure_conv +
            W_TOPO       * topo_conv +
            W_COMPLEXITY * complexity +
            W_WATER      * water
        )
        supply = np.clip(supply, 0.0, 100.0).astype(np.float32)

        # Invalid where core inputs are missing
        invalid = lc_nd | tc_nd | sh_nd | ps_nd
        supply[invalid] = OUT_NODATA

        # ---- Write output ----
        delete_if_exists(OUT_AESTHETIC_TIF)
        logging.info(f"Writing aesthetic supply raster: {OUT_AESTHETIC_TIF}")
        with rasterio.open(OUT_AESTHETIC_TIF, "w", **prof) as dst:
            for i, (_, win) in enumerate(dst.block_windows(1), start=1):
                sl_ = win.toslices()
                dst.write(supply[sl_], 1, window=win)
                if i % PROGRESS_EVERY_N_BLOCKS == 0 or i == total_blocks:
                    logging.info(f"  Write progress: {i:,}/{total_blocks:,} blocks ({100*i/total_blocks:.1f}%)")

    logging.info("Done. Aesthetic supply raster written successfully.")


if __name__ == "__main__":
    main()