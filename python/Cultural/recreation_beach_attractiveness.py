"""
Beach Attractiveness Surface Generator

Creates a raster "attractiveness surface" for beach visits within a coastal buffer, based on:
1) Surrounding landcover (neighbourhood influence via a distance-decay moving window),
2) Road access (distance-to-road raster),
3) Distance to the coastline (distance-to-coast raster).

Inputs:
- Landcover raster (categorical codes), already clipped to ~1 km coastal buffer (or containing NoData outside)
- CSV lookup table mapping landcover class -> attractiveness score (0–1)
- Road proximity raster (distance to nearest road in metres)
- Coast proximity raster (distance to coastline in metres)

Output:
- Single-band float32 GeoTIFF attractiveness raster

Notes:
- The landcover contribution is computed as a weighted mean of landcover scores in a window around each cell,
  using an exponential distance-decay kernel.
- Accessibility and coastline proximity are applied as multiplicative scalars in [0, 1].
- All rasters must share the same grid (CRS, transform, width/height). The script will check this.

Dependencies:
- rasterio
- numpy
- pandas
- scipy (for fast convolution). If scipy is not available, you can replace the convolution section with a slower method.

Author: (you)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

try:
    from scipy.signal import fftconvolve
except ImportError as e:
    raise ImportError(
        "scipy is required for fast convolution (scipy.signal.fftconvolve). "
        "Install scipy or replace convolution with an alternative."
    ) from e


# =========================
# Constants (edit these)
# =========================

@dataclass(frozen=True)
class Config:
    # Paths
    COAST_DIR =  Path(r"<PROJECT_DIRECTORY>\Cultural\Recreation\Coastal")
    RECREATION_DIR = Path(r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate")
    COMMON_DIR =  Path(r"<PROJECT_DIRECTORY>\Common")
    OUTPUT_DIR = COAST_DIR

    landcover_raster = COAST_DIR / "coast_1km_lcdb.tif"
    road_prox_raster= COMMON_DIR / "road_proximity.tif"
    coast_prox_raster= COAST_DIR / "coastline_all_proximity.tif"
    landcover_scores_csv = RECREATION_DIR / "lcdb6_attractiveness.csv"
    output_raster = OUTPUT_DIR / "attractiveness_beach.tif"

    # CSV columns
    csv_class_col: str = "class"   # change if your key is numeric class codes
    csv_score_col: str = "coastal"

    # Neighbourhood kernel (landcover context)
    window_radius_m: float = 300.0         # radius of neighbourhood influence (metres)
    landcover_decay_m: float = 150.0       # exponential decay length (metres). Smaller = more local influence.
    min_valid_fraction: float = 0.2        # minimum fraction of valid cells in window to compute a score
    local_landcover_power: float = 2.0      # Controls how strongly the focal cell’s own landcover suitability limits attractiveness.
    neighbourhood_landcover_power: float = 1.0 #Controls how strongly the surrounding landcover context influences attractiveness.

    # Road accessibility mediation (distance to road in metres)
    road_access_half_distance_m: float = 400.0  # distance where access scalar ~ 0.5 (for logistic)
    road_access_steepness_m: float = 120.0      # larger = smoother transition

    # Coast proximity mediation (distance to coast in metres)
    coast_pref_decay_m: float = 120.0      # attractiveness decays with distance inland from coast

    # Combination weights
    # Final = (neighbourhood_landcover_score ** landcover_power) * (road_access ** road_power) * (coast_proximity ** coast_power)
    ocean_code: int = -9999
    landcover_power: float = 1.0
    road_power: float = 1.0
    coast_power: float = 1.0

    # Output / NoData
    output_nodata: float = -9999.0

    # Logging
    log_level: int = logging.INFO


CFG = Config()


# =========================
# Logging setup
# =========================

logging.basicConfig(
    level=CFG.log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("beach_attractiveness")


# =========================
# Helper functions
# =========================

def _check_same_grid(meta_a: dict, meta_b: dict, name_a: str, name_b: str) -> None:
    """Raise ValueError if rasters are not aligned."""
    keys = ["crs", "transform", "width", "height"]
    for k in keys:
        if meta_a.get(k) != meta_b.get(k):
            raise ValueError(
                f"Raster grids do not match for {name_a} vs {name_b} (mismatch in {k}). "
                f"{name_a}.{k}={meta_a.get(k)}; {name_b}.{k}={meta_b.get(k)}"
            )


def load_scores_lookup(csv_path: str) -> Dict[str, float]:
    """Load landcover attractiveness scores from CSV into a dict."""
    df = pd.read_csv(csv_path)
    if CFG.csv_class_col not in df.columns or CFG.csv_score_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{CFG.csv_class_col}' and '{CFG.csv_score_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    # Clean keys to reduce accidental mismatches (e.g., trailing spaces)
    keys = df[CFG.csv_class_col].astype(str).str.strip()
    scores = df[CFG.csv_score_col].astype(float)

    # Clamp scores to [0, 1] just in case
    scores = scores.clip(0.0, 1.0)

    lookup = dict(zip(keys, scores))
    logger.info(f"Loaded {len(lookup)} landcover attractiveness scores from CSV.")
    return lookup


def build_distance_decay_kernel(radius_m: float, decay_m: float, pixel_size_x: float, pixel_size_y: float) -> np.ndarray:
    """
    Create a circular exponential distance-decay kernel for convolution.

    Kernel weight at distance d: exp(-d / decay_m), for d <= radius_m, else 0
    Kernel is normalised later via valid-weight convolution, so no need to normalise here.
    """
    # Determine kernel size in pixels
    rx = int(math.ceil(radius_m / abs(pixel_size_x)))
    ry = int(math.ceil(radius_m / abs(pixel_size_y)))

    # Grid of pixel offsets
    y = np.arange(-ry, ry + 1) * abs(pixel_size_y)
    x = np.arange(-rx, rx + 1) * abs(pixel_size_x)
    xx, yy = np.meshgrid(x, y)
    dd = np.sqrt(xx * xx + yy * yy)

    kernel = np.exp(-dd / decay_m)
    kernel[dd > radius_m] = 0.0

    # Avoid a zero kernel in pathological cases
    if np.all(kernel == 0):
        raise ValueError("Kernel ended up all zeros. Check radius_m, decay_m, and pixel size.")

    logger.info(
        f"Built kernel with radius {radius_m} m, decay {decay_m} m, "
        f"shape={kernel.shape}, sum={kernel.sum():.3f}"
    )
    return kernel.astype(np.float32)


def logistic_access(distance_m: np.ndarray, half_distance_m: float, steepness_m: float) -> np.ndarray:
    """
    Logistic decay from 1 (near road) to 0 (far from road).

    access = 1 / (1 + exp((d - half) / steepness))
    """
    d = np.asarray(distance_m, dtype=np.float32)
    return 1.0 / (1.0 + np.exp((d - half_distance_m) / steepness_m))


def exponential_coast_preference(distance_m: np.ndarray, decay_m: float) -> np.ndarray:
    """Exponential decay with distance inland from the coast: exp(-d/decay)."""
    d = np.asarray(distance_m, dtype=np.float32)
    return np.exp(-d / decay_m)


def safe_divide(numer: np.ndarray, denom: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Elementwise division with safe handling of zeros."""
    out = np.full_like(numer, fill, dtype=np.float32)
    mask = denom > 0
    out[mask] = (numer[mask] / denom[mask]).astype(np.float32)
    return out


def map_landcover_to_score(
    lc: np.ndarray,
    lc_nodata: float | int | None,
    lookup: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map integer landcover codes directly to 0–1 attractiveness scores.

    - Excludes ocean cells (CFG.ocean_code) and raster NoData from contributing to the
      neighbourhood calculation.
    - Skips any negative class codes that may appear in the CSV (e.g., -9999).
    """

    # Convert CSV lookup (string keys) to int -> float, skipping negatives (e.g., ocean/-9999)
    code_to_score: Dict[int, float] = {}
    for k, v in lookup.items():
        try:
            code = int(k)
        except (TypeError, ValueError):
            continue

        if code < 0 or code == CFG.ocean_code:
            continue

        code_to_score[code] = float(v)

    if not code_to_score:
        raise ValueError(
            "No valid (non-negative) landcover codes found in CSV lookup. "
            f"Check {CFG.csv_class_col} in {CFG.landcover_scores_csv}."
        )

    lc = lc.astype(np.int32, copy=False)

    # Valid mask: exclude NoData and ocean
    valid = np.ones(lc.shape, dtype=bool)

    if lc_nodata is not None:
        valid &= (lc != lc_nodata)

    valid &= (lc != CFG.ocean_code)

    # Build LUT array from 0 to max code
    max_code = max(code_to_score.keys())
    lut = np.full(max_code + 1, np.nan, dtype=np.float32)

    for code, score in code_to_score.items():
        if 0 <= code <= max_code:
            lut[code] = np.clip(score, 0.0, 1.0)

    scores = np.full(lc.shape, np.nan, dtype=np.float32)

    in_range = (lc >= 0) & (lc <= max_code) & valid
    scores[in_range] = lut[lc[in_range]]

    mapped_valid = valid & np.isfinite(scores)

    # Helpful warning if many cells are unmapped
    unmapped = int(valid.sum() - mapped_valid.sum())
    if unmapped > 0:
        logger.warning(
            f"{unmapped} valid landcover cells had no score mapping (outside CSV codes). "
            "They will be treated as NoData for the landcover neighbourhood calculation."
        )

    return scores, mapped_valid

# =========================
# Main processing
# =========================

def main() -> None:
    logger.info("Starting beach attractiveness surface calculation.")

    lookup = load_scores_lookup(CFG.landcover_scores_csv)

    # Read rasters
    logger.info("Reading input rasters...")
    with rasterio.open(CFG.landcover_raster) as lc_ds, \
         rasterio.open(CFG.road_prox_raster) as road_ds, \
         rasterio.open(CFG.coast_prox_raster) as coast_ds:

        _check_same_grid(lc_ds.meta, road_ds.meta, "landcover", "road_proximity")
        _check_same_grid(lc_ds.meta, coast_ds.meta, "landcover", "coast_proximity")

        lc = lc_ds.read(1)
        road = road_ds.read(1).astype(np.float32)
        coast = coast_ds.read(1).astype(np.float32)

        lc_nodata = lc_ds.nodata
        road_nodata = road_ds.nodata
        coast_nodata = coast_ds.nodata

        meta = lc_ds.meta.copy()

        # Determine analysis mask: inside coastal buffer = valid landcover cells (or you can use a dedicated buffer mask raster)
        analysis_mask = np.ones(lc.shape, dtype=bool)
        if lc_nodata is not None:
            analysis_mask &= (lc != lc_nodata)
        if road_nodata is not None:
            analysis_mask &= (road != road_nodata)
        if coast_nodata is not None:
            analysis_mask &= (coast != coast_nodata)

        logger.info(f"Analysis cells: {int(analysis_mask.sum())} of {analysis_mask.size}")

        # Map landcover to base attractiveness score raster
        logger.info("Mapping landcover classes to attractiveness scores...")
        lc_score, lc_valid = map_landcover_to_score(lc, lc_nodata, lookup)

        # Build distance-decay kernel in pixel space
        pixel_size_x = meta["transform"].a
        pixel_size_y = meta["transform"].e  # typically negative
        kernel = build_distance_decay_kernel(
            CFG.window_radius_m,
            CFG.landcover_decay_m,
            pixel_size_x,
            pixel_size_y
        )

        # Neighbourhood weighted mean landcover score via convolution:
        # numerator = conv(score * valid, kernel)
        # denom     = conv(valid, kernel)
        logger.info("Computing neighbourhood landcover attractiveness (convolution)...")

        # Use float32 to keep memory reasonable
        valid_f = lc_valid.astype(np.float32)
        score_f = np.where(lc_valid, lc_score, 0.0).astype(np.float32)

        # Convolutions
        numer = fftconvolve(score_f, kernel, mode="same").astype(np.float32)
        denom = fftconvolve(valid_f, kernel, mode="same").astype(np.float32)

        neigh_score = safe_divide(numer, denom, fill=np.nan)

        # Require minimum valid fraction in the window to avoid edge artefacts
        # Approximate valid fraction as denom / conv(all_ones, kernel) within analysis domain
        logger.info("Applying minimum-valid-fraction threshold...")
        ones = np.ones(lc.shape, dtype=np.float32)
        denom_full = fftconvolve(ones, kernel, mode="same").astype(np.float32)
        valid_fraction = safe_divide(denom, denom_full, fill=0.0)

        neigh_score[valid_fraction < CFG.min_valid_fraction] = np.nan

        # Local suitability: prevent low-suitability focal cells (e.g., mangroves/wetlands)
        # from "borrowing" high attractiveness from surrounding landcover.
        # lc_score is the per-cell landcover score (0..1), NaN where unmapped/invalid.
        logger.info("Combining local suitability with neighbourhood context...")
        local_score = lc_score.astype(np.float32)  # already 0..1 (or NaN)

        landcover_component = (np.power(local_score, CFG.local_landcover_power) *
                               np.power(neigh_score, CFG.neighbourhood_landcover_power)).astype(np.float32)

        # Road and coast mediation (both in 0..1)
        logger.info("Calculating road-access and coast-proximity scalars...")
        road_access = logistic_access(road, CFG.road_access_half_distance_m, CFG.road_access_steepness_m)
        coast_pref = exponential_coast_preference(coast, CFG.coast_pref_decay_m)

        # Combine components
        logger.info("Combining components into final attractiveness surface...")
        final = (np.power(landcover_component, CFG.landcover_power) *
                 np.power(road_access, CFG.road_power) *
                 np.power(coast_pref, CFG.coast_power)).astype(np.float32)

        # Mask outside analysis area
        final[~analysis_mask] = np.nan

        # Clamp just in case
        final = np.clip(final, 0.0, 1.0)

        # Write output with nodata
        logger.info("Writing output raster...")
        out = np.where(np.isfinite(final), final, CFG.output_nodata).astype(np.float32)

        meta.update(
            dtype="float32",
            count=1,
            nodata=CFG.output_nodata,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            bigtiff="IF_SAFER"
        )

        with rasterio.open(CFG.output_raster, "w", **meta) as out_ds:
            out_ds.write(out, 1)

    logger.info(f"Done. Wrote: {CFG.output_raster}")


if __name__ == "__main__":
    main()