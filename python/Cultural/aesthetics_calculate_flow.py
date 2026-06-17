"""
Compute a national-scale Aesthetic Flow Value raster from:
- Aesthetic supply raster (0-100, scenic experience at each location)
- Adult population raster (persons per cell)
- Traffic raster (average daily vehicle count, nodata where no highway)

Flow model:
    1) exposure = traffic * TRAFFIC_COEF + pop_adults * POP_COEF
    2) exposure_field = GaussianSpread(exposure)
       (Gaussian kernel with sigma=SIGMA_M, truncated at TRUNCATE_M,
        normalised so total exposure is conserved)
    3) flow_value = VALUE_PER_HOUR * exposure_field * (supply / 100)

Where:
    TRAFFIC_COEF = 0.8517  person-hours/year per unit ADT (1.4 persons/vehicle * 6s/3600 * 365 days)
    POP_COEF     = 2920.0  person-hours/year per adult (8 hrs/day * 365)
    VALUE_PER_HOUR = 1.37  NZD per person-hour ($4000/ha/year for maximum scenic quality,divided by 2920 resident exposure hours/year)

    SIGMA_M      = Gaussian kernel standard deviation (metres)
    TRUNCATE_M   = Kernel radius (metres)

Supply already encodes:
    - Convolved structural score (landcover base * tree cover)
    - Convolved topographic drama
    - Complexity (SHDI + patch density, already neighbourhood-representative)
    - Water proximity (distance-decay centred on each cell)

Exposure is redistributed spatially before multiplying by supply, so aesthetic value accrues to scenic landscape cells
rather than being located only where people reside or roads exist.
Traffic nodata is treated as zero (no highway = no traffic, not invalid).
Output is invalid only where supply or population is nodata.

Outputs:
    aesthetic_flow_value.tif    (NZD/ha/year, float32)
    aesthetic_flow.tif          (normalised 0-100, float32)

Dependencies:
    rasterio, numpy, scipy
"""

from __future__ import annotations

from scipy.signal import fftconvolve
import os
import logging
from typing import Optional, Tuple

import numpy as np
import rasterio

# ----------------------------
# Paths
# ----------------------------
AESTHETIC_SUPPLY_TIF = r"<PROJECT_DIRECTORY>\Cultural\Aesthetics\Intermediate\aesthetic_supply.tif"
POP_ADULTS_TIF       = r"<PROJECT_DIRECTORY>\Common\pop_adults_expanded.tif"
TRAFFIC_TIF          = r"<PROJECT_DIRECTORY>\Cultural\Aesthetics\Intermediate\traffic.tif"

OUTPUT_DIR           = r"<PROJECT_DIRECTORY>\Cultural\Aesthetics\Output"
OUT_FLOW_TIF         = os.path.join(OUTPUT_DIR, "aesthetic_flow_value_v5.tif")
OUT_FLOW_NORM_TIF    = os.path.join(OUTPUT_DIR, "aesthetic_flow_v5.tif")
OUT_EXPOSURE_TIF     = os.path.join(OUTPUT_DIR, "aesthetic_exposure_weighted_v5.tif")  # NEW


# ----------------------------
# Model settings
# ----------------------------
# person-hours/year per unit ADT: 1.4 occupancy * 6s/3600 * 365
TRAFFIC_COEF       = 0.8517

# person-hours/year per adult: 8 hrs/day * 365
POP_COEF           = 2920.0

# NZD per person-hour: $4000/ha/year / 2920 hrs
VALUE_PER_HOUR     = 1.37

# Demand spreading kernel parameters
SIGMA_M     = 300.0    # Gaussian sigma in metres
TRUNCATE_M  = 1000.0   # Kernel truncation radius in metres

# Denominator stability
DENOM_MIN   = 1e-3     # below this, treat as "no attractive land nearby"

# Normalisation
NORM_P_LO          = 1
NORM_P_HI          = 99
MAX_SAMPLES        = 2_000_000

OUT_NODATA         = -9999.0
LOG_LEVEL          = logging.INFO
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


def gaussian_kernel(pixel_size: float, sigma_m: float = 300.0, truncate_m: float = 1000.0) -> np.ndarray:
    sigma_px = sigma_m / pixel_size
    radius_px = int(truncate_m / pixel_size)

    y, x = np.mgrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma_px**2))

    kernel_sum = kernel.sum()
    if kernel_sum <= 0 or not np.isfinite(kernel_sum):
        raise ValueError("Gaussian kernel sum is invalid; check sigma/truncate/pixel size.")
    kernel /= kernel_sum  # conserve exposure

    return kernel.astype(np.float32)


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


def compute_percentiles(
    arr: np.ndarray,
    invalid: np.ndarray,
    p_lo: float,
    p_hi: float,
    max_samples: int
) -> Tuple[float, float]:
    vals = arr[~invalid].ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size > max_samples:
        idx  = np.random.choice(vals.size, size=max_samples, replace=False)
        vals = vals[idx]
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    logging.info(f"Normalisation percentiles: p{p_lo}={lo:.4f}, p{p_hi}={hi:.4f}")
    return lo, hi


def write_raster(path: str, prof: dict, arr: np.ndarray) -> None:
    delete_if_exists(path)
    logging.info(f"Writing raster: {path}")
    with rasterio.open(path, "w", **prof) as dst:
        total_blocks = sum(1 for _ in dst.block_windows(1))
        for i, (_, win) in enumerate(dst.block_windows(1), start=1):
            dst.write(arr[win.toslices()], 1, window=win)
            if i % PROGRESS_EVERY_N_BLOCKS == 0 or i == total_blocks:
                logging.info(f"  Write progress: {i:,}/{total_blocks:,} ({100*i/total_blocks:.1f}%)")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    setup_logging()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("Opening rasters...")
    with rasterio.open(AESTHETIC_SUPPLY_TIF) as sup_ds, \
         rasterio.open(POP_ADULTS_TIF)       as pop_ds, \
         rasterio.open(TRAFFIC_TIF)          as trf_ds:

        ensure_same_grid(sup_ds, pop_ds, "pop_adults")
        ensure_same_grid(sup_ds, trf_ds, "traffic")

        prof = output_profile(sup_ds)

        # ---- Read rasters into memory ----
        logging.info("Reading supply raster...")
        supply = sup_ds.read(1).astype(np.float32)
        sup_nd = mask_nodata(supply, sup_ds.nodata) | ~np.isfinite(supply)

        logging.info("Reading population raster...")
        pop = pop_ds.read(1).astype(np.float32)

        pop_nodata = pop_ds.nodata
        if pop_nodata is None:
            pop_nodata = -9999.0
            logging.warning("pop_ds.nodata is None; assuming -9999.0 for population nodata.")
        pop_nd = mask_nodata(pop, pop_nodata) | ~np.isfinite(pop)

        pop[pop_nd] = 0.0
        pop = np.maximum(pop, 0.0).astype(np.float32)

        logging.info("Reading traffic raster (nodata -> 0)...")
        trf = trf_ds.read(1).astype(np.float32)
        trf_nd = mask_nodata(trf, trf_ds.nodata) | ~np.isfinite(trf)
        trf[trf_nd] = 0.0
        trf = np.maximum(trf, 0.0).astype(np.float32)

        # ---- Build kernel safely while dataset is open ----
        pixel_size = abs(sup_ds.transform.a)
        logging.info(
            f"Building Gaussian kernel (sigma={SIGMA_M}m, "
            f"truncate={TRUNCATE_M}m, pixel={pixel_size}m)"
        )
        K = gaussian_kernel(pixel_size, SIGMA_M, TRUNCATE_M)

    # ---- Compute exposure (person-hours/year) ----
    logging.info("Computing exposure...")
    exposure = trf * TRAFFIC_COEF + pop * POP_COEF
    exposure = np.maximum(exposure, 0.0).astype(np.float32)

    # ---- Attractiveness A in [0,1], with nodata -> 0 ----
    A = (supply / 100.0).astype(np.float32)
    A[sup_nd] = 0.0
    A = np.clip(A, 0.0, 1.0)

    # ---- Proximity-weighted exposure (person-hours/year) ----
    logging.info("Computing proximity-weighted exposure field...")
    exposure_weighted = fftconvolve(exposure, K, mode="same").astype(np.float32)

    # FFT numerical noise can produce tiny negatives; clamp
    exposure_weighted = np.maximum(exposure_weighted, 0.0).astype(np.float32)

    # Write distance-weighted exposure raster (person-hours/year)
    exp_out = exposure_weighted.copy()
    exp_out[sup_nd] = OUT_NODATA
    write_raster(OUT_EXPOSURE_TIF, prof, exp_out)

    # ---- Flow accrues where exposure is nearby AND scenery is high ----
    flow_value = (VALUE_PER_HOUR * A * exposure_weighted).astype(np.float32)
    flow_value = np.maximum(flow_value, 0.0).astype(np.float32)
    flow_value[sup_nd] = OUT_NODATA

    # Final clamp to avoid tiny negatives anywhere, including "no population" areas
    flow_value = np.maximum(flow_value, 0.0).astype(np.float32)

    invalid = sup_nd
    flow_value[invalid] = OUT_NODATA

    # ---- Write flow value ----
    write_raster(OUT_FLOW_TIF, prof, flow_value)

    # ---- Normalise to 0-100 ----
    logging.info("Normalising to 0-100...")
    lo, hi = compute_percentiles(flow_value, invalid, NORM_P_LO, NORM_P_HI, MAX_SAMPLES)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        logging.warning("Percentiles collapsed, falling back to max normalisation.")
        lo = 0.0
        hi = float(np.nanmax(flow_value[~invalid]))

    norm = np.clip((flow_value - lo) / (hi - lo) * 100.0, 0.0, 100.0).astype(np.float32)
    norm[invalid] = OUT_NODATA

    write_raster(OUT_FLOW_NORM_TIF, prof, norm)

    logging.info("Done. Aesthetic flow rasters written successfully.")


if __name__ == "__main__":
    main()