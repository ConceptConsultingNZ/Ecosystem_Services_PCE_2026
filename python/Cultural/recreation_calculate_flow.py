"""
Allocate demand to supply cells using a gravity model with distance decay and attractiveness.

Optional accessibility adjustment:
- If ACCESS_HALF_M is not None, read a proximity raster (distance-to-access, metres)
  and downweight supply attractiveness where access is poor.

Allocation:
w_ij ∝ (A_j^alpha) * exp(-(d_ij/lambda)^beta)

Outputs:
- allocated raster (Float32): allocated demand received by each supply cell
- (optional) unallocated demand raster (Float32): demand in cells with Z==0

Fixes included (striping / artefacts):
1) FFT convolution uses ifftshift(kernel) before FFT (correct kernel centring).
2) FFT done in float64 for numerical stability; cast back to float32 at the end.
3) Explicit NaN/Inf cleanup after divisions and convolutions.
4) "Clip to zero" is tolerance-based (only tiny negatives are zeroed), reducing structured speckle/banding.
5) Optional: safer handling of nodata as 0 is retained, but NaNs are removed before writing.
"""

import os
import math
import time
import logging
import csv
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path

# ==============================
# Constants
# ==============================
RECREATION_TYPE = "freshwater_fishing" #Options: "mountain_biking", "short_walk","day_tramp","overnight_tramp","freshwater","freshwater_fishing",marine_boating"

# Directories
WORKING_DIR = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate"
OUTPUT_DIR = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Output"

# Input filenames (located in WORKING_DIR)
DEMAND_FILENAME = f"demand_{RECREATION_TYPE}.tif"
SUPPLY_ATTR_FILENAME = f"attractiveness_{RECREATION_TYPE}.tif"
PARAMETER_FILENAME = "distance_parameters.csv"

ACCESS_PROX_RASTER = r"<PROJECT_DIRECTORY>\Common\road_proximity.tif"
#ACCESS_PROX_RASTER = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Marine_boating\ramp_proximity.tif"

# Output filenames (written to OUTPUT_DIR)
OUTPUT_ALLOC_FILENAME = f"recreation_{RECREATION_TYPE}_value.tif"
OUTPUT_UNALLOC_FILENAME = f"unallocated_{RECREATION_TYPE}.tif"  # set to None to skip

# Build full paths
DEMAND_RASTER = os.path.join(WORKING_DIR, DEMAND_FILENAME)
SUPPLY_ATTR_RASTER = os.path.join(WORKING_DIR, SUPPLY_ATTR_FILENAME)
PARAMETER_CSV = os.path.join(WORKING_DIR, PARAMETER_FILENAME)

OUTPUT_ALLOC_RASTER = os.path.join(OUTPUT_DIR, OUTPUT_ALLOC_FILENAME)
OUTPUT_UNALLOC_RASTER = (
    None if OUTPUT_UNALLOC_FILENAME is None
    else os.path.join(OUTPUT_DIR, OUTPUT_UNALLOC_FILENAME)
)

# Gravity parameters
MEDIAN_DISTANCE_KM = None          # half-weight distance
KERNEL_CUTOFF_KM = None            # truncate kernel at this distance
ALPHA = 1.0                        # attractiveness exponent
BETA = 1.0                         # distance exponent (1.0 = exponential)

# Accessibility parameters (loaded from CSV in main())
ACCESS_HALF_M = None          # distance-to-access (m) where weight ~= 0.5
ACCESS_GAMMA = None           # steepness (higher = sharper drop-off)

# Output / IO
OUTPUT_DTYPE = "float32"
NODATA_OUT = 0.0
COMPRESS = "DEFLATE"
PREDICTOR = 3
TILED = True
BLOCK = 512

# Numerical stability
EPS_REL = 1e-10   # relative tolerance for treating small negative noise as 0
CLEAN_INF_TO_ZERO = True

# Threshold: treat tiny Z as 0 (unreachable)
Z_EPS_REL = 1e-10  # try 1e-10 if you see absurd spikes
Z_EPS_ABS = 0.0  # set >0 only if there is a meaningful absolute floor

# Logging
LOG_LEVEL = logging.INFO  # DEBUG for more detail

# ==============================
# Helpers
# ==============================

def _setup_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def _log_array_stats(name: str, arr: np.ndarray) -> None:
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    arr = np.asarray(arr)
    logging.debug(
        "%s stats: shape=%s dtype=%s min=%.6g max=%.6g sum=%.6g nan=%d inf=%d neg=%d",
        name, arr.shape, arr.dtype,
        float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nansum(arr)),
        int(np.isnan(arr).sum()), int(np.isinf(arr).sum()), int((arr < 0).sum())
    )

def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"Expected a number or blank, got '{value}'.") from e

def load_distance_params(csv_path: str, activity_type: str) -> tuple[float, float, float | None, float | None]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Parameter CSV not found: {csv_path}")

    target = (activity_type or "").strip().lower()
    if not target:
        raise ValueError("activity_type must be a non-empty string.")

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Parameter CSV has no header row: {csv_path}")

        fieldnames = [h.strip() for h in reader.fieldnames]
        header_map = {h.lower(): h for h in fieldnames}

        med_col = header_map.get("median_distance_km")
        cut_col = header_map.get("kernel_cutoff_km")
        if med_col is None or cut_col is None:
            raise ValueError(
                "Parameter CSV must contain columns 'median_distance_km' and 'kernel_cutoff_km'. "
                f"Found headers: {fieldnames}"
            )

        road_half_col = header_map.get("access_half")
        road_gamma_col = header_map.get("access_gamma")
        key_col = header_map.get("activity_type", fieldnames[0])

        for row in reader:
            row_key = (row.get(key_col, "") or "").strip().lower()
            if row_key != target:
                continue

            try:
                median_km = float(row[med_col])
                cutoff_km = float(row[cut_col])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid numeric values for '{activity_type}' in {csv_path}. "
                    f"Got median='{row.get(med_col)}', cutoff='{row.get(cut_col)}'."
                ) from e

            if median_km <= 0 or cutoff_km <= 0:
                raise ValueError(
                    f"median_distance_km and kernel_cutoff_km must be > 0. "
                    f"Got median={median_km}, cutoff={cutoff_km} for '{activity_type}'."
                )

            road_half_m = None
            road_gamma = None

            if road_half_col is not None:
                road_half_m = _parse_optional_float(row.get(road_half_col))
                if road_half_m is not None:
                    if road_half_m <= 0:
                        raise ValueError(
                            f"access_half must be > 0 or blank. Got {road_half_m} for '{activity_type}'."
                        )
                    if road_gamma_col is None:
                        road_gamma = 2.0
                        logging.warning("access_half provided but 'access_gamma' missing; defaulting gamma=2.0")
                    else:
                        road_gamma = _parse_optional_float(row.get(road_gamma_col))
                        if road_gamma is None:
                            road_gamma = 2.0
                            logging.warning("access_half provided but access_gamma blank; defaulting gamma=2.0")
                        elif road_gamma <= 0:
                            raise ValueError(
                                f"access_gamma must be > 0 or blank. Got {road_gamma} for '{activity_type}'."
                            )
                else:
                    road_gamma = None

            return median_km, cutoff_km, road_half_m, road_gamma

    raise KeyError(f"Activity type '{activity_type}' not found in {csv_path}")

def _raster_extent(src: "rasterio.io.DatasetReader"):
    b = src.bounds
    return (b.left, b.bottom, b.right, b.top)

def _fmt_extent(ext):
    l, b, r, t = ext
    return f"left={l:.3f}, bottom={b:.3f}, right={r:.3f}, top={t:.3f}"

def _describe_raster(src, label: str) -> str:
    ext = _raster_extent(src)
    return (
        f"{label}:\n"
        f"  path: {getattr(src, 'name', '(unknown)')}\n"
        f"  crs: {src.crs}\n"
        f"  shape (h,w): ({src.height}, {src.width})\n"
        f"  transform: {src.transform}\n"
        f"  pixel size: (x={src.transform.a:.6f}, y={src.transform.e:.6f})\n"
        f"  extent: {_fmt_extent(ext)}"
    )

def _alignment_report(dsrc, ssrc) -> str:
    lines = []
    lines.append(_describe_raster(dsrc, "DEMAND"))
    lines.append(_describe_raster(ssrc, "SUPPLY"))

    diffs = []
    if (dsrc.width, dsrc.height) != (ssrc.width, ssrc.height):
        diffs.append(
            f"shape mismatch: demand (h,w)=({dsrc.height},{dsrc.width}) vs "
            f"supply (h,w)=({ssrc.height},{ssrc.width})"
        )
    if dsrc.crs != ssrc.crs:
        diffs.append(f"CRS mismatch: demand {dsrc.crs} vs supply {ssrc.crs}")
    if dsrc.transform != ssrc.transform:
        diffs.append(f"transform mismatch:\n  demand {dsrc.transform}\n  supply {ssrc.transform}")

    dext = _raster_extent(dsrc)
    sext = _raster_extent(ssrc)
    if dext != sext:
        diffs.append(f"extent mismatch:\n  demand {_fmt_extent(dext)}\n  supply {_fmt_extent(sext)}")

    if diffs:
        lines.append("\nNOT ALIGNED. Differences:")
        for d in diffs:
            lines.append(f"  - {d}")
    else:
        lines.append("\nALIGNED: shape, CRS, transform, and extent match.")

    return "\n".join(lines)

def road_accessibility_weight(dist_to_access_m: np.ndarray, half_m: float, gamma: float) -> np.ndarray:
    d = np.asarray(dist_to_access_m, dtype=np.float32)
    d = np.clip(d, 0.0, None)
    half = max(float(half_m), 1e-6)
    g = max(float(gamma), 1e-6)
    w = 1.0 / (1.0 + np.power(d / half, g, dtype=np.float32))
    return w.astype(np.float32)

def build_decay_kernel(pixel_size_m: float, median_km: float, beta: float, cutoff_km: float) -> np.ndarray:
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive.")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if median_km <= 0:
        raise ValueError("median_km must be positive.")
    if cutoff_km <= 0:
        raise ValueError("cutoff_km must be positive.")
    if cutoff_km < median_km:
        logging.warning(
            "kernel_cutoff_km (%.3f) is less than median_distance_km (%.3f). "
            "That will truncate the decay early.",
            cutoff_km, median_km
        )

    median_m = median_km * 1000.0
    cutoff_m = cutoff_km * 1000.0
    lam = median_m / (math.log(2.0) ** (1.0 / beta))

    radius_px = int(math.ceil(cutoff_m / pixel_size_m))
    size = radius_px * 2 + 1

    logging.info(
        "Building decay kernel: median=%.3fkm cutoff=%.3fkm beta=%.2f pixel=%.2fm radius_px=%d size=%dx%d lambda=%.3fkm",
        median_km, cutoff_km, beta, pixel_size_m, radius_px, size, size, lam / 1000.0
    )

    xs = (np.arange(-radius_px, radius_px + 1) * pixel_size_m).astype(np.float32)
    ys = xs.copy()
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx * xx + yy * yy)

    decay = np.exp(-((dist / lam) ** beta)).astype(np.float32)
    decay[dist > cutoff_m] = 0.0

    s = float(decay.sum())
    if s > 0:
        decay /= s

    logging.info("Kernel built. Sum=%.6g nonzero=%d", float(decay.sum()), int(np.count_nonzero(decay)))
    _log_array_stats("Kernel", decay)
    return decay

def fft_convolve2d_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    FFT convolution with 'same' output size as a.

    Fixes:
    - Use float64 in FFT for stability.
    - ifftshift(kernel) so its centre corresponds to the FFT origin (0,0).
    """
    t0 = time.time()
    a = np.asarray(a, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    ah, aw = a.shape
    kh, kw = k.shape
    fh = ah + kh - 1
    fw = aw + kw - 1

    logging.info("FFT conv: input=%sx%s kernel=%sx%s fft=%sx%s", ah, aw, kh, kw, fh, fw)

    k0 = np.fft.ifftshift(k)  # critical for correct kernel centring in FFT space

    A = np.fft.rfft2(a, s=(fh, fw))
    K = np.fft.rfft2(k0, s=(fh, fw))
    out_full = np.fft.irfft2(A * K, s=(fh, fw)).real  # float64

    r0 = (kh - 1) // 2
    c0 = (kw - 1) // 2
    out = out_full[r0:r0 + ah, c0:c0 + aw]

    logging.info("FFT conv done in %.1fs", time.time() - t0)
    return out.astype(np.float32)

def safe_write_raster(path, profile, array):
    """
    Write a raster, falling back to incremented filenames if the target
    cannot be overwritten. Also creates statistics and overview pyramids
    for faster display in GIS software.
    """

    def _finalise_raster(raster_path, data, nodata):
        logging.info("Calculating statistics and pyramids for %s...", raster_path)

        with rasterio.open(raster_path, "r+") as dst:
            band = data.astype(np.float64)

            valid = np.isfinite(band)
            if nodata is not None:
                valid &= (band != nodata)

            if np.any(valid):
                valid_data = band[valid]

                dst.update_tags(
                    1,
                    STATISTICS_MINIMUM=str(float(np.min(valid_data))),
                    STATISTICS_MAXIMUM=str(float(np.max(valid_data))),
                    STATISTICS_MEAN=str(float(np.mean(valid_data))),
                    STATISTICS_STDDEV=str(float(np.std(valid_data))),
                    STATISTICS_VALID_PERCENT=str(
                        100.0 * float(np.count_nonzero(valid)) / float(valid.size)
                    )
                )

                logging.info(
                    "Stats: min=%.6g max=%.6g mean=%.6g std=%.6g",
                    float(np.min(valid_data)),
                    float(np.max(valid_data)),
                    float(np.mean(valid_data)),
                    float(np.std(valid_data))
                )

            overview_levels = [2, 4, 8, 16, 32]

            max_dim = max(dst.width, dst.height)
            overview_levels = [
                level for level in overview_levels
                if max_dim // level >= 256
            ]

            if overview_levels:
                logging.info("Building pyramids: %s", overview_levels)

                dst.build_overviews(
                    overview_levels,
                    Resampling.average
                )

                dst.update_tags(
                    ns="rio_overview",
                    resampling="average"
                )

    try:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(np.float32), 1)

        _finalise_raster(
            path,
            array,
            profile.get("nodata")
        )

        return path

    except Exception as e:
        logging.warning(
            "Failed to write %s (%s). Creating a new file.",
            path,
            e
        )

        p = Path(path)
        i = 1

        while True:
            new_path = p.with_name(f"{p.stem}_{i}{p.suffix}")

            try:
                with rasterio.open(new_path, "w", **profile) as dst:
                    dst.write(array.astype(np.float32), 1)

                _finalise_raster(
                    new_path,
                    array,
                    profile.get("nodata")
                )

                logging.info("Wrote raster to %s instead.", new_path)
                return str(new_path)

            except Exception:
                i += 1
def _clean_finite_inplace(arr: np.ndarray, fill_value: float = 0.0) -> None:
    """Replace NaN/Inf in-place."""
    if not np.issubdtype(arr.dtype, np.floating):
        return
    # numpy handles in-place for nan_to_num when out is provided
    np.nan_to_num(arr, copy=False, nan=fill_value,
                  posinf=(fill_value if CLEAN_INF_TO_ZERO else None),
                  neginf=(fill_value if CLEAN_INF_TO_ZERO else None))

def _clip_small_negatives_to_zero(arr: np.ndarray, eps_rel: float = EPS_REL) -> np.ndarray:
    """
    Only zero-out small negative numerical noise. Keep meaningful negatives for debugging if needed.
    Returns a float32 array.
    """
    arr = np.asarray(arr)
    _clean_finite_inplace(arr, fill_value=0.0)

    maxv = float(np.nanmax(arr)) if arr.size else 0.0
    eps = max(eps_rel * maxv, 0.0)

    neg_count = int((arr < 0).sum())
    if neg_count > 0:
        logging.info("Clipping negatives: count=%d min=%g eps=%g", neg_count, float(np.min(arr)), eps)

    if eps > 0:
        arr[arr < -eps] = 0.0
        arr[(arr >= -eps) & (arr < 0)] = 0.0
    else:
        arr[arr < 0] = 0.0

    return arr.astype(np.float32)

# ==============================
# Main
# ==============================

def main() -> None:
    global MEDIAN_DISTANCE_KM, KERNEL_CUTOFF_KM, ACCESS_HALF_M, ACCESS_GAMMA

    _setup_logging()
    logging.info("Starting allocation run")

    # Load distance-decay and optional access-adjustment parameters for this
    # recreation type. These control how far demand can travel to supply, and
    # whether supply is discounted when it is far from road/access infrastructure.
    MEDIAN_DISTANCE_KM, KERNEL_CUTOFF_KM, ACCESS_HALF_M, ACCESS_GAMMA = load_distance_params(
        PARAMETER_CSV, RECREATION_TYPE
    )

    logging.info(
        "Loaded distance parameters for '%s': median_distance_km=%.3f, kernel_cutoff_km=%.3f (from %s)",
        RECREATION_TYPE, MEDIAN_DISTANCE_KM, KERNEL_CUTOFF_KM, PARAMETER_CSV
    )

    if ACCESS_HALF_M is None:
        logging.info("Access adjustment: not required (access_half is blank).")
    else:
        logging.info(
            "Access adjustment: enabled (half=%.1fm gamma=%.2f). Proximity raster: %s",
            float(ACCESS_HALF_M),
            float(ACCESS_GAMMA if ACCESS_GAMMA is not None else 2.0),
            ACCESS_PROX_RASTER
        )

    logging.info("Demand raster: %s", DEMAND_RASTER)
    logging.info("Supply raster: %s", SUPPLY_ATTR_RASTER)
    logging.info("Output (alloc): %s", OUTPUT_ALLOC_RASTER)
    logging.info("Output (unalloc): %s", OUTPUT_UNALLOC_RASTER if OUTPUT_UNALLOC_RASTER else "(skipped)")

    t_all = time.time()

    logging.info("Opening rasters...")
    with rasterio.open(DEMAND_RASTER) as dsrc, rasterio.open(SUPPLY_ATTR_RASTER) as ssrc:

        # The allocation maths assumes demand and supply are on exactly the same
        # grid. Same extent, resolution, transform, dimensions and CRS. If not,
        # cell-by-cell operations would silently compare the wrong places.
        logging.info("Checking raster alignment...")
        report = _alignment_report(dsrc, ssrc)
        if "NOT ALIGNED" in report:
            logging.error("\n%s", report)
            raise ValueError("Demand and supply rasters are not aligned. See log above.")
        logging.info("\n%s", report)

        # Read both rasters into memory. D is demand. A is raw attractiveness.
        logging.info("Reading demand and supply arrays into memory...")
        t0 = time.time()
        D = dsrc.read(1).astype(np.float32)
        A = ssrc.read(1).astype(np.float32)
        logging.info("Read arrays in %.1fs", time.time() - t0)

        _log_array_stats("Demand (raw)", D)
        _log_array_stats("Supply (raw)", A)

        # Treat input NoData as zero contribution. For demand, NoData means no
        # population/demand to allocate. For supply, NoData means no available
        # attractiveness.
        logging.info("Applying nodata masks (treated as 0)...")
        if dsrc.nodata is not None:
            D = np.where(D == dsrc.nodata, 0.0, D)
        if ssrc.nodata is not None:
            A = np.where(A == ssrc.nodata, 0.0, A)

        # Negative demand or attractiveness has no meaningful interpretation here,
        # so force it to zero before the allocation starts.
        logging.info("Clipping negatives to 0 for inputs...")
        D = np.clip(D, 0.0, None)
        A = np.clip(A, 0.0, None)

        # Optional access adjustment. This discounts attractive sites that are far
        # from the relevant access raster, using the half-distance and gamma
        # parameters loaded above.
        if ACCESS_HALF_M is not None:
            if not os.path.exists(ACCESS_PROX_RASTER):
                raise FileNotFoundError(
                    f"Access adjustment enabled but proximity raster not found: {ACCESS_PROX_RASTER}"
                )

            logging.info("Applying accessibility adjustment to attractiveness...")
            with rasterio.open(ACCESS_PROX_RASTER) as rsrc:
                if (
                    rsrc.width != dsrc.width or
                    rsrc.height != dsrc.height or
                    rsrc.transform != dsrc.transform or
                    rsrc.crs != dsrc.crs
                ):
                    raise ValueError("Access proximity raster must be aligned with demand/supply rasters.")

                R = rsrc.read(1).astype(np.float32)
                if rsrc.nodata is not None:
                    R = np.where(R == rsrc.nodata, np.nan, R)

            Wacc = road_accessibility_weight(
                np.where(np.isnan(R), 1e9, R),
                half_m=float(ACCESS_HALF_M),
                gamma=float(ACCESS_GAMMA if ACCESS_GAMMA is not None else 2.0),
            )

            _log_array_stats("Access distance (m)", np.where(np.isnan(R), 0.0, R))
            _log_array_stats("Accessibility weight", Wacc)

            A = (A * Wacc).astype(np.float32)
            A = np.clip(A, 0.0, None)
            _log_array_stats("Supply (after access adj)", A)

        _log_array_stats("Demand (clean)", D)
        _log_array_stats("Supply (clean)", A)

        # Convert attractiveness into effective supply. Alpha allows stronger or
        # weaker concentration of demand around high-attractiveness cells.
        logging.info("Computing supply weights S = A^alpha (alpha=%.2f)...", ALPHA)
        if ALPHA != 1.0:
            S = np.power(A, ALPHA, dtype=np.float32)
        else:
            S = A.copy()

        _log_array_stats("S", S)

        # Build a distance-decay kernel in raster cells. This controls how demand
        # spreads across nearby supply cells.
        logging.info("Building distance decay kernel...")
        pixel_size_m = abs(dsrc.transform.a)
        if dsrc.crs and dsrc.crs.is_geographic:
            raise ValueError("CRS is geographic (degrees). Reproject to a projected CRS (metres).")

        K = build_decay_kernel(pixel_size_m, MEDIAN_DISTANCE_KM, BETA, KERNEL_CUTOFF_KM)

        # Stage 1:
        # Z is the total accessible weighted supply around each demand cell.
        # It is the denominator used to split demand among nearby supply cells.
        logging.info("Stage 1/3: Computing Z = conv(S, K)...")
        Z = fft_convolve2d_same(S, K)
        _clean_finite_inplace(Z, fill_value=0.0)
        _log_array_stats("Z", Z)

        # Stage 2:
        # P is demand divided by accessible supply. Cells with no meaningful nearby
        # supply are left as zero to avoid division by tiny numerical scraps.
        logging.info("Stage 2/3: Computing P = D / Z where Z > z_eps...")

        P = np.zeros_like(D, dtype=np.float32)
        _clean_finite_inplace(Z, fill_value=0.0)

        z_max = float(np.max(Z)) if Z.size else 0.0
        z_eps = max(float(Z_EPS_ABS), float(Z_EPS_REL) * z_max)

        mask = Z > z_eps
        reachable_pct = 100.0 * float(np.count_nonzero(mask)) / float(mask.size)
        logging.info("Z max=%g | z_eps=%g | reachable cells: %.2f%%", z_max, z_eps, reachable_pct)

        P_div = np.zeros_like(D, dtype=np.float64)
        P_div[mask] = D[mask].astype(np.float64) / Z[mask].astype(np.float64)

        P = P_div.astype(np.float32)
        _clean_finite_inplace(P, fill_value=0.0)
        _log_array_stats("P", P)

        # Stage 3:
        # Spread P back over the landscape with the same distance kernel, then
        # multiply by supply. This gives the final allocated demand captured by
        # each supply cell.
        logging.info("Stage 3/3: Computing KP = conv(P, K) and Alloc = S * KP...")
        KP = fft_convolve2d_same(P, K)
        _clean_finite_inplace(KP, fill_value=0.0)

        Alloc64 = S.astype(np.float64) * KP.astype(np.float64)
        _clean_finite_inplace(Alloc64, fill_value=0.0)
        Alloc = _clip_small_negatives_to_zero(Alloc64, eps_rel=EPS_REL)

        _log_array_stats("KP", KP)
        _log_array_stats("Alloc (clean)", Alloc)

        # Optional diagnostic output:
        # Unallocated demand is demand in cells that had no reachable supply.
        Unalloc = None
        has_unallocated = False

        if OUTPUT_UNALLOC_RASTER:
            logging.info("Checking for unallocated demand (where Z == 0 and D > 0)...")
            unalloc_mask = (~mask) & (D > 0)

            if np.any(unalloc_mask):
                has_unallocated = True
                Unalloc = np.zeros_like(D, dtype=np.float32)
                Unalloc[unalloc_mask] = D[unalloc_mask]

                logging.info(
                    "Unallocated demand found in %d cells (total demand = %.3f)",
                    int(np.count_nonzero(unalloc_mask)),
                    float(np.sum(Unalloc))
                )
                _log_array_stats("Unalloc", Unalloc)
            else:
                logging.info("No unallocated demand detected. Skipping unallocated raster.")

        # Prepare GeoTIFF profile for outputs. This keeps the demand raster grid
        # and CRS, but changes datatype, NoData, compression and tiling settings.
        logging.info("Writing outputs...")
        profile = dsrc.profile.copy()
        profile.update(
            driver="GTiff",
            dtype=OUTPUT_DTYPE,
            count=1,
            nodata=NODATA_OUT,
            compress=COMPRESS,
            predictor=PREDICTOR,
            tiled=TILED,
            blockxsize=BLOCK if TILED else None,
            blockysize=BLOCK if TILED else None,
            BIGTIFF="IF_SAFER",
        )
        profile.pop("photometric", None)

        # Ensure strictly finite output before writing. Any NaN or Inf is treated
        # as zero because rogue numerical gremlins do not get voting rights.
        _clean_finite_inplace(Alloc, fill_value=0.0)

        t_write = time.time()
        written_path = safe_write_raster(
            OUTPUT_ALLOC_RASTER,
            profile,
            Alloc
        )
        logging.info("Wrote alloc raster to %s in %.1fs", written_path, time.time() - t_write)

        if OUTPUT_UNALLOC_RASTER and has_unallocated:
            logging.info("Writing unallocated raster...")
            _clean_finite_inplace(Unalloc, fill_value=0.0)

            with rasterio.open(OUTPUT_UNALLOC_RASTER, "w", **profile) as dst:
                dst.write(Unalloc.astype(np.float32), 1)

            logging.info("Unallocated raster written.")

            safe_write_raster(
                OUTPUT_UNALLOC_RASTER,
                Unalloc,
                nodata_value=NODATA_OUT
            )

    logging.info("Done. Total runtime: %.1fs", time.time() - t_all)

    print("Wrote:", OUTPUT_ALLOC_RASTER)
    if OUTPUT_UNALLOC_RASTER and has_unallocated:
        print("Wrote:", OUTPUT_UNALLOC_RASTER)


if __name__ == "__main__":
    main()