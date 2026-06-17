"""
# These NZEnvDS layers provide annual precipitation plus precipitation totals for
# selected climate-defined quarters, but they do not identify which calendar
# months make up the wettest, driest, warmest, or coldest quarter at each pixel.
# The script therefore uses assumed calendar quarters as a seasonal template,
# then fits monthly proxy rasters so that their annual total matches annual
# precipitation and their three-month sums approximate the available quarterly
# precipitation constraints. PET is derived as annual precipitation divided by
# the rainfall-to-PET ratio, then distributed across months using an assumed
# seasonal PET curve. These outputs are therefore proxy monthly climate layers
# for InVEST SWY, not observed monthly climatology rasters.

Inputs:
    - annual precipitation raster
    - precipitation of warmest quarter raster
    - precipitation of coldest quarter raster
    - precipitation of wettest quarter raster
    - precipitation of driest quarter raster
    - rainfall-to-potential-evapotranspiration ratio raster

Outputs:
    - 12 monthly precipitation rasters
    - 12 monthly PET rasters

Important:
    This creates proxy monthly layers. It cannot infer the actual wettest/driest months
    unless those month identities are supplied separately.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import logging
import time

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
INPUT_DIR = Path(r"D:\Data\LRIS\nzenvds")

OUTPUT_DIR_PRECIP = Path(r"D:\InVEST\Seasonal water yield\precip")
OUTPUT_DIR_EVT = Path(r"D:\InVEST\Seasonal water yield\evt")

INPUTS = {
    "precip_annual": INPUT_DIR / "nzenvds-total-annual-precipitation-v10" / "nzenvds-total-annual-precipitation-v10.tif",
    "precip_warm_q": INPUT_DIR / "nzenvds-precipitation-of-the-warmest-quarter-v10" / "nzenvds-precipitation-of-the-warmest-quarter-v10.tif",
    "precip_cold_q": INPUT_DIR / "nzenvds-precipitation-of-the-coldest-quarter-v10" / "nzenvds-precipitation-of-the-coldest-quarter-v10.tif",
    "precip_wet_q": INPUT_DIR / "nzenvds-precipitation-of-the-wettest-quarter-v10" / "nzenvds-precipitation-of-the-wettest-quarter-v10.tif",
    "precip_dry_q": INPUT_DIR / "nzenvds-precipitation-of-the-driest-quarter-v10" / "nzenvds-precipitation-of-the-driest-quarter-v10.tif",
    "rain_to_pet_ratio": INPUT_DIR / "nzenvds-rainfall-to-potential-evapotranspiration-ratio-v10" / "nzenvds-rainfall-to-potential-evapotranspiration-ratio-v10.tif",
}

OUTPUT_DIR = Path("output_monthly_climate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Month numbers: Jan = 1, Feb = 2, ..., Dec = 12.
#
# These are assumptions, not facts from the raster inputs.
# Change these if you want a different seasonal pattern.
WARMEST_QUARTER_START_MONTH = 1   # Jan-Feb-Mar
COLDEST_QUARTER_START_MONTH = 6   # Jun-Jul-Aug
DRIEST_QUARTER_START_MONTH = 1    # Jan-Feb-Mar
WETTEST_QUARTER_START_MONTH = 6   # Jun-Jul-Aug

# Relative importance of matching each quarterly precipitation constraint.
# Annual is always treated strongly.
#
# For SWY, wettest/driest quarter information is usually more relevant than
# warmest/coldest quarter precipitation, so those get slightly higher weights.
CONSTRAINT_WEIGHTS = {
    "annual": 100.0,
    "warm_q": 10.0,
    "cold_q": 10.0,
    "wet_q": 25.0,
    "dry_q": 25.0,
}

# PET seasonal shape.
# Higher values mean stronger summer/winter PET contrast.
# 0.35 is moderate. Try 0.5 for stronger seasonality.
PET_SEASONAL_AMPLITUDE = 0.45

# PET peak month. For New Zealand, January is usually a reasonable proxy.
PET_PEAK_MONTH = 1

OUTPUT_NODATA = -9999.0

LOG_EVERY_N_BLOCKS = 25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def read_aligned(ds, ref, window, resampling=Resampling.bilinear):
    """
    Read a source raster aligned to the reference raster window.

    This lets input NZEnvDS rasters have different dimensions, transforms,
    or extents. The source raster is reprojected/resampled on the fly onto
    the annual precipitation grid, which is used as the modelling grid.
    """
    destination = np.full(
        (window.height, window.width),
        OUTPUT_NODATA,
        dtype=np.float64
    )

    window_transform = ref.window_transform(window)

    reproject(
        source=rasterio.band(ds, 1),
        destination=destination,
        src_transform=ds.transform,
        src_crs=ds.crs,
        src_nodata=ds.nodata,
        dst_transform=window_transform,
        dst_crs=ref.crs,
        dst_nodata=OUTPUT_NODATA,
        resampling=resampling,
    )

    return destination

def quarter_month_indices(start_month: int) -> list[int]:
    """
    Return zero-based month indices for a 3-month quarter.
    Jan = 1, Dec = 12.
    """
    return [((start_month - 1 + i) % 12) for i in range(3)]

def build_constraint_matrix() -> tuple[np.ndarray, list[str]]:
    """
    Build the matrix A where A @ monthly_values gives:
        annual total
        warm quarter total
        cold quarter total
        wet quarter total
        dry quarter total
    """
    constraints = []

    annual = np.ones(12, dtype=np.float64)
    constraints.append(("annual", annual))

    warm = np.zeros(12, dtype=np.float64)
    warm[quarter_month_indices(WARMEST_QUARTER_START_MONTH)] = 1.0
    constraints.append(("warm_q", warm))

    cold = np.zeros(12, dtype=np.float64)
    cold[quarter_month_indices(COLDEST_QUARTER_START_MONTH)] = 1.0
    constraints.append(("cold_q", cold))

    wet = np.zeros(12, dtype=np.float64)
    wet[quarter_month_indices(WETTEST_QUARTER_START_MONTH)] = 1.0
    constraints.append(("wet_q", wet))

    dry = np.zeros(12, dtype=np.float64)
    dry[quarter_month_indices(DRIEST_QUARTER_START_MONTH)] = 1.0
    constraints.append(("dry_q", dry))

    names = [name for name, _ in constraints]
    A = np.vstack([row for _, row in constraints])

    return A, names

def seasonal_pet_weights() -> np.ndarray:
    """
    Create monthly PET weights using a cosine curve peaking in PET_PEAK_MONTH.
    The weights are positive and sum to 1.
    """
    months = np.arange(1, 13, dtype=np.float64)

    # Cosine equals 1 at peak month and -1 six months later.
    raw = 1.0 + PET_SEASONAL_AMPLITUDE * np.cos(
        2.0 * np.pi * (months - PET_PEAK_MONTH) / 12.0
    )

    raw = np.maximum(raw, 0.01)
    return raw / raw.sum()

def precipitation_template_weights() -> np.ndarray:
    """
    Starting monthly precipitation pattern before fitting constraints.

    This uses the assumed wettest and driest quarters to create a smooth-ish
    seasonal prior. The final monthly values are adjusted against the available
    quarterly and annual rasters.
    """
    months = np.arange(1, 13, dtype=np.float64)

    # Peak rainfall around the middle month of the assumed wettest quarter.
    wet_peak = ((WETTEST_QUARTER_START_MONTH + 1 - 1) % 12) + 1

    raw = 1.0 + 0.25 * np.cos(
        2.0 * np.pi * (months - wet_peak) / 12.0
    )

    raw = np.maximum(raw, 0.01)
    return raw / raw.sum()


def prepare_solver(A: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Create a fixed linear operator for weighted ridge-style adjustment:

        minimise ||x - x0||^2 + sum_j weight_j * (A_j x - b_j)^2

    where:
        x  = 12 monthly values
        x0 = initial smooth monthly template
        b  = target annual/quarterly totals

    This does not guarantee exact quarterly matches if constraints conflict,
    but it is stable and fast for raster block processing.
    """
    weights = np.array(
        [CONSTRAINT_WEIGHTS[name] for name in names],
        dtype=np.float64
    )

    W = np.diag(weights)
    lhs = np.eye(12, dtype=np.float64) + A.T @ W @ A
    solver = np.linalg.inv(lhs) @ A.T @ W

    return solver

def fit_monthly_precip(
    p_annual: np.ndarray,
    p_warm: np.ndarray,
    p_cold: np.ndarray,
    p_wet: np.ndarray,
    p_dry: np.ndarray,
    valid: np.ndarray,
    A: np.ndarray,
    solver: np.ndarray,
    template_w: np.ndarray,
) -> np.ndarray:
    """
    Return array of shape (12, rows, cols) with monthly precipitation proxies.
    """
    rows, cols = p_annual.shape
    n = rows * cols

    annual_flat = p_annual.reshape(n)

    x0 = template_w[:, None] * annual_flat[None, :]

    b = np.vstack([
        p_annual.reshape(n),
        p_warm.reshape(n),
        p_cold.reshape(n),
        p_wet.reshape(n),
        p_dry.reshape(n),
    ])

    adjustment = solver @ (b - A @ x0)
    x = x0 + adjustment

    # Avoid negative monthly precipitation.
    x = np.maximum(x, 0.0)

    # Force annual precipitation to match exactly after clipping.
    monthly_sum = x.sum(axis=0)
    scale = np.divide(
        annual_flat,
        monthly_sum,
        out=np.zeros_like(annual_flat, dtype=np.float64),
        where=monthly_sum > 0
    )
    x *= scale[None, :]

    x[:, ~valid.reshape(n)] = OUTPUT_NODATA

    return x.reshape(12, rows, cols)


def create_monthly_pet(
    p_annual: np.ndarray,
    rain_to_pet_ratio: np.ndarray,
    valid: np.ndarray,
    pet_w: np.ndarray,
) -> np.ndarray:
    """
    Estimate annual PET from:
        PET_annual = precipitation_annual / rainfall_to_pet_ratio

    Then distribute annual PET across months using seasonal PET weights.
    """
    pet_annual = np.divide(
        p_annual,
        rain_to_pet_ratio,
        out=np.full_like(p_annual, np.nan, dtype=np.float64),
        where=(rain_to_pet_ratio > 0)
    )

    monthly_pet = pet_w[:, None, None] * pet_annual[None, :, :]
    monthly_pet[:, ~valid] = OUTPUT_NODATA

    return monthly_pet


def raster_valid_mask(arrays: list[np.ndarray], nodatas: list[float | None]) -> np.ndarray:
    """
    Build valid mask across all input rasters.
    """
    valid = np.ones(arrays[0].shape, dtype=bool)

    for arr, nodata in zip(arrays, nodatas):
        valid &= np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata

    # Physical plausibility checks.
    valid &= arrays[0] >= 0  # annual precip
    valid &= arrays[-1] > 0  # rainfall-to-PET ratio

    return valid


# ---------------------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------------------

def main():
    start_time = time.perf_counter()

    logger.info("Starting monthly precipitation and PET proxy raster creation.")
    logger.info("Opening input rasters.")

    input_paths = {k: Path(v) for k, v in INPUTS.items()}

    for name, path in input_paths.items():
        logger.info("Input %s: %s", name, path)

    datasets = {k: rasterio.open(v) for k, v in input_paths.items()}

    try:
        ref = datasets["precip_annual"]
        profile = ref.profile.copy()

        logger.info(
            "Reference raster: width=%s, height=%s, CRS=%s, nodata=%s",
            ref.width,
            ref.height,
            ref.crs,
            ref.nodata,
        )

        # Basic alignment check.
        # This reports differences only. It does not stop the script.
        # Downstream code must resample/reproject mismatched rasters if exact
        # alignment is required for block-wise raster operations.
        logger.info("Checking raster alignment against annual precipitation.")

        for name, ds in datasets.items():
            issues = []

            if ds.width != ref.width or ds.height != ref.height:
                issues.append(
                    f"dimensions differ: {ds.width}x{ds.height} "
                    f"vs reference {ref.width}x{ref.height}"
                )

            if ds.transform != ref.transform:
                issues.append(
                    f"transform differs: {ds.transform} "
                    f"vs reference {ref.transform}"
                )

            if ds.crs != ref.crs:
                issues.append(
                    f"CRS differs: {ds.crs} "
                    f"vs reference {ref.crs}"
                )

            if issues:
                logger.info(
                    "%s does not exactly match the annual precipitation raster:",
                    name,
                )
                for issue in issues:
                    logger.info("  - %s", issue)
            else:
                logger.info("%s matches the annual precipitation raster.", name)

        profile.update(
            dtype="float32",
            count=1,
            nodata=OUTPUT_NODATA,
            compress="deflate",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        logger.info("Creating output rasters.")
        logger.info("Precipitation output folder: %s", OUTPUT_DIR_PRECIP)
        logger.info("PET output folder: %s", OUTPUT_DIR_EVT)

        Path(OUTPUT_DIR_PRECIP).mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_DIR_EVT).mkdir(parents=True, exist_ok=True)

        precip_outputs = []
        pet_outputs = []

        for month in range(1, 13):
            precip_path = Path(OUTPUT_DIR_PRECIP) / f"precip_{month:02d}.tif"
            pet_path = Path(OUTPUT_DIR_EVT) / f"pet_{month:02d}.tif"

            precip_outputs.append(rasterio.open(precip_path, "w", **profile))
            pet_outputs.append(rasterio.open(pet_path, "w", **profile))

        logger.info("Preparing monthly allocation solver.")

        A, names = build_constraint_matrix()
        solver = prepare_solver(A, names)
        precip_w = precipitation_template_weights()
        pet_w = seasonal_pet_weights()

        block_windows = list(ref.block_windows(1))
        total_blocks = len(block_windows)

        logger.info("Processing %s raster blocks.", total_blocks)

        cells_total = ref.width * ref.height
        cells_processed = 0

        for block_index, (_, window) in enumerate(block_windows, start=1):
            block_start = time.perf_counter()

            p_annual = datasets["precip_annual"].read(1, window=window).astype(np.float64)
            p_warm = datasets["precip_warm_q"].read(1, window=window).astype(np.float64)
            p_cold = datasets["precip_cold_q"].read(1, window=window).astype(np.float64)
            p_wet = datasets["precip_wet_q"].read(1, window=window).astype(np.float64)
            p_dry = datasets["precip_dry_q"].read(1, window=window).astype(np.float64)
            ratio = datasets["rain_to_pet_ratio"].read(1, window=window).astype(np.float64)

            arrays = [p_annual, p_warm, p_cold, p_wet, p_dry, ratio]
            nodatas = [
                datasets["precip_annual"].nodata,
                datasets["precip_warm_q"].nodata,
                datasets["precip_cold_q"].nodata,
                datasets["precip_wet_q"].nodata,
                datasets["precip_dry_q"].nodata,
                datasets["rain_to_pet_ratio"].nodata,
            ]

            valid = raster_valid_mask(arrays, nodatas)

            monthly_precip = fit_monthly_precip(
                p_annual=p_annual,
                p_warm=p_warm,
                p_cold=p_cold,
                p_wet=p_wet,
                p_dry=p_dry,
                valid=valid,
                A=A,
                solver=solver,
                template_w=precip_w,
            )

            monthly_pet = create_monthly_pet(
                p_annual=p_annual,
                rain_to_pet_ratio=ratio,
                valid=valid,
                pet_w=pet_w,
            )

            for i in range(12):
                precip_outputs[i].write(monthly_precip[i].astype(np.float32), 1, window=window)
                pet_outputs[i].write(monthly_pet[i].astype(np.float32), 1, window=window)

            cells_processed += window.width * window.height

            block_elapsed = time.perf_counter() - block_start

            if (
                block_index == 1
                or block_index % LOG_EVERY_N_BLOCKS == 0
                or block_index == total_blocks
            ):
                elapsed = time.perf_counter() - start_time
                progress = block_index / total_blocks
                cells_progress = cells_processed / cells_total

                if progress > 0:
                    estimated_total = elapsed / progress
                    estimated_remaining = estimated_total - elapsed
                else:
                    estimated_remaining = float("nan")

                logger.info(
                    "Processed block %s/%s (%.1f%% blocks, %.1f%% cells). "
                    "Last block %.2fs. Elapsed %.1fs. Estimated remaining %.1fs.",
                    block_index,
                    total_blocks,
                    progress * 100,
                    cells_progress * 100,
                    block_elapsed,
                    elapsed,
                    estimated_remaining,
                )

        logger.info("Closing output rasters.")

        for ds in precip_outputs + pet_outputs:
            ds.close()

    finally:
        logger.info("Closing input rasters.")
        for ds in datasets.values():
            ds.close()

    elapsed_total = time.perf_counter() - start_time

    logger.info("Monthly precipitation and PET proxy rasters created.")
    logger.info("Total elapsed time: %.1f seconds.", elapsed_total)
    logger.info("Precipitation output folder: %s", Path(OUTPUT_DIR_PRECIP).resolve())
    logger.info("PET output folder: %s", Path(OUTPUT_DIR_EVT).resolve())


if __name__ == "__main__":
    main()