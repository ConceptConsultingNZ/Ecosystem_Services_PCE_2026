"""
Spatially allocate hunting effort using a distance-decay (exponential) kernel per species.
This routine treats the input hunter population raster as the origin of potential hunting trips.
For each species, it builds an exponential distance-decay kernel where the *median trip distance*
(x50) is the distance at which the kernel weight falls to 0.5:

    w(d) = exp(-ln(2) * d / x50)

The hunter population raster is then convolved with this kernel (FFT-based) to produce a
spatially smoothed "hunting effort weight" surface. The output is finally masked to permit
areas only (cells with value 1 in the permit raster). Cells outside permit areas are written
as nodata.

Parameters
----------
base_dir        Directory where outputs will be written.
pop_raster        Path to a raster of hunter population (can be fractional). Nodata is treated as zero.
permit_raster        Path to a raster defining permit areas (1 in permit areas, nodata elsewhere).
x50_km_by_species        Mapping of species name -> median trip distance (kilometres).
radius_mult        Kernel radius as a multiple of x50. For example, 3.0 gives ~12.5% of peak weight at the edge.
out_nodata        Nodata value for output rasters.

Outputs
-------
Writes one GeoTIFF per species to:        {base_dir}/hunting_weight_{species}.tif

Notes
- Assumes a projected CRS in metres (e.g., EPSG:2193) for correct distance calculations.
- Uses FFT convolution with zero-padding beyond raster bounds.
"""

import os
import math
import time
import logging
import numpy as np
import rasterio

# ------------------------------------------------------------
# Constants (edit these)
# ------------------------------------------------------------
BASE_DIR = r"<PROJECT_DIRECTORY>\Provisioning\Hunting\Intermediate"

POP_RASTER = os.path.join(BASE_DIR, "big_game_hunter_pop.tif")       # hunter population raster (can be fractional)
PERMIT_RASTER = os.path.join(BASE_DIR, "hunting_permit_areas.tif")   # 1 in permit areas, else nodata
OUTPUT_DIR = BASE_DIR
OUTPUT_PREFIX = "hunting_weight_"

# Species-specific median trip distances (50% of trips within x km)
X50_KM = {
    "deer": 150,
    "pigs": 50,
    "goats": 100,
    "tahr_chamois": 300,
}

# Kernel radius as a multiple of x50 (3x gives ~12.5% weight at the edge)
RADIUS_MULT = 3.0

# Output nodata and GeoTIFF creation options
OUT_NODATA = -9999.0
TILED = True
BLOCKXSIZE = 256
BLOCKYSIZE = 256
COMPRESS = "DEFLATE"
PREDICTOR = 3
ZLEVEL = 9

# Atomic-write behaviour (prevents zombie/corrupt GeoTIFFs if a write fails)
WRITE_TEMP_SUFFIX = ".tmp"

def allocate_hunting_effort_distance_decay(
    pop_raster: str,
    permit_raster: str,
    output_dir: str,
    x50_km_by_species: dict,
    radius_mult: float = RADIUS_MULT,
    out_nodata: float = OUT_NODATA,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:

    # -----------------------------
    # Logging setup
    # -----------------------------
    logger = logging.getLogger("hunting_distance_decay")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def build_exponential_kernel(pixel_size_m: float, x50_km: float, radius_mult_local: float) -> np.ndarray:
        """
        Exponential kernel w(d)=exp(-ln(2)*d/x50) so w(x50)=0.5.
        Builds a square kernel with radius = radius_mult * x50.
        Kernel is normalised to sum to 1.
        """
        x50_m = x50_km * 1000.0
        k = math.log(2.0) / x50_m

        radius_m = radius_mult_local * x50_m
        radius_px = int(math.ceil(radius_m / pixel_size_m))
        size = 2 * radius_px + 1  # (kept for readability)

        y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        dist_m = np.sqrt(x * x + y * y) * pixel_size_m

        kernel = np.exp(-k * dist_m).astype(np.float32)

        s = float(kernel.sum())
        if s > 0:
            kernel /= s

        return kernel

    def fft_convolve2d_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        FFT-based 2D convolution returning output cropped to the same shape as `a`.
        Uses zero-padding outside bounds.
        """
        a = a.astype(np.float32, copy=False)
        k = k.astype(np.float32, copy=False)

        out_shape = (a.shape[0] + k.shape[0] - 1, a.shape[1] + k.shape[1] - 1)

        fa = np.fft.rfftn(a, s=out_shape, axes=(0, 1))
        fk = np.fft.rfftn(k, s=out_shape, axes=(0, 1))
        conv_full = np.fft.irfftn(fa * fk, s=out_shape, axes=(0, 1)).astype(np.float32)

        ky, kx = k.shape
        y0 = (ky - 1) // 2
        x0 = (kx - 1) // 2
        return conv_full[y0:y0 + a.shape[0], x0:x0 + a.shape[1]]

    def assert_same_grid(a_path: str, b_path: str) -> None:
        with rasterio.open(a_path) as A, rasterio.open(b_path) as B:
            if (A.width != B.width) or (A.height != B.height):
                raise ValueError("Input rasters do not have the same width/height.")
            if A.transform != B.transform:
                raise ValueError("Input rasters do not have the same transform (grid alignment).")
            if A.crs != B.crs:
                raise ValueError("Input rasters do not have the same CRS.")

    def atomic_write_geotiff(path: str, profile: dict, array2d: np.ndarray) -> None:
        """
        Write to a temp file then atomically replace the target, avoiding corrupted leftovers.
        """
        tmp_path = path + WRITE_TEMP_SUFFIX

        # Clean up any prior temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(array2d, 1)

        # Atomic replace on Windows (also overwrites if target exists)
        os.replace(tmp_path, path)

    # -----------------------------
    # Main
    # -----------------------------
    t0 = time.perf_counter()
    logger.info("Starting hunting effort allocation (distance-decay).")
    logger.info("Population raster: %s", pop_raster)
    logger.info("Permit raster: %s", permit_raster)
    logger.info("Output dir: %s", output_dir)
    logger.info("Species count: %d", len(x50_km_by_species))

    logger.info("Checking grid alignment...")
    assert_same_grid(pop_raster, permit_raster)
    logger.info("Grid alignment OK.")

    logger.info("Reading population raster...")
    with rasterio.open(pop_raster) as pop_src:
        pop = pop_src.read(1).astype(np.float32)
        pop_profile = pop_src.profile.copy()
        pop_nodata = pop_src.nodata
        transform = pop_src.transform
    logger.info("Population raster read (shape=%s, nodata=%s).", pop.shape, str(pop_nodata))

    if pop_nodata is not None:
        pop = np.where(pop == pop_nodata, 0.0, pop)
    else:
        pop = np.nan_to_num(pop, nan=0.0)

    logger.info("Reading permit raster...")
    with rasterio.open(permit_raster) as perm_src:
        permit = perm_src.read(1)
        perm_nodata = perm_src.nodata
    logger.info("Permit raster read (nodata=%s).", str(perm_nodata))

    if perm_nodata is None:
        permit_mask = (permit == 1)
    else:
        permit_mask = (permit != perm_nodata) & (permit == 1)

    permit_cells = int(np.count_nonzero(permit_mask))
    total_cells = int(permit_mask.size)
    logger.info("Permit mask: %d of %d cells (%.2f%%).", permit_cells, total_cells, 100.0 * permit_cells / total_cells)

    pixel_size_m = float(abs(transform.a))
    if not np.isclose(abs(transform.e), pixel_size_m):
        logger.warning(
            "Non-square pixels detected (x=%s, y=%s). Using x pixel size for kernel distances.",
            str(abs(transform.a)),
            str(abs(transform.e)),
        )
    logger.info("Pixel size used for distance calculations: %.3f m.", pixel_size_m)

    # Output profile
    out_profile = pop_profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=out_nodata,
        compress=COMPRESS,
        predictor=PREDICTOR,
        zlevel=ZLEVEL,
        tiled=TILED,
    )
    if TILED:
        out_profile.update(
            blockxsize=BLOCKXSIZE,
            blockysize=BLOCKYSIZE,
        )

    for i, (species, x50_km) in enumerate(x50_km_by_species.items(), start=1):
        s0 = time.perf_counter()
        logger.info("[%d/%d] Species=%s | x50=%.1f km | radius_mult=%.2f", i, len(x50_km_by_species), species, x50_km, radius_mult)

        logger.info("Building kernel...")
        kernel = build_exponential_kernel(pixel_size_m=pixel_size_m, x50_km=x50_km, radius_mult_local=radius_mult)
        logger.info("Kernel built (size=%d x %d, sum=%.6f).", kernel.shape[0], kernel.shape[1], float(kernel.sum()))

        logger.info("Convolving population raster (FFT)...")
        c0 = time.perf_counter()
        smoothed = fft_convolve2d_same(pop, kernel)
        logger.info("Convolution done in %.2f s.", time.perf_counter() - c0)

        logger.info("Applying permit mask...")
        out = np.full(smoothed.shape, out_nodata, dtype=np.float32)
        out[permit_mask] = smoothed[permit_mask]

        out_path = os.path.join(output_dir, f"{output_prefix}{species}.tif")
        logger.info("Writing output (atomic replace)...")
        atomic_write_geotiff(out_path, out_profile, out)

        logger.info("Wrote: %s (species time: %.2f s)", out_path, time.perf_counter() - s0)

    logger.info("All species weight rasters created in %.2f s.", time.perf_counter() - t0)


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
allocate_hunting_effort_distance_decay(
    pop_raster=POP_RASTER,
    permit_raster=PERMIT_RASTER,
    output_dir=OUTPUT_DIR,
    x50_km_by_species=X50_KM,
)
