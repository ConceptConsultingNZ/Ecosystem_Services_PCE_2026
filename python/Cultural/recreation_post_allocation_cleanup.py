"""
Post-process an allocation raster by zeroing values below a magnitude threshold.

The threshold is defined as the maximum of:
1) A relative cutoff (fraction of the raster’s maximum value), and
2) An optional absolute cutoff in allocation units.

NaN and infinite values are set to zero. The output preserves the input
raster’s georeferencing and uses Float32 with configurable GeoTIFF
compression and tiling.

Intended to remove low-magnitude numerical artefacts that appear as
striping or speckle in low-demand areas without altering meaningful
allocation signal.
"""

import logging
import numpy as np
import rasterio

# ==============================
# User settings
# ==============================

INPUT_RASTER = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Output\recreation_short_walk_flow_value.tif"
OUTPUT_RASTER = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Output\recreation_short_walk_flow_value_clamped.tif"

ALLOC_CLAMP_REL = 1e-7
ALLOC_CLAMP_ABS = 0.0

COMPRESS = "DEFLATE"
PREDICTOR = 3
TILED = True
BLOCK = 512

LOG_LEVEL = logging.INFO

def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def main():
    setup_logging()

    with rasterio.open(INPUT_RASTER) as src:
        alloc = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        in_nodata = src.nodata

    # Clean NaN/Inf
    np.nan_to_num(alloc, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    max_val = float(np.max(alloc)) if alloc.size else 0.0
    thr = max(float(ALLOC_CLAMP_ABS), float(ALLOC_CLAMP_REL) * max_val)

    before_nz = int((alloc != 0).sum())
    before_pos = int((alloc > 0).sum())

    logging.info("Max allocation: %g", max_val)
    logging.info("Clamp threshold thr: %g", thr)
    logging.info("Nonzero before: %d | Positive before: %d", before_nz, before_pos)

    alloc[alloc < thr] = 0.0

    after_nz = int((alloc != 0).sum())
    after_pos = int((alloc > 0).sum())

    logging.info("Nonzero after: %d | Positive after: %d", after_nz, after_pos)

    # Choose output nodata:
    # - Preserve input nodata if it exists and is not 0.
    # - Otherwise use -9999 so that zeros are not treated as nodata by viewers.
    if in_nodata is not None and float(in_nodata) != 0.0:
        out_nodata = float(in_nodata)
    else:
        out_nodata = -9999.0

    profile.update(
        dtype="float32",
        nodata=out_nodata,
        compress=COMPRESS,
        predictor=PREDICTOR,
        tiled=TILED,
        blockxsize=BLOCK if TILED else None,
        blockysize=BLOCK if TILED else None,
        BIGTIFF="IF_SAFER",
    )
    profile.pop("photometric", None)

    with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
        dst.write(alloc.astype(np.float32), 1)

    logging.info("Wrote %s (nodata=%s)", OUTPUT_RASTER, out_nodata)

if __name__ == "__main__":
    main()