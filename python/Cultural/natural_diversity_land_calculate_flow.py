"""
Calculate land biodiversity / natural heritage value rasters.

The script combines wetland and native forest spatial layers with scarcity
multipliers derived from percentage remaining layers. Wetlands and native forest
each receive low, central, and high base values in NZD/ha/year. Scarcity
multipliers increase the value where a smaller percentage of the relevant
ecosystem remains.

Inputs are expected to be aligned rasters with the same extent, resolution,
projection, and cell order. Wetland and native forest rasters are assumed to be
binary masks, where 1 indicates presence and 0 indicates absence.
The pct remaining rasters range from 0 to 100.

Where wetland and native forest overlap, wetland takes priority.
"""
import os
import time
import logging
import numpy as np
import rasterio
import sys
from pathlib import Path

SCRIPT_FOLDER = Path(__file__).resolve().parent
PARENT_FOLDER = SCRIPT_FOLDER.parent

if str(PARENT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PARENT_FOLDER))

from raster_utils import write_float32_raster


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)
start_time = time.perf_counter()


INPUT_FOLDER = r"<PROJECT_DIRECTORY>\Cultural\Natural heritage\Intermediate"
OUTPUT_FOLDER = r"<PROJECT_DIRECTORY>\Cultural\Natural heritage\Output"
OUTPUT_FILENAME = "land_biodiversity_flow_value"

# All values adjusted to NZD 2026
WETLAND_VALUE_LOW = 1666.0  # Patterson & Cole 1999 passive value for wetland
WETLAND_VALUE_CENTRAL = 2814.0  # de Groot 2007 - genetic diversity of fresh wetlands
WETLAND_VALUE_HIGH = 16000.0  # Kirkland 1988 valuation study for Whangamarino

FOREST_VALUE_LOW = 346.86  # Patterson & Cole 2013 passive value for forest park
FOREST_VALUE_CENTRAL = 2077.0  # de Groot 2007 - genetic diversity of temperate forest
FOREST_VALUE_HIGH = 4128.48  # Patterson & Cole 2013 passive value for national parks

NODATA = -9999.0
DEFAULT_WETLAND_REMAINING_PCT = 25.0
DEFAULT_NATURAL_REMAINING_PCT = 25.0

wetland_path = os.path.join(INPUT_FOLDER, "wetland.tif")
native_forest_path = os.path.join(INPUT_FOLDER, "native_forest.tif")
wetland_remaining_path = os.path.join(INPUT_FOLDER, "wetland_remaining_pct.tif")
natural_pct_path = os.path.join(INPUT_FOLDER, "natural_pct.tif")


def fill_nodata_with_default(arr, nodata, default_value, name):
    """
    Replace NoData and non-finite values in an array with a default value.

    This is used for scarcity percentage rasters so that missing scarcity data
    defaults to no scarcity adjustment rather than forcing the output cell to
    NoData.
    """
    logger.info(
        "Filling NoData in %s with default value %.2f",
        name,
        default_value,
    )

    t0 = time.perf_counter()

    missing = ~np.isfinite(arr)

    if nodata is not None:
        missing |= arr == nodata

    missing_count = int(missing.sum())

    if missing_count > 0:
        arr = arr.copy()
        arr[missing] = default_value

    logger.info(
        "Filled %s missing cells in %s in %.1fs",
        f"{missing_count:,}",
        name,
        time.perf_counter() - t0,
    )

    log_array_stats(f"{name}_filled", arr)

    return arr.astype("float32")

def elapsed():
    """Return elapsed runtime as a readable string."""
    seconds = time.perf_counter() - start_time
    return f"{seconds:,.1f}s"


def log_array_stats(name, arr, nodata=None):
    """Log basic array statistics, excluding NoData and non-finite values."""
    valid = np.isfinite(arr)

    if nodata is not None:
        valid &= arr != nodata

    valid_count = int(valid.sum())
    total_count = arr.size

    if valid_count == 0:
        logger.info("%s: no valid cells", name)
        return

    logger.info(
        "%s: shape=%s, valid=%s/%s, min=%.4f, max=%.4f, mean=%.4f",
        name,
        arr.shape,
        f"{valid_count:,}",
        f"{total_count:,}",
        float(np.nanmin(arr[valid])),
        float(np.nanmax(arr[valid])),
        float(np.nanmean(arr[valid])),
    )


def read_raster(path, name):
    """Read a single-band raster as Float32 and return array, NoData value, and profile."""
    logger.info("Reading %s: %s", name, path)

    t0 = time.perf_counter()
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        arr = src.read(1).astype("float32")
        nodata = src.nodata

    logger.info("Finished reading %s in %.1fs", name, time.perf_counter() - t0)
    log_array_stats(name, arr, nodata)

    return arr, nodata, profile


def scarcity_multiplier(pct_remaining):
    """Return scarcity multiplier based on percentage of ecosystem remaining."""
    logger.info("Calculating scarcity multiplier")

    return np.select(
        [
            pct_remaining < 1,
            pct_remaining < 5,
            pct_remaining < 10,
            pct_remaining < 25,
            pct_remaining < 50,
        ],
        [1, 0.95, 0.85, 0.75, 0.6],
        default=0.5,
    ).astype("float32")


def calculate_biodiversity_value(
    label,
    wetland,
    native_forest,
    wetland_multiplier,
    forest_multiplier,
    valid,
    wetland_base_value,
    forest_base_value,
):
    """Calculate biodiversity value raster for a given wetland and forest base value."""
    logger.info(
        "Calculating %s raster: wetland value=%s, forest value=%s",
        label,
        wetland_base_value,
        forest_base_value,
    )

    t0 = time.perf_counter()

    result = np.zeros(wetland.shape, dtype="float32")

    wetland_mask = (wetland == 1) & valid
    forest_mask = (native_forest == 1) & valid & (wetland != 1)

    logger.info(
        "%s masks: wetland cells=%s, forest cells=%s",
        label,
        f"{int(wetland_mask.sum()):,}",
        f"{int(forest_mask.sum()):,}",
    )

    result[wetland_mask] = wetland_base_value * wetland_multiplier[wetland_mask]
    result[forest_mask] = forest_base_value * forest_multiplier[forest_mask]
    result[~valid] = NODATA

    logger.info("Finished calculating %s raster in %.1fs", label, time.perf_counter() - t0)
    log_array_stats(f"{label}_result", result, NODATA)

    return result.astype("float32")


logger.info("Starting land biodiversity calculation")
logger.info("Input folder: %s", INPUT_FOLDER)
logger.info("Output folder: %s", OUTPUT_FOLDER)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

wetland, wetland_nodata, profile = read_raster(wetland_path, "wetland")
native_forest, forest_nodata, _ = read_raster(native_forest_path, "native_forest")
wetland_remaining_pct, wetland_remaining_nodata, _ = read_raster(
    wetland_remaining_path,
    "wetland_remaining_pct",
)
natural_pct, natural_pct_nodata, _ = read_raster(natural_pct_path, "natural_pct")

#Fill nodata values
wetland_remaining_pct = fill_nodata_with_default(
    arr=wetland_remaining_pct,
    nodata=wetland_remaining_nodata,
    default_value=DEFAULT_WETLAND_REMAINING_PCT,
    name="wetland_remaining_pct",
)

natural_pct = fill_nodata_with_default(
    arr=natural_pct,
    nodata=natural_pct_nodata,
    default_value=DEFAULT_NATURAL_REMAINING_PCT,
    name="natural_pct",
)

logger.info("Building valid-data mask")

#Convert nodata to zero
wetland = np.where(wetland == wetland_nodata, 0, wetland)
native_forest = np.where(native_forest == forest_nodata, 0, native_forest)

valid = np.isfinite(wetland) & np.isfinite(native_forest)

logger.info("Final valid cells: %s/%s", f"{int(valid.sum()):,}", f"{valid.size:,}")

logger.info("Calculating wetland scarcity multiplier")
wetland_mult = scarcity_multiplier(wetland_remaining_pct)
log_array_stats("wetland_mult", wetland_mult)

logger.info("Calculating forest scarcity multiplier")
forest_mult = scarcity_multiplier(natural_pct)
log_array_stats("forest_mult", forest_mult)

central_result = calculate_biodiversity_value(
    label="central",
    wetland=wetland,
    native_forest=native_forest,
    wetland_multiplier=wetland_mult,
    forest_multiplier=forest_mult,
    valid=valid,
    wetland_base_value=WETLAND_VALUE_CENTRAL,
    forest_base_value=FOREST_VALUE_CENTRAL,
)

low_result = calculate_biodiversity_value(
    label="low",
    wetland=wetland,
    native_forest=native_forest,
    wetland_multiplier=wetland_mult,
    forest_multiplier=forest_mult,
    valid=valid,
    wetland_base_value=WETLAND_VALUE_LOW,
    forest_base_value=FOREST_VALUE_LOW,
)

high_result = calculate_biodiversity_value(
    label="high",
    wetland=wetland,
    native_forest=native_forest,
    wetland_multiplier=wetland_mult,
    forest_multiplier=forest_mult,
    valid=valid,
    wetland_base_value=WETLAND_VALUE_HIGH,
    forest_base_value=FOREST_VALUE_HIGH,
)

central_path = write_float32_raster(
    array=central_result,
    output_folder=OUTPUT_FOLDER,
    output_filename= OUTPUT_FILENAME + ".tif",
    profile=profile,
    nodata=NODATA,
)
logger.info("Wrote central raster to: %s", central_path)

low_path = write_float32_raster(
    array=low_result,
    output_folder=OUTPUT_FOLDER,
    output_filename=OUTPUT_FILENAME + "_low.tif",
    profile=profile,
    nodata=NODATA,
)
logger.info("Wrote low raster to: %s", low_path)

high_path = write_float32_raster(
    array=high_result,
    output_folder=OUTPUT_FOLDER,
    output_filename=OUTPUT_FILENAME + "_high.tif",
    profile=profile,
    nodata=NODATA,
)
logger.info("Wrote high raster to: %s", high_path)

logger.info("Land biodiversity low, central, and high layers calculated successfully in %s", elapsed())