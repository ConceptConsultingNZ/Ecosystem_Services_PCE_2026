"""
Build an attractiveness raster by mapping LCDB class codes in a raster
to attractiveness values from a CSV lookup table.

Inputs:
- LCDB class raster clipped/masked to track influence zone (Int16/Int32)
- CSV with columns: class_code, attractiveness  (at minimum)

Output:
- attractiveness raster (Float32) aligned to input raster
"""

import os
import logging
import numpy as np
import pandas as pd
import rasterio

# ==============================
# Constants (edit these)
# ==============================
INPUT_RASTER = r"D:\tmp\doc_track_lcdb.tif"
ATTRACTIVENESS_CSV = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate\lcdb6_attractiveness.csv"

OUTPUT_DIR = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate"
OUTPUT_FILENAME = "attractiveness_tramping.tif"
OUTPUT_RASTER = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# CSV column names
CSV_CLASS_COL = "class"
CSV_ATTR_COL = "tramping"

# Output settings
OUTPUT_DTYPE = "float32"
NODATA_OUT = -9999.0
COMPRESS = "DEFLATE"
PREDICTOR = 3  # good for float
ZLEVEL = 9
TILED = True
BLOCK = 512

# Logging
LOG_LEVEL = logging.INFO

# ==============================
# Implementation
# ==============================

def setup_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_lookup(csv_path: str) -> dict[int, float]:
    """
    Loads a lookup dictionary: {class_code: attractiveness}.
    CSV must include CSV_CLASS_COL and CSV_ATTR_COL.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if CSV_CLASS_COL not in df.columns or CSV_ATTR_COL not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{CSV_CLASS_COL}' and '{CSV_ATTR_COL}'. "
            f"Found: {list(df.columns)}"
        )

    df = df[[CSV_CLASS_COL, CSV_ATTR_COL]].copy()
    df[CSV_CLASS_COL] = df[CSV_CLASS_COL].astype(int)
    df[CSV_ATTR_COL] = df[CSV_ATTR_COL].astype(float)

    lookup = dict(zip(df[CSV_CLASS_COL].values, df[CSV_ATTR_COL].values))

    if len(lookup) == 0:
        raise ValueError("Lookup table is empty.")

    return lookup


def apply_lookup(arr: np.ndarray, lookup: dict[int, float], nodata_in: float | int | None) -> np.ndarray:
    """
    Map class codes in arr to attractiveness values using lookup.
    Any code not found in lookup gets NODATA_OUT.
    """
    out = np.full(arr.shape, NODATA_OUT, dtype=np.float32)

    # Valid mask: exclude nodata if defined
    if nodata_in is None:
        valid = np.ones(arr.shape, dtype=bool)
    else:
        valid = arr != nodata_in

    # Also treat zeros as "no track / no class" by default.
    # If 0 is a valid class code in your raster, remove this line.
    valid &= (arr != 0)

    if not np.any(valid):
        return out

    vals = arr[valid].astype(np.int32)

    # Fast mapping via vectorised unique codes
    uniq = np.unique(vals)
    missing = [int(c) for c in uniq if int(c) not in lookup]

    if missing:
        logging.warning(
            "Found %d class codes in raster that are missing from the CSV lookup. "
            "They will be set to NODATA_OUT. Examples: %s",
            len(missing),
            missing[:20],
        )

    for code in uniq:
        code_int = int(code)
        if code_int in lookup:
            out[valid & (arr == code)] = float(lookup[code_int])

    return out


def main() -> None:
    setup_logging()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("Input raster: %s", INPUT_RASTER)
    logging.info("Lookup CSV:   %s", ATTRACTIVENESS_CSV)
    logging.info("Output raster:%s", OUTPUT_RASTER)

    lookup = load_lookup(ATTRACTIVENESS_CSV)
    logging.info("Loaded %d class->attractiveness mappings", len(lookup))

    with rasterio.open(INPUT_RASTER) as src:
        profile = src.profile.copy()
        nodata_in = src.nodata
        logging.info("Input nodata: %s", str(nodata_in))

        arr = src.read(1)

        out = apply_lookup(arr, lookup, nodata_in)

        profile.update(
            driver="GTiff",
            dtype=OUTPUT_DTYPE,
            count=1,
            nodata=NODATA_OUT,
            compress=COMPRESS,
            predictor=PREDICTOR,
            zlevel=ZLEVEL,
            tiled=TILED,
            blockxsize=BLOCK if TILED else None,
            blockysize=BLOCK if TILED else None,
            BIGTIFF="IF_SAFER",
        )
        profile.pop("photometric", None)

        with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
            dst.write(out.astype(np.float32), 1)

    print("Attractiveness raster written:", OUTPUT_RASTER)


if __name__ == "__main__":
    main()