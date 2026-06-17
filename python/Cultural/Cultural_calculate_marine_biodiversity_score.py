#!/usr/bin/env python3
"""
Marine biodiversity scoring + monetary allocation across an SCC raster.

CSV format (wide taxa matrix):
    ["Taxa type", "Scientific name", 33, 34, 35, ...]
where each class column contains mean abundance/occurrence (blank = 0).

Method
- For each taxa group and each class:
    Shannon H over species with x>0
    Normalise to 0–1 with Hmax = ln(S_nonzero), where S_nonzero = count(x_i>0)
- Aggregate across taxa groups with geometric mean (non-zero handling).
- Allocate TOTAL_MONETARY_VALUE across raster cells proportional to biodiversity score.

Outputs
1) CSV: class_id, biodiversity_score, biodiversity_value (TOTAL by class)
2) Raster: biodiversity score (float32)
3) Raster: biodiversity value per cell (float32)
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import rasterio


# =========================
# User constants (edit me)
# =========================
TAXA_CSV = r"<PROJECT_DIRECTORY>\Cultural\Natural heritage\seafloor_mean_taxa.csv"
SCC_RASTER = r"D:\Data\DOC\New Zealand Seafloor Community Classification\SCC_combined_100m_EPSG2193.tif"

OUTPUT_DIR = r"<PROJECT_DIRECTORY>\Cultural\Natural heritage\Intermediate"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CLASS_CSV = os.path.join(OUTPUT_DIR, "marine_biodiversity_by_class.csv")
OUTPUT_SCORE_RASTER = os.path.join(OUTPUT_DIR, "heritage_marine_biodiversity_score.tif")
OUTPUT_VALUE_RASTER = os.path.join(OUTPUT_DIR, "heritage_marine_biodiversity_value.tif")

TOTAL_MONETARY_VALUE = 39171594.0  # total $ across the entire raster extent. Based on $22/household from Chunn 2013 and Rojas-Nazar 2022

# Taxa groups to include (must match CSV 'Taxa type' values)
TAXA_GROUPS = ["Macroalgae", "Demersal Fish", "Benthic invertebrates", "Reef fish"]

# Treat tiny values as 0 (optional noise gate). Set to 0.0 to disable.
MIN_VALUE_THRESHOLD = 0.0

# Output nodata for float rasters
OUTPUT_NODATA = -9999.0

# Raster compression (conservative)
OUTPUT_COMPRESS = "DEFLATE"
OUTPUT_PREDICTOR = 3  # good for float32


# =========================
# Logging
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# =========================
# Maths helpers
# =========================
def shannon_normalised(values: np.ndarray) -> float:
    """
    values: 1D non-negative array for species in ONE taxa group, ONE class.
    Uses only values > 0 for richness and Shannon.
    Returns 0..1.
    """
    if values.size == 0:
        return 0.0

    if MIN_VALUE_THRESHOLD > 0.0:
        values = values.copy()
        values[values < MIN_VALUE_THRESHOLD] = 0.0

    positive = values[values > 0]
    s = int(positive.size)
    if s <= 1:
        return 0.0

    total = float(positive.sum())
    if total <= 0.0:
        return 0.0

    p = positive / total
    h = float(-(p * np.log(p)).sum())
    hmax = math.log(s)
    if hmax <= 0.0:
        return 0.0
    return max(0.0, min(1.0, h / hmax))


def geometric_mean_nonzero(scores: Iterable[float]) -> float:
    """
    Geometric mean across group scores in [0,1].
    Uses only scores > 0 to avoid log(0). If all are 0, returns 0.
    """
    vals = [s for s in scores if s > 0.0]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def build_lut(keys: np.ndarray, values: np.ndarray) -> np.ndarray | None:
    """
    Fast lookup table for integer class IDs if max key isn't huge.
    """
    if keys.size == 0:
        return None
    max_key = int(keys.max())
    if max_key > 10_000_000:
        return None
    lut = np.zeros(max_key + 1, dtype=np.float32)
    lut[keys.astype(np.int64)] = values.astype(np.float32)
    return lut


def map_classes(arr: np.ndarray, lut: np.ndarray | None, keys_sorted: np.ndarray, vals_sorted: np.ndarray) -> np.ndarray:
    """
    Map integer class raster -> float array using LUT if available, else searchsorted.
    """
    if lut is not None:
        return lut[arr.astype(np.int64)]

    flat = arr.ravel().astype(np.int64)
    idx = np.searchsorted(keys_sorted, flat)
    out = np.zeros_like(flat, dtype=np.float32)
    ok = (idx < keys_sorted.size) & (keys_sorted[idx] == flat)
    out[ok] = vals_sorted[idx[ok]].astype(np.float32)
    return out.reshape(arr.shape)


# =========================
# Main
# =========================
def main() -> None:
    setup_logging()

    logging.info("Reading taxa CSV: %s", TAXA_CSV)
    df = pd.read_csv(TAXA_CSV)

    # Validate the first two columns exist
    if df.shape[1] < 3:
        raise ValueError("CSV must have at least 3 columns: 'Taxa type', 'Scientific name', and >=1 class column.")

    taxa_col = df.columns[0]          # "Taxa type"
    sci_col = df.columns[1]           # "Scientific name"
    class_cols = list(df.columns[2:]) # numeric headings like 33, 34, ...

    logging.info("Assuming taxa column = '%s', scientific name column = '%s'.", taxa_col, sci_col)
    logging.info("Found %d class columns.", len(class_cols))

    # Ensure class column names are ints (handles '33' vs 33)
    class_ids: List[int] = []
    class_col_rename: Dict[str, int] = {}
    for c in class_cols:
        try:
            cid = int(str(c).strip())
        except ValueError:
            raise ValueError(f"Class column heading '{c}' is not an integer. Expected headings like 33, 34, ...")
        class_ids.append(cid)
        class_col_rename[c] = cid

    df = df.rename(columns=class_col_rename)
    class_cols_int = class_ids  # now actual int column labels

    # Force numeric on class columns, blanks to 0
    df[class_cols_int] = df[class_cols_int].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Filter taxa groups
    df = df[df[taxa_col].isin(TAXA_GROUPS)].copy()
    logging.info("Rows after filtering taxa groups %s: %d", TAXA_GROUPS, len(df))

    if df.empty:
        raise ValueError("No rows left after filtering TAXA_GROUPS. Check spelling/case in TAXA_GROUPS vs CSV.")

    # Compute group scores per class
    logging.info("Computing Shannon-normalised scores per taxa group and class...")
    score_by_group: Dict[Tuple[str, int], float] = {}

    for taxa in TAXA_GROUPS:
        sub = df[df[taxa_col] == taxa]
        if sub.empty:
            logging.warning("No rows found for taxa group '%s' in CSV.", taxa)
            continue

        vals_matrix = sub[class_cols_int].to_numpy(dtype=np.float64)  # species x classes

        for j, cid in enumerate(class_cols_int):
            v = vals_matrix[:, j]
            score_by_group[(taxa, cid)] = shannon_normalised(v)

        logging.info("  Done: %s (species rows: %d)", taxa, sub.shape[0])

    # Aggregate across groups
    logging.info("Aggregating taxa-group scores to a single biodiversity score per class (geometric mean)...")
    biodiversity_score_by_class: Dict[int, float] = {}
    for cid in class_cols_int:
        group_scores = [score_by_group.get((taxa, cid), 0.0) for taxa in TAXA_GROUPS]
        biodiversity_score_by_class[cid] = geometric_mean_nonzero(group_scores)

    # Read SCC raster, count cells per class
    logging.info("Reading SCC raster and counting cells per class: %s", SCC_RASTER)
    class_cell_counts: Dict[int, int] = defaultdict(int)

    with rasterio.open(SCC_RASTER) as src:
        scc_nodata = src.nodata
        profile = src.profile.copy()
        logging.info("Raster: %d x %d, nodata=%s", src.width, src.height, str(scc_nodata))

        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            if scc_nodata is None:
                mask = np.ones(arr.shape, dtype=bool)
            else:
                mask = arr != scc_nodata

            if not np.any(mask):
                continue

            data = arr[mask].astype(np.int64)
            u, cts = np.unique(data, return_counts=True)
            for k, ct in zip(u.tolist(), cts.tolist()):
                class_cell_counts[int(k)] += int(ct)

    logging.info("Unique classes in raster (excluding nodata): %d", len(class_cell_counts))

    raster_class_ids = np.array(sorted(class_cell_counts.keys()), dtype=np.int64)
    raster_scores = np.array([biodiversity_score_by_class.get(int(cid), 0.0) for cid in raster_class_ids], dtype=np.float64)
    raster_counts = np.array([class_cell_counts[int(cid)] for cid in raster_class_ids], dtype=np.float64)

    denom = float(np.sum(raster_scores * raster_counts))
    if denom <= 0.0:
        logging.warning("All biodiversity scores are zero across raster. Values will be zero everywhere.")
        per_cell_value_by_class = {int(cid): 0.0 for cid in raster_class_ids}
    else:
        per_cell_value_by_class = {
            int(cid): float(TOTAL_MONETARY_VALUE * biodiversity_score_by_class.get(int(cid), 0.0) / denom)
            for cid in raster_class_ids
        }

    total_value_by_class = {
        int(cid): float(per_cell_value_by_class[int(cid)] * class_cell_counts[int(cid)])
        for cid in raster_class_ids
    }
    allocated_total = float(sum(total_value_by_class.values()))
    logging.info("Allocated total = %.6f (target %.6f)", allocated_total, TOTAL_MONETARY_VALUE)

    # Output CSV
    out_rows = []
    for cid in raster_class_ids.tolist():
        out_rows.append(
            {
                "class_id": int(cid),
                "biodiversity_score": float(biodiversity_score_by_class.get(int(cid), 0.0)),
                "biodiversity_value": float(total_value_by_class.get(int(cid), 0.0)),
            }
        )
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CLASS_CSV, index=False)
    logging.info("Wrote class CSV: %s", OUTPUT_CLASS_CSV)

    # Prepare raster mapping
    keys = raster_class_ids.astype(np.int64)
    score_vals = np.array([biodiversity_score_by_class.get(int(k), 0.0) for k in keys], dtype=np.float32)
    value_vals = np.array([per_cell_value_by_class.get(int(k), 0.0) for k in keys], dtype=np.float32)

    score_lut = build_lut(keys, score_vals)
    value_lut = build_lut(keys, value_vals)

    keys_sorted = keys
    score_sorted = score_vals
    value_sorted = value_vals

    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress=OUTPUT_COMPRESS,
        predictor=OUTPUT_PREDICTOR,
        tiled=True,
    )

    # Write rasters blockwise
    logging.info("Writing score raster: %s", OUTPUT_SCORE_RASTER)
    logging.info("Writing value raster: %s", OUTPUT_VALUE_RASTER)

    with rasterio.open(SCC_RASTER) as src, \
         rasterio.open(OUTPUT_SCORE_RASTER, "w", **out_profile) as dst_score, \
         rasterio.open(OUTPUT_VALUE_RASTER, "w", **out_profile) as dst_value:

        block_total = sum(1 for _ in src.block_windows(1))
        processed = 0

        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            if src.nodata is None:
                mask = np.ones(arr.shape, dtype=bool)
            else:
                mask = arr != src.nodata

            score_out = np.full(arr.shape, OUTPUT_NODATA, dtype=np.float32)
            value_out = np.full(arr.shape, OUTPUT_NODATA, dtype=np.float32)

            if np.any(mask):
                data = arr[mask].astype(np.int64)
                score_out[mask] = map_classes(data, score_lut, keys_sorted, score_sorted)
                value_out[mask] = map_classes(data, value_lut, keys_sorted, value_sorted)

            dst_score.write(score_out, 1, window=window)
            dst_value.write(value_out, 1, window=window)

            processed += 1
            if processed % 50 == 0 or processed == block_total:
                logging.info("Processed %d / %d blocks...", processed, block_total)

    logging.info("Done.")


if __name__ == "__main__":
    main()