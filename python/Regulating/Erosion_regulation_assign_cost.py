#!/usr/bin/env python3
"""
Compute erosion-regulation value raster:
    value_raster = clip(avoided_erosion, 0..MAX_TONNES) * cost_per_tonne(cell)

Rules:
- If NZLUM class is DAIRY_CLASS -> cost_per_t = DAIRY_PROFIT_RASTER * DAIRY_SCALAR
- If NZLUM class is in SNB_CLASSES -> cost_per_t = SNB_PROFIT_RASTER * SNB_SCALAR
- Otherwise -> cost_per_t comes from COST_LOOKUP_CSV
- If NZLUM class has no lookup value -> assume cost_per_t = 0 (not NoData)
- If NZLUM is NoData -> assume cost_per_t = 0 (not NoData)
- If avoided_erosion is NoData/NaN -> output NoData

Notes:
- NZLUM class raster is resampled with nearest-neighbour (categorical).
- Profit rasters are resampled with bilinear (continuous).
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import fiona
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.warp import reproject

WORKING_DIR = Path(r"<PROJECT_DIRECTORY>\Regulating\Erosion regulation\Intermediate")
OUT_DIR = Path(r"<PROJECT_DIRECTORY>\Regulating\Erosion regulation\Output")

AVOIDED_EROSION_RASTER = WORKING_DIR / "NZEEM_avoided_erosion_v2.tif"
NZLUM_CLASS_RASTER = Path(r"D:\Data\LRIS\lris-new-zealand-land-use-management-version-03-nzlum-v03-FGDB\nzlum.tif")
COST_LOOKUP_CSV = WORKING_DIR / "erosion_costs_nzlum.csv"
DAIRY_PROFIT_RASTER = WORKING_DIR / "dairy_profit.tif"
SNB_PROFIT_RASTER = WORKING_DIR / "snb_profit.tif"

MASK_VECTOR = Path(r"D:\Data\LINZ\lds-nz-coastlines-and-islands-polygons-topo-150k\nz-coastlines-and-islands-polygons-topo-150k.gpkg")
MASK_LAYER = "nz_coastlines_and_islands_polygons_topo_150k"

OUTPUTS = {
    "low": OUT_DIR / "erosion_regulation_value_low.tif",
    "central": OUT_DIR / "erosion_regulation_value.tif",
    "high": OUT_DIR / "erosion_regulation_value_high.tif",
}

SCENARIOS = ("low", "central", "high")

CSV_CLASS_COL = "class"
OUTPUT_NODATA = 0

DAIRY_CLASS = 221
SNB_CLASSES = {220, 222, 223}

#These parameters are calculated as (EBIT/ha) / (cost/t) from Soliman & Walsh 2005, converted to 2026 prices
DAIRY_SCALARS = {
    "low": 0.0002940 ,
    "central": 0.000511222,
    "high": 0.0014765 ,
}
SNB_SCALARS = {
    "low": 0.0002339,
    "central": 0.000393185,
    "high": 0.0006373,
}

MAX_TONNES =1995.0 #Uses the 99.9th percentile to ignore extreme values
CLAMP_NEGATIVE_TO_ZERO = True
LOG_LEVEL = logging.INFO


def setup_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_paths(paths: list[Path]) -> None:
    logging.info("Checking input paths.")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

def same_grid(ds: rasterio.io.DatasetReader, target_profile: dict) -> bool:
    return (
        ds.crs == target_profile["crs"]
        and ds.transform == target_profile["transform"]
        and ds.width == target_profile["width"]
        and ds.height == target_profile["height"]
    )

def align_raster_to_target(src_ds, target_profile, *, dst_dtype, resampling, dst_nodata):
    dst = np.empty((target_profile["height"], target_profile["width"]), dtype=dst_dtype)
    reproject(
        source=rasterio.band(src_ds, 1),
        destination=dst,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        resampling=resampling,
        src_nodata=src_ds.nodata,
        dst_nodata=dst_nodata,
    )
    return dst

def read_or_align_raster(path: Path, target_profile: dict, *, dtype, resampling, dst_nodata):
    logging.info("Opening raster: %s", path)
    with rasterio.open(path) as ds:
        if same_grid(ds, target_profile):
            logging.info("Raster already aligned: %s", path)
            arr = ds.read(1).astype(dtype, copy=False)
        else:
            logging.info("Aligning raster to avoided-erosion grid: %s", path)
            arr = align_raster_to_target(
                ds,
                target_profile,
                dst_dtype=dtype,
                resampling=resampling,
                dst_nodata=dst_nodata,
            )
        return arr, ds.nodata

def rasterize_mask_to_target(mask_vector_path: Path, mask_layer: str, target_profile: dict) -> np.ndarray:
    logging.info("Rasterising mask layer: %s | layer=%s", mask_vector_path, mask_layer)
    with fiona.open(mask_vector_path, layer=mask_layer, mode="r") as src:
        shapes = [(feat["geometry"], 1) for feat in src if feat["geometry"] is not None]

    if not shapes:
        raise ValueError(f"No valid geometries found in mask layer: {mask_layer}")

    return rasterize(
        shapes=shapes,
        out_shape=(target_profile["height"], target_profile["width"]),
        transform=target_profile["transform"],
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=False,
    )

def load_cost_lookup(csv_path: Path) -> dict[str, dict[int, float]]:
    logging.info("Reading cost lookup CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    required = [CSV_CLASS_COL, *SCENARIOS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[required].dropna(subset=[CSV_CLASS_COL])
    df[CSV_CLASS_COL] = df[CSV_CLASS_COL].astype(int)

    if df.duplicated(subset=[CSV_CLASS_COL]).any():
        logging.warning("Duplicate class codes found. Keeping last occurrence.")
        df = df.drop_duplicates(subset=[CSV_CLASS_COL], keep="last")

    lookups = {}
    for scenario in SCENARIOS:
        clean = df[[CSV_CLASS_COL, scenario]].dropna()
        lookups[scenario] = dict(zip(clean[CSV_CLASS_COL].tolist(), clean[scenario].astype(float).tolist()))
        logging.info("Loaded %d class->%s cost mappings.", len(lookups[scenario]), scenario)

    return lookups

def nodata_mask(arr: np.ndarray, nodata) -> np.ndarray:
    mask = np.zeros(arr.shape, dtype=bool)
    if nodata is not None:
        try:
            if not np.isnan(nodata):
                mask |= arr == nodata
        except TypeError:
            mask |= arr == nodata
    if np.issubdtype(arr.dtype, np.floating):
        mask |= np.isnan(arr)
    return mask

def clean_profit(profit_arr: np.ndarray, profit_nodata) -> np.ndarray:
    logging.info("Cleaning profit raster nodata values.")
    p = profit_arr.astype(np.float32, copy=False)
    bad = nodata_mask(p, profit_nodata)
    out = p.copy()
    out[bad] = 0.0
    return out

def prepare_avoided_erosion(ae: np.ndarray, ae_nodata) -> tuple[np.ndarray, np.ndarray]:
    logging.info("Preparing avoided erosion raster.")
    ae_is_nodata = nodata_mask(ae, ae_nodata)
    ae_clip = ae.astype(np.float32, copy=True)

    if CLAMP_NEGATIVE_TO_ZERO:
        ae_clip = np.maximum(ae_clip, 0.0)

    if MAX_TONNES is not None:
        ae_clip = np.minimum(ae_clip, float(MAX_TONNES))

    return ae_clip, ae_is_nodata

def build_costs_for_scenario(
    scenario: str,
    lu: np.ndarray,
    lu_nodata,
    computable: np.ndarray,
    dairy_profit_clean: np.ndarray,
    snb_profit_clean: np.ndarray,
    cost_lookup: dict[int, float],
) -> np.ndarray:
    logging.info("Building cost-per-tonne raster for scenario: %s", scenario)

    costs = np.zeros(lu.shape, dtype=np.float32)
    lu_is_nodata = nodata_mask(lu, lu_nodata)

    special_classes = {DAIRY_CLASS, *SNB_CLASSES}
    normal_mask = computable & ~lu_is_nodata & ~np.isin(lu, list(special_classes))

    if normal_mask.any():
        unique_classes = np.unique(lu[normal_mask])
        missing_classes = []

        for c in unique_classes.tolist():
            value = cost_lookup.get(int(c))
            if value is None:
                missing_classes.append(int(c))
            else:
                costs[(lu == c) & normal_mask] = float(value)

        if missing_classes:
            logging.warning(
                "%s: no CSV cost for %d NZLUM classes. Treating as zero. Sample: %s",
                scenario,
                len(missing_classes),
                missing_classes[:25],
            )

    dairy_mask = computable & ~lu_is_nodata & (lu == DAIRY_CLASS)
    if dairy_mask.any():
        dairy_missing = dairy_mask & (dairy_profit_clean == 0)
        dairy_valid = dairy_mask & ~dairy_missing
        costs[dairy_valid] = dairy_profit_clean[dairy_valid] * DAIRY_SCALARS[scenario]

        if dairy_missing.any():
            fallback = cost_lookup.get(DAIRY_CLASS, 0.0)
            logging.warning(
                "%s: dairy profit missing for %d pixels. Falling back to CSV value %.6f.",
                scenario,
                int(np.sum(dairy_missing)),
                fallback,
            )
            costs[dairy_missing] = float(fallback)

    snb_mask = computable & ~lu_is_nodata & np.isin(lu, list(SNB_CLASSES))
    if snb_mask.any():
        snb_missing = snb_mask & (snb_profit_clean == 0)
        snb_valid = snb_mask & ~snb_missing
        costs[snb_valid] = snb_profit_clean[snb_valid] * SNB_SCALARS[scenario]

        if snb_missing.any():
            fallback_classes = np.unique(lu[snb_missing]).tolist()
            logging.warning(
                "%s: SNB profit missing for %d pixels. Falling back to CSV lookup. Classes: %s",
                scenario,
                int(np.sum(snb_missing)),
                fallback_classes,
            )
            for c in fallback_classes:
                fallback = cost_lookup.get(int(c), 0.0)
                costs[(lu == c) & snb_missing] = float(fallback)

    return costs


def make_output_raster(ae_clip: np.ndarray, costs: np.ndarray, computable: np.ndarray, lu: np.ndarray) -> np.ndarray:
    out = np.full(ae_clip.shape, OUTPUT_NODATA, dtype=np.float32)

    values = ae_clip * costs
    valid_value = computable & np.isfinite(values) & (values != 0)

    out[valid_value] = values[valid_value]

    neg_out = valid_value & (out < 0)
    if np.any(neg_out):
        logging.warning("Negative output pixels detected: %d. Setting to nodata.", int(np.sum(neg_out)))
        logging.warning("NZLUM classes involved: %s", np.unique(lu[neg_out]).tolist()[:25])
        out[neg_out] = OUTPUT_NODATA

    return out

def write_raster(path: Path, arr: np.ndarray, profile: dict, computable: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logging.info("Deleting existing output: %s", path)
        path.unlink()

    logging.info("Writing raster: %s", path)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)

    vals = arr[(arr != OUTPUT_NODATA) & computable]
    if vals.size:
        logging.info(
            "Summary for %s: min=%.3f, mean=%.3f, max=%.3f, sum=%.3f",
            path.name,
            float(np.min(vals)),
            float(np.mean(vals)),
            float(np.max(vals)),
            float(np.sum(vals)),
        )
    else:
        logging.warning("No valid output values for %s.", path.name)


def main() -> int:
    setup_logging()
    logging.info("Starting erosion regulation valuation.")

    validate_paths([
        AVOIDED_EROSION_RASTER,
        NZLUM_CLASS_RASTER,
        COST_LOOKUP_CSV,
        DAIRY_PROFIT_RASTER,
        SNB_PROFIT_RASTER,
        MASK_VECTOR,
    ])

    cost_lookups = load_cost_lookup(COST_LOOKUP_CSV)

    logging.info("Opening avoided erosion raster: %s", AVOIDED_EROSION_RASTER)
    with rasterio.open(AVOIDED_EROSION_RASTER) as ae_ds:
        ae = ae_ds.read(1, masked=False)
        ae_profile = ae_ds.profile.copy()
        ae_nodata = ae_ds.nodata

    out_profile = ae_profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    mask_arr = rasterize_mask_to_target(MASK_VECTOR, MASK_LAYER, ae_profile)

    lu, lu_nodata = read_or_align_raster(
        NZLUM_CLASS_RASTER,
        ae_profile,
        dtype=np.int32,
        resampling=Resampling.nearest,
        dst_nodata=0,
    )

    dairy_profit, dairy_profit_nodata = read_or_align_raster(
        DAIRY_PROFIT_RASTER,
        ae_profile,
        dtype=np.float32,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )

    snb_profit, snb_profit_nodata = read_or_align_raster(
        SNB_PROFIT_RASTER,
        ae_profile,
        dtype=np.float32,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )

    ae_clip, ae_is_nodata = prepare_avoided_erosion(ae, ae_nodata)
    dairy_profit_clean = clean_profit(dairy_profit, dairy_profit_nodata)
    snb_profit_clean = clean_profit(snb_profit, snb_profit_nodata)

    inside_mask = mask_arr == 1
    computable = ~ae_is_nodata & inside_mask

    logging.info("Computable pixels: %d", int(np.sum(computable)))
    logging.info("AE clipped max: %.3f", float(np.nanmax(ae_clip[computable])))

    for scenario in SCENARIOS:
        logging.info("Processing scenario: %s", scenario)

        costs = build_costs_for_scenario(
            scenario=scenario,
            lu=lu,
            lu_nodata=lu_nodata,
            computable=computable,
            dairy_profit_clean=dairy_profit_clean,
            snb_profit_clean=snb_profit_clean,
            cost_lookup=cost_lookups[scenario],
        )

        logging.info("%s costs max: %.6f", scenario, float(np.nanmax(costs[computable])))

        out = make_output_raster(ae_clip, costs, computable, lu)
        write_raster(OUTPUTS[scenario], out, out_profile, computable)

    logging.info("Erosion regulation valuation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())