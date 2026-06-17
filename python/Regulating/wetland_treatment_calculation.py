
"""
Estimate wetland contaminant removal and write spatial raster outputs.

This script combines wetland ID rasters, wetland-river intersection data,
upstream contaminant loads, wetland area, catchment area, climate class, and
wetland performance curves to estimate contaminant removal by individual
wetlands. It can process TN, TP, TSS, and E. coli, although the contaminants
run are controlled by CONTAMINANTS_TO_RUN.

For each wetland, the script:
1. identifies eligible intersecting river segments, retaining only stream order 1 or 2 segments;
2. excludes Bog wetlands from stream-load treatment;
3. sums upstream flow and contaminant loads across the remaining eligible segments;
4. estimates the proportion of catchment area represented by the wetland;
5. estimates the maximum flow treatable by the wetland using a fixed treatable-flow-per-hectare assumption;
6. scales upstream loads by the fraction of flow treated, capped at 100% of upstream flow;
7. interpolates contaminant removal rates from wetland performance tables;
8. back-calculates removed loads or concentration reductions from observed post-treatment values;
9. allocates wetland-level results back to raster cells; and
10. writes GeoTIFF outputs for input loads, treated flow, and low, central, and high removal estimates.

TN removal rates are selected separately for warm and cool climate wetlands
using a warm-climate dummy raster. TP, TSS, and E. coli use removal-rate
curves that do not vary by climate. TN, TP, TSS, and treated-flow totals are
distributed equally across all raster cells belonging to each wetland so that
the raster sum equals the wetland-level total. E. coli values are treated as
concentration-like metrics and are assigned as constant values across each
wetland.

Inputs are expected to include:
- a wetland_id raster;
- a warm-climate dummy raster aligned to the wetland raster;
- a wetland-river intersection table;
- an upstream segment load table; and
- a wetland performance table containing catchment-proportion removal curves.

Outputs are compressed float32 GeoTIFF rasters using the wetland_id raster as
the spatial template. No-data values are written as -9999.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import os

# =============================================================================
# FILES AND DIRECTORIES
# =============================================================================
CONTAMINANTS_TO_RUN = ["TN", "TP", "TSS", "Ecoli"]  # e.g. ["TN", "TP", "TSS", "Ecoli"]

BASE_DIR = Path(r"<PROJECT_DIRECTORY>\Wetlands")
INPUT_DIR = BASE_DIR / "Inputs"
OUTPUT_DIR = BASE_DIR / "Intermediate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Categorical raster with the wetland_id
WETLAND_RASTER = INPUT_DIR / "wetland_id.tif"

# This raster has 1 if warm climate, otherwise 0
WARM_CLIMATE_DUMMY_RASTER = INPUT_DIR / "warm_climate_dummy.tif"

# CSV files
# This file has the many-to-many relationship wetland_id, wetland_area_ha, nzsegment, catchment_ha
# Note that wetland_area_ha is repeated across all rows with the same wetland. catchment_ha applies to the nzsegment
#Fields: wetland_id,Name_2018,Class_2018,Wetland_18,LCDB_UID,WONI,wetland_area_ha,nzsegment,catchment_ha,stream_order,wetland_type
WETLAND_INTERSECTIONS = INPUT_DIR / "wetland_riverline_intersections.csv"

# This file has the loads going into each segment
# Expected fields now include at least:
# nzsegment, upstream_total_flow, upstream_TN_load, upstream_TP_load,
# upstream_Ecoli_median_fw, upstream_Ecoli_Q95_fw, upstream_Ecoli_G540_fw,
# upstream_suspended_sed_load
SEGMENT_LOADS = INPUT_DIR / "upstream_quality_nzsegment.csv"

# Removal rate CSV
REMOVAL_RATES_CSV = INPUT_DIR / "wetland_performance.csv"

# Constant specifying how much flow can be treated per hectare of wetland (same units as segment median_flow / upstream_total_flow)
# No longer used because it's now including only order 1 and 2 streams
MAX_TREATABLE_FLOW_PER_HA = 9999 #0.0056


# Output rasters
OUTPUT_FILES = {
    "TN": {
        "central": OUTPUT_DIR / "tn_kg_removed_central.tif",
        "low": OUTPUT_DIR / "tn_kg_removed_low.tif",
        "high": OUTPUT_DIR / "tn_kg_removed_high.tif",
    },
    "TP": {
        "central": OUTPUT_DIR / "tp_kg_removed_central.tif",
        "low": OUTPUT_DIR / "tp_kg_removed_low.tif",
        "high": OUTPUT_DIR / "tp_kg_removed_high.tif",
    },
    "TSS": {
        "central": OUTPUT_DIR / "tss_t_removed_central.tif",
        "low": OUTPUT_DIR / "tss_t_removed_low.tif",
        "high": OUTPUT_DIR / "tss_t_removed_high.tif",
    },
    "Ecoli": {
        "central": OUTPUT_DIR / "ecoli_median_reduction_central.tif",
        "low": OUTPUT_DIR / "ecoli_median_reduction_low.tif",
        "high": OUTPUT_DIR / "ecoli_median_reduction_high.tif",
    },
    "Ecoli_pre": {
        "central": OUTPUT_DIR / "ecoli_median_no_wetland_central.tif",
        "low": OUTPUT_DIR / "ecoli_median_no_wetland_low.tif",
        "high": OUTPUT_DIR / "ecoli_median_no_wetland_high.tif",
    },
    "TN_input": OUTPUT_DIR / "tn_input_load_per_cell.tif",
    "TP_input": OUTPUT_DIR / "tp_input_load_per_cell.tif",
    "TSS_input": OUTPUT_DIR / "tss_input_load_per_cell.tif",
    "Ecoli_input": OUTPUT_DIR / "ecoli_fw_median_per_cell.tif",
    "Ecoli_Q95_input": OUTPUT_DIR / "ecoli_q95_fw_per_cell.tif",
    "Ecoli_G540_input": OUTPUT_DIR / "ecoli_g540_fw_per_cell.tif",
    "treated_flow": OUTPUT_DIR / "treated_flow_per_cell.tif"
}


OUT_NODATA = -9999.0


# =============================================================================
# HELPERS
# =============================================================================
def should_run(name):
    if not CONTAMINANTS_TO_RUN:
        return True
    return name in CONTAMINANTS_TO_RUN


def read_raster(path):
    """Read first band as float32 and return array + profile + nodata."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
    return arr, profile, nodata


def build_valid_mask(arr, nodata):
    """Return True where raster has valid data."""
    if nodata is None:
        return np.isfinite(arr)

    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isfinite(arr)

    return np.isfinite(arr) & (arr != nodata)


def interp_rates(proportions, x_table, y_table):
    """Linear interpolation of removal rates after clipping to table range."""
    p = np.clip(proportions, x_table.min(), x_table.max())
    return np.interp(p, x_table, y_table).astype("float32")


def write_raster(path, arr, template_profile, max_attempts=5):
    profile = template_profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=OUT_NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        bigtiff="if_safer",
        blockxsize=256,
        blockysize=256,
    )

    path = str(path)
    base, ext = os.path.splitext(path)

    last_error = None
    for attempt in range(max_attempts):
        out_path = path if attempt == 0 else f"{base}_{attempt}{ext}"
        try:
            print(f"  writing: {out_path}")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr.astype("float32"), 1)
            #print(f"  wrote: {out_path}")
            return out_path
        except PermissionError as e:
            last_error = e
            print(f"  permission denied: {out_path}")
        except Exception:
            raise

    raise RuntimeError(f"Permission denied after {max_attempts} attempts for: {path}") from last_error

def print_raster_summary(name, arr, nodata):
    valid = build_valid_mask(arr, nodata)
    n_valid = int(valid.sum())
    n_total = arr.size
    n_invalid = n_total - n_valid

    print(f"\n{name}")
    print(f"  shape: {arr.shape}")
    print(f"  nodata: {nodata}")
    print(f"  valid cells: {n_valid:,} / {n_total:,}")
    print(f"  invalid cells: {n_invalid:,}")

    if n_valid > 0:
        vals = arr[valid]
        print(f"  min: {np.min(vals):.6g}")
        print(f"  max: {np.max(vals):.6g}")
        print(f"  mean: {np.mean(vals):.6g}")
    else:
        print("  WARNING: no valid cells")

def check_alignment(name, profile, ref_profile):
    problems = []

    if profile["width"] != ref_profile["width"] or profile["height"] != ref_profile["height"]:
        problems.append(
            f"shape differs: {(profile['height'], profile['width'])} vs {(ref_profile['height'], ref_profile['width'])}"
        )

    if profile.get("crs") != ref_profile.get("crs"):
        problems.append(f"CRS differs: {profile.get('crs')} vs {ref_profile.get('crs')}")

    if profile.get("transform") != ref_profile.get("transform"):
        problems.append("transform differs")

    if problems:
        raise ValueError(f"{name} is not aligned with reference raster: " + "; ".join(problems))
    else:
        print(f"{name} alignment check passed")

def print_mask_summary(name, mask):
    print(f"{name}: {int(mask.sum()):,} true cells")

def safe_divide(numerator, denominator, default=0.0):
    if pd.isna(denominator) or denominator <= 0:
        return default
    return numerator / denominator

def allocate_wetland_values_to_raster(
    wetland_ids,
    wetland_mask,
    wetland_valid_mask,
    wetland_values,
    allocation="distributed",
    default_nodata=OUT_NODATA,
):
    """
    Create raster from wetland-level values.

    allocation="distributed":
        divide each wetland total equally across its cells so the raster sums
        back to the wetland total.

    allocation="constant":
        assign the same wetland-level value to every cell in the wetland.
        Use this for concentrations such as E. coli.
    """
    out = np.full(wetland_ids.shape, default_nodata, dtype="float32")

    wetland_ids_int = wetland_ids.astype(np.int64, copy=False)
    ids_in_raster = wetland_ids_int[wetland_mask]

    if ids_in_raster.size == 0:
        print("  no wetland cells found")
        out[~wetland_valid_mask] = default_nodata
        return out

    print("  counting cells per wetland...")
    unique_ids, counts = np.unique(ids_in_raster, return_counts=True)
    cell_count_by_id = dict(zip(unique_ids.tolist(), counts.tolist()))

    per_wetland_ids = []
    per_cell_values = []

    total = len(wetland_values)
    processed = 0
    skipped = 0
    step = max(1, total // 10)

    for wetland_id, wetland_value in wetland_values.items():
        processed += 1
        #if processed % step == 0 or processed == total:
            #print(f"  progress: {processed:,}/{total:,} wetlands ({processed/total:.0%})")

        n_cells = cell_count_by_id.get(int(wetland_id), 0)
        if n_cells <= 0:
            skipped += 1
            continue

        if allocation == "distributed":
            value_for_cells = np.float32(wetland_value / n_cells)
        elif allocation == "constant":
            value_for_cells = np.float32(wetland_value)
        else:
            raise ValueError(f"Unknown allocation mode: {allocation}")

        per_wetland_ids.append(int(wetland_id))
        per_cell_values.append(value_for_cells)

    if not per_wetland_ids:
        print("  no matching wetland IDs between lookup and raster")
        out[~wetland_valid_mask] = default_nodata
        return out

    per_wetland_ids = np.array(per_wetland_ids, dtype=np.int64)
    per_cell_values = np.array(per_cell_values, dtype=np.float32)

    print("  assigning values to raster cells...")
    sort_idx = np.argsort(per_wetland_ids)
    sorted_ids = per_wetland_ids[sort_idx]
    sorted_vals = per_cell_values[sort_idx]

    positions = np.searchsorted(sorted_ids, ids_in_raster)
    matched = (positions < len(sorted_ids)) & (sorted_ids[positions] == ids_in_raster)

    allocated_values = np.full(ids_in_raster.shape, default_nodata, dtype="float32")
    allocated_values[matched] = sorted_vals[positions[matched]]

    out[wetland_mask] = allocated_values
    out[~wetland_valid_mask] = default_nodata

    print(f"  wetlands with values: {len(per_wetland_ids):,}")
    print(f"  wetlands skipped: {skipped:,}")
    print(f"  cells assigned: {int(np.sum(matched)):,}")

    return out


# =============================================================================
# MAIN
# =============================================================================
def main():

    # Logic implemented:
    # Unit of analysis: wetland_id
    # For each wetland_id, find intersecting stream_order 1 or 2 segments
    # Exclude Bog wetlands from stream-load treatment
    # Sum the upstream load and flow for eligible segments
    # Calculate how much flow the wetland can treat: MAX_TREATABLE_FLOW_PER_HA * wetland_area_ha
    # Scale the loads by treatable / upstream_total_flow
    # Calculate removals using the rates in the CSV
    # Allocate removals equally to each cell with the same wetland_id
    # Output removal rasters for each contaminant

    print("\nReading rasters...")
    wetland, profile, wetland_nodata = read_raster(WETLAND_RASTER)
    warm_dummy, warm_profile, warm_dummy_nodata = read_raster(WARM_CLIMATE_DUMMY_RASTER)

    check_alignment("warm_climate_dummy", warm_profile, profile)
    print_raster_summary("wetland_id", wetland, wetland_nodata)
    print_raster_summary("warm_climate_dummy", warm_dummy, warm_dummy_nodata)

    wetland_valid_mask = build_valid_mask(wetland, wetland_nodata)
    warm_valid_mask = build_valid_mask(warm_dummy, warm_dummy_nodata)
    common_valid = wetland_valid_mask & warm_valid_mask
    wetland_mask = common_valid & (wetland > 0)

    print("\nBuilding masks...")
    print_mask_summary("wetland_valid_mask", wetland_valid_mask)
    print_mask_summary("warm_valid_mask", warm_valid_mask)
    print_mask_summary("common_valid", common_valid)
    print_mask_summary("wetland_mask", wetland_mask)

    if not np.any(wetland_mask):
        raise ValueError("No valid wetland cells found in wetland raster")

    print("\nReading removal rate table...")
    rates = pd.read_csv(REMOVAL_RATES_CSV)
    rates.columns = [c.strip() for c in rates.columns]

    required_rate_cols = [
        "catchment_proportion",
        "TN_warm", "TN_cool", "TN_warm_low", "TN_warm_high", "TN_cool_low", "TN_cool_high",
        "TP", "TP_low", "TP_high",
        "TSS", "TSS_low", "TSS_high",
        "Ecoli", "Ecoli_low", "Ecoli_high",
    ]
    missing_cols = [c for c in required_rate_cols if c not in rates.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in removal table: {missing_cols}")

    if rates["catchment_proportion"].isna().any():
        raise ValueError("catchment_proportion contains NA values")

    x = rates["catchment_proportion"].to_numpy(dtype="float32")
    if np.any(~np.isfinite(x)):
        raise ValueError("catchment_proportion contains non-finite values")
    if np.any(np.diff(x) < 0):
        raise ValueError("catchment_proportion must be sorted ascending for np.interp")

    tn_warm = rates["TN_warm"].to_numpy(dtype="float32")
    tn_cool = rates["TN_cool"].to_numpy(dtype="float32")
    tn_warm_low = rates["TN_warm_low"].to_numpy(dtype="float32")
    tn_warm_high = rates["TN_warm_high"].to_numpy(dtype="float32")
    tn_cool_low = rates["TN_cool_low"].to_numpy(dtype="float32")
    tn_cool_high = rates["TN_cool_high"].to_numpy(dtype="float32")

    tp_rate = rates["TP"].to_numpy(dtype="float32")
    tp_low = rates["TP_low"].to_numpy(dtype="float32")
    tp_high = rates["TP_high"].to_numpy(dtype="float32")

    tss_rate = rates["TSS"].to_numpy(dtype="float32")
    tss_low = rates["TSS_low"].to_numpy(dtype="float32")
    tss_high = rates["TSS_high"].to_numpy(dtype="float32")

    ecoli_rate = rates["Ecoli"].to_numpy(dtype="float32")
    ecoli_low = rates["Ecoli_low"].to_numpy(dtype="float32")
    ecoli_high = rates["Ecoli_high"].to_numpy(dtype="float32")

    print("\nReading wetland intersections...")
    intersections = pd.read_csv(WETLAND_INTERSECTIONS)
    intersections.columns = [c.strip() for c in intersections.columns]

    required_intersection_cols = [
        "wetland_id",
        "wetland_area_ha",
        "nzsegment",
        "catchment_ha",
        "stream_order",
        "wetland_type",
    ]
    missing_cols = [c for c in required_intersection_cols if c not in intersections.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in wetland intersections table: {missing_cols}")

    print("\nReading segment loads...")
    seg = pd.read_csv(SEGMENT_LOADS)
    seg.columns = [c.strip() for c in seg.columns]

    required_seg_cols = [
        "nzsegment",
        "upstream_total_flow",
        "upstream_TN_load",
        "upstream_TP_load",
        "upstream_Ecoli_median_fw",
        "upstream_Ecoli_Q95_fw",
        "upstream_Ecoli_G540_fw",
        "upstream_suspended_sed_load",
    ]
    missing_cols = [c for c in required_seg_cols if c not in seg.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in segment loads table: {missing_cols}")

    intersections["wetland_id"] = pd.to_numeric(intersections["wetland_id"], errors="coerce")
    intersections["nzsegment"] = pd.to_numeric(intersections["nzsegment"], errors="coerce")
    intersections["wetland_area_ha"] = pd.to_numeric(intersections["wetland_area_ha"], errors="coerce")
    intersections["catchment_ha"] = pd.to_numeric(intersections["catchment_ha"], errors="coerce")
    intersections["stream_order"] = pd.to_numeric(intersections["stream_order"], errors="coerce")
    intersections["wetland_type"] = intersections["wetland_type"].astype("string").str.strip()

    seg["nzsegment"] = pd.to_numeric(seg["nzsegment"], errors="coerce")
    for c in [
        "upstream_total_flow",
        "upstream_TN_load",
        "upstream_TP_load",
        "upstream_Ecoli_median_fw",
        "upstream_Ecoli_Q95_fw",
        "upstream_Ecoli_G540_fw",
        "upstream_suspended_sed_load",
    ]:
        seg[c] = pd.to_numeric(seg[c], errors="coerce")

    intersections = intersections.dropna(
        subset=["wetland_id", "nzsegment", "wetland_area_ha", "catchment_ha", "stream_order"]
    ).copy()
    seg = seg.dropna(subset=["nzsegment"]).copy()

    intersections["wetland_id"] = intersections["wetland_id"].astype(np.int64)
    intersections["nzsegment"] = intersections["nzsegment"].astype(np.int64)
    intersections["stream_order"] = intersections["stream_order"].astype(np.int64)
    seg["nzsegment"] = seg["nzsegment"].astype(np.int64)

    print("\nFiltering wetland-stream intersections...")
    n_before_filter = intersections.shape[0]
    is_bog = intersections["wetland_type"].str.casefold().eq("bog").fillna(False)

    #Exclude higher order streams and bogs
    intersections = intersections[
        intersections["stream_order"].isin([1, 2])
        & ~is_bog
    ].copy()

    print(f"  rows before filter: {n_before_filter:,}")
    print(f"  rows after stream_order/Bog filter: {intersections.shape[0]:,}")
    print(f"  rows removed: {n_before_filter - intersections.shape[0]:,}")

    print("\nSummarising climate by wetland_id from raster...")
    wetland_ids_int = wetland.astype(np.int64, copy=False)
    wetland_cell_df = pd.DataFrame({
        "wetland_id": wetland_ids_int[wetland_mask].ravel(),
        "warm_flag": (warm_dummy[wetland_mask] >= 0.5).astype(np.int8).ravel(),
    })

    wetland_climate = (
        wetland_cell_df.groupby("wetland_id", as_index=False)["warm_flag"]
        .mean()
        .rename(columns={"warm_flag": "warm_share"})
    )
    wetland_climate["is_warm"] = wetland_climate["warm_share"] >= 0.5

    print(f"  wetlands found in raster: {wetland_climate.shape[0]:,}")

    print("\nJoining intersections to segment loads...")
    df = intersections.merge(seg, on="nzsegment", how="left", validate="many_to_one")

    n_missing_load_rows = int(df["upstream_total_flow"].isna().sum())
    print(f"  eligible intersection rows: {df.shape[0]:,}")
    print(f"  rows with missing segment load match: {n_missing_load_rows:,}")

    for c in [
        "upstream_total_flow",
        "upstream_TN_load",
        "upstream_TP_load",
        "upstream_Ecoli_median_fw",
        "upstream_Ecoli_Q95_fw",
        "upstream_Ecoli_G540_fw",
        "upstream_suspended_sed_load",
    ]:
        df[c] = df[c].fillna(0.0)

    print("\nAggregating to wetland_id...")

    wetland_area = (
        df.groupby("wetland_id", as_index=False)["wetland_area_ha"]
        .first()
    )

    wetland_sums = (
        df.groupby("wetland_id", as_index=False)
        .apply(
            lambda g: pd.Series({
                "upstream_total_flow": g["upstream_total_flow"].sum(),
                "upstream_TN_load": g["upstream_TN_load"].sum(),
                "upstream_TP_load": g["upstream_TP_load"].sum(),
                "upstream_suspended_sed_load": g["upstream_suspended_sed_load"].sum(),
                "catchment_ha_sum": g["catchment_ha"].sum(),
                "intersecting_segment_count": g["nzsegment"].nunique(),
                "upstream_Ecoli_median_fw": (
                    (g["upstream_Ecoli_median_fw"] * g["upstream_total_flow"]).sum()
                    / g["upstream_total_flow"].sum()
                    if g["upstream_total_flow"].sum() > 0 else 0.0
                ),
                "upstream_Ecoli_Q95_fw": (
                    (g["upstream_Ecoli_Q95_fw"] * g["upstream_total_flow"]).sum()
                    / g["upstream_total_flow"].sum()
                    if g["upstream_total_flow"].sum() > 0 else 0.0
                ),
                "upstream_Ecoli_G540_fw": (
                    (g["upstream_Ecoli_G540_fw"] * g["upstream_total_flow"]).sum()
                    / g["upstream_total_flow"].sum()
                    if g["upstream_total_flow"].sum() > 0 else 0.0
                ),
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    wetland_df = wetland_area.merge(wetland_sums, on="wetland_id", how="left")
    wetland_df = wetland_df.merge(wetland_climate[["wetland_id", "is_warm"]], on="wetland_id", how="left")

    wetland_df["is_warm"] = wetland_df["is_warm"].astype("boolean").fillna(False).astype(bool)

    wetland_df["catchment_proportion"] = np.where(
        wetland_df["catchment_ha_sum"] > 0,
        wetland_df["wetland_area_ha"] / wetland_df["catchment_ha_sum"],
        0.0
    ).astype("float32")

    wetland_df["treatable_flow"] = (
        wetland_df["wetland_area_ha"] * MAX_TREATABLE_FLOW_PER_HA
    ).astype("float32")

    wetland_df["flow_scale"] = np.where(
        wetland_df["upstream_total_flow"] > 0,
        wetland_df["treatable_flow"] / wetland_df["upstream_total_flow"],
        0.0
    )
    wetland_df["flow_scale"] = wetland_df["flow_scale"].clip(0.0, 1.0).astype("float32")

    wetland_df["treated_flow"] = np.minimum(
        wetland_df["treatable_flow"],
        wetland_df["upstream_total_flow"]
    ).astype("float32")

    treated_flow_map = dict(zip(wetland_df["wetland_id"], wetland_df["treated_flow"].astype("float32")))
    tn_input_map = dict(zip(wetland_df["wetland_id"], wetland_df["upstream_TN_load"].astype("float32")))
    tp_input_map = dict(zip(wetland_df["wetland_id"], wetland_df["upstream_TP_load"].astype("float32")))
    tss_input_map = dict(zip(wetland_df["wetland_id"], wetland_df["upstream_suspended_sed_load"].astype("float32")))
    ecoli_input_map = dict(zip(wetland_df["wetland_id"], wetland_df["upstream_Ecoli_median_fw"].astype("float32")))
    ecoli_q95_input_map = dict(zip(wetland_df["wetland_id"], wetland_df["upstream_Ecoli_Q95_fw"].astype("float32")))
    ecoli_g540_input_map = dict(zip(wetland_df["wetland_id"], wetland_df["upstream_Ecoli_G540_fw"].astype("float32")))

    print(f"  wetlands after aggregation: {wetland_df.shape[0]:,}")
    if wetland_df.shape[0] > 0:
        print(f"  catchment proportion min/max: {wetland_df['catchment_proportion'].min():.6g} / {wetland_df['catchment_proportion'].max():.6g}")
        print(f"  flow scale min/max: {wetland_df['flow_scale'].min():.6g} / {wetland_df['flow_scale'].max():.6g}")

    print("\nLooking up rates at wetland level...")

    wetland_df["TN_rate_c"] = np.where(
        wetland_df["is_warm"],
        interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tn_warm),
        interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tn_cool),
    )
    wetland_df["TN_rate_l"] = np.where(
        wetland_df["is_warm"],
        interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tn_warm_low),
        interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tn_cool_low),
    )
    wetland_df["TN_rate_h"] = np.where(
        wetland_df["is_warm"],
        interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tn_warm_high),
        interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tn_cool_high),
    )

    wetland_df["TP_rate_c"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tp_rate)
    wetland_df["TP_rate_l"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tp_low)
    wetland_df["TP_rate_h"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tp_high)

    wetland_df["TSS_rate_c"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tss_rate)
    wetland_df["TSS_rate_l"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tss_low)
    wetland_df["TSS_rate_h"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, tss_high)

    wetland_df["Ecoli_rate_c"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, ecoli_rate)
    wetland_df["Ecoli_rate_l"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, ecoli_low)
    wetland_df["Ecoli_rate_h"] = interp_rates(wetland_df["catchment_proportion"].to_numpy(dtype="float32"), x, ecoli_high)

    print("\nCalculating wetland-level removals...")

    wetland_df["TN_treatable_load"] = wetland_df["upstream_TN_load"] * wetland_df["flow_scale"]
    wetland_df["TP_treatable_load"] = wetland_df["upstream_TP_load"] * wetland_df["flow_scale"]
    wetland_df["TSS_treatable_load"] = wetland_df["upstream_suspended_sed_load"] * wetland_df["flow_scale"]
    wetland_df["Ecoli_treatable"] = wetland_df["upstream_Ecoli_median_fw"] * wetland_df["flow_scale"]

    def removed_from_post_treatment(current_amount, rate):
        denom = 1.0 - rate
        return np.where(
            denom > 0,
            current_amount * rate / denom,
            np.nan
        )

    wetland_df["TN_removed_c"] = removed_from_post_treatment(wetland_df["TN_treatable_load"], wetland_df["TN_rate_c"])
    wetland_df["TN_removed_l"] = removed_from_post_treatment(wetland_df["TN_treatable_load"], wetland_df["TN_rate_l"])
    wetland_df["TN_removed_h"] = removed_from_post_treatment(wetland_df["TN_treatable_load"], wetland_df["TN_rate_h"])

    wetland_df["TP_removed_c"] = removed_from_post_treatment(wetland_df["TP_treatable_load"], wetland_df["TP_rate_c"])
    wetland_df["TP_removed_l"] = removed_from_post_treatment(wetland_df["TP_treatable_load"], wetland_df["TP_rate_l"])
    wetland_df["TP_removed_h"] = removed_from_post_treatment(wetland_df["TP_treatable_load"], wetland_df["TP_rate_h"])

    wetland_df["TSS_removed_c"] = removed_from_post_treatment(wetland_df["TSS_treatable_load"], wetland_df["TSS_rate_c"])
    wetland_df["TSS_removed_l"] = removed_from_post_treatment(wetland_df["TSS_treatable_load"], wetland_df["TSS_rate_l"])
    wetland_df["TSS_removed_h"] = removed_from_post_treatment(wetland_df["TSS_treatable_load"], wetland_df["TSS_rate_h"])

    wetland_df["Ecoli_removed_c"] = removed_from_post_treatment(wetland_df["Ecoli_treatable"], wetland_df["Ecoli_rate_c"])
    wetland_df["Ecoli_removed_l"] = removed_from_post_treatment(wetland_df["Ecoli_treatable"], wetland_df["Ecoli_rate_l"])
    wetland_df["Ecoli_removed_h"] = removed_from_post_treatment(wetland_df["Ecoli_treatable"], wetland_df["Ecoli_rate_h"])

    wetland_df["Ecoli_pre_treatment_c"] = wetland_df["upstream_Ecoli_median_fw"] / (
        1.0 - wetland_df["flow_scale"] * wetland_df["Ecoli_rate_c"]
    )
    wetland_df["Ecoli_pre_treatment_l"] = wetland_df["upstream_Ecoli_median_fw"] / (
        1.0 - wetland_df["flow_scale"] * wetland_df["Ecoli_rate_l"]
    )
    wetland_df["Ecoli_pre_treatment_h"] = wetland_df["upstream_Ecoli_median_fw"] / (
        1.0 - wetland_df["flow_scale"] * wetland_df["Ecoli_rate_h"]
    )

    wetland_df["wetland_id"] = wetland_df["wetland_id"].astype(np.int64)

    tn_c_map = dict(zip(wetland_df["wetland_id"], wetland_df["TN_removed_c"].astype("float32")))
    tn_l_map = dict(zip(wetland_df["wetland_id"], wetland_df["TN_removed_l"].astype("float32")))
    tn_h_map = dict(zip(wetland_df["wetland_id"], wetland_df["TN_removed_h"].astype("float32")))

    tp_c_map = dict(zip(wetland_df["wetland_id"], wetland_df["TP_removed_c"].astype("float32")))
    tp_l_map = dict(zip(wetland_df["wetland_id"], wetland_df["TP_removed_l"].astype("float32")))
    tp_h_map = dict(zip(wetland_df["wetland_id"], wetland_df["TP_removed_h"].astype("float32")))

    tss_c_map = dict(zip(wetland_df["wetland_id"], wetland_df["TSS_removed_c"].astype("float32")))
    tss_l_map = dict(zip(wetland_df["wetland_id"], wetland_df["TSS_removed_l"].astype("float32")))
    tss_h_map = dict(zip(wetland_df["wetland_id"], wetland_df["TSS_removed_h"].astype("float32")))

    ecoli_c_map = dict(zip(wetland_df["wetland_id"], wetland_df["Ecoli_removed_c"].astype("float32")))
    ecoli_l_map = dict(zip(wetland_df["wetland_id"], wetland_df["Ecoli_removed_l"].astype("float32")))
    ecoli_h_map = dict(zip(wetland_df["wetland_id"], wetland_df["Ecoli_removed_h"].astype("float32")))

    ecoli_pre_c_map = dict(zip(wetland_df["wetland_id"], wetland_df["Ecoli_pre_treatment_c"].astype("float32")))
    ecoli_pre_l_map = dict(zip(wetland_df["wetland_id"], wetland_df["Ecoli_pre_treatment_l"].astype("float32")))
    ecoli_pre_h_map = dict(zip(wetland_df["wetland_id"], wetland_df["Ecoli_pre_treatment_h"].astype("float32")))

    print("\nAllocating wetland-level removals equally across cells...")

    if should_run("TN"):
        print("\nTN allocation")
        tn_input_out = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tn_input_map)
        tn_out_c = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tn_c_map)
        tn_out_l = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tn_l_map)
        tn_out_h = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tn_h_map)

    if should_run("TP"):
        print("\nTP allocation")
        tp_input_out = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tp_input_map)
        tp_out_c = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tp_c_map)
        tp_out_l = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tp_l_map)
        tp_out_h = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tp_h_map)

    if should_run("TSS"):
        print("\nTSS allocation")
        tss_input_out = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tss_input_map)
        tss_out_c = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tss_c_map)
        tss_out_l = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tss_l_map)
        tss_out_h = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, tss_h_map)

    if should_run("Ecoli"):
        print("\nEcoli allocation")
        ecoli_input_out = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_input_map, allocation="constant")
        ecoli_q95_input_out = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_q95_input_map, allocation="constant")
        ecoli_g540_input_out = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_g540_input_map, allocation="constant")
        ecoli_out_c = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_c_map, allocation="constant")
        ecoli_out_l = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_l_map, allocation="constant")
        ecoli_out_h = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_h_map, allocation="constant")
        ecoli_pre_out_c = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_pre_c_map, allocation="constant")
        ecoli_pre_out_l = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_pre_l_map, allocation="constant")
        ecoli_pre_out_h = allocate_wetland_values_to_raster(wetland_ids_int, wetland_mask, wetland_valid_mask, ecoli_pre_h_map, allocation="constant")

    print("\nTreated flow allocation")
    treated_flow_out = allocate_wetland_values_to_raster(
        wetland_ids_int,
        wetland_mask,
        wetland_valid_mask,
        treated_flow_map
    )

    print("\nWriting outputs...")

    if should_run("TN"):
        write_raster(OUTPUT_FILES["TN_input"], tn_input_out, profile)
        write_raster(OUTPUT_FILES["TN"]["central"], tn_out_c, profile)
        write_raster(OUTPUT_FILES["TN"]["low"], tn_out_l, profile)
        write_raster(OUTPUT_FILES["TN"]["high"], tn_out_h, profile)

    if should_run("TP"):
        write_raster(OUTPUT_FILES["TP_input"], tp_input_out, profile)
        write_raster(OUTPUT_FILES["TP"]["central"], tp_out_c, profile)
        write_raster(OUTPUT_FILES["TP"]["low"], tp_out_l, profile)
        write_raster(OUTPUT_FILES["TP"]["high"], tp_out_h, profile)

    if should_run("TSS"):
        write_raster(OUTPUT_FILES["TSS_input"], tss_input_out, profile)
        write_raster(OUTPUT_FILES["TSS"]["central"], tss_out_c, profile)
        write_raster(OUTPUT_FILES["TSS"]["low"], tss_out_l, profile)
        write_raster(OUTPUT_FILES["TSS"]["high"], tss_out_h, profile)

    if should_run("Ecoli"):
        write_raster(OUTPUT_FILES["Ecoli_input"], ecoli_input_out, profile)
        write_raster(OUTPUT_FILES["Ecoli_Q95_input"], ecoli_q95_input_out, profile)
        write_raster(OUTPUT_FILES["Ecoli_G540_input"], ecoli_g540_input_out, profile)
        write_raster(OUTPUT_FILES["Ecoli"]["central"], ecoli_out_c, profile)
        write_raster(OUTPUT_FILES["Ecoli"]["low"], ecoli_out_l, profile)
        write_raster(OUTPUT_FILES["Ecoli"]["high"], ecoli_out_h, profile)
        write_raster(OUTPUT_FILES["Ecoli_pre"]["central"], ecoli_pre_out_c, profile)
        write_raster(OUTPUT_FILES["Ecoli_pre"]["low"], ecoli_pre_out_l, profile)
        write_raster(OUTPUT_FILES["Ecoli_pre"]["high"], ecoli_pre_out_h, profile)

    write_raster(OUTPUT_FILES["treated_flow"], treated_flow_out, profile)

    print("\nDone.")


if __name__ == "__main__":
    main()