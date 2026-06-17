#!/usr/bin/env python3
"""
Back-calculate contaminant loads removed by wetlands from observed load rasters.

This script estimates the amount of contaminant removed by wetlands by
inverting a removal efficiency function. Given observed loads (post-removal),
it reconstructs the counterfactual load without wetlands and derives the
removed component:

    removed = observed * rate / (1 - rate)

Removal rates are interpolated from a lookup table as a function of
catchment wetland proportion. Nitrogen (TN) rates vary by climate
(warm vs cool), while TP, TSS, and E. coli use a single relationship.

Inputs
------
- wetland.tif:
    Binary or continuous wetland extent. Cells > 0 are treated as wetlands.
- wetland_proportion.tif:
    Proportion of upstream catchment in wetlands (used for rate lookup).
- warm_climate_dummy.tif:
    Climate classification (>= 0.5 = warm, < 0.5 = cool).
- Observed contaminant rasters:
    TN (N_load.tif), TP (P_load.tif), TSS (sed_load.tif), E. coli.
- wetland_performance.csv:
    Lookup table of removal rates by catchment proportion, including
    central, low, and high estimates.

Outputs
-------
For each contaminant, three rasters (central, low, high) representing
the estimated load removed by wetlands. Cells outside wetlands or with
invalid inputs are assigned nodata (-9999).

Key assumptions
---------------
- All rasters are perfectly aligned (same CRS, transform, resolution, extent).
- Observed loads represent post-wetland conditions.
- Removal rates are bounded below 1 to avoid division by zero.
- Linear interpolation between table values is appropriate.

Diagnostics
-----------
The script prints detailed diagnostics to help identify data issues:
- Raster validity summaries (valid vs nodata cells)
- Alignment checks (CRS, transform, shape)
- Mask sizes (wetland, climate classes)
- Rate distributions after interpolation
- Overlap between wetlands and valid observed data

Notes
-----
- Rates are clipped to [0, 0.999999] for numerical stability.
- GeoTIFF outputs use DEFLATE compression with tiling.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import os

# =============================================================================
# FILES AND DIRECTORIES
# =============================================================================
CONTAMINANTS_TO_RUN = ["TSS"]   # e.g. ["TSS"] or ["TN","TP"]

BASE_DIR = Path(r"<PROJECT_DIRECTORY>\Wetlands")
INPUT_DIR = BASE_DIR / "Intermediate"
OUTPUT_DIR = BASE_DIR / "Intermediate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input rasters
WETLAND_RASTER = INPUT_DIR / "wetland.tif"
WETLAND_PROPORTION_RASTER = INPUT_DIR / "wetland_proportion.tif"
WARM_CLIMATE_DUMMY_RASTER = INPUT_DIR / "warm_climate_dummy.tif"

# Observed contaminant rasters
TN_RASTER = INPUT_DIR / "N_load.tif"
TP_RASTER = INPUT_DIR / "P_load.tif"
TSS_RASTER = INPUT_DIR / "sed_load.tif"
ECOLI_RASTER = INPUT_DIR / "ecoli_median.tif"

# Removal rate CSV
REMOVAL_RATES_CSV = INPUT_DIR / "wetland_performance.csv"

# Output rasters
OUTPUT_FILES = {
    "TN": {
        "central": OUTPUT_DIR / "tn_removed_central.tif",
        "low": OUTPUT_DIR / "tn_removed_low.tif",
        "high": OUTPUT_DIR / "tn_removed_high.tif",
    },
    "TP": {
        "central": OUTPUT_DIR / "tp_removed_central.tif",
        "low": OUTPUT_DIR / "tp_removed_low.tif",
        "high": OUTPUT_DIR / "tp_removed_high.tif",
    },
    "TSS": {
        "central": OUTPUT_DIR / "tss_removed_central.tif",
        "low": OUTPUT_DIR / "tss_removed_low.tif",
        "high": OUTPUT_DIR / "tss_removed_high.tif",
    },
    "Ecoli": {
        "central": OUTPUT_DIR / "ecoli_median_no_wetland_central.tif",
        "low": OUTPUT_DIR / "ecoli_median_no_wetland_low.tif",
        "high": OUTPUT_DIR / "ecoli_median_no_wetland_high.tif",
    },
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


def write_raster(path, arr, template_profile):
    """Write float32 GeoTIFF. Falls back to new filename if write fails."""
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
    attempt = 0

    while True:
        try:
            out_path = path if attempt == 0 else f"{base}_{attempt}{ext}"
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr.astype("float32"), 1)
            return out_path
        except Exception as e:
            print(f"Write failed for {path} on attempt {attempt + 1}: {e}")
            attempt += 1


def backcalculate_removed(observed, rate):
    """
    removed = observed/(1-rate) - observed
            = observed * rate / (1-rate)

    Rates are clipped just below 1 to avoid division by zero.
    """
    safe_rate = np.clip(rate, 0.0, 0.999999)
    return observed * safe_rate / (1.0 - safe_rate)


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
        problems.append(f"shape differs: {(profile['height'], profile['width'])} vs {(ref_profile['height'], ref_profile['width'])}")

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


def print_rate_summary(name, rate_arr, mask):
    n = int(mask.sum())
    print(f"\n{name} rates")
    print(f"  cells with rates: {n:,}")
    if n > 0:
        vals = rate_arr[mask]
        print(f"  min: {np.min(vals):.6g}")
        print(f"  max: {np.max(vals):.6g}")
        print(f"  mean: {np.mean(vals):.6g}")
        if np.any(~np.isfinite(vals)):
            print("  WARNING: non-finite rates found")
    else:
        print("  WARNING: no cells selected for rate assignment")


def diagnose_contaminant(name, obs, obs_nodata, wetland_mask, rate_c):
    obs_valid = build_valid_mask(obs, obs_nodata)
    overlap = wetland_mask & obs_valid

    print(f"\n{name} diagnostics")
    print(f"  observed valid cells: {int(obs_valid.sum()):,}")
    print(f"  wetland cells: {int(wetland_mask.sum()):,}")
    print(f"  overlap of observed valid and wetland: {int(overlap.sum()):,}")

    if np.any(obs_valid):
        vals = obs[obs_valid]
        print(f"  observed min/max on valid cells: {np.min(vals):.6g} / {np.max(vals):.6g}")

    if np.any(overlap):
        overlap_obs = obs[overlap]
        overlap_rate = rate_c[overlap]
        print(f"  overlap observed min/max: {np.min(overlap_obs):.6g} / {np.max(overlap_obs):.6g}")
        print(f"  overlap rate min/max: {np.min(overlap_rate):.6g} / {np.max(overlap_rate):.6g}")
    else:
        print("  WARNING: no valid overlap, output will be all nodata")

        wetland_only_invalid_obs = wetland_mask & ~obs_valid
        print(f"  wetland cells where observed raster is invalid: {int(wetland_only_invalid_obs.sum()):,}")

        if np.any(wetland_mask):
            wetland_obs = obs[wetland_mask]
            finite_on_wetlands = np.isfinite(wetland_obs)
            print(f"  finite observed values on wetland cells: {int(finite_on_wetlands.sum()):,} / {int(wetland_mask.sum()):,}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Reading rasters...")

    wetland, profile, wetland_nodata = read_raster(WETLAND_RASTER)
    wetland_prop, wetland_prop_profile, wetland_prop_nodata = read_raster(WETLAND_PROPORTION_RASTER)
    warm_dummy, warm_dummy_profile, warm_dummy_nodata = read_raster(WARM_CLIMATE_DUMMY_RASTER)

    tn_obs, tn_profile, tn_nodata = read_raster(TN_RASTER)
    tp_obs, tp_profile, tp_nodata = read_raster(TP_RASTER)
    tss_obs, tss_profile, tss_nodata = read_raster(TSS_RASTER)
    ecoli_obs, ecoli_profile, ecoli_nodata = read_raster(ECOLI_RASTER)

    print("\nChecking raster alignment...")
    check_alignment("wetland_proportion", wetland_prop_profile, profile)
    check_alignment("warm_climate_dummy", warm_dummy_profile, profile)
    check_alignment("TN", tn_profile, profile)
    check_alignment("TP", tp_profile, profile)
    check_alignment("TSS", tss_profile, profile)
    check_alignment("Ecoli", ecoli_profile, profile)

    print("\nRaster summaries...")
    print_raster_summary("wetland", wetland, wetland_nodata)
    print_raster_summary("wetland_proportion", wetland_prop, wetland_prop_nodata)
    print_raster_summary("warm_climate_dummy", warm_dummy, warm_dummy_nodata)
    print_raster_summary("TN observed", tn_obs, tn_nodata)
    print_raster_summary("TP observed", tp_obs, tp_nodata)
    print_raster_summary("TSS observed", tss_obs, tss_nodata)
    print_raster_summary("Ecoli observed", ecoli_obs, ecoli_nodata)

    print("\nReading removal rate table...")
    rates = pd.read_csv(REMOVAL_RATES_CSV)
    rates.columns = [c.strip() for c in rates.columns]

    required_cols = [
        "catchment_proportion",
        "TN_warm", "TN_cool", "TN_warm_low", "TN_warm_high", "TN_cool_low", "TN_cool_high",
        "TP", "TP_low", "TP_high",
        "TSS", "TSS_low", "TSS_high",
        "Ecoli", "Ecoli_low", "Ecoli_high",
    ]
    missing_cols = [c for c in required_cols if c not in rates.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in removal table: {missing_cols}")

    if rates["catchment_proportion"].isna().any():
        raise ValueError("catchment_proportion contains NA values")

    x = rates["catchment_proportion"].to_numpy(dtype="float32")
    if np.any(~np.isfinite(x)):
        raise ValueError("catchment_proportion contains non-finite values")

    if np.any(np.diff(x) < 0):
        raise ValueError("catchment_proportion must be sorted ascending for np.interp")

    # TN depends on warm/cool climate
    tn_warm = rates["TN_warm"].to_numpy(dtype="float32")
    tn_cool = rates["TN_cool"].to_numpy(dtype="float32")
    tn_warm_low = rates["TN_warm_low"].to_numpy(dtype="float32")
    tn_warm_high = rates["TN_warm_high"].to_numpy(dtype="float32")
    tn_cool_low = rates["TN_cool_low"].to_numpy(dtype="float32")
    tn_cool_high = rates["TN_cool_high"].to_numpy(dtype="float32")

    # Others do not vary by climate in the table
    tp_rate = rates["TP"].to_numpy(dtype="float32")
    tp_low = rates["TP_low"].to_numpy(dtype="float32")
    tp_high = rates["TP_high"].to_numpy(dtype="float32")

    tss_rate = rates["TSS"].to_numpy(dtype="float32")
    tss_low = rates["TSS_low"].to_numpy(dtype="float32")
    tss_high = rates["TSS_high"].to_numpy(dtype="float32")

    ecoli_rate = rates["Ecoli"].to_numpy(dtype="float32")
    ecoli_low = rates["Ecoli_low"].to_numpy(dtype="float32")
    ecoli_high = rates["Ecoli_high"].to_numpy(dtype="float32")

    for name, arr in {
        "TN_warm": tn_warm,
        "TN_cool": tn_cool,
        "TP": tp_rate,
        "TSS": tss_rate,
        "Ecoli": ecoli_rate,
    }.items():
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} rates contain non-finite values")

    print("\nBuilding masks...")
    common_valid = (
        build_valid_mask(wetland, wetland_nodata) &
        build_valid_mask(wetland_prop, wetland_prop_nodata) &
        build_valid_mask(warm_dummy, warm_dummy_nodata)
    )

    wetland_mask = common_valid & (wetland > 0)
    warm_mask = wetland_mask & (warm_dummy >= 0.5)
    cool_mask = wetland_mask & (warm_dummy < 0.5)

    print_mask_summary("common_valid", common_valid)
    print_mask_summary("wetland_mask", wetland_mask)
    print_mask_summary("warm_mask", warm_mask)
    print_mask_summary("cool_mask", cool_mask)

    if np.any(wetland_mask):
        wetland_props = wetland_prop[wetland_mask]
        print("\nWetland proportion on wetland cells")
        print(f"  min: {np.min(wetland_props):.6g}")
        print(f"  max: {np.max(wetland_props):.6g}")
        print(f"  mean: {np.mean(wetland_props):.6g}")
        print(f"  table range: {x.min():.6g} to {x.max():.6g}")
        print(f"  below table min: {int((wetland_props < x.min()).sum()):,}")
        print(f"  above table max: {int((wetland_props > x.max()).sum()):,}")
    else:
        print("WARNING: wetland_mask has no true cells, all outputs will be nodata")

    # -------------------------------------------------------------------------
    # LOOK UP RATES
    # -------------------------------------------------------------------------
    print("\nLooking up rates...")

    tn_rate_c = np.zeros(wetland.shape, dtype="float32")
    tn_rate_l = np.zeros(wetland.shape, dtype="float32")
    tn_rate_h = np.zeros(wetland.shape, dtype="float32")

    tp_rate_c = np.zeros(wetland.shape, dtype="float32")
    tp_rate_l = np.zeros(wetland.shape, dtype="float32")
    tp_rate_h = np.zeros(wetland.shape, dtype="float32")

    tss_rate_c = np.zeros(wetland.shape, dtype="float32")
    tss_rate_l = np.zeros(wetland.shape, dtype="float32")
    tss_rate_h = np.zeros(wetland.shape, dtype="float32")

    ecoli_rate_c = np.zeros(wetland.shape, dtype="float32")
    ecoli_rate_l = np.zeros(wetland.shape, dtype="float32")
    ecoli_rate_h = np.zeros(wetland.shape, dtype="float32")

    if np.any(warm_mask):
        p = wetland_prop[warm_mask]
        tn_rate_c[warm_mask] = interp_rates(p, x, tn_warm)
        tn_rate_l[warm_mask] = interp_rates(p, x, tn_warm_low)
        tn_rate_h[warm_mask] = interp_rates(p, x, tn_warm_high)

    if np.any(cool_mask):
        p = wetland_prop[cool_mask]
        tn_rate_c[cool_mask] = interp_rates(p, x, tn_cool)
        tn_rate_l[cool_mask] = interp_rates(p, x, tn_cool_low)
        tn_rate_h[cool_mask] = interp_rates(p, x, tn_cool_high)

    if np.any(wetland_mask):
        p = wetland_prop[wetland_mask]

        tp_rate_c[wetland_mask] = interp_rates(p, x, tp_rate)
        tp_rate_l[wetland_mask] = interp_rates(p, x, tp_low)
        tp_rate_h[wetland_mask] = interp_rates(p, x, tp_high)

        tss_rate_c[wetland_mask] = interp_rates(p, x, tss_rate)
        tss_rate_l[wetland_mask] = interp_rates(p, x, tss_low)
        tss_rate_h[wetland_mask] = interp_rates(p, x, tss_high)

        ecoli_rate_c[wetland_mask] = interp_rates(p, x, ecoli_rate)
        ecoli_rate_l[wetland_mask] = interp_rates(p, x, ecoli_low)
        ecoli_rate_h[wetland_mask] = interp_rates(p, x, ecoli_high)

    print_rate_summary("TN central", tn_rate_c, wetland_mask)
    print_rate_summary("TP central", tp_rate_c, wetland_mask)
    print_rate_summary("TSS central", tss_rate_c, wetland_mask)
    print_rate_summary("Ecoli central", ecoli_rate_c, wetland_mask)

    # -------------------------------------------------------------------------
    # DIAGNOSTICS BEFORE PROCESSING
    # -------------------------------------------------------------------------
    if should_run("TN"):
        diagnose_contaminant("TN", tn_obs, tn_nodata, wetland_mask, tn_rate_c)
    if should_run("TP"):
        diagnose_contaminant("TP", tp_obs, tp_nodata, wetland_mask, tp_rate_c)
    if should_run("TSS"):
        diagnose_contaminant("TSS", tss_obs, tss_nodata, wetland_mask, tss_rate_c)
    if should_run("Ecoli"):
        diagnose_contaminant("Ecoli", ecoli_obs, ecoli_nodata, wetland_mask, ecoli_rate_c)

    # Extra TSS-specific checks
    tss_valid = build_valid_mask(tss_obs, tss_nodata)
    tss_overlap = wetland_mask & tss_valid

    print("\nExtra TSS checks")
    print(f"  TSS nodata value: {tss_nodata}")
    print(f"  finite TSS cells: {int(np.isfinite(tss_obs).sum()):,}")
    print(f"  valid TSS cells under nodata rule: {int(tss_valid.sum()):,}")
    print(f"  wetland/TSS overlap cells: {int(tss_overlap.sum()):,}")

    if np.any(tss_valid):
        print(f"  TSS unique sample (first 10 valid values): {np.unique(tss_obs[tss_valid])[:10]}")
    else:
        print("  WARNING: TSS raster has no valid cells at all")

    if np.any(wetland_mask) and not np.any(tss_overlap):
        print("  WARNING: TSS has no valid cells where wetlands exist")
        print("  This usually means one of these:")
        print("    1. TSS raster nodata value is wrong")
        print("    2. TSS raster is spatially misaligned despite matching dimensions")
        print("    3. Wetland cells fall only in TSS nodata area")
        print("    4. sed_load.tif is not the raster you think it is")

    # -------------------------------------------------------------------------
    # BACK-CALCULATE REMOVED LOADS
    # -------------------------------------------------------------------------
    print("\nCalculating removed loads...")

    def process_contaminant(name, obs, obs_nodata, rate_c, rate_l, rate_h):
        valid = wetland_mask & build_valid_mask(obs, obs_nodata)

        out_c = np.full(obs.shape, OUT_NODATA, dtype="float32")
        out_l = np.full(obs.shape, OUT_NODATA, dtype="float32")
        out_h = np.full(obs.shape, OUT_NODATA, dtype="float32")

        n_valid = int(valid.sum())
        print(f"{name}: cells to write = {n_valid:,}")

        if n_valid == 0:
            print(f"WARNING: {name} output will be all nodata")

        out_c[valid] = backcalculate_removed(obs[valid], rate_c[valid])
        out_l[valid] = backcalculate_removed(obs[valid], rate_l[valid])
        out_h[valid] = backcalculate_removed(obs[valid], rate_h[valid])

        for label, arr in [("central", out_c), ("low", out_l), ("high", out_h)]:
            valid_out = arr != OUT_NODATA
            if np.any(valid_out):
                vals = arr[valid_out]
                print(f"  {name} {label} min/max: {np.min(vals):.6g} / {np.max(vals):.6g}")
            else:
                print(f"  {name} {label}: no valid output cells")

        return out_c, out_l, out_h

    if should_run("TN"):
        tn_out_c, tn_out_l, tn_out_h = process_contaminant(
            "TN", tn_obs, tn_nodata, tn_rate_c, tn_rate_l, tn_rate_h
        )

    if should_run("TP"):
        tp_out_c, tp_out_l, tp_out_h = process_contaminant(
            "TP", tp_obs, tp_nodata, tp_rate_c, tp_rate_l, tp_rate_h
        )

    if should_run("TSS"):
        tss_out_c, tss_out_l, tss_out_h = process_contaminant(
            "TSS", tss_obs, tss_nodata, tss_rate_c, tss_rate_l, tss_rate_h
        )

    if should_run("Ecoli"):
        ecoli_out_c, ecoli_out_l, ecoli_out_h = process_contaminant(
            "Ecoli", ecoli_obs, ecoli_nodata, ecoli_rate_c, ecoli_rate_l, ecoli_rate_h
        )
    # -------------------------------------------------------------------------
    # WRITE OUTPUTS
    # -------------------------------------------------------------------------
    print("\nWriting outputs...")

    if should_run("TN"):
        write_raster(OUTPUT_FILES["TN"]["central"], tn_out_c, profile)
        write_raster(OUTPUT_FILES["TN"]["low"], tn_out_l, profile)
        write_raster(OUTPUT_FILES["TN"]["high"], tn_out_h, profile)

    if should_run("TP"):
        write_raster(OUTPUT_FILES["TP"]["central"], tp_out_c, profile)
        write_raster(OUTPUT_FILES["TP"]["low"], tp_out_l, profile)
        write_raster(OUTPUT_FILES["TP"]["high"], tp_out_h, profile)

    if should_run("TSS"):
        write_raster(OUTPUT_FILES["TSS"]["central"], tss_out_c, profile)
        write_raster(OUTPUT_FILES["TSS"]["low"], tss_out_l, profile)
        write_raster(OUTPUT_FILES["TSS"]["high"], tss_out_h, profile)

    if should_run("Ecoli"):
        write_raster(OUTPUT_FILES["Ecoli"]["central"], ecoli_out_c, profile)
        write_raster(OUTPUT_FILES["Ecoli"]["low"], ecoli_out_l, profile)
        write_raster(OUTPUT_FILES["Ecoli"]["high"], ecoli_out_h, profile)

    print("\nDone.")


if __name__ == "__main__":
    main()