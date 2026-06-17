from __future__ import annotations

import math
import os
import time
import numpy as np
import rasterio

"""
    Calculate rasterised contaminant values (N, P, sediment, E. coli) and total value.

    For each cell, nitrogen, phosphorus, and sediment values are calculated using
    removal quantities and unit values. E. coli value is derived from the change in
    NOF band classification with and without wetland treatment. The total value is
    the sum of all four components.

    Outputs:
        - total: combined value across all contaminants
        - N: nitrogen value
        - P: phosphorus value
        - S: sediment value
        - E: E. coli value

    Important:
        A single "valid" mask is applied across all outputs, requiring valid inputs
        for the E. coli calculation (median, median_no_wetland, q95, and g540).
        This means cells with valid N/P/S removal values but invalid E. coli inputs
        would be set to nodata for all outputs.
        This is not a problem with the contaminant data sourced from NIWA Rivermaps because it is complete. 
        If you were doing the analysis with some potentially missing data, you might want to change this.
    """

# -----------------------------
# User-defined constants
# -----------------------------
VARIATION = "central"
USE_LOW_HIGH_INPUT_RASTERS = True

LOG_EVERY_N_ROWS = 500  # progress interval

WORKING_DIR = r"<PROJECT_DIRECTORY>\Regulating\Wetlands\Intermediate"
OUT_DIR = r"<PROJECT_DIRECTORY>\Regulating\Wetlands\Output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_NODATA = -9999.0

OUTPUT_PATHS = {
    "central": {
        "total": os.path.join(OUT_DIR, "wetland_treatment_flow_value.tif"),
        "N": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_N.tif"),
        "P": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_P.tif"),
        "S": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_S.tif"),
        "E": os.path.join(OUT_DIR, "wetland_treatment_flow_value_E.tif"),
    },
    "low": {
        "total": os.path.join(OUT_DIR, "wetland_treatment_flow_value_low.tif"),
        "N": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_low_N.tif"),
        "P": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_low_P.tif"),
        "S": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_low_S.tif"),
        "E": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_low_E.tif"),
    },
    "high": {
        "total": os.path.join(OUT_DIR, "wetland_treatment_flow_value_high.tif"),
        "N": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_high_N.tif"),
        "P": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_high_P.tif"),
        "S": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_high_S.tif"),
        "E": os.path.join(WORKING_DIR, "wetland_treatment_flow_value_high_E.tif"),
    },
}

UNIT_VALUES = {
    "low": {"N": 13.6, "P": 68.0, "S": 2.04},
    "central": {"N": 27.2, "P": 136.0, "S": 4.08},
    "high": {"N": 54.4, "P": 272.0, "S": 8.16},
}

NOF_BAND_VALUES_BY_SCENARIO = {
    "central": {"A": 69.04, "B": 23.99, "C": 14.60, "D": 0.0, "E": 0.0},
    "low": {"A": 58.19, "B": 13.56, "C": 4.59, "D": 0.0, "E": 0.0},
    "high": {"A": 79.89, "B": 34.42, "C": 26.70, "D": 0.0, "E": 0.0},
}


# -----------------------------
# Logging helper
# -----------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# -----------------------------
# Raster paths
# -----------------------------
def build_input_paths(variation: str):
    return {
        "N_REMOVE": os.path.join(WORKING_DIR, f"tn_kg_removed_{variation}.tif"),
        "P_REMOVE": os.path.join(WORKING_DIR, f"tp_kg_removed_{variation}.tif"),
        "S_REMOVE": os.path.join(WORKING_DIR, f"tss_t_removed_{variation}.tif"),
        "E_MEDIAN_NO": os.path.join(WORKING_DIR, f"ecoli_median_no_wetland_{variation}.tif"),
        "E_MEDIAN": os.path.join(WORKING_DIR, "ecoli_fw_median_per_cell.tif"),
        "E_Q95": os.path.join(WORKING_DIR, "ecoli_q95_fw_per_cell.tif"),
        "E_G540": os.path.join(WORKING_DIR, "ecoli_g540_fw_per_cell.tif"),
    }


# -----------------------------
# Core logic
# -----------------------------
def get_nof_band(g540, q95):
    if g540 < 0.05:
        band_g = "A"
    elif g540 < 0.10:
        band_g = "B"
    elif g540 < 0.20:
        band_g = "C"
    elif g540 < 0.30:
        band_g = "D"
    else:
        band_g = "E"

    if q95 <= 540:
        band_q = "A"
    elif q95 <= 1000:
        band_q = "B"
    elif q95 <= 1200:
        band_q = "C"
    else:
        band_q = "D"

    bands = ["A", "B", "C", "D", "E"]
    return bands[max(bands.index(band_g), bands.index(band_q))]


def classify_stream(median, median_no, g540, q95):
    factor = median_no / median
    q95_no = q95 * factor

    sigma = math.log(q95 / median) / 1.645
    z = (math.log(540) - math.log(median_no)) / sigma
    g540_no = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    g540_no = max(0.0, min(1.0, g540_no))

    return (
        get_nof_band(g540, q95),
        get_nof_band(g540_no, q95_no),
    )


def read_raster(path):
    log(f"Reading {os.path.basename(path)}")
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        return arr, src.profile.copy(), src.nodata


def write_raster(path, arr, profile):
    log(f"Writing {os.path.basename(path)}")
    profile.update(dtype="float32", count=1, nodata=OUT_NODATA)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


# -----------------------------
# Main calc
# -----------------------------
def calculate_value_rasters(input_variation, value_variation):
    t0 = time.time()
    log(f"Starting calculation: input={input_variation}, value={value_variation}")

    paths = build_input_paths(input_variation)
    unit_vals = UNIT_VALUES[value_variation]
    nof_vals = NOF_BAND_VALUES_BY_SCENARIO[value_variation]

    # Read rasters
    n, profile, _ = read_raster(paths["N_REMOVE"])
    p, _, _ = read_raster(paths["P_REMOVE"])
    s, _, _ = read_raster(paths["S_REMOVE"])
    med_no, _, _ = read_raster(paths["E_MEDIAN_NO"])
    med, _, _ = read_raster(paths["E_MEDIAN"])
    q95, _, _ = read_raster(paths["E_Q95"])
    g540, _, _ = read_raster(paths["E_G540"])

    rows, cols = n.shape
    log(f"Raster size: {rows} x {cols}")

    out_total = np.full_like(n, OUT_NODATA, dtype="float64")
    out_n = np.full_like(n, OUT_NODATA, dtype="float64")
    out_p = np.full_like(n, OUT_NODATA, dtype="float64")
    out_s = np.full_like(n, OUT_NODATA, dtype="float64")
    out_e = np.full_like(n, OUT_NODATA, dtype="float64")

    # Per-contaminant values
    base_n = n * unit_vals["N"]
    base_p = p * unit_vals["P"]
    base_s = s * unit_vals["S"]

    valid = (
        (med > 0)
        & (med_no > 0)
        & (q95 > 0)
        & np.isfinite(g540)
    )

    log(f"med > 0: {np.sum(med > 0)}")
    log(f"med_no > 0: {np.sum(med_no > 0)}")
    log(f"q95 > 0: {np.sum(q95 > 0)}")
    log(f"finite g540: {np.sum(np.isfinite(g540))}")
    log(f"valid total: {np.sum(valid)}")
    log(f"valid %: {100 * np.mean(valid):.4f}%")

    log(f"med min/max: {np.nanmin(med)} / {np.nanmax(med)}")
    log(f"med_no min/max: {np.nanmin(med_no)} / {np.nanmax(med_no)}")
    log(f"q95 min/max: {np.nanmin(q95)} / {np.nanmax(q95)}")
    log(f"g540 min/max: {np.nanmin(g540)} / {np.nanmax(g540)}")

    log(f"Valid cells: {np.sum(valid)} ({np.mean(valid)*100:.1f}%)")

    # Row-wise loop (for logging)
    for i in range(rows):
        if i % LOG_EVERY_N_ROWS == 0:
            log(f"Processing row {i}/{rows}")

        row_mask = valid[i]

        if not np.any(row_mask):
            continue

        for j in np.where(row_mask)[0]:
            band_with, band_without = classify_stream(
                med[i, j], med_no[i, j], g540[i, j], q95[i, j]
            )

            ecoli_val = nof_vals[band_with] - nof_vals[band_without]

            out_n[i, j] = base_n[i, j]
            out_p[i, j] = base_p[i, j]
            out_s[i, j] = base_s[i, j]
            out_e[i, j] = ecoli_val
            out_total[i, j] = base_n[i, j] + base_p[i, j] + base_s[i, j] + ecoli_val

    # Prevent negative values in valid output cells
    for arr in [out_total, out_n, out_p, out_s, out_e]:
        valid_out = arr != OUT_NODATA
        arr[valid_out] = np.maximum(arr[valid_out], 0.0)

    log(f"Finished in {time.time() - t0:.1f}s")
    return {
        "total": out_total,
        "N": out_n,
        "P": out_p,
        "S": out_s,
        "E": out_e,
    }, profile


# -----------------------------
# Run
# -----------------------------
for scenario in ["central", "low", "high"]:
    input_var = scenario if USE_LOW_HIGH_INPUT_RASTERS else VARIATION

    outputs, prof = calculate_value_rasters(input_var, scenario)

    for key, arr in outputs.items():
        write_raster(OUTPUT_PATHS[scenario][key], arr, prof)

    log(f"Completed scenario: {scenario}")