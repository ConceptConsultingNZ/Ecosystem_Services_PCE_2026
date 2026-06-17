from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

"""
Carbon Sequestration Raster Processing Script

This script calculates spatial estimates of carbon sequestration (tonnes) and
associated monetary values ($) from multiple habitat rasters (e.g. mangrove,
saltmarsh, seagrass).

For each input raster, habitat extent or presence is multiplied by habitat-
specific sequestration rates (central, low, high scenarios). Outputs are then
combined across habitats on a per-pixel basis using a maximum-value rule,
such that overlapping habitats are resolved by selecting the highest-value
habitat contribution per cell.

Outputs include:
    - Tonnes of carbon sequestered per year (central, low, high)
    - Monetary value of sequestration (central, low, high)

Monetary values are calculated using a single shadow price of carbon applied
consistently across all habitats, in line with standard cost–benefit analysis.

Key assumptions:
    - Input rasters are spatially aligned or will be reprojected to a common grid
    - Raster values represent habitat extent (e.g. hectares per pixel) or a
      proportional proxy of habitat presence
    - Sequestration rates are expressed per unit of raster value (e.g. tCO₂e/ha/year)
    - Overlapping habitats are mutually exclusive in the final output via a
      maximum-value selection (not summed)
    - No double counting of carbon occurs across habitats
"""

# =========================
# Constants
# =========================

INPUT_DIR = Path(r"<PROJECT_DIRECTORY>\Regulating\Blue carbon\Intermediate")
OUTPUT_DIR = Path(r"<PROJECT_DIRECTORY>\Regulating\Blue carbon\Output")

MANGROVE_RASTER = INPUT_DIR / "mangrove.tif"
SALTMARSH_RASTER = INPUT_DIR / "saltmarsh.tif"
SEAGRASS_RASTER = INPUT_DIR / "seagrass.tif"

CARBON_DOLLAR_VALUE = 106.0 #Treasury shadow price of carbon for 2026

#Rates from Berthelsen et al. 2025. Converted to CO2
SEQUESTRATION_RATES = {
    "mangrove": {"file": MANGROVE_RASTER, "central": 1.43, "low": 0.909333333, "high": 3.784},
    "saltmarsh": {"file": SALTMARSH_RASTER, "central": 2.163333333, "low": 1.8979, "high": 5.325466667},
    "seagrass": {"file": SEAGRASS_RASTER, "central": 0.146666667, "low": 0.0748, "high": 0.218533333},
}

OUTPUT_TONNES_CENTRAL = OUTPUT_DIR / "regulating_blue_carbon_flow_t.tif"
OUTPUT_TONNES_LOW = OUTPUT_DIR / "regulating_blue_carbon_flow_t_low.tif"
OUTPUT_TONNES_HIGH = OUTPUT_DIR / "regulating_blue_carbon_flow_t_high.tif"

OUTPUT_VALUE_CENTRAL = OUTPUT_DIR / "regulating_blue_carbon_flow_value.tif"
OUTPUT_VALUE_LOW = OUTPUT_DIR / "regulating_blue_carbon_flow_value_low.tif"
OUTPUT_VALUE_HIGH = OUTPUT_DIR / "regulating_blue_carbon_flow_value_high.tif"


def read_align(path, profile, transform, crs):
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata

        same = (
            src.crs == crs
            and src.transform == transform
            and src.width == profile["width"]
            and src.height == profile["height"]
        )

        if same:
            if nodata is not None:
                data[data == nodata] = np.nan
            return data

        dst = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")

        reproject(
            source=data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        return dst


def write(path, arr, profile):
    nodata = -9999.0
    out = np.where(np.isnan(arr), nodata, arr).astype("float32")

    profile = profile.copy()
    profile.update(dtype="float32", count=1, compress="deflate", nodata=nodata)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    template_path = next(iter(SEQUESTRATION_RATES.values()))["file"]
    with rasterio.open(template_path) as t:
        profile = t.profile
        transform = t.transform
        crs = t.crs

    shape = (profile["height"], profile["width"])

    # Initialise outputs as nan (important for max logic)
    ton_c = np.full(shape, np.nan, dtype="float32")
    ton_l = np.full(shape, np.nan, dtype="float32")
    ton_h = np.full(shape, np.nan, dtype="float32")

    for name, info in SEQUESTRATION_RATES.items():
        arr = read_align(info["file"], profile, transform, crs)

        # Calculate tonnes for this habitat
        t_c = arr * info["central"]
        t_l = arr * info["low"]
        t_h = arr * info["high"]

        # Stack with existing and take max
        ton_c = np.nanmax(np.stack([ton_c, t_c]), axis=0)
        ton_l = np.nanmax(np.stack([ton_l, t_l]), axis=0)
        ton_h = np.nanmax(np.stack([ton_h, t_h]), axis=0)

        print(f"Processed {name}")

    # Convert to value
    val_c = ton_c * CARBON_DOLLAR_VALUE
    val_l = ton_l * CARBON_DOLLAR_VALUE
    val_h = ton_h * CARBON_DOLLAR_VALUE

    # Write outputs
    write(OUTPUT_TONNES_CENTRAL, ton_c, profile)
    write(OUTPUT_TONNES_LOW, ton_l, profile)
    write(OUTPUT_TONNES_HIGH, ton_h, profile)

    write(OUTPUT_VALUE_CENTRAL, val_c, profile)
    write(OUTPUT_VALUE_LOW, val_l, profile)
    write(OUTPUT_VALUE_HIGH, val_h, profile)

    print("Done (max-priority overlap handling).")


if __name__ == "__main__":
    main()