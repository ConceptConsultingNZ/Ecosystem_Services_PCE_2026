import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
sys.path.append(str(Path(__file__).resolve().parents[1]))
from raster_utils import write_float32_raster


OUT_DIR = r"<PROJECT_DIRECTORY>\Provisioning\Timber\Output"
WORK_DIR = r"<PROJECT_DIRECTORY>\Provisioning\Timber\Intermediate"

REGION_ID_RASTER = r"<PROJECT_DIRECTORY>\Common\region.tif"
INDEX_300_RASTER = os.path.join(WORK_DIR, "300_index.tif")
FOREST_EXTENT_RASTER = os.path.join(WORK_DIR, "nzlum_plantation_forest_prop.tif")
REGION_VALUE_CSV = os.path.join(WORK_DIR, "region_values.csv")

CENTRAL_OUTPUT_FILENAME = "timber_value.tif"
LOW_OUTPUT_FILENAME = "timber_value_low.tif"
HIGH_OUTPUT_FILENAME = "timber_value_high.tif"

def report_raster_stats(array, label):
    valid_mask = np.isfinite(array) & (array != 0)

    total_value = np.sum(array[valid_mask], dtype=np.float64)
    valid_cells = np.count_nonzero(valid_mask)
    total_cells = array.size
    percent_valid = 100 * valid_cells / total_cells

    print(f"{label} total value: {total_value:,.2f}")
    print(f"{label} valid cells: {valid_cells:,} of {total_cells:,} ({percent_valid:.2f}%)")


def main():
    print("Reading region values CSV...")
    values_df = pd.read_csv(REGION_VALUE_CSV)

    required_cols = {"region_id", "central", "low", "high"}
    missing_cols = required_cols - set(values_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {missing_cols}")

    values_df["region_id"] = values_df["region_id"].astype(int)

    value_maps = {
        "central": dict(zip(values_df["region_id"], values_df["central"])),
        "low": dict(zip(values_df["region_id"], values_df["low"])),
        "high": dict(zip(values_df["region_id"], values_df["high"])),
    }

    output_filenames = {
        "central": CENTRAL_OUTPUT_FILENAME,
        "low": LOW_OUTPUT_FILENAME,
        "high": HIGH_OUTPUT_FILENAME,
    }

    print("Reading rasters...")
    with rasterio.open(REGION_ID_RASTER) as region_src, \
            rasterio.open(INDEX_300_RASTER) as index_src, \
            rasterio.open(FOREST_EXTENT_RASTER) as forest_src:

        region = region_src.read(1)
        index_300 = index_src.read(1).astype(np.float32)
        forest_prop = forest_src.read(1).astype(np.float32)

        profile = index_src.profile.copy()

        if region.shape != index_300.shape or forest_prop.shape != index_300.shape:
            raise ValueError("Input rasters do not have matching dimensions.")

    forest_prop = np.nan_to_num(forest_prop, nan=0.0)
    forest_prop = np.clip(forest_prop, 0.0, 1.0)

    #Lookup the default 300-index to use if the raster has nodata
    index_lookup = dict(zip(values_df["region_id"], values_df["300_index"]))
    index_nodata = profile.get("nodata", None)

    if index_nodata is not None:
        index_invalid = index_300 == index_nodata
    else:
        index_invalid = ~np.isfinite(index_300)

    for region_id, value in index_lookup.items():
        index_300[(region == region_id) & index_invalid] = value

    index_300 = np.nan_to_num(index_300, nan=0.0)

    print("Creating output rasters...")
    for field_name, lookup in value_maps.items():
        print(f"Processing {field_name}...")

        region_values = np.zeros(region.shape, dtype=np.float32)

        for region_id, value in lookup.items():
            region_values[region == region_id] = value

        output_array = region_values * index_300 * forest_prop
        output_array = np.nan_to_num(output_array, nan=0.0).astype(np.float32)

        report_raster_stats(output_array, field_name)

        output_filename = output_filenames[field_name]

        write_float32_raster(
            output_array,
            OUT_DIR,
            output_filename,
            profile,
            nodata=0.0
        )

        print(f"Saved {output_filename}")

    print("Timber provisioning rasters calculated successfully.")


if __name__ == "__main__":
    main()