"""
Create an additive sum raster from multiple input rasters.

The script can either use a manually supplied list of raster files or search an
input directory for GeoTIFFs whose filenames contain a specified partial match.

Each input raster is warped onto a common 100 m output grid defined by
TARGET_EXTENT. Input rasters may have different extents, but must use the same
CRS as the first raster. NoData cells are treated as zero during summation. Cells
where all inputs are NoData are written as OUTPUT_NODATA.

The output raster is written as float32 to OUTPUT_DIR.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

# Option 1: find rasters from a directory using a filename partial match
INPUT_DIR = r"D:\tmp\Processed rasters\Aggregated"
FILENAME_PARTIAL_MATCH = ""

# Option 2: provide rasters directly
INPUT_RASTERS = [
    # r"D:\tmp\Unprocessed rasters\raster_1.tif",
    # r"D:\tmp\Unprocessed rasters\raster_2.tif",
]

OUTPUT_DIR = r"D:\tmp\Processed rasters\Aggregated"
#OUTPUT_FILENAME = f"{FILENAME_PARTIAL_MATCH}_value.tif"
OUTPUT_FILENAME = "Total_value.tif"

# Extent format: xmin, ymin, xmax, ymax
#Land-only: 1089300,4162800,2470400,6223200
#Marine: 715100,3728500,2893500,7142400
TARGET_EXTENT = (715100, 3728500, 2893500, 7142400)

CELL_SIZE = 100
OUTPUT_NODATA = 0.0


def get_input_rasters():
    if INPUT_RASTERS:
        print(f"Using manually supplied raster list with {len(INPUT_RASTERS)} files.")
        return INPUT_RASTERS

    if not INPUT_DIR:
        raise ValueError("No INPUT_RASTERS or INPUT_DIR provided. The rasters are looking at you expectantly.")

    matched_files = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            fname = file.lower()

            if fname.endswith(".tif") and FILENAME_PARTIAL_MATCH.lower() in fname:
                matched_files.append(os.path.join(root, file))

    matched_files.sort()

    if not matched_files:
        raise ValueError(
            f"No raster files found in INPUT_DIR matching '{FILENAME_PARTIAL_MATCH}'. "
            "The directory has produced a dramatic silence."
        )

    print(f"Found {len(matched_files)} matching raster files.")
    return matched_files

def build_target_grid():
    xmin, ymin, xmax, ymax = TARGET_EXTENT

    width = int(round((xmax - xmin) / CELL_SIZE))
    height = int(round((ymax - ymin) / CELL_SIZE))

    if width <= 0 or height <= 0:
        raise ValueError("TARGET_EXTENT produces invalid raster dimensions. The extent has gone rogue.")

    transform = from_origin(xmin, ymax, CELL_SIZE, CELL_SIZE)

    return width, height, transform

def check_crs(src, reference_crs):
    if src.crs != reference_crs:
        raise ValueError(
            f"CRS differs from the first raster: {src.name}. "
            "Same grid, different universe. Refusing politely."
        )

def main():
    input_rasters = get_input_rasters()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    target_width, target_height, target_transform = build_target_grid()

    print("Reading reference raster...")
    with rasterio.open(input_rasters[0]) as ref:
        reference_crs = ref.crs
        reference_profile = ref.profile.copy()

    sum_array = np.zeros((target_height, target_width), dtype=np.float32)
    any_valid = np.zeros((target_height, target_width), dtype=bool)

    for raster_path in input_rasters:
        print(f"Processing: {raster_path}")

        with rasterio.open(raster_path) as src:
            check_crs(src, reference_crs)

            source_nodata = src.nodata
            INTERNAL_NODATA = -3.4028235e38 #temporary placeholder value used during processing

            warped = np.full(
                (target_height, target_width),
                INTERNAL_NODATA,
                dtype=np.float32
            )

            reproject(
                source=rasterio.band(src, 1),
                destination=warped,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=source_nodata,
                dst_transform=target_transform,
                dst_crs=reference_crs,
                dst_nodata=INTERNAL_NODATA,
                resampling=Resampling.nearest
            )

            valid_mask = np.isfinite(warped) & (warped != INTERNAL_NODATA)

            if source_nodata is not None:
                valid_mask &= warped != source_nodata

            # Optional, but useful if some rasters have bad/missing NoData metadata
            valid_mask &= ~np.isin(warped, [-9999, -32768, -3.4e38])

            sum_array[valid_mask] += warped[valid_mask]
            any_valid |= valid_mask

    output_array = np.where(any_valid, sum_array, OUTPUT_NODATA).astype(np.float32)

    output_profile = reference_profile.copy()
    output_profile.update(
        driver="GTiff",
        height=target_height,
        width=target_width,
        transform=target_transform,
        crs=reference_crs,
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize = 256,
        bigtiff="IF_SAFER"
    )

    print(f"Writing output raster: {output_path}")

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(output_array, 1)

        print("Calculating internal raster statistics...")

        valid_values = output_array[np.isfinite(output_array) & (output_array != OUTPUT_NODATA)]
        if valid_values.size > 0:
            raster_sum = float(valid_values.sum())

            dst.update_tags(
                1,
                STATISTICS_MINIMUM=float(valid_values.min()),
                STATISTICS_MAXIMUM=float(valid_values.max()),
                STATISTICS_MEAN=float(valid_values.mean()),
                STATISTICS_STDDEV=float(valid_values.std())
            )
        else:
            raster_sum = 0.0

        print("Building internal raster pyramids...")

        overview_levels = [2, 4, 8, 16, 32]
        dst.build_overviews(
            overview_levels,
            Resampling.average
        )
        dst.update_tags(ns="rio_overview", resampling="average")

    print(f"Additive sum raster created successfully. Raster sum: {raster_sum:,.2f}")

if __name__ == "__main__":
    main()