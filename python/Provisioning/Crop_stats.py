import os
import sys
import glob
import numpy as np
import rasterio
import csv

"""
    Summarise the contents of all the crop suitability GeoTIFF rasters in the input directory.
    (These came from https://landuseopportunities.nz/)

    This function scans `INPUT_DIR` for `.tif` files, reads each raster using
    Rasterio, and calculates the minimum, maximum, and mean values of all
    finite, non-nodata pixels in the first band. Results are written to a CSV
    file specified by `OUTPUT_CSV`.

    Output CSV columns:
        raster : Name of the raster file.
        min    : Minimum valid pixel value.
        max    : Maximum valid pixel value.
        mean   : Mean valid pixel value.

    Notes:
        - Rasters with no valid pixels are skipped.
        - Multi-band rasters are skipped.
        - CRS mismatches generate a warning but do not prevent processing.
        - Statistics are calculated only from finite, non-masked values.

    Returns:
        None
    """

# -----------------------------
# Settings
# -----------------------------
INPUT_DIR = r"D:\Data\Crop suitability"
OUTPUT_CSV = os.path.join(INPUT_DIR, "crop_suitability_summary.csv")

SRC_EPSG = 4326

# -----------------------------
# Main
# -----------------------------
def main():

    tif_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    if not tif_paths:
        print(f"No .tif files found in: {INPUT_DIR}")
        sys.exit(0)

    print(f"Found {len(tif_paths)} GeoTIFF(s) in: {INPUT_DIR}")
    print(f"Writing summary to: {OUTPUT_CSV}\n")

    rows = []

    for i, in_path in enumerate(tif_paths, start=1):
        filename = os.path.basename(in_path)
        print(f"[{i}/{len(tif_paths)}] Processing: {filename}")

        with rasterio.open(in_path) as src:

            if src.count != 1:
                print("  - Skipping (expected 1 band)")
                continue

            if src.crs is None or src.crs.to_epsg() != SRC_EPSG:
                print(f"  - WARNING: CRS is {src.crs}, expected EPSG:{SRC_EPSG}")

            band = src.read(1, masked=True).astype(np.float32)

            valid = (~band.mask) & np.isfinite(band.data)

            if not np.any(valid):
                print("  - Skipping (no finite, non-nodata pixels)")
                continue

            min_val = float(np.nanmin(band.data[valid]))
            max_val = float(np.nanmax(band.data[valid]))
            mean_val = float(np.nanmean(band.data[valid]))

            rows.append({
                "raster": filename,
                "min": min_val,
                "max": max_val,
                "mean": mean_val
            })

            print(f"  - min: {min_val}, max: {max_val}, mean: {mean_val}")

    # -----------------------------
    # Write CSV
    # -----------------------------
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["raster", "min", "max", "mean"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Summary written to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
