import os
import sys
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.crs import CRS

# -----------------------------
# Settings
# -----------------------------
INPUT_DIR = r"D:\Data\Crop suitability"
OUTPUT_SUBDIR = "NZGD2000"
OUTPUT_DIR = os.path.join(INPUT_DIR, OUTPUT_SUBDIR)

SRC_EPSG = 4326
DST_EPSG = 2193

# Extent is "minx,miny,maxx,maxy" in EPSG:2193 metres
EXTENT_STR = "1090000.0000,2089400.0000,4748100.0000,6194000.0000"
MINX, MINY, MAXX, MAXY = map(float, EXTENT_STR.split(","))

PIXEL_SIZE = 100.0  # metres

# Output grid: define transform and shape
width = int(np.ceil((MAXX - MINX) / PIXEL_SIZE))
height = int(np.ceil((MAXY - MINY) / PIXEL_SIZE))
transform_dst = from_origin(MINX, MAXY, PIXEL_SIZE, PIXEL_SIZE)  # origin at top-left

src_crs = CRS.from_epsg(SRC_EPSG)
dst_crs = CRS.from_epsg(DST_EPSG)

# -----------------------------
# Helpers
# -----------------------------
def prompt_continue():
    resp = input("Continue processing remaining rasters? (y/n): ").strip().lower()
    return resp in ("y", "yes")

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tif_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    if not tif_paths:
        print(f"No .tif files found in: {INPUT_DIR}")
        sys.exit(0)

    print(f"Found {len(tif_paths)} GeoTIFF(s) in: {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Target CRS: EPSG:{DST_EPSG}, cell size: {PIXEL_SIZE} m")
    print(f"Target extent (EPSG:{DST_EPSG}): {MINX}, {MINY}, {MAXX}, {MAXY}")
    print(f"Target grid: {width} cols x {height} rows\n")

    for i, in_path in enumerate(tif_paths, start=1):
        filename = os.path.basename(in_path)
        out_path = os.path.join(OUTPUT_DIR, filename)

        print(f"[{i}/{len(tif_paths)}] Processing: {filename}")

        with rasterio.open(in_path) as src:
            # Basic checks
            if src.count != 1:
                print(f"  - Skipping (expected 1 band, found {src.count})")
                continue

            if src.crs is None or src.crs.to_epsg() != SRC_EPSG:
                print(f"  - WARNING: Source CRS is {src.crs}, expected EPSG:{SRC_EPSG}. Proceeding anyway.")

            # Calculate max value (ignoring nodata)
            band = src.read(1, masked=True).astype(np.float32)

            # Build a clean validity mask: not nodata AND finite (so excludes NaN/Inf)
            valid = (~band.mask) & np.isfinite(band.data)

            if not np.any(valid):
                print("  - Skipping (no finite, non-nodata pixels)")
                continue

            max_val = float(np.nanmax(band.data[valid]))
            print(f"  - Max value (finite & non-nodata): {max_val}")

            if max_val <= 0 or not np.isfinite(max_val):
                print("  - WARNING: Max value is <= 0 or non-finite; output will be all zeros where data exists.")
                scaled = np.full(band.shape, np.nan, dtype=np.float32)
                scaled[valid] = 0.0
            else:
                scaled = np.full(band.shape, np.nan, dtype=np.float32)
                scaled[valid] = np.clip(band.data[valid] / max_val, 0.0, 1.0)


            # Prepare destination array
            dst_arr = np.full((height, width), np.nan, dtype=np.float32)

            # Reproject into fixed grid/extent
            reproject(
                source=scaled,
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs or src_crs,
                dst_transform=transform_dst,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

            # Write output
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                dtype=rasterio.float32,
                count=1,
                crs=dst_crs,
                transform=transform_dst,
                width=width,
                height=height,
                nodata=np.nan,
                compress="deflate",
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(dst_arr, 1)

        print(f"  - Saved: {out_path}\n")

        # Ask after the first raster only
        if i == 1:
            if not prompt_continue():
                print("Stopped after first raster.")
                break

    print("Done.")

if __name__ == "__main__":
    main()
