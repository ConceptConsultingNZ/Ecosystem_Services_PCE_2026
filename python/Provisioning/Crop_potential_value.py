"""
Crop provisioning ES: maximum potential value across crops, with mainland mask.

What it does
------------
1) Reads multiple crop "yield suitability" rasters (0..1) on the same grid.
2) Multiplies each raster by a crop-specific gross output value (NZ$/ha) from a CSV.
3) Takes the per-cell maximum across crops to create an upper-bound potential supply raster.
4) Masks the result to the Mainland polygon (everything outside becomes nodata).

Inputs
------
- rasters_dir: folder containing the yield rasters (filenames match CSV yield_file column)
- csv_path: CSV with columns [yield_file, value]
- mainland_shp: polygon shapefile defining the mainland mask

Output
------
- crop_supply.tif: Int32 GeoTIFF, DEFLATE-compressed, masked to mainland
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import fiona


def main():
    # -------------------------
    # User inputs
    # -------------------------
    rasters_dir = r"D:\Data\Crop suitability\NZGD2000"
    csv_path = r"<PROJECT_DIRECTORY>\Provisioning\Crops\potential_value.csv"
    out_path = r"<PROJECT_DIRECTORY>\Provisioning\Crops\crop_supply.tif"
    mainland_shp = r"D:\Data\LINZ\lds-nz-coastlines-and-islands-polygons-topo-150k-GPKG\Mainland.shp"

    # Output settings (conservative, widely compatible)
    OUT_DTYPE = np.int32
    OUT_NODATA = np.iinfo(np.int32).min  # -2147483648
    COMPRESS = "DEFLATE"
    PREDICTOR = 2
    TILED = True
    BLOCKXSIZE = 512
    BLOCKYSIZE = 512

    # Write to a temporary file first, then mask into final output
    tmp_out_path = os.path.splitext(out_path)[0] + "_tmp.tif"
    if os.path.exists(tmp_out_path):
        os.remove(tmp_out_path)

    # -------------------------
    # Load crop values table
    # -------------------------
    df = pd.read_csv(csv_path)
    required_cols = {"yield_file", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    crop_entries = []
    for _, row in df.iterrows():
        fname = str(row["yield_file"]).strip()
        val = float(row["value"])
        fpath = os.path.join(rasters_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Raster not found: {fpath}")
        crop_entries.append((fname, fpath, val))

    if not crop_entries:
        raise ValueError("No crop entries found in CSV.")

    # -------------------------
    # Open first raster to define grid
    # -------------------------
    with rasterio.open(crop_entries[0][1]) as src0:
        width = src0.width
        height = src0.height
        transform = src0.transform
        crs = src0.crs
        if src0.is_tiled and src0.block_shapes:
            block_h, block_w = src0.block_shapes[0]
        else:
            block_h, block_w = (BLOCKYSIZE, BLOCKXSIZE)

    out_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int32",
        "crs": crs,
        "transform": transform,
        "nodata": OUT_NODATA,
        "compress": COMPRESS,
        "predictor": PREDICTOR,
        "tiled": TILED,
        "blockxsize": block_w,
        "blockysize": block_h,
        "BIGTIFF": "IF_SAFER",
    }

    # Progress counters
    n_rows = (height + block_h - 1) // block_h
    n_cols = (width + block_w - 1) // block_w
    total_windows = n_rows * n_cols
    processed_windows = 0
    print(f"Starting crop max-potential calculation: {total_windows} windows, {len(crop_entries)} crops")

    # -------------------------
    # Processing (windowed max across crops)
    # -------------------------
    sources = []
    try:
        # Open all sources once
        for fname, fpath, val in crop_entries:
            src = rasterio.open(fpath)
            if src.width != width or src.height != height:
                raise ValueError(f"Raster size mismatch: {fname}")
            if src.transform != transform:
                raise ValueError(f"Transform mismatch: {fname}")
            if src.crs != crs:
                raise ValueError(f"CRS mismatch: {fname}")
            sources.append((fname, src, val))

        with rasterio.open(tmp_out_path, "w", **out_profile) as dst:
            for row_off in range(0, height, block_h):
                win_h = min(block_h, height - row_off)
                for col_off in range(0, width, block_w):
                    win_w = min(block_w, width - col_off)
                    window = Window(col_off, row_off, win_w, win_h)

                    max_val = np.full((win_h, win_w), -np.inf, dtype=np.float32)
                    any_data = np.zeros((win_h, win_w), dtype=bool)

                    for fname, src, value_per_ha in sources:
                        arr = src.read(1, window=window).astype(np.float32)

                        # Valid only if finite and not nodata
                        mask_valid = np.isfinite(arr)
                        src_nodata = src.nodata
                        if src_nodata is not None:
                            mask_valid &= (arr != src_nodata)

                        # Compute only where valid
                        pot = np.full(arr.shape, -np.inf, dtype=np.float32)
                        pot[mask_valid] = arr[mask_valid] * value_per_ha

                        max_val = np.maximum(max_val, pot)
                        any_data |= mask_valid

                    # Convert safely to int, avoiding NaN/Inf casts
                    out = np.full((win_h, win_w), OUT_NODATA, dtype=np.int32)
                    valid = any_data & np.isfinite(max_val)
                    if np.any(valid):
                        out_vals = np.rint(max_val[valid]).astype(np.int64)
                        out[valid] = np.clip(
                            out_vals,
                            np.iinfo(np.int32).min + 1,
                            np.iinfo(np.int32).max,
                        ).astype(np.int32)

                    dst.write(out, 1, window=window)

                    processed_windows += 1
                    if processed_windows % 100 == 0 or processed_windows == total_windows:
                        pct = 100.0 * processed_windows / total_windows
                        print(f"Processed {processed_windows}/{total_windows} windows ({pct:.1f}%)")

        print("Finished max-potential raster. Applying mainland mask...")

    finally:
        for _, src, _ in sources:
            try:
                src.close()
            except Exception:
                pass

    # -------------------------
    # Mask to mainland polygon (outside becomes nodata)
    # -------------------------
    with fiona.open(mainland_shp, "r") as shp:
        geometries = [feat["geometry"] for feat in shp]

    with rasterio.open(tmp_out_path) as src:
        masked_data, masked_transform = mask(
            src,
            geometries,
            crop=False,          # keep same extent
            nodata=OUT_NODATA,
            filled=True
        )
        final_profile = src.profile.copy()
        final_profile.update(transform=masked_transform)

    # Write final output (overwrite if exists)
    if os.path.exists(out_path):
        os.remove(out_path)

    with rasterio.open(out_path, "w", **final_profile) as dst:
        dst.write(masked_data)

    # Clean up temp
    try:
        os.remove(tmp_out_path)
    except Exception:
        pass

    print("Finished writing masked output:")
    print(out_path)


if __name__ == "__main__":
    main()
