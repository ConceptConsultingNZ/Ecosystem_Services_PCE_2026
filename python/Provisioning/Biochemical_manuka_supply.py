"""
Create (1) a raw Float32 mānuka supply raster and (2) an Int16 0–100 index.

Why: rounding a product of 0–1 factors to Int16 (0–100) can create lots of zeros,
especially when manuka proportion is small and/or moisture suitability is low.
The raw Float32 output lets you inspect the continuous values before scaling.

Outputs:
  - OUT_RAW (Float32): raw supply in [0,1] (NaN/NoData where manuka raster is NoData)
  - OUT_IDX (Int16): 0–100 index (NoData where manuka raster is NoData)
"""

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling


MANUKA = r"<PROJECT_DIRECTORY>\Provisioning\Biochemical\Intermediate\manuka_proportion_5km.tif"
SOLAR  = r"D:\Data\LRIS\nzenvds\nzenvds-mean-annual-solar-radiation-v10\nzenvds-mean-annual-solar-radiation-v10.tif"
TEMP   = r"D:\Data\LRIS\nzenvds\nzenvds-mean-temperature-of-the-warmest-quarter-v10\nzenvds-mean-temperature-of-the-warmest-quarter-v10.tif"
DEF    = r"D:\Data\LRIS\nzenvds\nzenvds-annual-water-deficit-v10\nzenvds-annual-water-deficit-v10.tif"

OUT_IDX = r"<PROJECT_DIRECTORY>\Provisioning\Biochemical\Output\biochemical_supply.tif"
OUT_RAW = r"<PROJECT_DIRECTORY>\Provisioning\Biochemical\Output\biochemical_supply_raw.tif"

NODATA_OUT = -9999

# Known range for annual water deficit (mm)
DEF_MIN = -1.5
DEF_MAX = 394.0

# Moisture "sweet spot" parameters in normalised deficit space
D_STAR = 0.5
WIDTH  = 0.5

# Percentile scaling (robust normalisation)
P_LOW  = 5
P_HIGH = 95


def read_to_template(src_path, template_ds, window=None, resampling=Resampling.bilinear):
    dst_crs = template_ds.crs
    dst_transform = template_ds.window_transform(window) if window is not None else template_ds.transform
    dst_height = window.height if window is not None else template_ds.height
    dst_width  = window.width  if window is not None else template_ds.width

    dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src:
        if window is not None:
            b = rasterio.windows.bounds(window, template_ds.transform)
            src_window = from_bounds(*b, transform=src.transform)
        else:
            b = rasterio.transform.array_bounds(template_ds.height, template_ds.width, template_ds.transform)
            src_window = from_bounds(b[0], b[1], b[2], b[3], transform=src.transform)

        src_window = src_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        if src_window.width <= 0 or src_window.height <= 0:
            return dst

        src_arr = src.read(1, window=src_window).astype(np.float32)
        src_nodata = src.nodata
        if src_nodata is not None:
            src_arr = np.where(src_arr == src_nodata, np.nan, src_arr)

        src_transform = src.window_transform(src_window)

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

    return dst


def percentile_normalise(x, p_low=5, p_high=95):
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)

    lo = np.nanpercentile(x[valid], p_low)
    hi = np.nanpercentile(x[valid], p_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def deficit_sweetspot(def_mm, dmin, dmax, d_star=0.5, width=0.5):
    dn = (def_mm - dmin) / (dmax - dmin)
    dn = np.clip(dn, 0.0, 1.0)
    m = 1.0 - (np.abs(dn - d_star) / width)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def main():
    with rasterio.open(MANUKA) as mds:
        manuka = mds.read(1).astype(np.float32)
        manuka_nodata = mds.nodata if mds.nodata is not None else -9999

        # Treat NoData as NaN, clip proportion to [0,1]
        manuka = np.where(manuka == manuka_nodata, np.nan, manuka)
        manuka = np.clip(manuka, 0.0, 1.0)

        # Valid template cells are those with manuka proportion data
        template_valid = np.isfinite(manuka)

        # "Mānuka exists" where proportion > 0 (and not NaN)
        manuka_mask = template_valid & (manuka > 0)

        solar = read_to_template(SOLAR, mds, resampling=Resampling.bilinear)
        temp  = read_to_template(TEMP,  mds, resampling=Resampling.bilinear)
        wdef  = read_to_template(DEF,   mds, resampling=Resampling.bilinear)

        solar_n = percentile_normalise(solar, P_LOW, P_HIGH)
        temp_n  = percentile_normalise(temp,  P_LOW, P_HIGH)
        moist_n = deficit_sweetspot(wdef, DEF_MIN, DEF_MAX, D_STAR, WIDTH)

        # -------------------------
        # RAW SUPPLY (Float32)
        # -------------------------
        supply_raw = np.full((mds.height, mds.width), np.nan, dtype=np.float32)
        supply_raw[manuka_mask] = (
            manuka[manuka_mask] *
            solar_n[manuka_mask] *
            temp_n[manuka_mask] *
            moist_n[manuka_mask]
        )
        # keep NaN where template is NoData
        supply_raw = np.where(template_valid, supply_raw, np.nan)

        # -------------------------
        # INDEX (Int16 0–100)
        # -------------------------
        idx = np.rint(supply_raw * 100.0)
        idx = np.clip(idx, 0, 100)

        out_idx = np.full((mds.height, mds.width), NODATA_OUT, dtype=np.int16)

        valid_idx = manuka_mask & np.isfinite(idx)
        out_idx[valid_idx] = idx[valid_idx].astype(np.int16)

        # -------------------------
        # Write RAW Float32
        # -------------------------
        profile_raw = mds.profile.copy()
        profile_raw.update(
            dtype=rasterio.float32,
            nodata=np.nan,              # GeoTIFF "nodata=NaN" is supported by rasterio
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        with rasterio.open(OUT_RAW, "w", **profile_raw) as dst:
            dst.write(supply_raw.astype(np.float32), 1)

        # -------------------------
        # Write Int16 index
        # -------------------------
        profile_idx = mds.profile.copy()
        profile_idx.update(
            dtype=rasterio.int16,
            nodata=NODATA_OUT,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        with rasterio.open(OUT_IDX, "w", **profile_idx) as dst:
            dst.write(out_idx, 1)

    print("Raw Float32 supply written:", OUT_RAW)
    print("Int16 0–100 supply index written:", OUT_IDX)


if __name__ == "__main__":
    main()
