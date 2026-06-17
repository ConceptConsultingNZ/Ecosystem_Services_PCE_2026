"""
Allocate marine farm value to surrounding ocean cells using:
    w(d) = exp(-d / lambda_m)
- Only allocates to ocean cells (sea_raster != sea_nodata)
- Allows overlapping radii (values sum)
- Output matches seafloor classification raster grid
"""
import os
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds

# ============================
# USER PARAMETERS
# ============================

FARMS_PATH = r"<PROJECT_DIRECTORY>\Provisioning\Aquaculture\marine_farms.shp"
VALUE_FIELD = "value"
SEA_RASTER_PATH = r"D:\Data\DOC\New Zealand Seafloor Community Classification\SCC_GF75_100m_EPSG2193_Int16.tif"
OUTPUT_PATH = r"D:\tmp\aquaculture_value.tif"

ATTRIBUTION_PROPORTION = 0.6 #Value attributed to ES. Sensitivity: low=40%, central = 60%, high = 80%

RADIUS_M = 5000.0
LAMBDA_M = 2000.0
SEA_NODATA = -9999
PROGRESS_EVERY = 25


def allocate_marine_farm_value_distance_decay():

    print("Opening sea raster...")
    with rasterio.open(SEA_RASTER_PATH) as src:
        sea_crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
        profile = src.profile.copy()

        sea = src.read(1)
        ocean_mask = sea != SEA_NODATA

        acc = np.zeros((height, width), dtype=np.float64)

    print("Reading farms layer...")
    farms = gpd.read_file(FARMS_PATH)

    if farms.empty:
        raise ValueError("Farms layer is empty.")
    if VALUE_FIELD not in farms.columns:
        raise ValueError(f"Field '{VALUE_FIELD}' not found.")

    farms = farms[farms.geometry.notnull()].copy()
    farms = farms[farms[VALUE_FIELD].notnull()].copy()

    if farms.crs is None:
        raise ValueError("Farms layer has no CRS defined.")
    if farms.crs != sea_crs:
        farms = farms.to_crs(sea_crs)

    r2 = RADIUS_M * RADIUS_M
    total_in = float(farms[VALUE_FIELD].sum())
    total_attributed = total_in * ATTRIBUTION_PROPORTION

    def window_cell_centres(win):
        rows = np.arange(win.row_off, win.row_off + win.height)
        cols = np.arange(win.col_off, win.col_off + win.width)

        xs = transform.c + (cols + 0.5) * transform.a
        ys = transform.f + (rows + 0.5) * transform.e

        return np.meshgrid(xs, ys)

    for i, row in enumerate(farms.itertuples(index=False), start=1):

        geom = row.geometry
        if geom is None:
            continue

        x0, y0 = geom.centroid.x, geom.centroid.y
        v = float(getattr(row, VALUE_FIELD)) * ATTRIBUTION_PROPORTION

        if not np.isfinite(v) or v == 0:
            continue

        minx, miny = x0 - RADIUS_M, y0 - RADIUS_M
        maxx, maxy = x0 + RADIUS_M, y0 + RADIUS_M

        win = from_bounds(minx, miny, maxx, maxy, transform=transform)

        row_off = int(max(0, math.floor(win.row_off)))
        col_off = int(max(0, math.floor(win.col_off)))
        row_max = int(min(height, math.ceil(win.row_off + win.height)))
        col_max = int(min(width, math.ceil(win.col_off + win.width)))

        if row_off >= row_max or col_off >= col_max:
            continue

        win = rasterio.windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=col_max - col_off,
            height=row_max - row_off,
        )

        local_ocean = ocean_mask[row_off:row_max, col_off:col_max]
        if not local_ocean.any():
            continue

        X, Y = window_cell_centres(win)

        dx = X - x0
        dy = Y - y0
        d2 = dx * dx + dy * dy

        within = (d2 <= r2) & local_ocean
        if not within.any():
            continue

        d = np.sqrt(d2)
        w = np.exp(-d / LAMBDA_M)
        w[~within] = 0.0

        wsum = float(w.sum())
        if wsum <= 0:
            continue

        contrib = v * (w / wsum)

        acc[row_off:row_max, col_off:col_max] += contrib

        if i % PROGRESS_EVERY == 0 or i == len(farms):
            print(f"Processed {i}/{len(farms)} farms")

    print("Finalising output...")

    out = acc.astype(np.float32)

    # Set land to nodata
    out[~ocean_mask] = SEA_NODATA

    # Set zero ocean cells to nodata
    zero_ocean = (ocean_mask) & (out == 0)
    out[zero_ocean] = SEA_NODATA

    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=SEA_NODATA,
        compress="DEFLATE",
        predictor=3,
        zlevel=9,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(out, 1)

    total_out = float(np.where(out != SEA_NODATA, out, 0.0).sum())

    print(f"Input farm sum:        {total_in:,.6f}")
    print(f"Attributed farm sum:   {total_attributed:,.6f}")
    print(f"Output raster sum:     {total_out:,.6f}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    allocate_marine_farm_value_distance_decay()
