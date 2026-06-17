"""
Spatially allocate catchment-level demand to supply cells, weighted by supply value.

Logic (per catchment c):
  output_j = EFFECTIVENESS * demand_c * (supply_j / sum_supply_c)  for j inside c

- catchment demand comes from a GPKG polygon layer (one row per catchment).
- supply is a raster (Float32/Float64 is fine).
- output is a raster on the same grid as the supply raster.

Notes:
- If a catchment has demand but sum_supply_c == 0 (or no valid supply pixels), the output is 0 for that catchment.
- Supply nodata is excluded from sums and output.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling

# -----------------------------
# Constants (edit these)
# -----------------------------
SUPPLY_RASTER = r"<PROJECT_DIRECTORY>\Regulating\Natural hazard regulation\Output\flood_regulation_supply.tif"
CATCHMENT_GPKG = r"<PROJECT_DIRECTORY>\Regulating\Natural hazard regulation\natural_hazard_working.gpkg"
CATCHMENT_LAYER = "ecoindex_catchment_demand"   # layer name in the gpkg
CATCHMENT_ID_FIELD = "catch_id"               # unique catchment identifier field
DEMAND_FIELD = "demand_sum"                   # catchment-level demand field (NZD, total per catchment)
OUTPUT_RASTER = r"<PROJECT_DIRECTORY>\Regulating\Natural hazard regulation\Output\flood_regulation_flow_value_v2.tif"

EFFECTIVENESS = 0.20  # 20% conservative cap
OUTPUT_NODATA = -9999.0

# Optional: treat negative/invalid supply as zero for weighting
CLIP_SUPPLY_MIN = 0.0

# -----------------------------
# Helper functions
# -----------------------------
def _build_catchment_id_raster(
    gdf: gpd.GeoDataFrame,
    id_field: str,
    out_shape: tuple[int, int],
    transform,
    all_touched: bool = False,
    dtype=np.int32,
) -> np.ndarray:
    """Rasterise polygons into an int catchment-id raster."""
    shapes = ((geom, int(cid)) for geom, cid in zip(gdf.geometry, gdf[id_field]))
    id_raster = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,  # 0 means "no catchment"
        all_touched=all_touched,
        dtype=dtype,
    )
    return id_raster


def _sum_supply_by_catchment(
    supply_path: str,
    catch_id_raster: np.ndarray,
    clip_min: float | None = None,
) -> dict[int, float]:
    """First pass: compute sum of supply per catchment id across the whole raster."""
    sums: dict[int, float] = {}

    with rasterio.open(supply_path) as src:
        supply_nodata = src.nodata

        # Process in blocks/windows to avoid loading everything at once
        for _, window in src.block_windows(1):
            s = src.read(1, window=window, masked=False).astype(np.float32)
            ids = catch_id_raster[
                window.row_off : window.row_off + window.height,
                window.col_off : window.col_off + window.width,
            ]

            if supply_nodata is not None:
                valid = (s != supply_nodata)
            else:
                valid = np.isfinite(s)

            # Exclude pixels outside catchments
            valid &= (ids != 0)

            if clip_min is not None:
                valid &= (s > clip_min)
                s = np.where(s > clip_min, s, 0.0)

            if not np.any(valid):
                continue

            ids_v = ids[valid].astype(np.int64)
            s_v = s[valid].astype(np.float64)

            # Group sums by catchment id within this window
            uniq, inv = np.unique(ids_v, return_inverse=True)
            win_sums = np.bincount(inv, weights=s_v)

            for k, v in zip(uniq, win_sums):
                sums[int(k)] = sums.get(int(k), 0.0) + float(v)

    return sums


def _make_output(
    supply_path: str,
    catch_id_raster: np.ndarray,
    demand_by_id: dict[int, float],
    sum_supply_by_id: dict[int, float],
    out_path: str,
    effectiveness: float,
    out_nodata: float,
    clip_min: float | None = None,
):
    """Second pass: write output raster where demand is allocated to supply pixels by supply weights."""
    with rasterio.open(supply_path) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=out_nodata,
            compress="DEFLATE",
            predictor=2,
            zlevel=9,
            tiled=True,
            blockxsize=256,
            blockysize=256
        )

        supply_nodata = src.nodata

        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                s = src.read(1, window=window, masked=False).astype(np.float32)
                ids = catch_id_raster[
                    window.row_off : window.row_off + window.height,
                    window.col_off : window.col_off + window.width,
                ]

                # Default output is 0 (not nodata) inside catchments; nodata outside supply-valid areas
                out = np.zeros((window.height, window.width), dtype=np.float32)

                if supply_nodata is not None:
                    valid_supply = (s != supply_nodata)
                else:
                    valid_supply = np.isfinite(s)

                if clip_min is not None:
                    s = np.where(s > clip_min, s, 0.0)

                # Only allocate where we have a catchment id and valid supply pixel
                valid = valid_supply & (ids != 0)

                if np.any(valid):
                    ids_v = ids[valid].astype(np.int64)
                    s_v = s[valid].astype(np.float64)

                    # Compute per-pixel factor = effectiveness * demand_c / sum_supply_c
                    # Then multiply by supply pixel value.
                    # Build vectorised arrays via mapping from id -> factor
                    factors = np.zeros_like(s_v, dtype=np.float64)

                    # Vectorised-ish: unique ids in this window, then fill factors
                    uniq = np.unique(ids_v)
                    for cid in uniq:
                        cid_int = int(cid)
                        d = float(demand_by_id.get(cid_int, 0.0))
                        denom = float(sum_supply_by_id.get(cid_int, 0.0))
                        if d <= 0.0 or denom <= 0.0:
                            continue
                        f = effectiveness * d / denom
                        factors[ids_v == cid] = f

                    out_vals = (s_v * factors).astype(np.float32)
                    out[valid] = out_vals

                # Assign nodata where supply is invalid (keeps edge behaviour clean)
                out[~valid_supply] = out_nodata

                dst.write(out, 1, window=window)


# -----------------------------
# Main
# -----------------------------
def main():
    # Open supply raster to get grid + CRS
    with rasterio.open(SUPPLY_RASTER) as src:
        supply_crs = src.crs
        out_shape = (src.height, src.width)
        transform = src.transform

    # Read catchments + demand
    gdf = gpd.read_file(CATCHMENT_GPKG, layer=CATCHMENT_LAYER)

    # Keep only necessary fields and valid geometries
    gdf = gdf[[CATCHMENT_ID_FIELD, DEMAND_FIELD, "geometry"]].copy()
    gdf = gdf[gdf.geometry.notnull()].copy()

    # Reproject to supply CRS if needed
    if gdf.crs != supply_crs:
        gdf = gdf.to_crs(supply_crs)

    # Demand mapping (catchment id -> total demand)
    demand_by_id = {
        int(row[CATCHMENT_ID_FIELD]): float(row[DEMAND_FIELD]) if row[DEMAND_FIELD] is not None else 0.0
        for _, row in gdf.iterrows()
    }

    # Rasterise catchment IDs to the supply grid
    catch_id_raster = _build_catchment_id_raster(
        gdf=gdf,
        id_field=CATCHMENT_ID_FIELD,
        out_shape=out_shape,
        transform=transform,
        all_touched=False,  # set True if you prefer "paint all touched pixels"
        dtype=np.int32,
    )

    # Pass 1: sum supply per catchment
    sum_supply_by_id = _sum_supply_by_catchment(
        supply_path=SUPPLY_RASTER,
        catch_id_raster=catch_id_raster,
        clip_min=CLIP_SUPPLY_MIN,
    )

    # Pass 2: allocate demand to supply pixels and write output
    _make_output(
        supply_path=SUPPLY_RASTER,
        catch_id_raster=catch_id_raster,
        demand_by_id=demand_by_id,
        sum_supply_by_id=sum_supply_by_id,
        out_path=OUTPUT_RASTER,
        effectiveness=EFFECTIVENESS,
        out_nodata=OUTPUT_NODATA,
        clip_min=CLIP_SUPPLY_MIN,
    )

    print("Spatial allocation complete.")
    print(f"Output written to: {OUTPUT_RASTER}")


if __name__ == "__main__":
    main()