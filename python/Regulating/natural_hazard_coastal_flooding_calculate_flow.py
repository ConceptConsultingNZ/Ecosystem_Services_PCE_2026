"""
Avoided coastal flood damage from mangroves and saltmarsh (screening model).

This script estimates the expected annual flood damage to properties within
1-in-100 coastal flood extent polygons, and the portion of that damage avoided
due to the presence of mangroves and saltwater wetlands. The approach is a
simplified ecosystem service model intended for cost–benefit analysis when
detailed hydrodynamic modelling is unavailable.

Method summary
--------------
1. Property values are rasterised and converted to expected annual flood damage
   using a fixed damage ratio and annual exceedance probability (AEP).

2. Flood extent polygons are rasterised to create unique flood-zone IDs. All
   calculations are performed at the flood-zone level.

3. For each flood extent feature:
   - Total expected damage is calculated from property values within the zone.
   - Demand (expected damage) is averaged uniformly across all cells in the
     flood extent.

4. The proportion of mangrove and saltwater wetland area within each flood
   extent is used as a proxy for ecosystem service supply (attenuation capacity).

5. Flood damage reduction is estimated using:
       r(P) = min(beta * P, r_max)
   where P is wetland proportion within the flood extent, beta is an effectiveness
   parameter, and r_max is an upper-bound reduction derived from literature.

6. A distance decay function is applied to represent diminishing protection with
   increasing distance from wetlands. This is used as an adjustment factor at the
   flood-zone level.

7. Avoided damage is calculated for each flood extent feature and then allocated
   evenly across mangrove and wetland cells within that feature to produce a
   realised flow raster.

Inputs
------
- Mangrove binary raster (1 = mangrove, 0 = no mangrove)
- Saltwater wetland binary raster (1 = wetland, 0 = no wetland)
- Flood extent polygon layer (.gpkg)
- Property point layer (.gpkg) with a capital value field

All rasters must be aligned (same CRS, resolution, extent, and grid). Flood
polygons must be in a compatible CRS and will be rasterised internally.

Outputs
-------
- Demand raster:
    Expected annual flood damage (NZD per pixel), averaged across each flood
    extent feature.

- Realised flow raster:
    Avoided flood damage attributable to mangrove and wetland cells (NZD per
    pixel), with values distributed evenly across ecosystem cells within each
    flood extent feature.

- Flood zone raster:
    Integer raster identifying flood extent features.

Key assumptions
---------------
- Flood damages are approximated as a fixed proportion of property capital value.
- Wetland effectiveness scales linearly with its proportion of the flood extent,
  subject to a maximum cap.
- Protection effects decline with distance from wetlands and are incorporated as
  a zone-level adjustment factor.
- Avoided damages are distributed uniformly across wetlands within each flood
  extent rather than modelled via detailed flow paths.

Limitations
-----------
- Does not model flood depth, velocity, or hydrodynamic processes.
- Wetland effects are approximated using benefit transfer from literature.
- Spatial connectivity between wetlands and properties is simplified to flood
  extent grouping.
- Distance decay is applied in an aggregated manner rather than along explicit
  flow paths.
- Results are sensitive to parameter choices; sensitivity analysis is recommended.

References (conceptual basis)
----------------------------
- Barbier et al. (2013) – Coastal ecosystem services and flood protection
- Narayan et al. (2017) – Global flood damage reduction by coastal wetlands
- Fairchild et al. (2021) – Saltmarsh effects on flood extent and damages
- Freeman et al. (2014) – Avoided damage valuation framework
"""
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import logging
import time


# =========================
# CONSTANTS
# =========================

WORKING_DIR = Path(r"<PROJECT_DIRECTORY>\Regulating\Natural hazard regulation\Intermediate")
OUTPUT_DIR = Path(r"<PROJECT_DIRECTORY>\Regulating\Natural hazard regulation\Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANGROVE_RASTER = Path(r"<PROJECT_DIRECTORY>\Regulating\Blue carbon\Intermediate\mangrove.tif")
SALTWATER_WETLAND_RASTER = WORKING_DIR / "saltwater_wetlands.tif"

FLOOD_POLYGONS = r"D:\Data\NIWA\CoastalFloodLayersAEP1percent_gdb\CoastalFloodLayers.gpkg"
FLOOD_LAYER = "SLR0_fixed_geometries"
FLOOD_ID_FIELD = "OBJECTID"

PROPERTIES_GPKG = Path(r"<PROJECT_DIRECTORY>\Common\properties.gpkg")
PROPERTIES_LAYER = "properties_2020"
CAPITAL_VALUE_FIELD = "capital_value"

OUTPUT_DEMAND_RASTER = OUTPUT_DIR / "natural_hazard_coastal_flooding_demand.tif"
OUTPUT_FLOW_RASTER_CENTRAL = OUTPUT_DIR / "Regulating_coastal_storm_surge_value.tif"
OUTPUT_FLOW_RASTER_LOW = OUTPUT_DIR / "Regulating_coastal_storm_surge_value_low.tif"
OUTPUT_FLOW_RASTER_HIGH = OUTPUT_DIR / "Regulating_coastal_storm_surge_value_high.tif"

AEP = 0.01

SCENARIOS = {
    "low": {
        "damage_ratio": 0.05,
        "beta": 0.25,
        "max_reduction": 0.05,
        "flow_raster": OUTPUT_FLOW_RASTER_LOW,
    },
    "central": {
        "damage_ratio": 0.10,
        "beta": 0.50,
        "max_reduction": 0.10,
        "flow_raster": OUTPUT_FLOW_RASTER_CENTRAL,
    },
    "high": {
        "damage_ratio": 0.20,
        "beta": 1.00,
        "max_reduction": 0.20,
        "flow_raster": OUTPUT_FLOW_RASTER_HIGH,
    },
}

DISTANCE_DECAY_METRES = 500
MAX_BENEFIT_DISTANCE = 1000

PROPERTY_RASTER_ALL_TOUCHED = True
WETLAND_BINARY_THRESHOLD = 0.5

FLOAT_RASTER_DTYPE = "float32"
FLOAT_NODATA = 0.0

RASTER_CREATION_OPTIONS = {
    "compress": "deflate",
    "predictor": 3,
}


# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# =========================
# FUNCTIONS
# =========================

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    return arr, profile


def check_alignment(reference_profile, *rasters):
    for path in rasters:
        with rasterio.open(path) as src:
            if (
                src.transform != reference_profile["transform"]
                or src.crs != reference_profile["crs"]
                or src.width != reference_profile["width"]
                or src.height != reference_profile["height"]
            ):
                raise ValueError(f"{path} is not aligned with the template raster.")


def rasterise_flood_polygons(path, layer, id_field, reference_profile):
    logger.info("Reading flood polygons")
    gdf = gpd.read_file(path, layer=layer)

    logger.info(f"Loaded {len(gdf):,} flood extent features")

    if gdf.crs != reference_profile["crs"]:
        logger.info("Reprojecting flood polygons to raster CRS")
        gdf = gdf.to_crs(reference_profile["crs"])

    gdf = gdf[gdf.geometry.notnull()].copy()

    if id_field is None:
        gdf["_zone_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
        id_field = "_zone_id"
    else:
        gdf[id_field] = gdf[id_field].astype(np.int32)

    shapes = zip(gdf.geometry, gdf[id_field])

    zone_raster = rasterize(
        shapes=shapes,
        out_shape=(reference_profile["height"], reference_profile["width"]),
        transform=reference_profile["transform"],
        fill=0,
        dtype="int32",
        all_touched=True,
    )

    return zone_raster


def rasterise_property_damage(gpkg, layer, value_field, reference_profile, damage_ratio):
    logger.info("Reading property points")
    gdf = gpd.read_file(gpkg, layer=layer)

    logger.info(f"Loaded {len(gdf):,} property features")

    if gdf.crs != reference_profile["crs"]:
        logger.info("Reprojecting properties to raster CRS")
        gdf = gdf.to_crs(reference_profile["crs"])

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf[value_field] = gdf[value_field].fillna(0).astype(float)

    damage = np.zeros(
        (reference_profile["height"], reference_profile["width"]),
        dtype=FLOAT_RASTER_DTYPE,
    )

    chunk_size = 200_000

    for i in range(0, len(gdf), chunk_size):
        chunk = gdf.iloc[i:i + chunk_size]
        logger.info(f"Rasterising properties {i:,} to {i + len(chunk):,}")

        shapes = zip(
            chunk.geometry,
            chunk[value_field] * damage_ratio * AEP,
        )

        chunk_raster = rasterize(
            shapes=shapes,
            out_shape=(reference_profile["height"], reference_profile["width"]),
            transform=reference_profile["transform"],
            fill=0.0,
            merge_alg=rasterio.enums.MergeAlg.add,
            dtype=FLOAT_RASTER_DTYPE,
            all_touched=PROPERTY_RASTER_ALL_TOUCHED,
        )

        damage += chunk_raster

    return damage


def calculate_distance_decay(wetland_mask, profile):
    pixel_width = abs(profile["transform"].a)
    pixel_height = abs(profile["transform"].e)

    distance = distance_transform_edt(
        ~wetland_mask,
        sampling=(pixel_height, pixel_width),
    )

    decay = np.exp(-distance / DISTANCE_DECAY_METRES)
    decay[distance > MAX_BENEFIT_DISTANCE] = 0.0
    decay[wetland_mask] = 1.0

    return decay


def average_demand_and_flow_by_flood_feature(
    raw_property_damage,
    zone_raster,
    wetland_mask,
    distance_decay,
    beta,
    max_reduction,
):
    logger.info("Averaging demand and flow by flood extent feature using vectorised zonal stats")

    zone_flat = zone_raster.ravel().astype(np.int64)
    damage_flat = raw_property_damage.ravel().astype("float64")
    wetland_flat = wetland_mask.ravel().astype("float64")
    decay_flat = distance_decay.ravel().astype("float64")

    valid = zone_flat > 0

    if not valid.any():
        logger.warning("No valid flood zones found. Returning zero rasters.")
        return (
            np.zeros_like(raw_property_damage, dtype="float64"),
            np.zeros_like(raw_property_damage, dtype="float64"),
        )

    zone_ids = zone_flat[valid]
    max_zone_id = int(zone_ids.max())

    logger.info(f"Calculating zonal totals for {max_zone_id:,} possible zone IDs")

    zone_cell_count = np.bincount(
        zone_ids,
        minlength=max_zone_id + 1,
    ).astype("float64")

    zone_damage_total = np.bincount(
        zone_ids,
        weights=damage_flat[valid],
        minlength=max_zone_id + 1,
    )

    zone_wetland_count = np.bincount(
        zone_ids,
        weights=wetland_flat[valid],
        minlength=max_zone_id + 1,
    )

    zone_damage_decay_total = np.bincount(
        zone_ids,
        weights=damage_flat[valid] * decay_flat[valid],
        minlength=max_zone_id + 1,
    )

    wetland_proportion = np.zeros(max_zone_id + 1, dtype="float64")
    reduction_rate = np.zeros(max_zone_id + 1, dtype="float64")
    decay_adjustment = np.zeros(max_zone_id + 1, dtype="float64")
    avoided_damage_total = np.zeros(max_zone_id + 1, dtype="float64")
    demand_per_cell = np.zeros(max_zone_id + 1, dtype="float64")
    flow_per_wetland_cell = np.zeros(max_zone_id + 1, dtype="float64")

    has_cells = zone_cell_count > 0
    has_damage = zone_damage_total > 0
    has_wetland = zone_wetland_count > 0

    wetland_proportion[has_cells] = (
        zone_wetland_count[has_cells] / zone_cell_count[has_cells]
    )

    reduction_rate = np.minimum(beta * wetland_proportion, max_reduction)

    decay_adjustment[has_damage] = (
        zone_damage_decay_total[has_damage] / zone_damage_total[has_damage]
    )

    avoided_damage_total = zone_damage_total * reduction_rate * decay_adjustment

    demand_per_cell[has_cells] = (
        zone_damage_total[has_cells] / zone_cell_count[has_cells]
    )

    can_allocate_flow = has_wetland & (avoided_damage_total > 0)

    flow_per_wetland_cell[can_allocate_flow] = (
        avoided_damage_total[can_allocate_flow]
        / zone_wetland_count[can_allocate_flow]
    )

    logger.info("Mapping per-zone values back to raster cells")

    demand = np.zeros_like(raw_property_damage, dtype="float64")
    flow = np.zeros_like(raw_property_damage, dtype="float64")

    zone_safe = zone_raster.astype(np.int64)

    flood_cells = zone_safe > 0
    wetland_flood_cells = flood_cells & wetland_mask

    demand[flood_cells] = demand_per_cell[zone_safe[flood_cells]]
    flow[wetland_flood_cells] = flow_per_wetland_cell[zone_safe[wetland_flood_cells]]

    logger.info(f"Demand total: ${demand.sum():,.2f}")
    logger.info(f"Flow total: ${flow.sum():,.2f}")

    return demand, flow


def write_raster(path, array, profile, dtype=FLOAT_RASTER_DTYPE, nodata=FLOAT_NODATA):
    def get_new_path(p):
        i = 1
        while True:
            new_path = p.with_name(f"{p.stem}_{i}{p.suffix}")
            if not new_path.exists():
                return new_path
            i += 1

    output_path = path

    try:
        if output_path.exists():
            logger.warning(f"Overwriting existing raster: {output_path}")
            output_path.unlink()
    except PermissionError:
        logger.warning(f"Cannot delete {output_path}. Creating new file instead.")
        output_path = get_new_path(output_path)

    out_profile = profile.copy()
    out_profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        **RASTER_CREATION_OPTIONS,
    )

    logger.info(f"Writing raster as {dtype}: {output_path}")

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)

    return output_path


# =========================
# MAIN
# =========================

def main():
    start_total = time.time()
    logger.info("Starting avoided damage calculation. Tiny flood goblin is awake.")

    logger.info("Reading template raster")
    mangrove, profile = read_raster(MANGROVE_RASTER)

    check_alignment(profile, SALTWATER_WETLAND_RASTER)

    logger.info("Reading saltwater wetland raster")
    saltmarsh, _ = read_raster(SALTWATER_WETLAND_RASTER)

    wetland_mask = (
        (mangrove > WETLAND_BINARY_THRESHOLD)
        | (saltmarsh > WETLAND_BINARY_THRESHOLD)
    )

    logger.info(f"Total wetland/mangrove cells: {wetland_mask.sum():,}")

    zone_raster = rasterise_flood_polygons(
        FLOOD_POLYGONS,
        FLOOD_LAYER,
        FLOOD_ID_FIELD,
        profile,
    )

    logger.info("Calculating distance decay from wetlands")
    distance_decay = calculate_distance_decay(wetland_mask, profile)

    results = {}

    for scenario_name, params in SCENARIOS.items():
        logger.info(f"==== Running {scenario_name.upper()} scenario ====")

        raw_property_damage = rasterise_property_damage(
            PROPERTIES_GPKG,
            PROPERTIES_LAYER,
            CAPITAL_VALUE_FIELD,
            profile,
            params["damage_ratio"],
        )

        raw_property_damage = np.where(zone_raster > 0, raw_property_damage, 0.0)

        logger.info(
            f"{scenario_name.capitalize()} raw expected annual property damage: "
            f"${raw_property_damage.sum():,.2f}"
        )

        demand, flow = average_demand_and_flow_by_flood_feature(
            raw_property_damage,
            zone_raster,
            wetland_mask,
            distance_decay,
            params["beta"],
            params["max_reduction"],
        )

        flow_path = write_raster(params["flow_raster"], flow, profile)

        results[scenario_name] = {
            "demand_total": demand.sum(),
            "flow_total": flow.sum(),
            "flow_path": flow_path,
        }

        if scenario_name == "central":
            demand_path = write_raster(OUTPUT_DEMAND_RASTER, demand, profile)

    logger.info("==== SUMMARY ====")
    logger.info(f"Central demand raster written to: {demand_path}")

    for scenario_name, result in results.items():
        logger.info(
            f"{scenario_name.capitalize()} demand total: "
            f"${result['demand_total']:,.2f}"
        )
        logger.info(
            f"{scenario_name.capitalize()} flow total: "
            f"${result['flow_total']:,.2f}"
        )
        logger.info(
            f"{scenario_name.capitalize()} flow raster written to: "
            f"{result['flow_path']}"
        )

    logger.info(f"Total runtime: {time.time() - start_total:.1f}s. No rasters were harmed.")


if __name__ == "__main__":
    main()