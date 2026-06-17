"""
Create hunting demand, supply, trip-flow, and value-flow rasters.

Workflow:
1. create_hunter_population_rasters()
   Estimates the adult population participating in big game and game bird hunting by
   combining the adult population raster with regional hunting participation rates.

2. create_game_bird_supply_raster()
   Creates a relative game bird supply raster by mapping land-cover classes to
   game bird supply scores from the land-cover lookup table.

3. create_big_game_supply_rasters()
   Creates relative supply rasters for each big game species by masking species
   abundance rasters to valid hunting permit areas. Cells outside permit areas are
   assigned zero supply.

4. allocate_hunting_flow_values()
   Allocates annual hunting trips and monetary values from hunter population cells
   to species supply cells using an exponential distance-decay kernel based on mean
   one-way travel distance. Species-level flow value rasters are summed to produce
   total central, low, and high hunting flow value rasters, along with total allocated
   hunting trips and central demand value rasters.
"""

import os
import logging
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.signal import fftconvolve

#General parameters
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
WORKING_DIR = r"<PROJECT_DIRECTORY>\Cultural\Hunting\Intermediate"
OUTPUT_DIR = r"<PROJECT_DIRECTORY>\Cultural\Hunting\Output"
RASTER_DRIVER = "GTiff"
RASTER_COMPRESS = "deflate"
RASTER_PREDICTOR = 3

#Run controls
#Set these to False when the corresponding source rasters have not changed.
RUN_HUNTER_POPULATION_RASTERS = False
RUN_BIG_GAME_SUPPLY_RASTERS = False
RUN_VALUE_ALLOCATION = True

#Species whose flow rasters should be recalculated from the value lookup.
#Species not listed here are read from their existing flow rasters and included in the totals.
#Use None to recalculate every species.
SPECIES_TO_REALLOCATE = ("game_bird",)

#When True, skipped species are still included in total rasters using their existing flow rasters.
USE_EXISTING_FLOW_RASTERS_FOR_SKIPPED_SPECIES = True

#Supply inputs
GAME_BIRD_SUPPLY = r"<PROJECT_DIRECTORY>\Cultural\Hunting\Intermediate\hunting_supply_game_bird.tif" #class_2023
PERMIT_AREA_RASTER = os.path.join(WORKING_DIR, "hunting_permit_areas.tif") #Raster values 1 or nodata
BIG_GAME_SPECIES = ("red_deer", "sika_deer", "fallow_deer", "pig", "tahr", "chamois")
HUNTING_SPECIES = ("game_bird",) + BIG_GAME_SPECIES
EXTENT_RASTER_TEMPLATE =os.path.join(WORKING_DIR, "extent_{species}.tif")

#Participation inputs
POPULATION_RASTER = r"<PROJECT_DIRECTORY>\Common\pop_adults.tif"
REGION_RASTER = r"D:\Data\Stats NZ geography\region.tif"  # REGC2020
REGION_LOOKUP_CSV = os.path.join(WORKING_DIR, "region_participation.csv")

#Participation outputs
POPULATION_BIG_GAME_HUNTERS_RASTER = os.path.join(WORKING_DIR, "population_big_game_hunters.tif")
POPULATION_BIRD_GAME_HUNTERS_RASTER = os.path.join(WORKING_DIR, "population_bird_game_hunters.tif")
HUNTER_POPULATION_BY_SPECIES = {
    "game_bird": POPULATION_BIRD_GAME_HUNTERS_RASTER,
    **{species_name: POPULATION_BIG_GAME_HUNTERS_RASTER for species_name in BIG_GAME_SPECIES},
}

#Supply output
SUPPLY_SPECIES_TEMPLATE = os.path.join(WORKING_DIR, "hunting_supply_{species}.tif")

#Value inputs
#Fields: species,trips_per_year,mean_one_way_travel_distance_km,total_value_central,total_value_low,total_value_high
TOTAL_VALUE_LOOKUP_CSV = os.path.join(WORKING_DIR, "total_value.csv")

#Value outputs
HUNTING_DEMAND_VALUE_RASTER = os.path.join(WORKING_DIR, "hunting_demand_valuev3.tif")
HUNTING_TRIPS_RASTER = os.path.join(WORKING_DIR, "hunting_tripsv3.tif")
HUNTING_FLOW_VALUE_TOTAL_RASTER = os.path.join(OUTPUT_DIR, "hunting_flow_valuev3.tif")
HUNTING_FLOW_VALUE_LOW_RASTER = os.path.join(OUTPUT_DIR, "hunting_flow_value_lowv3.tif")
HUNTING_FLOW_VALUE_HIGH_RASTER = os.path.join(OUTPUT_DIR, "hunting_flow_value_highv3.tif")
HUNTING_FLOW_VALUE_SPECIES_TEMPLATE = os.path.join(WORKING_DIR, "hunting_flow_value_{species}.tif")

def read_raster_aligned_to_template(raster_path, template_profile, template_shape):
    # Reprojects raster to match template, treating nodata and negative values as zero.
    with rasterio.open(raster_path) as src:
        src_nodata = src.nodata

        src_array = src.read(1, masked=True).astype("float32")
        src_array = src_array.filled(0)

        if src_nodata is not None:
            src_array = np.where(src_array == src_nodata, 0, src_array)

        src_array = np.where(np.isfinite(src_array), src_array, 0)
        src_array = np.where(src_array < 0, 0, src_array).astype("float32")

        aligned = np.zeros(template_shape, dtype="float32")

        reproject(
            source=src_array,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    aligned = np.where(np.isfinite(aligned), aligned, 0)
    aligned = np.where(aligned < 0, 0, aligned).astype("float32")

    return aligned

def save_raster(array, output_path, profile):
    final_path = output_path

    if os.path.exists(final_path):
        try:
            os.remove(final_path)
        except PermissionError:
            base, ext = os.path.splitext(output_path)
            i = 1

            while os.path.exists(final_path):
                final_path = f"{base}_{i}{ext}"
                i += 1

    with rasterio.open(final_path, "w", **profile) as dst:
        dst.write(array, 1)

    return final_path

def create_hunter_population_rasters():
    os.makedirs(WORKING_DIR, exist_ok=True)

    out_big_game = POPULATION_BIG_GAME_HUNTERS_RASTER
    out_bird_game = POPULATION_BIRD_GAME_HUNTERS_RASTER

    lookup = pd.read_csv(REGION_LOOKUP_CSV)
    lookup["REGC2020"] = lookup["REGC2020"].astype(int)

    big_game_lookup = dict(zip(lookup["REGC2020"], lookup["participation_big_game"]))
    bird_game_lookup = dict(zip(lookup["REGC2020"], lookup["participation_bird_game"]))

    with rasterio.open(POPULATION_RASTER) as pop_src:
        pop = pop_src.read(1).astype("float32")
        pop_profile = pop_src.profile.copy()
        pop_nodata = pop_src.nodata
        target_shape = pop.shape

        with rasterio.open(REGION_RASTER) as reg_src:
            region = np.empty(target_shape, dtype="int32")

            reproject(
                source=rasterio.band(reg_src, 1),
                destination=region,
                src_transform=reg_src.transform,
                src_crs=reg_src.crs,
                src_nodata=reg_src.nodata,
                dst_transform=pop_src.transform,
                dst_crs=pop_src.crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

    big_participation = np.zeros(target_shape, dtype="float32")
    bird_participation = np.zeros(target_shape, dtype="float32")

    for reg_code, rate in big_game_lookup.items():
        big_participation[region == reg_code] = rate

    for reg_code, rate in bird_game_lookup.items():
        bird_participation[region == reg_code] = rate

    valid = np.isfinite(pop)
    if pop_nodata is not None:
        valid &= pop != pop_nodata

    population_big_game_hunters = np.where(valid, pop * big_participation, np.nan).astype("float32")
    population_bird_hunters = np.where(valid, pop * bird_participation, np.nan).astype("float32")

    out_profile = pop_profile.copy()
    out_profile.update(
        driver=RASTER_DRIVER,
        dtype="float32",
        nodata=np.nan,
        compress=RASTER_COMPRESS,
        predictor=RASTER_PREDICTOR,
    )

    big_game_path = save_raster(population_big_game_hunters, out_big_game, out_profile)
    bird_game_path = save_raster(population_bird_hunters, out_bird_game, out_profile)

    logging.info(f"Hunter population rasters calculated successfully: {big_game_path}, {bird_game_path}")

def create_big_game_supply_rasters():
    output_paths = []

    with rasterio.open(PERMIT_AREA_RASTER) as permit_src:
        permit_area = permit_src.read(1)
        permit_nodata = permit_src.nodata

        permit_valid = np.isfinite(permit_area) & (permit_area == 1)
        if permit_nodata is not None:
            permit_valid &= permit_area != permit_nodata

        for species_name in BIG_GAME_SPECIES:
            species_abundance_raster = EXTENT_RASTER_TEMPLATE.format(species=species_name)
            out_supply = SUPPLY_SPECIES_TEMPLATE.format(species=species_name)

            with rasterio.open(species_abundance_raster) as abundance_src:
                abundance = abundance_src.read(1).astype("float32")
                abundance_profile = abundance_src.profile.copy()
                abundance_nodata = abundance_src.nodata

                if (
                    abundance_src.shape != permit_src.shape
                    or abundance_src.transform != permit_src.transform
                    or abundance_src.crs != permit_src.crs
                ):
                    aligned_permit = np.empty(abundance.shape, dtype="float32")

                    reproject(
                        source=permit_area,
                        destination=aligned_permit,
                        src_transform=permit_src.transform,
                        src_crs=permit_src.crs,
                        src_nodata=permit_nodata,
                        dst_transform=abundance_src.transform,
                        dst_crs=abundance_src.crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.nearest,
                    )

                    species_permit_valid = np.isfinite(aligned_permit) & (aligned_permit == 1)
                    if permit_nodata is not None:
                        species_permit_valid &= aligned_permit != permit_nodata
                else:
                    species_permit_valid = permit_valid

            abundance_valid = np.isfinite(abundance)
            if abundance_nodata is not None:
                abundance_valid &= abundance != abundance_nodata

            big_game_supply = np.where(species_permit_valid & abundance_valid, abundance, 0).astype("float32")

            out_profile = abundance_profile.copy()
            out_profile.update(
                driver=RASTER_DRIVER,
                dtype="float32",
                nodata=0,
                compress=RASTER_COMPRESS,
                predictor=RASTER_PREDICTOR,
            )

            output_path = save_raster(big_game_supply, out_supply, out_profile)
            output_paths.append(output_path)
            logging.info(f"Big game supply raster calculated successfully: {output_path}")

    return output_paths

def create_exponential_decay_kernel(mean_distance_km, pixel_size_km):
    radius_pixels = int(np.ceil((mean_distance_km * 5) / pixel_size_km))
    y, x = np.ogrid[-radius_pixels:radius_pixels + 1, -radius_pixels:radius_pixels + 1]
    distance_km = np.sqrt(x**2 + y**2) * pixel_size_km

    kernel = np.exp(-distance_km / mean_distance_km)
    kernel /= kernel.sum()

    return kernel.astype("float32")

def get_hunter_population_raster_for_species(species_name):
    try:
        return HUNTER_POPULATION_BY_SPECIES[species_name]
    except KeyError as exc:
        raise ValueError(f"No hunter population raster has been defined for species: {species_name}") from exc

def allocate_hunting_flow_values():
    value_lookup = pd.read_csv(TOTAL_VALUE_LOOKUP_CSV)

    required_fields = {
        "species",
        "trips_per_year",
        "mean_one_way_travel_distance_km",
        "total_value_central",
        "total_value_low",
        "total_value_high",
    }

    missing_fields = required_fields - set(value_lookup.columns)
    if missing_fields:
        raise ValueError(f"Missing fields in {TOTAL_VALUE_LOOKUP_CSV}: {missing_fields}")

    with rasterio.open(POPULATION_BIG_GAME_HUNTERS_RASTER) as template_src:
        template_profile = template_src.profile.copy()
        template_shape = template_src.shape
        pixel_size_km = abs(template_src.transform.a) / 1000

    total_demand_value = np.zeros(template_shape, dtype="float32")
    total_trips_allocated = np.zeros(template_shape, dtype="float32")
    total_flow_central = np.zeros(template_shape, dtype="float32")
    total_flow_low = np.zeros(template_shape, dtype="float32")
    total_flow_high = np.zeros(template_shape, dtype="float32")

    out_profile = template_profile.copy()
    out_profile.update(
        driver=RASTER_DRIVER,
        dtype="float32",
        nodata=0,
        compress=RASTER_COMPRESS,
        predictor=RASTER_PREDICTOR,
    )

    species_to_reallocate = None if SPECIES_TO_REALLOCATE is None else set(SPECIES_TO_REALLOCATE)

    for species_name in HUNTING_SPECIES:
        row_match = value_lookup[value_lookup["species"] == species_name]

        if row_match.empty:
            logging.warning(f"No value lookup row found for {species_name}. Skipping.")
            continue

        row = row_match.iloc[0]

        if species_name not in HUNTING_SPECIES:
            continue

        if species_to_reallocate is not None and species_name not in species_to_reallocate:
            if USE_EXISTING_FLOW_RASTERS_FOR_SKIPPED_SPECIES:
                existing_species_flow_raster = HUNTING_FLOW_VALUE_SPECIES_TEMPLATE.format(species=species_name)

                if not os.path.exists(existing_species_flow_raster):
                    raise FileNotFoundError(
                        f"Existing flow raster not found for skipped species {species_name}: "
                        f"{existing_species_flow_raster}"
                    )

                existing_species_flow = read_raster_aligned_to_template(
                    raster_path=existing_species_flow_raster,
                    template_profile=template_profile,
                    template_shape=template_shape,
                )

                total_value_central = float(row["total_value_central"])
                total_value_low = float(row["total_value_low"])
                total_value_high = float(row["total_value_high"])

                if total_value_central > 0:
                    low_scale = total_value_low / total_value_central
                    high_scale = total_value_high / total_value_central
                else:
                    low_scale = 0
                    high_scale = 0

                total_flow_central += existing_species_flow
                total_flow_low += (existing_species_flow * low_scale).astype("float32")
                total_flow_high += (existing_species_flow * high_scale).astype("float32")

                logging.info(
                    f"Included existing flow raster for {species_name}: "
                    f"{existing_species_flow_raster}"
                )

            continue

        hunter_population_raster = get_hunter_population_raster_for_species(species_name)

        hunter_population = read_raster_aligned_to_template(
            raster_path=hunter_population_raster,
            template_profile=template_profile,
            template_shape=template_shape,
        )

        hunter_population = np.where(np.isfinite(hunter_population), hunter_population, 0).astype("float32")
        total_hunters = hunter_population.sum()

        if total_hunters <= 0:
            raise ValueError(f"Total hunter population is zero for {species_name}. Demand cannot be allocated.")

        hunter_share = hunter_population / total_hunters

        trips_per_year = float(row["trips_per_year"])
        mean_distance_km = float(row["mean_one_way_travel_distance_km"])
        total_value_central = float(row["total_value_central"])
        total_value_low = float(row["total_value_low"])
        total_value_high = float(row["total_value_high"])

        species_supply_raster = SUPPLY_SPECIES_TEMPLATE.format(species=species_name)

        species_supply = read_raster_aligned_to_template(
            raster_path=species_supply_raster,
            template_profile=template_profile,
            template_shape=template_shape,
        )

        logging.info(
            f"{species_name}: supply min={np.nanmin(species_supply):,.2f}, "
            f"max={np.nanmax(species_supply):,.2f}, "
            f"sum={np.nansum(species_supply):,.2f}, "
            f"negative_cells={(species_supply < 0).sum():,}, "
            f"cells_lt_minus_1000000={(species_supply < -1_000_000).sum():,}"
        )

        #Check for negatives
        if np.any(species_supply < 0):
            min_value = np.nanmin(species_supply)

            raise ValueError(
                f"{species_name}: supply raster contains negative values. "
                f"Minimum value = {min_value:,.2f}"
            )

        species_trip_demand = (hunter_share * trips_per_year).astype("float32")
        species_value_demand = (hunter_share * total_value_central).astype("float32")

        kernel = create_exponential_decay_kernel(mean_distance_km, pixel_size_km)

        logging.info(f"Allocating hunting trips for {species_name}.")
        trip_accessibility = fftconvolve(species_trip_demand, kernel, mode="same").astype("float32")
        logging.info(f"Completed distance-decay convolution for {species_name}.")

        raw_trip_flow = species_supply * trip_accessibility

        if raw_trip_flow.sum() > 0:
            species_trip_flow = (raw_trip_flow * (trips_per_year / raw_trip_flow.sum())).astype("float32")
        else:
            species_trip_flow = np.zeros(template_shape, dtype="float32")

        value_per_trip_central = total_value_central / trips_per_year if trips_per_year > 0 else 0
        species_flow_central = (species_trip_flow * value_per_trip_central).astype("float32")

        if total_value_central > 0:
            low_scale = total_value_low / total_value_central
            high_scale = total_value_high / total_value_central
        else:
            low_scale = 0
            high_scale = 0

        species_flow_low = (species_flow_central * low_scale).astype("float32")
        species_flow_high = (species_flow_central * high_scale).astype("float32")

        species_output = save_raster(
            species_flow_central,
            HUNTING_FLOW_VALUE_SPECIES_TEMPLATE.format(species=species_name),
            out_profile,
        )
        logging.info(f"Hunting flow value raster created successfully: {species_output}")

        total_demand_value += species_value_demand
        total_trips_allocated += species_trip_flow
        total_flow_central += species_flow_central
        total_flow_low += species_flow_low
        total_flow_high += species_flow_high

        logging.info(
            f"{species_name}: trips={trips_per_year:,.2f}, "
            f"supply_sum={species_supply.sum():,.2f}, "
            f"raw_trip_flow_sum={raw_trip_flow.sum():,.2f}, "
            f"allocated_trips={species_trip_flow.sum():,.2f}, "
            f"allocated_value={species_flow_central.sum():,.2f}, "
            f"target_value={total_value_central:,.2f}"
        )

    demand_output = save_raster(total_demand_value, HUNTING_DEMAND_VALUE_RASTER, out_profile)
    logging.info(f"Hunting demand value raster created successfully: {demand_output}")

    trips_output = save_raster(total_trips_allocated, HUNTING_TRIPS_RASTER, out_profile)
    logging.info(f"Hunting trips raster created successfully: {trips_output}")

    total_output = save_raster(total_flow_central, HUNTING_FLOW_VALUE_TOTAL_RASTER, out_profile)
    logging.info(f"Hunting flow value raster created successfully: {total_output}")

    low_output = save_raster(total_flow_low, HUNTING_FLOW_VALUE_LOW_RASTER, out_profile)
    logging.info(f"Hunting low flow value raster created successfully: {low_output}")

    high_output = save_raster(total_flow_high, HUNTING_FLOW_VALUE_HIGH_RASTER, out_profile)
    logging.info(f"Hunting high flow value raster created successfully: {high_output}")

def main():
    if RUN_HUNTER_POPULATION_RASTERS:
        create_hunter_population_rasters()

    if RUN_BIG_GAME_SUPPLY_RASTERS:
        create_big_game_supply_rasters()

    if RUN_VALUE_ALLOCATION:
        allocate_hunting_flow_values()

if __name__ == "__main__":
    main()