"""
Calculate reach-level freshwater fishing attractiveness from NZRiverMaps fish presence
and water quality data.

Fishing attractiveness is calculated as:

    desirability_avg × MCI_scaled × clarity_scaled × WCC_scaled

A minimum MALF, and minimum stream order 3 threshold is applied before conversion to the final 0-100 index.
Rows where MALF is missing or <= MIN_MALF are assigned zero attractiveness.

The resulting river attractiveness raster is masked using the CONSERVED_LAND raster,
so river fishing attractiveness is retained only where CONSERVED_LAND > 0.

The script then:
    1. Rasterises lake fishing attractiveness.
    2. Writes the river raster without a NoData flag.
    3. Combines lake and river fishing attractiveness, giving priority to lakes.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from pathlib import Path
from rasterio.features import rasterize

# ==============================
# CONFIGURATION CONSTANTS
# ==============================

FISH_FILE = Path(r"D:\Data\NIWA\NZRiverMaps\NZRiverMaps_data_for_fishing.csv")
DESIRABILITY_FILE = Path(r"<PROJECT_DIRECTORY>\Cultural\Recreation\Fishing\species_desirability.csv")

RIVERLINES_GPKG = Path(r"D:\Data\NIWA\REC2_geodata_version_5\nzRec2_v5.gdb")
RIVERLINES_LAYER = "riverlines"

CONSERVED_LAND = Path(r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate\conserved_land_100m.tif")

OUTPUT_DIR = Path(r"<PROJECT_DIRECTORY>\Cultural\Recreation\Intermediate")
OUTPUT_CSV = OUTPUT_DIR / "rivermaps_fish_scores.csv"
OUTPUT_GPKG = OUTPUT_DIR / "Recreation_fishing.gpkg"

OUTPUT_LINES_LAYER = "fishing_attractiveness_lines"
LAKE_LAYER = "lake_spi"
LAKE_ATTR = "lake_spi"

OUTPUT_RIVER_RASTER = OUTPUT_DIR / "attractiveness_river_fishing.tif"
OUTPUT_LAKE_RASTER = OUTPUT_DIR / "attractiveness_lake_fishing.tif"
OUTPUT_FRESHWATER_RASTER = OUTPUT_DIR / "attractiveness_freshwater_fishing.tif"

JOIN_FIELD = "nzsegment"
LOCATION_COLS = [JOIN_FIELD, "NZTM_Easting", "NZTM_Northing"]

COL_MCI = "MCI_2021"
COL_WCC = "WCC_92"
COL_CLARITY = "clarity_median"
COL_MALF = "MALF"

LOWER_Q = 0.01
UPPER_Q = 0.99
MIN_MOD = 0.2
MIN_MALF = 0.01

RASTER_NODATA = -9999
RASTER_ATTR = "fishing_attractiveness"


# ==============================
# HELPER FUNCTIONS
# ==============================

def robust_minmax(
    series: pd.Series,
    lower_q: float,
    upper_q: float,
    invert: bool = False,
    floor: float = 0.0
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)

    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=series.index)

    scaled = (s - lo) / (hi - lo)
    scaled = scaled.clip(0, 1).fillna(0)

    if invert:
        scaled = 1 - scaled

    if floor > 0:
        non_missing = s.notna()
        scaled.loc[non_missing] = scaled.loc[non_missing].clip(lower=floor, upper=1)

    return scaled


def read_input_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Reading input data...")
    fish_df = pd.read_csv(FISH_FILE)
    des_df = pd.read_csv(DESIRABILITY_FILE)
    return fish_df, des_df

def calculate_fish_scores(fish_df: pd.DataFrame, des_df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating species desirability scores...")

    des_df = des_df.copy()
    des_df["Species"] = des_df["Species"].astype(str).str.strip()
    desirability_map = dict(zip(des_df["Species"], des_df["Desirability"]))

    matching_species = [col for col in fish_df.columns if col in desirability_map]

    if not matching_species:
        raise ValueError("No fish species columns matched the desirability table. The fish are not thrilled.")

    presence = fish_df[matching_species]
    weights = pd.Series({col: desirability_map[col] for col in matching_species})
    scored = presence.mul(weights, axis=1)

    output_df = fish_df[LOCATION_COLS].copy()

    output_df["desirability_sum"] = scored.sum(axis=1)
    present_count = presence.astype(float).gt(0).sum(axis=1)
    output_df["present_species_count"] = present_count
    output_df["desirability_avg"] = output_df["desirability_sum"].div(present_count).fillna(0)

    output_df[COL_MALF] = pd.to_numeric(fish_df[COL_MALF], errors="coerce")

    return output_df

def add_water_quality_modifiers(output_df: pd.DataFrame, fish_df: pd.DataFrame) -> pd.DataFrame:
    print("Scaling water quality modifiers...")

    output_df = output_df.copy()

    output_df["mci_scaled"] = robust_minmax(
        fish_df[COL_MCI], LOWER_Q, UPPER_Q, invert=False, floor=MIN_MOD
    )
    output_df["clarity_scaled"] = robust_minmax(
        fish_df[COL_CLARITY], LOWER_Q, UPPER_Q, invert=False, floor=MIN_MOD
    )
    output_df["wcc_scaled"] = robust_minmax(
        fish_df[COL_WCC], LOWER_Q, UPPER_Q, invert=True, floor=MIN_MOD
    )

    return output_df


def calculate_river_attractiveness(output_df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating river fishing attractiveness...")

    output_df = output_df.copy()

    output_df["fishing_attractiveness_raw"] = (
        output_df["desirability_avg"]
        * output_df["mci_scaled"]
        * output_df["clarity_scaled"]
        * output_df["wcc_scaled"]
    )

    print(f"Applying MALF threshold: MALF > {MIN_MALF}...")
    valid_malf = output_df[COL_MALF].notna() & (output_df[COL_MALF] > MIN_MALF)
    output_df.loc[~valid_malf, "fishing_attractiveness_raw"] = 0

    output_df[RASTER_ATTR] = (
        (output_df["fishing_attractiveness_raw"] * 100)
        .round()
        .clip(0, 100)
        .astype("Int16")
    )

    return output_df


def save_scores_csv(output_df: pd.DataFrame) -> None:
    print("Saving CSV output...")
    output_df.to_csv(OUTPUT_CSV, index=False)


def join_scores_to_riverlines(output_df: pd.DataFrame) -> gpd.GeoDataFrame:
    print("Joining scores to riverlines...")

    river_gdf = gpd.read_file(RIVERLINES_GPKG, layer=RIVERLINES_LAYER)

    print("Excluding streams with StreamOrde < 3...")
    river_gdf["StreamOrde"] = pd.to_numeric(
        river_gdf["StreamOrde"],
        errors="coerce"
    )
    river_gdf = river_gdf[river_gdf["StreamOrde"] >= 3].copy()

    join_cols = [
        JOIN_FIELD,
        RASTER_ATTR,
        "fishing_attractiveness_raw",
        "desirability_avg",
        "mci_scaled",
        "clarity_scaled",
        "wcc_scaled",
        COL_MALF,
        "present_species_count",
        "desirability_sum",
    ]

    join_df = output_df[join_cols].copy()

    river_gdf[JOIN_FIELD] = pd.to_numeric(river_gdf[JOIN_FIELD], errors="coerce")
    join_df[JOIN_FIELD] = pd.to_numeric(join_df[JOIN_FIELD], errors="coerce")

    joined_gdf = river_gdf.merge(join_df, on=JOIN_FIELD, how="left")
    joined_gdf[RASTER_ATTR] = joined_gdf[RASTER_ATTR].fillna(0).astype(np.int16)

    return joined_gdf


def write_joined_riverlines(joined_gdf: gpd.GeoDataFrame) -> None:
    print("Writing joined riverlines...")

    joined_gdf.to_file(
        OUTPUT_GPKG,
        layer=OUTPUT_LINES_LAYER,
        driver="GPKG"
    )


def read_template_raster() -> tuple[np.ndarray, dict, object, tuple[int, int], object]:
    print("Reading conserved land raster profile...")

    with rasterio.open(CONSERVED_LAND) as src:
        conserved_data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        out_shape = conserved_data.shape
        raster_crs = src.crs

    return conserved_data, profile, transform, out_shape, raster_crs


def reproject_to_raster_crs(gdf: gpd.GeoDataFrame, raster_crs) -> gpd.GeoDataFrame:
    if gdf.crs != raster_crs:
        print("Reprojecting layer to match raster grid...")
        return gdf.to_crs(raster_crs)

    return gdf


def rasterise_river_attractiveness(
    joined_gdf: gpd.GeoDataFrame,
    conserved_data: np.ndarray,
    transform,
    out_shape: tuple[int, int]
) -> np.ndarray:
    print("Rasterising river fishing attractiveness...")

    shapes = (
        (geom, int(value))
        for geom, value in zip(joined_gdf.geometry, joined_gdf[RASTER_ATTR])
        if geom is not None and not geom.is_empty and pd.notna(value)
    )

    fishing_data = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int16"
    )

    print("Applying conserved land mask...")
    fishing_data[conserved_data <= 0] = 0

    return fishing_data.astype(np.int16)


def write_river_raster(fishing_data: np.ndarray, profile: dict) -> None:
    river_profile = profile.copy()
    river_profile.update(
        dtype="int16",
        nodata=None,
        count=1,
        compress="deflate",
        predictor=2,
        zlevel=9
    )

    print("Writing river fishing raster with no NoData flag...")
    with rasterio.open(OUTPUT_RIVER_RASTER, "w", **river_profile) as dst:
        dst.write(fishing_data.astype(np.int16), 1)


def rasterise_lake_attractiveness(
    profile: dict,
    transform,
    out_shape: tuple[int, int],
    raster_crs
) -> np.ndarray:
    print("Reading lake fishing layer...")
    lake_gdf = gpd.read_file(OUTPUT_GPKG, layer=LAKE_LAYER)
    lake_gdf = reproject_to_raster_crs(lake_gdf, raster_crs)

    print("Rasterising lake fishing attractiveness...")

    if LAKE_ATTR not in lake_gdf.columns:
        raise ValueError(f"Lake layer does not contain expected field: {LAKE_ATTR}")

    lake_shapes = (
        (geom, int(value))
        for geom, value in zip(lake_gdf.geometry, lake_gdf[LAKE_ATTR])
        if geom is not None and not geom.is_empty and pd.notna(value)
    )

    lake_data = rasterize(
        shapes=lake_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=RASTER_NODATA,
        dtype="int16"
    )

    return lake_data.astype(np.int16)


def write_lake_raster(lake_data: np.ndarray, profile: dict) -> None:
    lake_profile = profile.copy()
    lake_profile.update(
        dtype="int16",
        nodata=RASTER_NODATA,
        count=1,
        compress="deflate",
        predictor=2,
        zlevel=9
    )

    print("Writing lake fishing raster...")
    with rasterio.open(OUTPUT_LAKE_RASTER, "w", **lake_profile) as dst:
        dst.write(lake_data.astype(np.int16), 1)


def combine_lake_and_river(lake_data: np.ndarray, river_data: np.ndarray) -> np.ndarray:
    print("Combining lake and river fishing attractiveness...")
    return np.where(lake_data > 0, lake_data, river_data).astype(np.int16)


def write_freshwater_raster(freshwater_data: np.ndarray, profile: dict) -> None:
    freshwater_profile = profile.copy()
    freshwater_profile.update(
        dtype="int16",
        nodata=RASTER_NODATA,
        count=1,
        compress="deflate",
        predictor=2,
        zlevel=9
    )

    print("Writing combined freshwater fishing raster...")
    with rasterio.open(OUTPUT_FRESHWATER_RASTER, "w", **freshwater_profile) as dst:
        dst.write(freshwater_data.astype(np.int16), 1)


def print_outputs() -> None:
    print("Fishing attractiveness scores calculated successfully. The fish are, presumably, thrilled.")
    print(f"CSV written to: {OUTPUT_CSV}")
    print(f"Lines written to: {OUTPUT_GPKG} (layer: {OUTPUT_LINES_LAYER})")
    print(f"River raster written to: {OUTPUT_RIVER_RASTER}")
    print(f"Lake raster written to: {OUTPUT_LAKE_RASTER}")
    print(f"Freshwater raster written to: {OUTPUT_FRESHWATER_RASTER}")


# ==============================
# MAIN PROCESS
# ==============================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fish_df, des_df = read_input_tables()

    output_df = calculate_fish_scores(fish_df, des_df)
    output_df = add_water_quality_modifiers(output_df, fish_df)
    output_df = calculate_river_attractiveness(output_df)

    save_scores_csv(output_df)

    joined_gdf = join_scores_to_riverlines(output_df)
    write_joined_riverlines(joined_gdf)

    conserved_data, profile, transform, out_shape, raster_crs = read_template_raster()

    joined_gdf = reproject_to_raster_crs(joined_gdf, raster_crs)

    river_data = rasterise_river_attractiveness(
        joined_gdf=joined_gdf,
        conserved_data=conserved_data,
        transform=transform,
        out_shape=out_shape
    )
    write_river_raster(river_data, profile)

    lake_data = rasterise_lake_attractiveness(
        profile=profile,
        transform=transform,
        out_shape=out_shape,
        raster_crs=raster_crs
    )
    write_lake_raster(lake_data, profile)

    freshwater_data = combine_lake_and_river(lake_data, river_data)
    write_freshwater_raster(freshwater_data, profile)

    print_outputs()


if __name__ == "__main__":
    main()