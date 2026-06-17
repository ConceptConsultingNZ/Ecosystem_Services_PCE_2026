"""
Summarise all matching rasters for one specified feature in a polygon layer,
disaggregated by a second, more detailed polygon layer.

Example:
    Summarise raster values for one Functional Urban Area, split by landcover class.

For each raster:
    1. Load one selected feature by ID.
    2. Load the detailed/disaggregation layer.
    3. Intersect detailed polygons with the selected feature.
    4. Read only the raster window overlapping the selected feature.
    5. Rasterise each intersected detailed polygon/group into that window.
    6. Summarise raster values inside each detailed class.

This is much faster than processing the full raster when the selected feature is
small relative to the raster extent. It also avoids summarising the whole selected
feature as one big geographic smoothie.
"""

import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform

# =========================
# User settings
# =========================
timestamp = datetime.now().strftime("%Y%m%d%H%M")
output_csv = rf"D:\tmp\Waihi_by_conservation_{timestamp}.csv"

#Rasters to include
base_dir = r"D:\tmp\updated_rasters"
ends_with = "value.tif"
max_rasters_to_process = None #None means all

# Main feature layer: this is the larger polygon to summarise within.
# Example: one urban area.
#feature_path = r"D:\Data\Stats NZ geography\statsnz-functional-urban-area-2023-generalised-GPKG\FUA_2023_singleparts.gpkg"
feature_path = r"D:\Data\Stats NZ geography\statsnz-statistical-area-3-2025\statistical-area-3-2025.gpkg"
feature_layer = None
feature_id_field = "SA32025_V1_00"
selected_feature_id = '52430'
feature_label_field = "SA32025_V1_NAME"

# Detailed/disaggregation layer: this is the layer used to split the selected feature.
#disaggregate_path = r"D:\Data\LRIS\lris-lcdb-v60-land-cover-database-version-60-mainland-new-zealand\lcdb-v60-land-cover-database-version-60-mainland-new-zealand.gpkg"
disaggregate_path = r"D:\Data\LINZ\lds-protected-areas-GPKG\protected-areas.gpkg"
#disaggregate_layer = "lcdb_v60_land_cover_database_version_60_mainland_new_zealand"
disaggregate_layer = None

# The field that defines the detailed classes, e.g. landcover class.
disaggregate_class_field = "napalis_id"

# Optional extra fields to carry through to the output, if they exist.
# Useful examples: ["LCDB5Name", "Class_2018", "landcover_code"]
disaggregate_extra_fields = ["name","type"]

# Optional filter for the disaggregation layer.
# Leave as None to use all detailed polygons overlapping the selected feature.
# Example: "class_group == 'Urban'"
disaggregate_query = None

aggregation_types = ["sum", "mean", "median", "count", "min", "max", "q05", "q95"]
all_touched = False

# If True, detailed polygons with the same class are dissolved within the selected feature
# before rasterisation. This gives one row per class per raster.
# If False, each detailed polygon fragment is summarised separately.
dissolve_disaggregate_classes = True


# =========================
# Helper functions
# =========================
def get_available_filename(path):
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    i = 1

    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def load_vector_layer(path, layer=None, label="layer"):
    print(f"Loading {label}...")

    if layer is None:
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(path, layer=layer)

    if gdf.empty:
        raise ValueError(f"{label} is empty.")

    if gdf.crs is None:
        raise ValueError(f"{label} has no CRS defined.")

    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError(f"{label} has no valid geometries.")

    return gdf

def load_selected_feature():
    gdf = load_vector_layer(feature_path, feature_layer, "main feature layer")

    if feature_id_field not in gdf.columns:
        raise ValueError(f"Field '{feature_id_field}' not found in main feature layer.")

    selected = gdf.loc[gdf[feature_id_field] == selected_feature_id].copy()

    if selected.empty:
        raise ValueError(
            f"No feature found where {feature_id_field} == {selected_feature_id}."
        )

    if len(selected) > 1:
        raise ValueError(
            f"More than one feature found where {feature_id_field} == "
            f"{selected_feature_id}. The ID field must be unique."
        )

    if feature_label_field is not None and feature_label_field in selected.columns:
        feature_label = str(selected.iloc[0][feature_label_field])
    else:
        feature_label = None

    print(f"Selected feature ID: {selected_feature_id}")
    print(f"Selected feature label: {feature_label}")

    return selected, feature_label

def load_disaggregation_features(selected_feature):
    detail = load_vector_layer(
        disaggregate_path,
        disaggregate_layer,
        "disaggregation layer"
    )
    """
    Load disaggregation features intersecting the selected feature and add
    an extra 'No overlap' class representing areas not covered by any
    disaggregation polygon.
    """

    if disaggregate_class_field not in detail.columns:
        raise ValueError(
            f"Field '{disaggregate_class_field}' not found in disaggregation layer."
        )

    missing_extra_fields = [
        field for field in disaggregate_extra_fields if field not in detail.columns
    ]
    if missing_extra_fields:
        raise ValueError(
            f"Extra disaggregation field(s) not found: {missing_extra_fields}"
        )

    if disaggregate_query is not None:
        print(f"Filtering disaggregation layer with query: {disaggregate_query}")
        detail = detail.query(disaggregate_query).copy()

    if detail.crs != selected_feature.crs:
        detail = detail.to_crs(selected_feature.crs)

    selected_geom = selected_feature.geometry.iloc[0]
    selected_bounds = selected_geom.bounds

    print("Clipping disaggregation layer to selected feature bounds...")

    detail = detail.cx[
        selected_bounds[0]:selected_bounds[2],
        selected_bounds[1]:selected_bounds[3]
    ].copy()

    selected_for_overlay = selected_feature[[feature_id_field, "geometry"]].copy()

    pieces_list = []

    if not detail.empty:
        keep_fields = [disaggregate_class_field] + disaggregate_extra_fields + ["geometry"]
        detail_for_overlay = detail[keep_fields].copy()

        print("Intersecting disaggregation features with selected feature...")

        pieces = gpd.overlay(
            detail_for_overlay,
            selected_for_overlay,
            how="intersection",
            keep_geom_type=True
        )

        pieces = pieces.loc[pieces.geometry.notna() & ~pieces.geometry.is_empty].copy()

        if not pieces.empty:
            pieces["disaggregate_class"] = pieces[disaggregate_class_field].astype(str)
            pieces_list.append(pieces)

    # Add the part of the selected feature not covered by any disaggregation feature.
    print("Finding area not covered by disaggregation features...")

    if pieces_list:
        covered_geom = gpd.GeoSeries(
            pd.concat([p.geometry for p in pieces_list], ignore_index=True),
            crs=selected_feature.crs
        ).union_all()

        uncovered_geom = selected_geom.difference(covered_geom)
    else:
        uncovered_geom = selected_geom

    if uncovered_geom is not None and not uncovered_geom.is_empty:
        uncovered_row = {
            disaggregate_class_field: "No overlap",
            "disaggregate_class": "No overlap",
            "geometry": uncovered_geom
        }

        for field in disaggregate_extra_fields:
            uncovered_row[field] = None

        uncovered = gpd.GeoDataFrame(
            [uncovered_row],
            geometry="geometry",
            crs=selected_feature.crs
        )

        pieces_list.append(uncovered)

    if not pieces_list:
        raise ValueError("Selected feature has no valid geometry to summarise.")

    pieces = pd.concat(pieces_list, ignore_index=True)
    pieces = gpd.GeoDataFrame(pieces, geometry="geometry", crs=selected_feature.crs)

    pieces = pieces.loc[pieces.geometry.notna() & ~pieces.geometry.is_empty].copy()

    if dissolve_disaggregate_classes:
        dissolve_fields = ["disaggregate_class"] + disaggregate_extra_fields

        print("Dissolving features by disaggregation class...")

        pieces = pieces.dissolve(
            by=dissolve_fields,
            as_index=False,
            dropna=False
        )

    pieces["disaggregate_part_id"] = np.arange(1, len(pieces) + 1)

    print(f"Prepared {len(pieces)} disaggregation feature(s)/class(es), including uncovered area.")

    return pieces

def find_rasters():
    print(f"Scanning for rasters ending with '{ends_with}'...")

    raster_files = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(ends_with.lower()):
                raster_files.append(os.path.join(root, file))

    raster_files.sort()

    if max_rasters_to_process is not None:
        raster_files = raster_files[:max_rasters_to_process]
        print(f"Limiting processing to first {len(raster_files)} raster(s).")

    print(f"Found {len(raster_files)} raster(s) to process.")

    return raster_files

def get_category_and_subcategory_from_filename(raster_path):
    raster_file = os.path.basename(raster_path)

    if not raster_file.lower().endswith("_value.tif"):
        raise ValueError(
            f"Raster filename must end with '_value.tif': {raster_file}"
        )

    name = raster_file[:-len("_value.tif")]

    if "_" not in name:
        category = name
        subcategory = None
    else:
        category, subcategory = name.split("_", 1)
        subcategory = subcategory.replace("_", " ").capitalize()

    return category, subcategory

def safe_window_from_feature_bounds(feature_bounds, src):
    raw_window = from_bounds(
        *feature_bounds,
        transform=src.transform
    )

    raster_window = raw_window.round_offsets().round_lengths()

    raster_window = raster_window.intersection(
        rasterio.windows.Window(
            col_off=0,
            row_off=0,
            width=src.width,
            height=src.height
        )
    )

    return raster_window

def calculate_stats(values):
    if values.size == 0:
        return {
            "sum": 0.0,
            "mean": None,
            "median": None,
            "count": 0,
            "min": None,
            "max": None,
            "q05": None,
            "q95": None,
            "positive_cell_count": 0,
        }

    return {
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "positive_cell_count": int((values > 0).sum()),
    }

def summarise_piece(piece_geom, data, local_transform):
    local_shape = data.shape

    piece_mask = rasterize(
        shapes=[(piece_geom, 1)],
        out_shape=local_shape,
        transform=local_transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched
    ).astype(bool)

    valid_value_mask = piece_mask & ~data.mask
    values = data.filled(0)[valid_value_mask].astype(float)

    pixel_area_ha = abs(local_transform.a * local_transform.e) / 10_000
    area_ha = float(piece_mask.sum() * pixel_area_ha)

    stats = calculate_stats(values)

    return area_ha, stats


# =========================
# Validate settings
# =========================
supported_stats = {"sum", "mean", "median", "count", "min", "max", "q05", "q95"}

unsupported = [s for s in aggregation_types if s not in supported_stats]
if unsupported:
    raise ValueError(
        f"Unsupported aggregation type(s): {unsupported}. "
        f"Currently supported: {sorted(supported_stats)}"
    )


# =========================
# Load vector inputs
# =========================
selected_feature, feature_label = load_selected_feature()
disaggregation_features = load_disaggregation_features(selected_feature)

# Find rasters
raster_files = find_rasters()

# Process rasters
results = []

for i, raster_path in enumerate(raster_files, start=1):
    raster_file = os.path.basename(raster_path)
    raster_modified_date = datetime.fromtimestamp(
        os.path.getmtime(raster_path)
    ).isoformat(sep=" ", timespec="seconds")

    category, subcategory = get_category_and_subcategory_from_filename(raster_path)

    print(f"\n[{i}/{len(raster_files)}] Processing: {raster_path}")

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError("Raster has no CRS defined.")

            if selected_feature.crs != src.crs:
                selected_for_raster = selected_feature.to_crs(src.crs)
            else:
                selected_for_raster = selected_feature

            if disaggregation_features.crs != src.crs:
                pieces_for_raster = disaggregation_features.to_crs(src.crs)
            else:
                pieces_for_raster = disaggregation_features

            selected_geom = selected_for_raster.geometry.iloc[0]
            raster_bounds = src.bounds
            feature_bounds = selected_geom.bounds

            if (
                feature_bounds[2] <= raster_bounds.left or
                feature_bounds[0] >= raster_bounds.right or
                feature_bounds[3] <= raster_bounds.bottom or
                feature_bounds[1] >= raster_bounds.top
            ):
                print("  Selected feature does not overlap raster extent.")

                for _, piece in pieces_for_raster.iterrows():
                    stats = calculate_stats(np.array([], dtype=float))

                    output_row = {
                        "category": category,
                        "subcategory": subcategory,
                        "raster_file": raster_file,
                        "raster_modified_date": raster_modified_date,
                        "feature_id": selected_feature_id,
                        "feature_label": feature_label,
                        "disaggregate_part_id": int(piece["disaggregate_part_id"]),
                        "disaggregate_class": piece["disaggregate_class"],
                        "area_ha": 0.0,
                        "positive_cell_count": stats["positive_cell_count"],
                    }

                    for field in disaggregate_extra_fields:
                        output_row[field] = piece.get(field)

                    for stat in aggregation_types:
                        output_row[stat] = stats[stat]

                    results.append(output_row)

                continue

            read_window = safe_window_from_feature_bounds(feature_bounds, src)

            if read_window.width <= 0 or read_window.height <= 0:
                print("  Selected feature overlap window has zero size.")
                data = None
                local_transform = None
            else:
                data = src.read(1, window=read_window, masked=True)
                local_transform = window_transform(read_window, src.transform)
                print(f"  Window shape: {data.shape}")

            for _, piece in pieces_for_raster.iterrows():
                piece_geom = piece.geometry

                if data is None:
                    area_ha = 0.0
                    stats = calculate_stats(np.array([], dtype=float))
                else:
                    area_ha, stats = summarise_piece(
                        piece_geom,
                        data,
                        local_transform
                    )

                output_row = {
                    "category": category,
                    "subcategory": subcategory,
                    "raster_file": raster_file,
                    "raster_modified_date": raster_modified_date,
                    "feature_id": selected_feature_id,
                    "feature_label": feature_label,
                    "disaggregate_part_id": int(piece["disaggregate_part_id"]),
                    "disaggregate_class": piece["disaggregate_class"],
                    "area_ha": area_ha,
                    "positive_cell_count": stats["positive_cell_count"],
                }

                for field in disaggregate_extra_fields:
                    output_row[field] = piece.get(field)

                for stat in aggregation_types:
                    output_row[stat] = stats[stat]

                results.append(output_row)

            print(f"  Added {len(pieces_for_raster)} disaggregated row(s).")
            print("  Done.")

    except Exception as e:
        print(f"  Failed: {e}")


# =========================
# Save output
# =========================
df = pd.DataFrame(results)

base_columns = [
    "category",
    "subcategory",
    "raster_file",
    "raster_modified_date",
    "feature_id",
    "feature_label",
    "disaggregate_part_id",
    "disaggregate_class",
] + disaggregate_extra_fields + [
    "area_ha",
    "positive_cell_count",
]

output_columns = base_columns + aggregation_types
df = df.reindex(columns=output_columns)

final_output_csv = get_available_filename(output_csv)
df.to_csv(final_output_csv, index=False, encoding="utf-8-sig")

print(f"\nSaved results to: {final_output_csv}")
print(f"Rows written: {len(df)}")
print("Finished. The polygons have been sliced, diced, and politely summarised.")

