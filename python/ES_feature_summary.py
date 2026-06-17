"""
Summarise all matching rasters for one specified feature in a polygon layer.

For each raster:
    1. Load one polygon feature by ID.
    2. Read only the raster window overlapping that feature's bounding box.
    3. Rasterise the selected feature into that window.
    4. Summarise raster values inside the feature.

This is much faster than processing the full raster when the selected feature is
small relative to the raster extent.
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
output_csv = rf"D:\tmp\single_feature_raster_summary_{timestamp}.csv"
base_dir = r"D:\tmp\updated_rasters"
ends_with = "value.tif"
max_rasters_to_process = None

#feature_path = r"D:\Data\MfE\mfe-prediction-of-wetlands-before-humans-arrived\prediction-of-wetlands-before-humans-arrived.gpkg"
#feature_layer = "prediction_of_wetlands_before_humans_arrived"
feature_path = r"D:\Data\Stats NZ geography\statsnz-functional-urban-area-2023-generalised-GPKG\functional-urban-area-2023-generalised.gpkg"
feature_layer = None

feature_id_field = "FUA2023_V1_00"
selected_feature_id = 1001
feature_label_field = "Pukekohe"

aggregation_types = ["sum", "mean", "median", "count", "min", "max", "q05", "q95"]
all_touched = False

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

def load_selected_feature():
    print("Loading polygon layer...")

    if feature_layer is None:
        gdf = gpd.read_file(feature_path)
    else:
        gdf = gpd.read_file(feature_path, layer=feature_layer)

    if gdf.empty:
        raise ValueError("Feature layer is empty.")

    if gdf.crs is None:
        raise ValueError("Feature layer has no CRS defined.")

    if feature_id_field not in gdf.columns:
        raise ValueError(f"Field '{feature_id_field}' not found in feature layer.")

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

    selected = selected.loc[
        selected.geometry.notna() & ~selected.geometry.is_empty
    ].copy()

    if selected.empty:
        raise ValueError("Selected feature has no valid geometry.")

    if feature_label_field is not None and feature_label_field in selected.columns:
        feature_label = str(selected.iloc[0][feature_label_field])
    else:
        feature_label = None

    print(f"Selected feature ID: {selected_feature_id}")
    print(f"Selected feature label: {feature_label}")

    return selected, feature_label

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
        subcategory = subcategory.replace("_", " ")

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
# Load selected feature
# =========================

selected_feature, feature_label = load_selected_feature()


# =========================
# Find rasters
# =========================

raster_files = find_rasters()


# =========================
# Process rasters
# =========================

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
                feature_for_raster = selected_feature.to_crs(src.crs)
            else:
                feature_for_raster = selected_feature

            geom = feature_for_raster.geometry.iloc[0]

            raster_bounds = src.bounds
            feature_bounds = geom.bounds

            if (
                feature_bounds[2] <= raster_bounds.left or
                feature_bounds[0] >= raster_bounds.right or
                feature_bounds[3] <= raster_bounds.bottom or
                feature_bounds[1] >= raster_bounds.top
            ):
                print("  Feature does not overlap raster extent.")

                stats = calculate_stats(np.array([], dtype=float))
                area_ha = 0.0

            else:
                read_window = safe_window_from_feature_bounds(feature_bounds, src)

                if read_window.width <= 0 or read_window.height <= 0:
                    print("  Feature overlap window has zero size.")

                    stats = calculate_stats(np.array([], dtype=float))
                    area_ha = 0.0

                else:
                    data = src.read(1, window=read_window, masked=True)

                    local_transform = window_transform(
                        read_window,
                        src.transform
                    )

                    local_shape = data.shape

                    feature_mask = rasterize(
                        shapes=[(geom, 1)],
                        out_shape=local_shape,
                        transform=local_transform,
                        fill=0,
                        dtype="uint8",
                        all_touched=all_touched
                    ).astype(bool)

                    valid_value_mask = feature_mask & ~data.mask

                    values = data.filled(0)[valid_value_mask].astype(float)

                    pixel_area_ha = abs(
                        local_transform.a * local_transform.e
                    ) / 10_000

                    area_ha = float(feature_mask.sum() * pixel_area_ha)

                    stats = calculate_stats(values)

                    print(f"  Window shape: {local_shape}")
                    print(f"  Area ha: {area_ha}")
                    print(f"  Count: {stats['count']}")
                    print(f"  Sum: {stats['sum']}")

        output_row = {
            "category": category,
            "subcategory": subcategory,
            "raster_file": raster_file,
            "raster_modified_date": raster_modified_date,
            "feature_id": selected_feature_id,
            "feature_label": feature_label,
            "area_ha": area_ha,
            "positive_cell_count": stats["positive_cell_count"],
        }

        for stat in aggregation_types:
            output_row[stat] = stats[stat]

        results.append(output_row)

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
    "area_ha",
    "positive_cell_count",
]

output_columns = base_columns + aggregation_types
df = df.reindex(columns=output_columns)

final_output_csv = get_available_filename(output_csv)
df.to_csv(final_output_csv, index=False, encoding="utf-8-sig")

print(f"\nSaved results to: {final_output_csv}")
print(f"Rows written: {len(df)}")
print("Finished.")