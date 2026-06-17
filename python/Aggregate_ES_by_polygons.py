"""
Summarise raster values by one required feature layer and one optional second
feature layer.

Feature layers are rasterised once to a user-specified target grid.
Each value raster must match that target grid exactly.
"""

import os
from datetime import datetime
from math import ceil

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

# =========================
# User settings
# =========================

timestamp = datetime.now().strftime("%Y%m%d%H%M")
output_csv = rf"D:\tmp\wetland_ES_{timestamp}.csv"
aggregation_types = ["sum", "mean", "median", "count", "min", "max", "q05", "q95"]
max_rasters_to_process = None
ends_with = "value.tif"
base_dir = r"D:\tmp\updated_rasters"

# Target grid settings
target_crs = "EPSG:2193"
# Bounds are xmin, ymin, xmax, ymax in target_crs units.
target_bounds = (1089300,4162800,2470400,6223200)
target_pixel_size = 100

# Required first feature layer
feature1_path = r"D:\Data\LRIS\Wetlands\WONI.gpkg"
feature1_layer = "WONI_with_type"

feature1_id_field = "feature_id"
feature1_label_field = "wetland_type"
feature1_unclassified_value = 0
feature1_unclassified_label = "Not wetland"
feature1_all_touched = False

# Optional second feature layer
feature2_path = None # r"D:\Data\eco-index\kx-eco-index-catchments-for-nz\eco-index-catchments-for-nz.gpkg"
feature2_layer = None #"eco_index_catchments_for_nz"
feature2_class_value_field = None
feature2_class_label_field = "Catchment"
feature2_unclassified_value = 0
feature2_unclassified_label = "No catchment"
feature2_all_touched = False


# =========================
# Helper functions
# =========================
def read_raster_on_target_grid(src, target_crs, target_transform, target_shape):
    dst = np.full(target_shape, np.nan, dtype="float64")

    src_nodata = src.nodata

    reproject(
        source=rasterio.band(src, 1),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src_nodata,
        dst_transform=target_transform,
        dst_crs=target_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest
    )

    return np.ma.masked_invalid(dst)

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

def build_target_grid(bounds, pixel_size):
    xmin, ymin, xmax, ymax = bounds

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("target_bounds must be xmin, ymin, xmax, ymax.")

    width = int(ceil((xmax - xmin) / pixel_size))
    height = int(ceil((ymax - ymin) / pixel_size))

    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    return transform, (height, width)

def load_feature_layer(path, layer, layer_description):
    if layer is None:
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.read_file(path, layer=layer)

    if gdf.empty:
        raise ValueError(f"No features found in {layer_description}.")

    if gdf.crs is None:
        raise ValueError(f"{layer_description} has no CRS defined.")

    gdf = gdf.copy()
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError(f"No valid geometries found in {layer_description}.")

    return gdf

def validate_integer_id_field(gdf, field_name, layer_description):
    if field_name not in gdf.columns:
        raise ValueError(f"Field '{field_name}' not found in {layer_description}.")

    if gdf[field_name].isna().any():
        raise ValueError(f"Field '{field_name}' in {layer_description} contains nulls.")

    try:
        gdf[field_name] = gdf[field_name].astype(np.int32)
    except Exception as e:
        raise ValueError(
            f"Field '{field_name}' in {layer_description} must be convertible "
            "to integer for rasterisation."
        ) from e

    return gdf

def validate_label_field(gdf, field_name, layer_description):
    if field_name not in gdf.columns:
        raise ValueError(f"Field '{field_name}' not found in {layer_description}.")
    return gdf

def make_raster_shapes(gdf, value_field):
    return [
        (geom, int(value))
        for geom, value in zip(gdf.geometry, gdf[value_field])
        if geom is not None and not geom.is_empty and pd.notna(value)
    ]

def rasters_match_target(src, target_crs, target_transform, target_shape):
    if src.crs is None:
        return False, "Raster has no CRS."

    if str(src.crs) != str(rasterio.crs.CRS.from_string(target_crs)):
        return False, f"CRS mismatch. Raster CRS is {src.crs}, target CRS is {target_crs}."

    if src.shape != target_shape:
        return False, f"Shape mismatch. Raster shape is {src.shape}, target shape is {target_shape}."

    if not np.allclose(tuple(src.transform), tuple(target_transform), rtol=0, atol=1e-9):
        return False, "Transform mismatch."

    return True, None

def prepare_feature1():
    feature1 = load_feature_layer(
        feature1_path,
        feature1_layer,
        "first feature layer"
    )

    feature1 = validate_label_field(
        feature1,
        feature1_label_field,
        "first feature layer"
    )

    keep_fields = [feature1_label_field, "geometry"]

    if feature1_id_field is not None:
        keep_fields.insert(0, feature1_id_field)

    feature1 = feature1[keep_fields].copy()

    feature1[feature1_label_field] = (
        feature1[feature1_label_field]
        .fillna("Missing feature1 label")
        .astype(str)
    )

    if feature1_id_field is None:
        generated_lookup_df = (
            feature1[[feature1_label_field]]
            .drop_duplicates()
            .sort_values(feature1_label_field)
            .reset_index(drop=True)
        )

        generated_lookup_df["generated_feature1_id"] = (
            np.arange(1, len(generated_lookup_df) + 1).astype(np.int32)
        )

        label_to_feature1_id = dict(
            zip(
                generated_lookup_df[feature1_label_field],
                generated_lookup_df["generated_feature1_id"]
            )
        )

        feature1["_feature1_id"] = (
            feature1[feature1_label_field]
            .map(label_to_feature1_id)
            .astype(np.int32)
        )

        feature1_id_field_for_raster = "_feature1_id"

        feature1_lookup = generated_lookup_df.rename(
            columns={
                "generated_feature1_id": "feature1_id",
                feature1_label_field: "feature1_label"
            }
        )

    else:
        feature1 = validate_integer_id_field(
            feature1,
            feature1_id_field,
            "first feature layer"
        )

        feature1_id_field_for_raster = feature1_id_field

        if (feature1[feature1_id_field_for_raster] == feature1_unclassified_value).any():
            raise ValueError(
                f"`feature1_unclassified_value` is set to {feature1_unclassified_value}, "
                "but that value already exists in the first feature layer."
            )

        feature1_lookup = (
            feature1[[feature1_id_field_for_raster, feature1_label_field]]
            .drop_duplicates()
            .copy()
        )

        duplicate_values = feature1_lookup.loc[
            feature1_lookup[feature1_id_field_for_raster].duplicated(),
            feature1_id_field_for_raster
        ].tolist()

        if duplicate_values:
            raise ValueError(
                f"First feature ID value(s) have multiple labels. "
                f"Example duplicate value(s): {duplicate_values[:10]}"
            )

        feature1_lookup = (
            feature1_lookup
            .sort_values(feature1_id_field_for_raster)
            .reset_index(drop=True)
            .rename(
                columns={
                    feature1_id_field_for_raster: "feature1_id",
                    feature1_label_field: "feature1_label"
                }
            )
        )

    if feature1_unclassified_value in feature1_lookup["feature1_id"].values:
        raise ValueError(
            f"`feature1_unclassified_value` is set to {feature1_unclassified_value}, "
            "but that value already exists in the first feature layer."
        )

    feature1_lookup = pd.concat(
        [
            feature1_lookup,
            pd.DataFrame({
                "feature1_id": [feature1_unclassified_value],
                "feature1_label": [feature1_unclassified_label]
            })
        ],
        ignore_index=True
    )

    feature1 = feature1[
        [feature1_id_field_for_raster, feature1_label_field, "geometry"]
    ].copy()

    return feature1, feature1_lookup, feature1_id_field_for_raster

def prepare_optional_feature2():
    if feature2_path is None:
        return None, {0: None}, None

    feature2 = load_feature_layer(
        feature2_path,
        feature2_layer,
        "second feature layer"
    )

    feature2 = validate_label_field(
        feature2,
        feature2_class_label_field,
        "second feature layer"
    )

    keep_fields = [feature2_class_label_field, "geometry"]

    if feature2_class_value_field is not None:
        keep_fields.insert(0, feature2_class_value_field)

    feature2 = feature2[keep_fields].copy()

    feature2[feature2_class_label_field] = (
        feature2[feature2_class_label_field]
        .fillna("Missing feature2 label")
        .astype(str)
    )

    if feature2_class_value_field is None:
        generated_lookup_df = (
            feature2[[feature2_class_label_field]]
            .drop_duplicates()
            .sort_values(feature2_class_label_field)
            .reset_index(drop=True)
        )

        generated_lookup_df["generated_feature2_value"] = (
            np.arange(1, len(generated_lookup_df) + 1).astype(np.int32)
        )

        label_to_value = dict(
            zip(
                generated_lookup_df[feature2_class_label_field],
                generated_lookup_df["generated_feature2_value"]
            )
        )

        feature2["_feature2_value"] = (
            feature2[feature2_class_label_field]
            .map(label_to_value)
            .astype(np.int32)
        )

        feature2_value_field_for_raster = "_feature2_value"

        feature2_lookup = dict(
            zip(
                generated_lookup_df["generated_feature2_value"],
                generated_lookup_df[feature2_class_label_field]
            )
        )

    else:
        if feature2_class_value_field not in feature2.columns:
            raise ValueError(
                f"Field '{feature2_class_value_field}' not found in second feature layer."
            )

        feature2[feature2_class_value_field] = pd.to_numeric(
            feature2[feature2_class_value_field],
            errors="raise"
        ).astype(np.int32)

        feature2_value_field_for_raster = feature2_class_value_field

        if (feature2[feature2_value_field_for_raster] == feature2_unclassified_value).any():
            raise ValueError(
                f"`feature2_unclassified_value` is set to {feature2_unclassified_value}, "
                "but that value already exists in the second feature layer."
            )

        feature2_lookup_df = (
            feature2[[feature2_value_field_for_raster, feature2_class_label_field]]
            .drop_duplicates()
            .copy()
        )

        duplicate_values = feature2_lookup_df.loc[
            feature2_lookup_df[feature2_value_field_for_raster].duplicated(),
            feature2_value_field_for_raster
        ].tolist()

        if duplicate_values:
            raise ValueError(
                f"Second feature class value(s) have multiple labels. "
                f"Example duplicate value(s): {duplicate_values[:10]}"
            )

        feature2_lookup = dict(
            zip(
                feature2_lookup_df[feature2_value_field_for_raster],
                feature2_lookup_df[feature2_class_label_field]
            )
        )

    feature2_lookup[feature2_unclassified_value] = feature2_unclassified_label

    return feature2, feature2_lookup, feature2_value_field_for_raster

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
# Build target grid
# =========================

target_transform, target_shape = build_target_grid(
    target_bounds,
    target_pixel_size
)

pixel_area_ha = abs(target_transform.a * target_transform.e) / 10_000

print("Target grid created.")
print(f"  CRS: {target_crs}")
print(f"  Bounds: {target_bounds}")
print(f"  Pixel size: {target_pixel_size}")
print(f"  Shape: {target_shape}")
print(f"  Pixel area ha: {pixel_area_ha}")


# =========================
# Load and rasterise feature layers once
# =========================

print("Loading first feature layer...")

feature1, feature1_lookup, feature1_id_field_for_raster = prepare_feature1()

print(f"Loaded {len(feature1)} first-layer feature(s).")
print(f"Loaded {len(feature1_lookup)} first-layer lookup row(s).")

if str(feature1.crs) != str(rasterio.crs.CRS.from_string(target_crs)):
    print("Reprojecting first feature layer to target CRS...")
    feature1_for_raster = feature1.to_crs(target_crs)
else:
    feature1_for_raster = feature1

print("Rasterising first feature layer once...")

feature1_for_raster = feature1_for_raster.sort_values(
    feature1_id_field_for_raster
)

feature1_shapes = make_raster_shapes(
    feature1_for_raster,
    feature1_id_field_for_raster
)

feature1_grid = rasterize(
    shapes=feature1_shapes,
    out_shape=target_shape,
    transform=target_transform,
    fill=feature1_unclassified_value,
    dtype="int32",
    all_touched=feature1_all_touched
)

feature2_enabled = feature2_path is not None

if feature2_enabled:
    print("Loading optional second feature layer...")

    feature2, feature2_lookup, feature2_value_field_for_raster = prepare_optional_feature2()

    print(f"Loaded {len(feature2)} second-layer feature(s).")
    print(f"Loaded {len(feature2_lookup)} second-layer class lookup row(s).")

    if str(feature2.crs) != str(rasterio.crs.CRS.from_string(target_crs)):
        print("Reprojecting second feature layer to target CRS...")
        feature2_for_raster = feature2.to_crs(target_crs)
    else:
        feature2_for_raster = feature2

    print("Rasterising second feature layer once...")

    feature2_for_raster = feature2_for_raster.sort_values(
        feature2_value_field_for_raster
    )

    feature2_shapes = make_raster_shapes(
        feature2_for_raster,
        feature2_value_field_for_raster
    )

    feature2_grid = rasterize(
        shapes=feature2_shapes,
        out_shape=target_shape,
        transform=target_transform,
        fill=feature2_unclassified_value,
        dtype="int32",
        all_touched=feature2_all_touched
    )

else:
    print("No second feature layer provided. Summarising by first feature layer only.")
    feature2_lookup = {0: None}
    feature2_grid = np.zeros(target_shape, dtype=np.int32)


# =========================
# Find rasters
# =========================

results = []

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


# =========================
# Process rasters
# =========================

for i, raster_path in enumerate(raster_files, start=1):
    raster_file = os.path.basename(raster_path)

    category, subcategory = get_category_and_subcategory_from_filename(raster_path)

    raster_modified_date = datetime.fromtimestamp(
        os.path.getmtime(raster_path)
    ).isoformat(sep=" ", timespec="seconds")

    root = os.path.dirname(raster_path)
    rel_path = os.path.relpath(root, base_dir)
    foldername = rel_path.split(os.sep)[0]

    print(f"\n[{i}/{len(raster_files)}] Processing: {raster_path}")

    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError("Raster has no CRS defined.")

            print("  Warping raster to target grid...")
            data = read_raster_on_target_grid(
                src,
                target_crs,
                target_transform,
                target_shape
            )

        area_mask = np.ones(target_shape, dtype=bool)
        value_mask = ~data.mask

        values = data.filled(0)
        raster_values = values[value_mask].astype(float)

        raster_total = float(raster_values.sum()) if raster_values.size else 0.0
        print(f"  Raster total: {raster_total}")

        value_stats_df = pd.DataFrame({
            "feature1_id": feature1_grid[value_mask].astype(np.int32),
            "feature2_value": feature2_grid[value_mask].astype(np.int32),
            "raster_value": raster_values
        })

        value_stats_df["positive"] = value_stats_df["raster_value"] > 0

        area_stats_df = pd.DataFrame({
            "feature1_id": feature1_grid[area_mask].astype(np.int32),
            "feature2_value": feature2_grid[area_mask].astype(np.int32),
            "pixel_area_ha": pixel_area_ha
        })

        group_fields = ["feature1_id", "feature2_value"]

        value_grouped = (
            value_stats_df
            .groupby(group_fields, as_index=False)
            .agg(
                sum=("raster_value", "sum"),
                mean=("raster_value", "mean"),
                median=("raster_value", "median"),
                count=("raster_value", "size"),
                min=("raster_value", "min"),
                max=("raster_value", "max"),
                q05=("raster_value", lambda x: x.quantile(0.05)),
                q95=("raster_value", lambda x: x.quantile(0.95)),
                positive_cell_count=("positive", "sum")
            )
        )

        area_grouped = (
            area_stats_df
            .groupby(group_fields, as_index=False)
            .agg(
                area_ha=("pixel_area_ha", "sum")
            )
        )

        grouped = area_grouped.merge(
            value_grouped,
            on=group_fields,
            how="left"
        )

        grouped["sum"] = grouped["sum"].fillna(0.0)
        grouped["count"] = grouped["count"].fillna(0).astype(int)
        grouped["positive_cell_count"] = (
            grouped["positive_cell_count"]
            .fillna(0)
            .astype(int)
        )

        grouped["feature2_label"] = grouped["feature2_value"].map(feature2_lookup)

        grouped = grouped.merge(
            feature1_lookup,
            on="feature1_id",
            how="left"
        )

        for _, row in grouped.iterrows():
            output_row = {
                "category": category,
                "subcategory": subcategory,
                "raster_file": raster_file,
                "raster_modified_date": raster_modified_date,
                "feature1_id": int(row["feature1_id"]),
                "feature1_label": row["feature1_label"],
                "feature2_label": row["feature2_label"],
                "area_ha": float(row["area_ha"]),
                "positive_cell_count": int(row["positive_cell_count"]),
            }

            for stat in aggregation_types:
                val = row[stat]
                output_row[stat] = None if pd.isna(val) else float(val)

            if "count" in aggregation_types and output_row["count"] is not None:
                output_row["count"] = int(output_row["count"])

            results.append(output_row)

        total_row = {
            "category": foldername,
            "subcategory": subcategory,
            "raster_file": raster_file,
            "raster_modified_date": raster_modified_date,
            "feature1_id": None,
            "feature1_label": "Total raster",
            "feature2_label": None,
            "area_ha": None,
            "positive_cell_count": int((raster_values > 0).sum()),
        }

        raster_stat_values = {
            "sum": float(raster_values.sum()) if raster_values.size else 0.0,
            "mean": float(raster_values.mean()) if raster_values.size else None,
            "median": float(np.median(raster_values)) if raster_values.size else None,
            "count": int(raster_values.size),
            "min": float(raster_values.min()) if raster_values.size else None,
            "max": float(raster_values.max()) if raster_values.size else None,
            "q05": float(np.quantile(raster_values, 0.05)) if raster_values.size else None,
            "q95": float(np.quantile(raster_values, 0.95)) if raster_values.size else None,
        }

        for stat in aggregation_types:
            total_row[stat] = raster_stat_values[stat]

        results.append(total_row)

        if "sum" in aggregation_types:
            grouped_sum_total = float(grouped["sum"].sum())
            print(f"  Sum of grouped sums: {grouped_sum_total}")
            print(f"  Difference vs raster total: {grouped_sum_total - raster_total}")

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
    "feature1_id",
    "feature1_label",
    "feature2_label",
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