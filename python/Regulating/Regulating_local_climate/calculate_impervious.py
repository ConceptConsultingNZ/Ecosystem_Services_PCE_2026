# ============================================================
# Do not use! This crashed my PC when it ran out of memory.
# It was quicker to do it manually in QGIS than solve the memory problem (probably the joins)
# ============================================================
STOP

import sys
import os
import time
import logging

from qgis.core import (
    QgsApplication,
    QgsVectorLayer,
    QgsProcessingFeedback,
    QgsCoordinateReferenceSystem
)

# ---- QGIS init ----
QGIS_PREFIX = r"D:\Program Files\QGIS 3.44.0\apps\qgis"
QgsApplication.setPrefixPath(QGIS_PREFIX, True)
qgs = QgsApplication([], False)
qgs.initQgis()

# ---- Make plugin modules importable ----
plugins_path = os.path.join(QGIS_PREFIX, "python", "plugins")
if plugins_path not in sys.path:
    sys.path.insert(0, plugins_path)

# ---- Initialise Processing plugin ----
import processing
from processing.core.Processing import Processing
Processing.initialize()

# ---- Ensure the native provider exists (native:* algorithms) ----
from qgis.analysis import QgsNativeAlgorithms
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

# ---- Logging + feedback ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()
feedback = QgsProcessingFeedback()

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()
feedback = QgsProcessingFeedback()

# ============================================================
# INPUTS
# ============================================================

# Use your pre-built road polygons (no buffering step)
ROADS_POLY_URI = r"D:\Data\LINZ\lds-nz-road-centrelines-topo-150k\road_polygons.gpkg|layername=roads_sa1"

BUILDINGS_URI = r"D:\Data\LINZ\lds-nz-building-outlines-FGDB\buildings_fixed_geometry.gpkg|layername=fixed_geometries"
MASK_URI = r"D:\Data\LINZ\lds-nz-coastlines-and-islands-polygons-topo-150k\Mainland.shp"

TARGET_EPSG = 2193
GRID_SPACING = 100.0

# Fixed raster extent (NZTM)
XMIN = 715100.0
YMIN = 3728500.0
XMAX = 2893500.0
YMAX = 7142400.0

OUTPUT_RASTER = r"<PROJECT_DIRECTORY>\Regulating\Local climate\impervious_percent.tif"

DST_NODATA = -9999.0

# ============================================================
# HELPERS
# ============================================================

def load_layer(uri: str, name: str) -> QgsVectorLayer:
    log.info(f"Loading {name}")
    lyr = QgsVectorLayer(uri, name, "ogr")
    if not lyr.isValid():
        raise RuntimeError(f"Failed to load {name}\nURI: {uri}")
    return lyr


def assert_crs(layer: QgsVectorLayer, name: str):
    crs = layer.crs()
    if crs.authid() != f"EPSG:{TARGET_EPSG}":
        raise RuntimeError(f"{name} CRS must be EPSG:{TARGET_EPSG}, got {crs.authid()}")
    log.info(f"{name} CRS OK ({crs.authid()})")


def sum_area_per_cell(input_layer: QgsVectorLayer, grid_layer: QgsVectorLayer, label: str) -> QgsVectorLayer:
    log.info(f"Intersecting {label} with grid")
    t0 = time.time()

    inter = processing.run(
        "native:intersection",
        {
            "INPUT": input_layer,
            "OVERLAY": grid_layer,
            "INPUT_FIELDS": [],
            "OVERLAY_FIELDS": ["cell_id"],
            "OVERLAY_FIELDS_PREFIX": "",
            "OUTPUT": "memory:"
        },
        feedback=feedback
    )["OUTPUT"]

    log.info(f"Calculating area parts for {label}")
    inter = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": inter,
            "FIELD_NAME": "part_area",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 20,
            "FIELD_PRECISION": 3,
            "FORMULA": "$area",
            "OUTPUT": "memory:"
        },
        feedback=feedback
    )["OUTPUT"]

    log.info(f"Aggregating {label} area by cell")
    stats = processing.run(
        "native:statisticsbycategories",
        {
            "INPUT": inter,
            "CATEGORIES_FIELD_NAME": ["cell_id"],
            "VALUES_FIELD_NAME": "part_area",
            "STATISTICS": [1],  # 1 = sum
            "OUTPUT": "memory:"
        },
        feedback=feedback
    )["OUTPUT"]

    log.info(f"{label} aggregation complete in {time.time() - t0:.1f} sec")
    return stats


# ============================================================
# MAIN
# ============================================================

start_total = time.time()
log.info("Starting imperviousness raster (100 m) using road polygons (no buffering).")

roads_poly = load_layer(ROADS_POLY_URI, "Road polygons")
bld = load_layer(BUILDINGS_URI, "Buildings")
mask = load_layer(MASK_URI, "Mainland mask")

assert_crs(roads_poly, "Road polygons")
assert_crs(bld, "Buildings")
assert_crs(mask, "Mask")

# ------------------------------------------------------------
# Create fixed 100 m grid (then mask to mainland)
# ------------------------------------------------------------
log.info("Creating 100 m grid")
extent_str = f"{XMIN},{XMAX},{YMIN},{YMAX}"

grid = processing.run(
    "native:creategrid",
    {
        "TYPE": 2,  # Rectangle (polygon)
        "EXTENT": extent_str,
        "HSPACING": GRID_SPACING,
        "VSPACING": GRID_SPACING,
        "HOVERLAY": 0,
        "VOVERLAY": 0,
        "CRS": QgsCoordinateReferenceSystem(f"EPSG:{TARGET_EPSG}"),
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

log.info("Clipping grid to mainland")
grid = processing.run(
    "native:clip",
    {"INPUT": grid, "OVERLAY": mask, "OUTPUT": "memory:"},
    feedback=feedback
)["OUTPUT"]

log.info(f"Grid cells after masking: {grid.featureCount():,}")

grid = processing.run(
    "native:fieldcalculator",
    {
        "INPUT": grid,
        "FIELD_NAME": "cell_id",
        "FIELD_TYPE": 1,  # Integer
        "FIELD_LENGTH": 12,
        "FIELD_PRECISION": 0,
        "FORMULA": "$id",
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

# ------------------------------------------------------------
# Intersect + sum areas per cell
# ------------------------------------------------------------
roads_sum = sum_area_per_cell(roads_poly, grid, "roads")
bld_sum = sum_area_per_cell(bld, grid, "buildings")

# Rename "sum" to stable names
roads_sum = processing.run(
    "native:renametablefield",
    {"INPUT": roads_sum, "FIELD": "sum", "NEW_NAME": "road_area", "OUTPUT": "memory:"},
    feedback=feedback
)["OUTPUT"]

bld_sum = processing.run(
    "native:renametablefield",
    {"INPUT": bld_sum, "FIELD": "sum", "NEW_NAME": "building_area", "OUTPUT": "memory:"},
    feedback=feedback
)["OUTPUT"]

# ------------------------------------------------------------
# Join + calculate impervious proportion
# ------------------------------------------------------------
log.info("Joining road areas to grid")
grid = processing.run(
    "native:joinattributestable",
    {
        "INPUT": grid,
        "FIELD": "cell_id",
        "INPUT_2": roads_sum,
        "FIELD_2": "cell_id",
        "FIELDS_TO_COPY": ["road_area"],
        "METHOD": 1,
        "DISCARD_NONMATCHING": False,
        "PREFIX": "",
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

log.info("Joining building areas to grid")
grid = processing.run(
    "native:joinattributestable",
    {
        "INPUT": grid,
        "FIELD": "cell_id",
        "INPUT_2": bld_sum,
        "FIELD_2": "cell_id",
        "FIELDS_TO_COPY": ["building_area"],
        "METHOD": 1,
        "DISCARD_NONMATCHING": False,
        "PREFIX": "",
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

log.info("Calculating impervious proportion per cell")
grid = processing.run(
    "native:fieldcalculator",
    {
        "INPUT": grid,
        "FIELD_NAME": "imperv_prop",
        "FIELD_TYPE": 0,
        "FIELD_LENGTH": 20,
        "FIELD_PRECISION": 6,
        "FORMULA": 'least( (coalesce("road_area",0) + coalesce("building_area",0)) / $area, 1.0)',
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

# ------------------------------------------------------------
# Rasterise
# ------------------------------------------------------------
log.info("Rasterising impervious proportion to 100 m raster")
raster_extent = f"{XMIN},{XMAX},{YMIN},{YMAX}"

processing.run(
    "gdal:rasterize",
    {
        "INPUT": grid,
        "FIELD": "imperv_prop",
        "BURN": 0,
        "UNITS": 1,  # Georeferenced units
        "WIDTH": GRID_SPACING,
        "HEIGHT": GRID_SPACING,
        "EXTENT": raster_extent,
        "NODATA": DST_NODATA,
        "INIT": DST_NODATA,
        "DATA_TYPE": 5,  # Float32
        "OPTIONS": "COMPRESS=DEFLATE|PREDICTOR=2|TILED=YES",
        "OUTPUT": OUTPUT_RASTER
    },
    feedback=feedback
)

log.info(f"Raster written to: {OUTPUT_RASTER}")
log.info(f"Total runtime: {time.time() - start_total:.1f} seconds")

# Clean shutdown
qgs.exitQgis()

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()
feedback = QgsProcessingFeedback()

# ============================================================
# INPUTS
# ============================================================

# Use your pre-built road polygons (no buffering step)
# NOTE: you had ".gpkg.gpkg" in the path. Fixed to ".gpkg".
ROADS_POLY_URI = r"D:\Data\LINZ\lds-nz-road-centrelines-topo-150k\road_polygons.gpkg|layername=roads_sa1"

BUILDINGS_URI = r"D:\Data\LINZ\lds-nz-building-outlines-FGDB\buildings_fixed_geometry.gpkg|layername=fixed_geometries"
MASK_URI = r"D:\Data\LINZ\lds-nz-coastlines-and-islands-polygons-topo-150k\Mainland.shp"

TARGET_EPSG = 2193
GRID_SPACING = 100.0

# Fixed raster extent (NZTM)
XMIN = 715100.0
YMIN = 3728500.0
XMAX = 2893500.0
YMAX = 7142400.0

OUTPUT_RASTER = r"<PROJECT_DIRECTORY>\Regulating\Local climate\impervious_percent.tif"

DST_NODATA = -9999.0

# ============================================================
# HELPERS
# ============================================================

def load_layer(uri: str, name: str) -> QgsVectorLayer:
    log.info(f"Loading {name}")
    lyr = QgsVectorLayer(uri, name, "ogr")
    if not lyr.isValid():
        raise RuntimeError(f"Failed to load {name}\nURI: {uri}")
    return lyr


def assert_crs(layer: QgsVectorLayer, name: str):
    crs = layer.crs()
    if crs.authid() != f"EPSG:{TARGET_EPSG}":
        raise RuntimeError(f"{name} CRS must be EPSG:{TARGET_EPSG}, got {crs.authid()}")
    log.info(f"{name} CRS OK ({crs.authid()})")


def sum_area_per_cell(input_layer: QgsVectorLayer, grid_layer: QgsVectorLayer, label: str) -> QgsVectorLayer:
    log.info(f"Intersecting {label} with grid")
    t0 = time.time()

    inter = processing.run(
        "native:intersection",
        {
            "INPUT": input_layer,
            "OVERLAY": grid_layer,
            "INPUT_FIELDS": [],
            "OVERLAY_FIELDS": ["cell_id"],
            "OVERLAY_FIELDS_PREFIX": "",
            "OUTPUT": "memory:"
        },
        feedback=feedback
    )["OUTPUT"]

    log.info(f"Calculating area parts for {label}")
    inter = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": inter,
            "FIELD_NAME": "part_area",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 20,
            "FIELD_PRECISION": 3,
            "FORMULA": "$area",
            "OUTPUT": "memory:"
        },
        feedback=feedback
    )["OUTPUT"]

    log.info(f"Aggregating {label} area by cell")
    stats = processing.run(
        "native:statisticsbycategories",
        {
            "INPUT": inter,
            "CATEGORIES_FIELD_NAME": ["cell_id"],
            "VALUES_FIELD_NAME": "part_area",
            "STATISTICS": [1],  # 1 = sum
            "OUTPUT": "memory:"
        },
        feedback=feedback
    )["OUTPUT"]

    log.info(f"{label} aggregation complete in {time.time() - t0:.1f} sec")
    return stats


# ============================================================
# MAIN
# ============================================================

start_total = time.time()
log.info("Starting imperviousness raster (100 m) using road polygons (no buffering).")

roads_poly = load_layer(ROADS_POLY_URI, "Road polygons")
bld = load_layer(BUILDINGS_URI, "Buildings")
mask = load_layer(MASK_URI, "Mainland mask")

assert_crs(roads_poly, "Road polygons")
assert_crs(bld, "Buildings")
assert_crs(mask, "Mask")

# ------------------------------------------------------------
# Create fixed 100 m grid (then mask to mainland)
# ------------------------------------------------------------
log.info("Creating 100 m grid")
extent_str = f"{XMIN},{XMAX},{YMIN},{YMAX}"

grid = processing.run(
    "native:creategrid",
    {
        "TYPE": 2,  # Rectangle (polygon)
        "EXTENT": extent_str,
        "HSPACING": GRID_SPACING,
        "VSPACING": GRID_SPACING,
        "HOVERLAY": 0,
        "VOVERLAY": 0,
        "CRS": QgsCoordinateReferenceSystem(f"EPSG:{TARGET_EPSG}"),
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

log.info("Clipping grid to mainland")
grid = processing.run(
    "native:clip",
    {"INPUT": grid, "OVERLAY": mask, "OUTPUT": "memory:"},
    feedback=feedback
)["OUTPUT"]

log.info(f"Grid cells after masking: {grid.featureCount():,}")

grid = processing.run(
    "native:fieldcalculator",
    {
        "INPUT": grid,
        "FIELD_NAME": "cell_id",
        "FIELD_TYPE": 1,  # Integer
        "FIELD_LENGTH": 12,
        "FIELD_PRECISION": 0,
        "FORMULA": "$id",
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

# ------------------------------------------------------------
# Intersect + sum areas per cell
# ------------------------------------------------------------
roads_sum = sum_area_per_cell(roads_poly, grid, "roads")
bld_sum = sum_area_per_cell(bld, grid, "buildings")

# Rename "sum" to stable names
roads_sum = processing.run(
    "native:renametablefield",
    {"INPUT": roads_sum, "FIELD": "sum", "NEW_NAME": "road_area", "OUTPUT": "memory:"},
    feedback=feedback
)["OUTPUT"]

bld_sum = processing.run(
    "native:renametablefield",
    {"INPUT": bld_sum, "FIELD": "sum", "NEW_NAME": "building_area", "OUTPUT": "memory:"},
    feedback=feedback
)["OUTPUT"]

# ------------------------------------------------------------
# Join + calculate impervious proportion
# ------------------------------------------------------------
log.info("Joining road areas to grid")
grid = processing.run(
    "native:joinattributestable",
    {
        "INPUT": grid,
        "FIELD": "cell_id",
        "INPUT_2": roads_sum,
        "FIELD_2": "cell_id",
        "FIELDS_TO_COPY": ["road_area"],
        "METHOD": 1,
        "DISCARD_NONMATCHING": False,
        "PREFIX": "",
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

log.info("Joining building areas to grid")
grid = processing.run(
    "native:joinattributestable",
    {
        "INPUT": grid,
        "FIELD": "cell_id",
        "INPUT_2": bld_sum,
        "FIELD_2": "cell_id",
        "FIELDS_TO_COPY": ["building_area"],
        "METHOD": 1,
        "DISCARD_NONMATCHING": False,
        "PREFIX": "",
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

log.info("Calculating impervious proportion per cell")
grid = processing.run(
    "native:fieldcalculator",
    {
        "INPUT": grid,
        "FIELD_NAME": "imperv_prop",
        "FIELD_TYPE": 0,
        "FIELD_LENGTH": 20,
        "FIELD_PRECISION": 6,
        "FORMULA": 'least( (coalesce("road_area",0) + coalesce("building_area",0)) / $area, 1.0)',
        "OUTPUT": "memory:"
    },
    feedback=feedback
)["OUTPUT"]

# ------------------------------------------------------------
# Rasterise
# ------------------------------------------------------------
log.info("Rasterising impervious proportion to 100 m raster")
raster_extent = f"{XMIN},{XMAX},{YMIN},{YMAX}"

processing.run(
    "native:rasterize",
    {
        "INPUT": grid,
        "FIELD": "imperv_prop",
        "BURN": 0,
        "UNITS": 1,  # Georeferenced units
        "WIDTH": GRID_SPACING,
        "HEIGHT": GRID_SPACING,
        "EXTENT": f"{XMIN},{XMAX},{YMIN},{YMAX}",
        "NODATA": DST_NODATA,
        "DATA_TYPE": 5,  # Float32
        "INIT": DST_NODATA,
        "OUTPUT": OUTPUT_RASTER
    },
    feedback=feedback
)

log.info(f"Raster written to: {OUTPUT_RASTER}")
log.info(f"Total runtime: {time.time() - start_total:.1f} seconds")

# Clean shutdown
qgs.exitQgis()