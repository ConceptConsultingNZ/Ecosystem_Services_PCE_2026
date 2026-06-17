#Reprojects to EPSG:2193 100m grid.
#Note this needs to be run in an environment with GDAL, such as OSGeo4W shell that comes packaged with QGIS.

import os
import math
from osgeo import gdal

# -----------------------------
# Inputs
# -----------------------------
DIR1 = r"D:\Data\DOC\New Zealand Seafloor Community Classification\New Zealand Seafloor Community Classification of the Exclusive Economic Zone"
DIR2 = r"D:\Data\DOC\New Zealand Seafloor Community Classification\New Zealand Seafloor Community Classification of the Territorial Sea"
OUTDIR = r"D:\Data\DOC\New Zealand Seafloor Community Classification"

in_raster1 = os.path.join(DIR1, "SCC_GF75_EEZ_1km_Final.tif")     # offshore / deep (1 km)
in_raster2 = os.path.join(DIR2, "SCC_GF75_TS_250m_Final.tif")     # nearshore (250 m)

out_raster = os.path.join(OUTDIR, "SCC_GF75_100m_EPSG2193_Int16.tif")

dst_epsg = 2193
target_res = 100
final_nodata = -9999

# Prefer TS (nearshore) where they overlap
# (TS will be applied second in the VRT build list)
PREFER_TS_OVER_EEZ = True

# -----------------------------
# Helpers
# -----------------------------
def get_extent(ds):
    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + xsize * gt[1]
    ymin = ymax + ysize * gt[5]
    return xmin, ymin, xmax, ymax

def round_extent_to_grid(xmin, ymin, xmax, ymax, grid):
    rxmin = math.floor(xmin / grid) * grid
    rymin = math.floor(ymin / grid) * grid
    rxmax = math.ceil(xmax / grid) * grid
    rymax = math.ceil(ymax / grid) * grid
    return rxmin, rymin, rxmax, rymax

def open_and_get_nodata(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open: {path}")
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    proj_snip = (ds.GetProjection() or "")[:80]
    ds = None
    return nd, proj_snip

# -----------------------------
# Inspect both sources
# -----------------------------
nd1, proj1 = open_and_get_nodata(in_raster1)
nd2, proj2 = open_and_get_nodata(in_raster2)

print("EEZ nodata:", nd1)
print("TS  nodata:", nd2)
print("EEZ proj starts:", proj1)
print("TS  proj starts:", proj2)

# -----------------------------
# Step 1: Warp each to EPSG:2193 @ 100 m, no forced bounds
# -----------------------------
tmp1 = os.path.join(OUTDIR, "SCC_EEZ_100m_2193_tmp.tif")
tmp2 = os.path.join(OUTDIR, "SCC_TS_100m_2193_tmp.tif")

warp_to_2193 = lambda src_nodata: gdal.WarpOptions(
    dstSRS=f"EPSG:{dst_epsg}",
    xRes=target_res,
    yRes=target_res,
    targetAlignedPixels=True,  # -tap
    resampleAlg="near",
    srcNodata=src_nodata,
    dstNodata=final_nodata,
    outputType=gdal.GDT_Int16,
    format="GTiff",
    creationOptions=[
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "ZLEVEL=9",
        "TILED=YES",
        "BIGTIFF=IF_SAFER",
    ],
    multithread=True
)

print("Warping EEZ to 2193/100m...")
ds_tmp1 = gdal.Warp(tmp1, in_raster1, options=warp_to_2193(nd1))
if ds_tmp1 is None:
    raise RuntimeError("Warp failed for EEZ.")
ds_tmp1 = None

print("Warping TS to 2193/100m...")
ds_tmp2 = gdal.Warp(tmp2, in_raster2, options=warp_to_2193(nd2))
if ds_tmp2 is None:
    raise RuntimeError("Warp failed for TS.")
ds_tmp2 = None

# -----------------------------
# Step 2: Build a VRT mosaic, preferring TS over EEZ in overlaps
# Order matters: later rasters win for nodata regions when using BuildVRT resolution rules.
# We'll then warp the VRT to a rounded 100 m extent.
# -----------------------------
vrt_path = os.path.join(OUTDIR, "SCC_mosaic_2193_100m_tmp.vrt")

src_list = [tmp1, tmp2] if PREFER_TS_OVER_EEZ else [tmp2, tmp1]

print("Building VRT mosaic...")
vrt_opts = gdal.BuildVRTOptions(
    resampleAlg="near",
    resolution="user",
    xRes=target_res,
    yRes=target_res,
    srcNodata=final_nodata,
    VRTNodata=final_nodata
)
vrt_ds = gdal.BuildVRT(vrt_path, src_list, options=vrt_opts)
if vrt_ds is None:
    raise RuntimeError("BuildVRT failed.")
vrt_ds = None

# -----------------------------
# Step 3: Determine rounded extent from the VRT and write final GeoTIFF
# -----------------------------
vrt_ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
if vrt_ds is None:
    raise RuntimeError("Could not open VRT mosaic.")

mxmin, mymin, mxmax, mymax = get_extent(vrt_ds)
print("Mosaic extent (2193):", mxmin, mymin, mxmax, mymax)

rxmin, rymin, rxmax, rymax = round_extent_to_grid(mxmin, mymin, mxmax, mymax, target_res)
print("Rounded extent (2193):", rxmin, rymin, rxmax, rymax)

vrt_ds = None

print("Warping mosaic VRT to final rounded extent...")
warp_final = gdal.WarpOptions(
    dstSRS=f"EPSG:{dst_epsg}",
    xRes=target_res,
    yRes=target_res,
    targetAlignedPixels=True,
    outputBounds=(rxmin, rymin, rxmax, rymax),
    resampleAlg="near",
    srcNodata=final_nodata,
    dstNodata=final_nodata,
    outputType=gdal.GDT_Int16,
    format="GTiff",
    creationOptions=[
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "ZLEVEL=9",
        "TILED=YES",
        "BIGTIFF=IF_SAFER",
    ],
    multithread=True
)

out_ds = gdal.Warp(out_raster, vrt_path, options=warp_final)
if out_ds is None:
    raise RuntimeError("Final warp failed.")
out_ds = None

print("Wrote:", out_raster)

# Optional cleanup (comment out if you want to inspect intermediates)
# for p in [tmp1, tmp2, vrt_path]:
#     try:
#         os.remove(p)
#     except OSError:
#         pass
