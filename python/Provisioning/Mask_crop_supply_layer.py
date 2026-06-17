import rasterio
from rasterio.mask import mask
import fiona

# -------------------------
# Inputs
# -------------------------
in_raster = r"<PROJECT_DIRECTORY>\Provisioning\Crops\crop_supply.tif"
mask_shp = r"D:\Data\LINZ\lds-nz-coastlines-and-islands-polygons-topo-150k-GPKG\Mainland.shp"
out_raster = r"<PROJECT_DIRECTORY>\Provisioning\Crops\crop_supply_masked.tif"

# -------------------------
# Read mask geometry
# -------------------------
with fiona.open(mask_shp, "r") as shp:
    geometries = [feature["geometry"] for feature in shp]

# -------------------------
# Mask raster
# -------------------------
with rasterio.open(in_raster) as src:
    masked_data, masked_transform = mask(
        src,
        geometries,
        crop=False,
        nodata=src.nodata
    )

    out_profile = src.profile.copy()
    out_profile.update(transform=masked_transform)

# -------------------------
# Write output
# -------------------------
with rasterio.open(out_raster, "w", **out_profile) as dst:
    dst.write(masked_data)

print("Masking complete. Apples have been evicted from the harbour.")
