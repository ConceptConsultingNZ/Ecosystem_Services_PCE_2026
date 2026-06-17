import os
from datetime import datetime
import numpy as np
import rasterio


def safe_output_path(output_folder, output_filename):
    """
    Return a writable output path.

    If the requested output file already exists, try to delete it. If deletion
    fails, return a timestamped filename instead.
    """
    output_path = os.path.join(output_folder, output_filename)

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            base, ext = os.path.splitext(output_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_folder, f"{base}_{timestamp}{ext}")

    return output_path


def write_float32_raster(array, output_folder, output_filename, profile, nodata=-9999.0):
    """
    Write a single-band Float32 GeoTIFF with conservative compression settings.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = safe_output_path(output_folder, output_filename)

    output_profile = profile.copy()
    output_profile.update(
        dtype="float32",
        nodata=nodata,
        compress="DEFLATE",
        predictor=3,
        zlevel=9,
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(array.astype(np.float32), 1)

    return output_path