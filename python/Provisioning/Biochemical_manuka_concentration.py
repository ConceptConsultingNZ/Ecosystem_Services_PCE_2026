"""
Calculate proportion of surrounding mānuka within a chosen radius for each mānuka cell

- manuka_extent.tif: 1 = mānuka/kānuka, NoData = -9999
- Non-mānuka cells treated as 0 in neighbourhood calculation
- Output: proportion (0–1, float32) for mānuka cells only; others = NoData
- Default radius suggested 5000 (5 km) because bees forage 3-5km
"""

import numpy as np
import rasterio
from scipy.ndimage import convolve
import time


MANUKA = r"<PROJECT_DIRECTORY>\Provisioning\Biochemical\Intermediate\manuka_extent.tif"
OUT_PROP = r"<PROJECT_DIRECTORY>\Provisioning\Biochemical\Intermediate\manuka_proportion_5km.tif"

NODATA_IN = -9999
NODATA_OUT = -9999

RADIUS_M = 5000


def circular_kernel(radius_cells: int) -> np.ndarray:
    print(f"Creating circular kernel with radius {radius_cells} cells...")
    y, x = np.ogrid[-radius_cells:radius_cells + 1,
                    -radius_cells:radius_cells + 1]
    mask = (x * x + y * y) <= (radius_cells * radius_cells)
    print(f"Kernel size: {mask.shape[0]} x {mask.shape[1]} "
          f"({mask.sum()} cells inside radius)")
    return mask.astype(np.int16)


def main():
    t0 = time.time()
    print("Opening mānuka raster...")

    with rasterio.open(MANUKA) as ds:
        man = ds.read(1)
        nodata = ds.nodata if ds.nodata is not None else NODATA_IN

        print("Raster loaded.")
        print(f"Dimensions: {ds.width} x {ds.height}")
        print("Calculating radius in cells...")

        px = abs(ds.transform.a)
        if px <= 0:
            raise ValueError("Could not determine pixel size.")

        radius_cells = int(round(RADIUS_M / px))
        if radius_cells < 1:
            raise ValueError("Radius too small relative to pixel size.")

        print(f"Pixel size: {px} m")
        print(f"Using radius: {RADIUS_M} m ({radius_cells} cells)")

        print("Building mānuka binary mask...")
        manuka_mask = (man == 1).astype(np.int16)

        print("Building neighbourhood kernel...")
        kern = circular_kernel(radius_cells)

        print("Starting convolution (this may take a while)...")
        t_conv_start = time.time()
        manuka_count = convolve(manuka_mask, kern, mode="constant", cval=0)
        t_conv_end = time.time()
        print(f"Convolution complete in {t_conv_end - t_conv_start:.2f} seconds.")

        total_cells = kern.sum()
        print(f"Total neighbourhood cells: {total_cells}")

        print("Calculating proportion...")
        manuka_prop = manuka_count.astype(np.float32) / float(total_cells)

        print("Applying mānuka mask to output...")
        out = np.full((ds.height, ds.width), NODATA_OUT, dtype=np.float32)
        out[man == 1] = manuka_prop[man == 1]

        print("Writing output raster...")
        profile = ds.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            nodata=NODATA_OUT,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256
        )

        with rasterio.open(OUT_PROP, "w", **profile) as dst:
            dst.write(out, 1)

    t1 = time.time()
    print("Done.")
    print(f"Total processing time: {t1 - t0:.2f} seconds.")
    print("Output written to:", OUT_PROP)


if __name__ == "__main__":
    main()
