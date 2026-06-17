import os
import logging
import pandas as pd
import rasterio
from rasterio.features import rasterize
import fiona
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

LANDUSE_GPKG = r"D:\Data\LRIS\lris-new-zealand-land-use-management-version-03-nzlum-v03-FGDB\NZLUM_subsets.gpkg"
LANDUSE_LAYER = "nzlum_cropping_hort"

CROP_LOOKUP_CSV = r"<PROJECT_DIRECTORY>\Provisioning\Crops\current_production.csv"

OUTDIR = r"<PROJECT_DIRECTORY>\Provisioning\Crops\Output"
os.makedirs(OUTDIR, exist_ok=True)

TEMPLATE_RASTER = os.path.join(OUTDIR, "crop_supply.tif")

OUTPUT_RASTER_CENTRAL = os.path.join(OUTDIR, "crop_value.tif")
OUTPUT_RASTER_LOW = os.path.join(OUTDIR, "crop_value_low.tif")
OUTPUT_RASTER_HIGH = os.path.join(OUTDIR, "crop_value_high.tif")


def get_crop_index(props, commod_to_crop_index):
    commod_1 = props.get("commod_1")

    if commod_1 not in [None, "", "NULL"]:
        commod_1 = str(commod_1).lower().strip()
        return commod_to_crop_index.get(commod_1)

    lu_code_secondary = props.get("lu_code_secondary")

    if lu_code_secondary == 5:
        return 27
    elif lu_code_secondary == 4:
        return 26
    else:
        return 25


def iter_shapes(contribution_field, lookup, commod_to_crop_index):
    total = 0
    matched = 0
    yielded = 0
    non_matches = {}

    with fiona.open(LANDUSE_GPKG, layer=LANDUSE_LAYER) as src:
        for feat in src:
            total += 1

            geom = feat["geometry"]
            props = dict(feat["properties"])

            if geom is None:
                continue

            crop_index = get_crop_index(props, commod_to_crop_index)

            if crop_index is None:
                raw_crop = props.get("commod_1")
                key = f"unmapped commod_1: {raw_crop!r}"
                non_matches[key] = non_matches.get(key, 0) + 1
                continue

            crop_index = str(crop_index).strip()
            crop_data = lookup.get(crop_index)

            if crop_data is None:
                key = f"missing crop_index in lookup: {crop_index!r}"
                non_matches[key] = non_matches.get(key, 0) + 1
                continue

            matched += 1

            crop_value = np.float32(crop_data["crop_value"])
            contribution = np.float32(crop_data[contribution_field])
            value = np.float32(crop_value * contribution)

            if np.isfinite(value):
                yielded += 1
                yield geom, float(value)

    logging.info(
        f"{contribution_field}: total={total:,}, matched={matched:,}, yielded={yielded:,}"
    )

    if non_matches:
        logging.warning("Top unmatched values:")
        for key, count in sorted(non_matches.items(), key=lambda x: -x[1])[:20]:
            logging.warning(f"{key}, count={count:,}")


def rasterise_value(contribution_field, out_raster, lookup, commod_to_crop_index, template_profile, transform, shape):
    logging.info(f"Rasterising {contribution_field} to {out_raster}")

    arr = rasterize(
        shapes=iter_shapes(contribution_field, lookup, commod_to_crop_index),
        out_shape=shape,
        transform=transform,
        fill=np.float32(-9999),
        dtype="float32",
        all_touched=True
    )

    profile = template_profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=np.float32(-9999),
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(out_raster, "w", **profile) as dst:
        dst.write(arr, 1)

    logging.info(f"Saved {out_raster}")


def main():
    logging.info("Reading crop lookup CSV")
    crop_lookup = pd.read_csv(CROP_LOOKUP_CSV)

    required_cols = {
        "crop_index",
        "commod",
        "crop_value",
        "contribution_central",
        "contribution_low",
        "contribution_high"
    }

    missing_cols = required_cols - set(crop_lookup.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in lookup CSV: {missing_cols}")

    crop_lookup["crop_index"] = crop_lookup["crop_index"].astype(str).str.strip()
    crop_lookup["commod"] = crop_lookup["commod"].astype(str).str.lower().str.strip()

    float_cols = [
        "crop_value",
        "contribution_central",
        "contribution_low",
        "contribution_high"
    ]

    crop_lookup[float_cols] = crop_lookup[float_cols].replace(",", "", regex=True)
    crop_lookup[float_cols] = crop_lookup[float_cols].apply(pd.to_numeric, errors="coerce").astype("float32")

    lookup = crop_lookup.set_index("crop_index").to_dict("index")
    commod_to_crop_index = crop_lookup.set_index("commod")["crop_index"].to_dict()

    logging.info("Reading template raster")
    with rasterio.open(TEMPLATE_RASTER) as template:
        profile = template.profile
        transform = template.transform
        shape = (template.height, template.width)

    rasterise_value(
        "contribution_central",
        OUTPUT_RASTER_CENTRAL,
        lookup,
        commod_to_crop_index,
        profile,
        transform,
        shape
    )

    rasterise_value(
        "contribution_low",
        OUTPUT_RASTER_LOW,
        lookup,
        commod_to_crop_index,
        profile,
        transform,
        shape
    )

    rasterise_value(
        "contribution_high",
        OUTPUT_RASTER_HIGH,
        lookup,
        commod_to_crop_index,
        profile,
        transform,
        shape
    )

    logging.info("Crop value rasters created successfully.")


if __name__ == "__main__":
    main()