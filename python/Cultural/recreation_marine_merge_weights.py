"""
Re-weight a gravity / attraction model raster using aerial survey density
inside the survey footprint, while preserving the overall total.

Survey raster has nodata = -9999 outside survey area.
"""

import numpy as np
import rasterio

# ==============================
# FILE PATH CONSTANTS
# ==============================

MODEL_RASTER_PATH = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Marine_boating\recreation_marine_boating_flow_value.tif"
SURVEY_RASTER_PATH =  r"<PROJECT_DIRECTORY>\Cultural\Recreation\Marine_boating\vessel_density.tif"
OUTPUT_RASTER_PATH = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Marine_boating\recreation_marine_boating_flow_value_reweighted.tif"

SURVEY_NODATA = -9999


def assert_same_grid(src_a, src_b):
    if (
        src_a.crs != src_b.crs
        or src_a.transform != src_b.transform
        or src_a.width != src_b.width
        or src_a.height != src_b.height
    ):
        raise ValueError(
            "Model and survey rasters are not aligned "
            "(CRS/transform/shape differ)."
        )


def main():
    with rasterio.open(MODEL_RASTER_PATH) as msrc, \
         rasterio.open(SURVEY_RASTER_PATH) as ssrc:

        assert_same_grid(msrc, ssrc)

        model = msrc.read(1).astype(np.float64)
        survey = ssrc.read(1).astype(np.float64)

        # Valid masks
        model_nodata = msrc.nodata
        model_valid = np.isfinite(model)
        if model_nodata is not None:
            model_valid &= (model != model_nodata)

        survey_valid = np.isfinite(survey) & (survey != SURVEY_NODATA)

        overlap = model_valid & survey_valid

        if not np.any(overlap):
            raise ValueError("No valid overlapping cells between model and survey.")

        model_sum_inside = np.sum(model[overlap])
        survey_sum_inside = np.sum(survey[overlap])

        if survey_sum_inside == 0:
            raise ValueError("Survey sum inside overlap is zero; cannot scale.")

        # Scale survey to match model mass inside survey footprint
        scale = model_sum_inside / survey_sum_inside
        survey_scaled = survey * scale

        # Replace model values inside survey area
        output = model.copy()
        output[overlap] = survey_scaled[overlap]

        # Preserve model nodata
        if model_nodata is not None:
            output[~model_valid] = model_nodata

        # Write output
        profile = msrc.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER"
        )

        with rasterio.open(OUTPUT_RASTER_PATH, "w", **profile) as dst:
            dst.write(output.astype(np.float32), 1)

        # Diagnostics
        total_model = np.sum(model[model_valid])
        total_output = np.sum(output[model_valid])

        print(f"Scale factor applied: {scale:.6f}")
        print(f"Original total: {total_model:.6f}")
        print(f"New total: {total_output:.6f}")
        print(f"Difference: {total_output - total_model:.6f}")
        print("Done.")


if __name__ == "__main__":
    main()