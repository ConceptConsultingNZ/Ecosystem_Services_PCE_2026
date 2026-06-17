"""
Compute per-pixel nutrient abatement cost from N & P load/retained rasters.

Logic (per pixel):
1) Compute reduction fractions:
   rN = retained_N / load_N
   rP = retained_P / load_P

2) Two engineered-abatement strategies:
   A) Optimise for N reduction:
      - Pay to remove retained_N at $/kgN(rN)
      - This also removes some P as a co-benefit: bonusP_frac(rN) * load_P
      - Remaining P to remove = max(0, retained_P - bonusP_amount)
      - Pay to remove remaining P at $/kgP(remainingP / load_P)

   B) Optimise for P reduction:
      - Pay to remove retained_P at $/kgP(rP)
      - This also removes some N as a co-benefit: bonusN_frac(rP) * load_N
      - Remaining N to remove = max(0, retained_N - bonusN_amount)
      - Pay to remove remaining N at $/kgN(remainingN / load_N)

3) Select the minimum cost of A and B and write to output raster.

Notes:
- Uses linear interpolation across the provided curve table.
- Handles divide-by-zero and nodata robustly.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import rasterio


# -----------------------------
# User-defined constants
# -----------------------------
WORKING_DIR = r"<PROJECT_DIRECTORY>\Regulating\Nutrient regulation\Intermediate"
N_LOAD_RASTER = os.path.join(WORKING_DIR,"load_n.tif")
P_LOAD_RASTER = os.path.join(WORKING_DIR,"load_p.tif")
N_RETAIN_RASTER = os.path.join(WORKING_DIR,"retained_n.tif")
P_RETAIN_RASTER = os.path.join(WORKING_DIR,"retained_p.tif")

COST_CURVE_CSV = os.path.join(WORKING_DIR,"marginal_costs.csv")  # Must contain columns shown below
COST_LEVEL = "high"  # one of: "low", "mid", "high"

OUTPUT_COST_RASTER = r"<PROJECT_DIRECTORY>\Regulating\Nutrient regulation\Output\Nutrient_regulation_value_high.tif"

# Optional: clamp reduction fractions to avoid weirdness for tiny denominators
CLAMP_MIN = 0.0
CLAMP_MAX = 0.999999

# -----------------------------
# Helpers
# -----------------------------
def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise num/den with 0 where den<=0 or num<0."""
    out = np.zeros_like(num, dtype=np.float32)
    m = (den > 0) & (num >= 0)
    out[m] = (num[m] / den[m]).astype(np.float32)
    return out


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi).astype(np.float32)


def _interp_from_curve(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Vectorised 1D linear interpolation for array x given tabulated xp, fp.
    Uses endpoint values outside range.
    """
    # np.interp is vectorised over x, but xp/fp must be 1D increasing
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1]).astype(np.float32)


def _load_cost_curve(csv_path: str, level: str) -> dict[str, np.ndarray]:
    """
    Load curve table and return arrays for interpolation.
    Expected columns:
      Target_reduction
      Bonus_P_reduction
      N_mid, N_low, N_high
      Bonus_N_reduction
      P_mid, P_low, P_high
    """
    df = pd.read_csv(csv_path)
    required = {
        "Target_reduction",
        "Bonus_P_reduction",
        "N_mid", "N_low", "N_high",
        "Bonus_N_reduction",
        "P_mid", "P_low", "P_high",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cost curve CSV missing columns: {sorted(missing)}")

    df = df.sort_values("Target_reduction")
    xp = df["Target_reduction"].to_numpy(dtype=np.float32)

    if level.lower() == "mid":
        n_cost = df["N_mid"].to_numpy(dtype=np.float32)
        p_cost = df["P_mid"].to_numpy(dtype=np.float32)
    elif level.lower() == "low":
        n_cost = df["N_low"].to_numpy(dtype=np.float32)
        p_cost = df["P_low"].to_numpy(dtype=np.float32)
    elif level.lower() == "high":
        n_cost = df["N_high"].to_numpy(dtype=np.float32)
        p_cost = df["P_high"].to_numpy(dtype=np.float32)
    else:
        raise ValueError("COST_LEVEL must be one of: 'low', 'mid', 'high'")

    bonus_p = df["Bonus_P_reduction"].to_numpy(dtype=np.float32)  # P co-benefit when optimising for N
    bonus_n = df["Bonus_N_reduction"].to_numpy(dtype=np.float32)  # N co-benefit when optimising for P

    return {
        "xp": xp,
        "n_cost": n_cost,
        "p_cost": p_cost,
        "bonus_p": bonus_p,
        "bonus_n": bonus_n,
    }


# -----------------------------
# Main computation
# -----------------------------
def main() -> None:
    curve = _load_cost_curve(COST_CURVE_CSV, COST_LEVEL)
    xp = curve["xp"]

    with rasterio.open(N_LOAD_RASTER) as nL, \
         rasterio.open(P_LOAD_RASTER) as pL, \
         rasterio.open(N_RETAIN_RASTER) as nR, \
         rasterio.open(P_RETAIN_RASTER) as pR:

        # Basic compatibility checks
        if (nL.width, nL.height) != (pL.width, pL.height) or (nL.transform != pL.transform):
            raise ValueError("N load and P load rasters do not align (shape/transform mismatch).")
        if (nL.width, nL.height) != (nR.width, nR.height) or (nL.transform != nR.transform):
            raise ValueError("N load and N retained rasters do not align (shape/transform mismatch).")
        if (nL.width, nL.height) != (pR.width, pR.height) or (nL.transform != pR.transform):
            raise ValueError("N load and P retained rasters do not align (shape/transform mismatch).")

        profile = nL.profile.copy()
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=np.int16(-9999.0),
            compress="deflate",
            predictor=2,  # safe default for float; GDAL will handle appropriately
            tiled=True,
            blocksizex = 256,
            blocksizey = 256,
            bigtiff="if_safer",
        )

        nodata_out = profile["nodata"]

        with rasterio.open(OUTPUT_COST_RASTER, "w", **profile) as dst:
            for _, window in dst.block_windows(1):
                load_n = nL.read(1, window=window).astype(np.float32)
                load_p = pL.read(1, window=window).astype(np.float32)
                ret_n = nR.read(1, window=window).astype(np.float32)
                ret_p = pR.read(1, window=window).astype(np.float32)

                #Sanity clip - cannot be larger than load
                ret_n = np.clip(ret_n, 0, load_n).astype(np.float32)
                ret_p = np.clip(ret_p, 0, load_p).astype(np.float32)

                # Treat negative as invalid
                valid = (load_n > 0) & (load_p > 0) & (ret_n >= 0) & (ret_p >= 0)

                # Reduction fractions represented by retention
                rN = _clamp(_safe_div(ret_n, load_n), CLAMP_MIN, CLAMP_MAX)
                rP = _clamp(_safe_div(ret_p, load_p), CLAMP_MIN, CLAMP_MAX)

                # Interpolate marginal costs and co-benefits for those fractions
                # Cost per kg for the nutrient being optimised
                cN_at_rN = _interp_from_curve(rN, xp, curve["n_cost"])
                cP_at_rP = _interp_from_curve(rP, xp, curve["p_cost"])

                # Co-benefit fractions (of the other nutrient's load)
                bonusP_frac_at_rN = _interp_from_curve(rN, xp, curve["bonus_p"])
                bonusN_frac_at_rP = _interp_from_curve(rP, xp, curve["bonus_n"])

                # -------------------------
                # Strategy A: optimise for N
                # -------------------------
                cost_A = ret_n * cN_at_rN  # $/kgN * kgN

                bonusP_amt = bonusP_frac_at_rN * load_p  # kg P removed as co-benefit
                remP = np.maximum(0.0, ret_p - bonusP_amt).astype(np.float32)
                remP_frac = _clamp(_safe_div(remP, load_p), CLAMP_MIN, CLAMP_MAX)
                cP_at_remP = _interp_from_curve(remP_frac, xp, curve["p_cost"])
                cost_A = cost_A + remP * cP_at_remP

                # -------------------------
                # Strategy B: optimise for P
                # -------------------------
                cost_B = ret_p * cP_at_rP  # $/kgP * kgP

                bonusN_amt = bonusN_frac_at_rP * load_n  # kg N removed as co-benefit
                remN = np.maximum(0.0, ret_n - bonusN_amt).astype(np.float32)
                remN_frac = _clamp(_safe_div(remN, load_n), CLAMP_MIN, CLAMP_MAX)
                cN_at_remN = _interp_from_curve(remN_frac, xp, curve["n_cost"])
                cost_B = cost_B + remN * cN_at_remN

                # Choose minimum cost
                out = np.minimum(cost_A, cost_B).astype(np.float32)

                #Check for extreme values
                if np.any(out > 9999):
                    i, j = np.argwhere(out > 9999)[0]
                    print(f"High cost >9999 at pixel ({i},{j}): "
                          f"cost={out[i, j]}, ret_n={ret_n[i, j]}, ret_p={ret_p[i, j]}, "
                          f"load_n={load_n[i, j]}, load_p={load_p[i, j]}, "
                          f"rN={rN[i, j]}, rP={rP[i, j]}, "
                          f"cN={cN_at_rN[i, j]}, cP={cP_at_rP[i, j]}")
                    sys.exit("Stopping due to extreme cost value.")

                # Apply nodata
                out_masked = np.where(valid, out, nodata_out).astype(np.float32)

                dst.write(out_masked, 1, window=window)

    print("Abatement cost raster written successfully.")
    print(f"Output: {OUTPUT_COST_RASTER}")


if __name__ == "__main__":
    main()