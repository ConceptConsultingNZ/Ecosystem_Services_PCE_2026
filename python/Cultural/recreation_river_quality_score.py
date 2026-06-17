#!/usr/bin/env python3
"""
General freshwater recreation quality score (weighted geometric mean)
- Ingest RiverMaps data
- Normalise: E. coli G260 (exceedance), Visual clarity median, CHLA 92nd percentile
- Output: nzsegment + quality_score to CSV
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ----------------------------
# Filepaths (edit these)
# ----------------------------
FILEPATH_IN = r"D:\Data\NIWA\NZRiverMaps\NZRiverMaps_data_2023-12-03.csv"
FILEPATH_OUT = r"<PROJECT_DIRECTORY>\Cultural\Recreation\Freshwater\nzsegment_water_quality.csv"

# ----------------------------
# Column names (edit if needed)
# ----------------------------
COL_NZSEGMENT = "nzsegment"
COL_ECOLI_G260 = "E. coli G260"              # expected as exceedance proportion (0-1) OR percent (0-100)
COL_CLARITY_MEDIAN = "Visual clarity median" # higher = better
COL_CHLA_92 = "CHLA 92%"                     # higher = worse (nuisance/blooms); in-river periphyton proxy

# ----------------------------
# Normalisation constants
# Map each metric into an acceptability score in [0,1]
# ----------------------------

# Floors/ceilings to keep logs stable and avoid exact 0 or 1
SCORE_FLOOR = 0.05
SCORE_CEIL = 0.95

# Visual clarity (median) thresholds in metres
# Below CLARITY_BAD => 0, above CLARITY_GOOD => 1 (linear between)
CLARITY_BAD = 0.6
CLARITY_GOOD = 2.0

# CHLA (92nd percentile)
# Below CHLA_GOOD => 1, above CHLA_BAD => 0 (linear between, inverted)
CHLA_GOOD = 50.0
CHLA_BAD = 200.0

# ----------------------------
# Weights (should sum to 1.0)
# ----------------------------
W_ECOLI = 0.45
W_CLARITY = 0.35
W_CHLA = 0.20


def read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input file type: {ext}. Use .csv, .parquet, or .xlsx")


def clamp01(x: pd.Series, lo: float = SCORE_FLOOR, hi: float = SCORE_CEIL) -> pd.Series:
    return x.clip(lower=lo, upper=hi)


def linear_score_higher_is_better(x: pd.Series, bad: float, good: float) -> pd.Series:
    """0 at/below bad; 1 at/above good; linear between."""
    s = (x - bad) / (good - bad)
    return s.clip(lower=0.0, upper=1.0)


def linear_score_lower_is_better(x: pd.Series, good: float, bad: float) -> pd.Series:
    """1 at/below good; 0 at/above bad; linear between."""
    s = (bad - x) / (bad - good)
    return s.clip(lower=0.0, upper=1.0)


def normalise_g260_exceedance(x: pd.Series) -> pd.Series:
    """
    Convert G260 exceedance proportion (0–1) into acceptability score.
    Assumes:
        x = proportion of samples exceeding 260 (0 = never exceeds, 1 = always exceeds)

    Acceptability score:
        1 - exceedance
    """
    x_num = pd.to_numeric(x, errors="coerce")

    # Ensure valid bounds
    p = x_num.clip(lower=0.0, upper=1.0)

    # Higher exceedance = worse; invert
    s = 1.0 - p

    return s.clip(lower=0.0, upper=1.0)


def weighted_geometric_mean(scores: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """
    Weighted geometric mean:
      exp( sum_i w_i * ln(clamp(score_i)) )
    """
    # Normalise weights in case they don't sum perfectly to 1
    wsum = float(sum(weights.values()))
    if not np.isclose(wsum, 1.0):
        weights = {k: v / wsum for k, v in weights.items()}

    log_sum = None
    for k, s in scores.items():
        w = weights[k]
        s_clamped = clamp01(s.astype(float))
        term = w * np.log(s_clamped)
        log_sum = term if log_sum is None else (log_sum + term)

    return np.exp(log_sum)


def main() -> None:
    df = read_any_table(FILEPATH_IN)

    # Basic checks
    missing_cols = [c for c in [COL_NZSEGMENT, COL_ECOLI_G260, COL_CLARITY_MEDIAN, COL_CHLA_92] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}\nAvailable columns: {list(df.columns)}")

    # Normalise to acceptability scores
    s_ecoli = normalise_g260_exceedance(df[COL_ECOLI_G260])

    clarity_raw = pd.to_numeric(df[COL_CLARITY_MEDIAN], errors="coerce")
    s_clarity = linear_score_higher_is_better(clarity_raw, bad=CLARITY_BAD, good=CLARITY_GOOD)

    chla_raw = pd.to_numeric(df[COL_CHLA_92], errors="coerce")
    s_chla = linear_score_lower_is_better(chla_raw, good=CHLA_GOOD, bad=CHLA_BAD)

    # Combine via weighted geometric mean
    weights = {"ecoli": W_ECOLI, "clarity": W_CLARITY, "chla": W_CHLA}
    scores = {"ecoli": s_ecoli, "clarity": s_clarity, "chla": s_chla}

    quality_score = weighted_geometric_mean(scores=scores, weights=weights)

    out = pd.DataFrame(
        {
            COL_NZSEGMENT: df[COL_NZSEGMENT],
            "ecoli_g260_raw": df[COL_ECOLI_G260],
            "clarity_median_raw": df[COL_CLARITY_MEDIAN],
            "chla_92_raw": df[COL_CHLA_92],
            "quality_score": quality_score,
        }
    )

    # Write output
    out.to_csv(FILEPATH_OUT, index=False)
    print(f"Saved {len(out):,} rows to: {FILEPATH_OUT}")


if __name__ == "__main__":
    main()