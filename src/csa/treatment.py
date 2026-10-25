"""Stage D: missing-value and outlier treatment. Fitted on build, applied to all years.

Treated values feed binning only. The current policy always uses raw values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def fit_apply(df: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config.load("prep")["treatment"]
    build = df["split"] == "build"
    out = pd.DataFrame(index=df.index)
    rows = []
    for var, kind in zip(features["variable"], features["kind"]):
        x = df[var]
        if kind == "categorical":
            out[var] = x.fillna(cfg["categorical_missing"])
            rows.append({"variable": var, "missing_build": int(x[build].isna().sum()),
                         "missing_filled_with": cfg["categorical_missing"],
                         "outlier_method": "n/a (categorical)", "lower_cap": np.nan,
                         "upper_cap": np.nan, "capped_build_pct": 0.0, "capped_oot_pct": 0.0})
            continue
        xb = x[build]
        fill = float(xb.median())
        q1, q3 = xb.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            k = cfg["outlier_iqr_k"]
            lo, hi, method = q1 - k * iqr, q3 + k * iqr, f"{k} x IQR fences"
        else:
            p_lo, p_hi = cfg["outlier_fallback"]
            lo, hi = xb.quantile([p_lo / 100, p_hi / 100])
            method = f"P{p_lo}-P{p_hi} (IQR = 0)"
        lo, hi = max(lo, xb.min()), min(hi, xb.max())
        filled = x.fillna(fill)
        capped = (filled < lo) | (filled > hi)
        out[var] = filled.clip(lo, hi)
        rows.append({"variable": var, "missing_build": int(xb.isna().sum()),
                     "missing_filled_with": round(fill, 4), "outlier_method": method,
                     "lower_cap": round(float(lo), 4), "upper_cap": round(float(hi), 4),
                     "capped_build_pct": round(100 * capped[build].mean(), 2),
                     "capped_oot_pct": round(100 * capped[~build].mean(), 2)})
    return out, pd.DataFrame(rows)


def apply_from_table(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Re-apply a saved treatment table (phase 1 output) without refitting."""
    out = pd.DataFrame(index=df.index)
    for r in table.itertuples(index=False):
        x = df[r.variable]
        if r.outlier_method.startswith("n/a"):
            out[r.variable] = x.fillna(r.missing_filled_with)
        else:
            out[r.variable] = x.fillna(float(r.missing_filled_with)).clip(r.lower_cap, r.upper_cap)
    return out
