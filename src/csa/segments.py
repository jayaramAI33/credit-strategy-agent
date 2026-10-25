"""Swap-in / swap-out segment catalogue.

A segment = a current-policy group (why a loan is approved or declined today)
            x up to two conditions on binned features (top-5 features + FICO).
Segments are FOUND on build vintages and CONFIRMED on the 2015 out-of-time year.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

# Current-policy groups. Order matters for declined loans: first matching reason wins.
DECLINED_GROUPS = {
    "KO_DTI":     "DTI > 35 (or missing)",
    "KO_LTI":     "loan-to-income > 0.40 (or missing)",
    "C_LOW_FICO": "grade C with FICO < 677",
    "D_LOW_FICO": "grade D with FICO < 697",
    "E":          "grade E",
    "FG":         "grade F or G",
}
APPROVED_GROUPS = {"A": "grade A", "B": "grade B", "C": "grade C with FICO >= 677",
                   "D": "grade D with FICO >= 697"}


def policy_group(df: pd.DataFrame, approved: pd.Series) -> pd.Series:
    g = pd.Series("", index=df.index, dtype=object)
    ko_dti = ~(df["dti"] <= 35).fillna(False)
    ko_lti = ~(df["loan_to_income"] <= 0.40).fillna(False)
    dec = ~approved
    g[dec & ko_dti] = "KO_DTI"
    g[dec & ~ko_dti & ko_lti] = "KO_LTI"
    rest = dec & ~ko_dti & ~ko_lti
    g[rest & (df.grade == "C")] = "C_LOW_FICO"
    g[rest & (df.grade == "D")] = "D_LOW_FICO"
    g[rest & (df.grade == "E")] = "E"
    g[rest & df.grade.isin(["F", "G"])] = "FG"
    for k in APPROVED_GROUPS:
        g[approved & (df.grade == k)] = k
    assert (g != "").all(), "every loan must fall in exactly one policy group"
    return g


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str          # ">" or "<="
    value: float

    def mask(self, x: pd.Series) -> np.ndarray:
        v = x.to_numpy()
        return v > self.value if self.op == ">" else v <= self.value

    def text(self) -> str:
        val = f"{self.value:,.0f}" if abs(self.value) >= 100 else f"{self.value:g}"
        return f"{self.feature} {self.op} {val}"


def tail_conditions(feature: str, spec: dict, side: str, max_bins: int = 2) -> list[Condition]:
    """Conditions selecting the 1 or 2 lowest-risk ('good') or highest-risk ('bad') bins."""
    edges, direction = spec["edges"], spec["direction"]
    if not edges:
        return []
    out = []
    for k in range(1, min(max_bins, len(edges)) + 1):
        high_end = (direction == "down") == (side == "good")   # good side is high values if risk falls
        if high_end:
            out.append(Condition(feature, ">", edges[-k]))
        else:
            out.append(Condition(feature, "<=", edges[k - 1]))
    return out


def candidate_rules(features: list[str], specs: dict, side: str) -> list[tuple[Condition, ...]]:
    singles = {f: tail_conditions(f, specs[f], side) for f in features}
    rules: list[tuple[Condition, ...]] = [()]
    for f in features:
        rules += [(c,) for c in singles[f]]
    for f1, f2 in combinations(features, 2):
        rules += [(a, b) for a in singles[f1] for b in singles[f2]]
    return rules
