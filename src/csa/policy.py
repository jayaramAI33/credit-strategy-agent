"""Current policy (champion) from config/strategy.yaml, applied to raw values."""
from __future__ import annotations

import pandas as pd

from . import config

_SAFE = {">": lambda x, v: x <= v, ">=": lambda x, v: x < v,
         "<": lambda x, v: x >= v, "<=": lambda x, v: x > v}


def approve(df: pd.DataFrame, policy=None) -> pd.Series:
    """True = approved. Missing values on a knockout field decline (NaN fails the safe test)."""
    policy = policy or config.load("strategy")["current_policy"]
    ok = pd.Series(True, index=df.index)
    for ko in policy["knockouts"]:
        ok &= _SAFE[ko["op"]](df[ko["field"]], ko["value"]).fillna(False).astype(bool)
    grade_ok = pd.Series(False, index=df.index)
    for g, rule in policy["grade_rules"].items():
        in_g = df["grade"] == g
        if rule == "approve":
            grade_ok |= in_g
        elif isinstance(rule, (dict,)) or hasattr(rule, "keys"):
            grade_ok |= in_g & (df["fico_mid"] >= rule["min_fico"])
    return ok & grade_ok


def baseline(df: pd.DataFrame, approved: pd.Series, by: str) -> pd.DataFrame:
    d = df.assign(approved=approved)
    rows = []
    for key, g in d.groupby(by, sort=True):
        a, dcl = g[g.approved], g[~g.approved]
        rows.append({by: key, "loans": len(g), "approval_pct": 100 * len(a) / len(g),
                     "approved_bad_pct": 100 * a.bad.mean(),
                     "declined_bad_pct": 100 * dcl.bad.mean() if len(dcl) else float("nan"),
                     "all_bad_pct": 100 * g.bad.mean()})
    return pd.DataFrame(rows)
