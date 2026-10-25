"""Stage C: variable screening with the user's three rules, on build years only.

Rule 1  missing values: reject if >= max_missing_pct
Rule 2  categorical levels: keep only if min < levels < max (exclusive)
Rule 3  single-value concentration: reject if one value covers > max_top_value_share_pct
The first failing rule is logged as the reason.
"""
from __future__ import annotations

import pandas as pd

from . import config


def _profile(s: pd.Series) -> dict:
    n = len(s)
    nonmiss = s.dropna()
    vc = nonmiss.value_counts()
    return {
        "missing_pct": 100 * (n - len(nonmiss)) / n,
        "levels": int(nonmiss.nunique()),
        "top_value": vc.index[0] if len(vc) else None,
        "top_share_pct": 100 * vc.iloc[0] / len(nonmiss) if len(vc) else 100.0,
    }


def _verdict(p: dict, kind: str, rules) -> tuple[str, str]:
    if p["missing_pct"] >= rules["max_missing_pct"]:
        return "rejected", f"R1 missing {p['missing_pct']:.2f}% >= {rules['max_missing_pct']}%"
    if kind == "categorical" and not (
            rules["cat_levels_min_exclusive"] < p["levels"] < rules["cat_levels_max_exclusive"]):
        return "rejected", (f"R2 {p['levels']} levels, needs > {rules['cat_levels_min_exclusive']}"
                            f" and < {rules['cat_levels_max_exclusive']}")
    if kind != "date" and p["top_share_pct"] > rules["max_top_value_share_pct"]:
        return "rejected", (f"R3 one value ({p['top_value']}) covers {p['top_share_pct']:.1f}%"
                            f" > {rules['max_top_value_share_pct']}%")
    return "kept", ""


def screen(build: pd.DataFrame) -> pd.DataFrame:
    rules = config.load("prep")["screening"]
    fields = config.load("fields")
    rows = []
    for kind, cols in config.candidate_fields().items():
        for col in cols:
            p = _profile(build[col])
            status, reason = _verdict(p, kind, rules)
            note = "date: used only through derived features" if kind == "date" else ""
            rows.append({"variable": col, "kind": kind, "source": "raw", **p,
                         "status": status, "reason": reason, "note": note})
    log = pd.DataFrame(rows)

    # Derived features: same rules; sources must be kept unless explicitly allowed.
    kept = set(log.loc[log.status == "kept", "variable"]) | set(config.role_fields())
    kept |= {"issue_d"}
    for name, spec in fields["derived"].items():
        allowed = dict(spec.get("allowed_rejected_sources", {}))
        bad_src = [s for s in spec["from"] if s not in kept and s not in allowed]
        p = _profile(build[name])
        if bad_src:
            status, reason = "rejected", f"source rejected: {', '.join(bad_src)}"
        else:
            status, reason = _verdict(p, "numeric", rules)
        note = "; ".join(f"uses rejected {k}: {v}" for k, v in allowed.items())
        rows.append({"variable": name, "kind": "numeric", "source": "derived", **p,
                     "status": status, "reason": reason, "note": note})
        for r in spec.get("replaces", []):
            for row in rows:
                if row["variable"] == r and row["status"] == "kept":
                    row["status"], row["note"] = "replaced", f"replaced by derived {name}"
    log = pd.DataFrame(rows)
    log["top_value"] = log["top_value"].astype(str)
    return log


def kept_features(log: pd.DataFrame) -> pd.DataFrame:
    """Kept, binnable features (dates are only used through derived features)."""
    return log[(log.status == "kept") & (log.kind != "date")][["variable", "kind", "source"]]
