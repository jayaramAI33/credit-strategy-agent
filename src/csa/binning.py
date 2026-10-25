"""Stage E/G: binning, WoE/IV, stability, and top-N feature selection.

Numeric: equal-frequency start -> merge to a strictly monotonic bad rate -> enforce
minimum bin size and bads -> cap the number of bins. Categorical: small levels merged
into "Other", then closest bad rates merged down to max_bins. Fitted on build only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

EPS = 0.5  # smoothing for WoE


@dataclass
class BinSpec:
    variable: str
    kind: str                       # numeric | categorical
    edges: list = field(default_factory=list)       # numeric: inner cut points, bins are (e_i-1, e_i]
    groups: dict = field(default_factory=dict)      # categorical: level -> group label
    direction: str = ""             # numeric: "up" (bad rate rises with value) or "down"

    def assign(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            return pd.Series(np.searchsorted(np.asarray(self.edges), x.to_numpy(), side="left"),
                             index=x.index)
        default = "Other" if "Other" in self.groups.values() else self._largest
        return x.map(self.groups).fillna(default)

    def labels(self) -> list:
        if self.kind == "categorical":
            return sorted(set(self.groups.values()), key=list(self.groups.values()).index)
        e = self.edges
        if not e:
            return ["all"]
        f = lambda v: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"
        return [f"<= {f(e[0])}"] + [f"({f(a)}, {f(b)}]" for a, b in zip(e[:-1], e[1:])] + [f"> {f(e[-1])}"]

    _largest: str = ""


def _stats(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    g = pd.DataFrame({"bin": bins.to_numpy(), "bad": y.to_numpy()}).groupby("bin")["bad"]
    return pd.DataFrame({"n": g.size(), "bads": g.sum()})


def _fit_numeric(var: str, x: pd.Series, y: pd.Series, cfg) -> BinSpec:
    qs = np.linspace(0, 1, cfg["initial_bins"] + 1)[1:-1]
    edges = sorted(set(np.quantile(x, qs)))
    edges = [e for e in edges if e < x.max()]          # drop an empty top bin
    xs, ys = x.to_numpy(), y.to_numpy()

    def table(ed):
        b = np.searchsorted(np.asarray(ed), xs, side="left")
        n = np.bincount(b, minlength=len(ed) + 1)
        bd = np.bincount(b, weights=ys, minlength=len(ed) + 1)
        return n, bd

    n, bd = table(edges)
    rate = bd / np.maximum(n, 1)
    idx = np.arange(len(n))
    direction = "up" if np.cov(idx, rate, aweights=np.maximum(n, 1))[0, 1] >= 0 else "down"
    sign = 1 if direction == "up" else -1

    changed = True
    while changed and edges:
        changed = False
        n, bd = table(edges)
        rate = bd / np.maximum(n, 1)
        for i in range(len(rate) - 1):              # strict monotonic: merge first violation
            if sign * (rate[i + 1] - rate[i]) <= 0:
                del edges[i]; changed = True; break
    N = len(xs)
    min_n, min_b = cfg["min_bin_share_pct"] / 100 * N, cfg["min_bin_bads"]
    while edges:
        n, bd = table(edges)
        small = [i for i in range(len(n)) if n[i] < min_n or bd[i] < min_b]
        if not small:
            break
        i = min(small, key=lambda j: n[j])
        rate = bd / np.maximum(n, 1)
        if i == 0:
            del edges[0]
        elif i == len(n) - 1:
            del edges[-1]
        else:
            left_gap, right_gap = abs(rate[i] - rate[i - 1]), abs(rate[i + 1] - rate[i])
            del edges[i - 1 if left_gap <= right_gap else i]
    while len(edges) + 1 > cfg["max_bins"]:
        n, bd = table(edges)
        rate = bd / np.maximum(n, 1)
        del edges[int(np.argmin(np.abs(np.diff(rate))))]
    return BinSpec(var, "numeric", edges=[float(e) for e in edges], direction=direction)


def _fit_categorical(var: str, x: pd.Series, y: pd.Series, cfg) -> BinSpec:
    share = x.value_counts(normalize=True)
    small = set(share[share < cfg["cat_small_level_share_pct"] / 100].index)
    groups = {lvl: ("Other" if lvl in small else lvl) for lvl in share.index}
    while True:
        st = _stats(x.map(groups), y)
        if len(st) <= cfg["max_bins"]:
            break
        st["rate"] = st.bads / st.n
        st = st.sort_values("rate")
        gaps = np.diff(st["rate"].to_numpy())
        i = int(np.argmin(gaps))
        a, b = st.index[i], st.index[i + 1]
        merged = f"{a}|{b}" if "Other" not in (a, b) else "Other"
        groups = {k: (merged if v in (a, b) else v) for k, v in groups.items()}
    spec = BinSpec(var, "categorical", groups=groups)
    spec._largest = x.map(groups).value_counts().index[0]
    return spec


def fit(var: str, kind: str, x: pd.Series, y: pd.Series) -> BinSpec:
    cfg = config.load("prep")["binning"]
    return (_fit_numeric if kind == "numeric" else _fit_categorical)(var, x, y, cfg)


def evaluate(spec: BinSpec, xb, yb, xo, yo) -> tuple[pd.DataFrame, dict]:
    """Per-bin table (build and OOT on build bins) plus summary metrics."""
    sb, so = _stats(spec.assign(xb), yb), _stats(spec.assign(xo), yo)
    keys = list(range(len(spec.edges) + 1)) if spec.kind == "numeric" else spec.labels()
    t = pd.DataFrame(index=keys)
    t["n_build"] = sb["n"].reindex(keys).fillna(0)
    t["bads_build"] = sb["bads"].reindex(keys).fillna(0)
    t["n_oot"] = so["n"].reindex(keys).fillna(0)
    t["bads_oot"] = so["bads"].reindex(keys).fillna(0)
    G, B = (t.n_build - t.bads_build).sum(), t.bads_build.sum()
    Go, Bo = (t.n_oot - t.bads_oot).sum(), t.bads_oot.sum()
    pg, pb = (t.n_build - t.bads_build + EPS) / G, (t.bads_build + EPS) / B
    pgo, pbo = (t.n_oot - t.bads_oot + EPS) / Go, (t.bads_oot + EPS) / Bo
    t["woe"] = np.log(pg / pb)
    t["vol_build_pct"] = 100 * t.n_build / t.n_build.sum()
    t["bad_build_pct"] = 100 * t.bads_build / t.n_build.clip(lower=1)
    t["vol_oot_pct"] = 100 * t.n_oot / t.n_oot.sum()
    t["bad_oot_pct"] = 100 * t.bads_oot / t.n_oot.clip(lower=1)
    iv_b = float(((pg - pb) * np.log(pg / pb)).sum())
    iv_o = float(((pgo - pbo) * np.log(pgo / pbo)).sum())
    p1, p2 = t.vol_build_pct / 100 + 1e-6, t.vol_oot_pct / 100 + 1e-6
    psi = float(((p2 - p1) * np.log(p2 / p1)).sum())
    mono = None
    if spec.kind == "numeric" and len(t) > 1:
        sign = 1 if spec.direction == "up" else -1
        mono = bool((sign * np.diff(t.bad_oot_pct.to_numpy()) >= 0).all())
    t.insert(0, "bin", spec.labels())
    t.insert(0, "variable", spec.variable)
    return t.reset_index(drop=True), {
        "variable": spec.variable, "kind": spec.kind, "n_bins": len(t),
        "direction": spec.direction or "n/a", "iv_build": iv_b, "iv_oot": iv_o, "psi": psi,
        "monotonic_oot": mono}


def woe_encode(spec: BinSpec, x: pd.Series, table: pd.DataFrame) -> pd.Series:
    keys = list(range(len(spec.edges) + 1)) if spec.kind == "numeric" else spec.labels()
    return spec.assign(x).map(dict(zip(keys, table["woe"]))).astype(float)


def select_top(iv: pd.DataFrame, woe_build: pd.DataFrame) -> pd.DataFrame:
    """Greedy top-N by build IV, subject to eligibility and a correlation cap."""
    cfg = config.load("prep")["selection"]
    policy_vars = set(config.load("fields")["policy_variables"])
    out, chosen = [], []
    for _, r in iv.sort_values("iv_build", ascending=False).iterrows():
        v = r["variable"]
        why = []
        if cfg["exclude_policy_variables"] and v in policy_vars:
            why.append("already in current policy")
        if r.iv_build < cfg["min_iv"] or r.iv_oot < cfg["min_iv"]:
            why.append(f"IV below {cfg['min_iv']}")
        if r.psi > cfg["max_psi"]:
            why.append(f"PSI {r.psi:.3f} > {cfg['max_psi']}")
        if r.monotonic_oot is False:
            why.append("bad-rate trend breaks in OOT")
        if not why and len(chosen) < cfg["top_n"]:
            corr = {c: abs(woe_build[v].corr(woe_build[c])) for c in chosen}
            worst = max(corr.items(), key=lambda kv: kv[1]) if corr else (None, 0)
            if worst[1] > cfg["max_abs_corr"]:
                why.append(f"corr {worst[1]:.2f} with {worst[0]}")
        if not why and len(chosen) >= cfg["top_n"]:
            why.append(f"top {cfg['top_n']} already filled")
        selected = not why
        if selected:
            chosen.append(v)
        out.append({"variable": v, "iv_build": r.iv_build, "selected": selected,
                    "rank": len(chosen) if selected else None, "reason": "; ".join(why)})
    return pd.DataFrame(out)
