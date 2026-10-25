"""Vectorised exhaustive benchmark: every subset of listed swap-ins x every set of up to
`max_outs` listed swap-outs, scored on 2015 against the full risk appetite.

Swap-ins (currently declined) and swap-outs (currently approved) are disjoint loan sets,
so approvals and bads add up independently: each plan is scored with scalar arithmetic,
which makes millions of plans cheap. Segment-level limits (swap-in bad rate, volume) hold
for every listed segment by construction, so only portfolio-level limits are checked.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .lab import StrategyLab


class Benchmark:
    def __init__(self, lab: StrategyLab, max_outs: int = 3):
        lim = lab.appetite["limits"]
        self.d_max = lim["portfolio_bad_rate_delta_pp"]["max"]
        self.st = lim["stressed_portfolio_bad_rate_delta_pp"]
        self.g_max = lim["net_approval_growth_pct"]["max"]
        self.c_max = lim["swap_in_concentration_pct"]["max"]
        oot = lab.split["oot"]; A = lab.approved[oot]; y = lab.bad[oot]; grp = lab.group[oot]
        nb, base_bads = A.sum(), y[A].sum()
        cat = lab.catalogue
        self.ins = list(cat[(cat.kind == "swap_in") & cat.listed].id)
        self.outs = list(cat[(cat.kind == "swap_out") & cat.listed].id)
        MI = [lab._masks[s][oot] & ~A for s in self.ins]
        MO = [lab._masks[s][oot] & A for s in self.outs]
        groups = sorted(set(grp))

        # all subsets of swap-ins. Each loan gets a bit code of the swap-in segments it belongs to;
        # a subset S includes the loan iff (code & S) != 0. Aggregating by code makes this fast.
        self.in_sets = [()] + [c for k in range(1, len(self.ins) + 1) for c in combinations(range(len(self.ins)), k)]
        code = np.zeros(len(y), dtype=np.int64)
        for j, m in enumerate(MI):
            code |= m.astype(np.int64) << j
        sel = code > 0
        uniq, inv = np.unique(code[sel], return_inverse=True)
        n_c = np.bincount(inv).astype(float)
        b_c = np.bincount(inv, weights=y[sel])
        g_c = np.stack([np.bincount(inv, weights=(grp[sel] == g).astype(float), minlength=len(uniq)) for g in groups])
        s_bits = np.array([sum(1 << j for j in s) for s in self.in_sets], dtype=np.int64)
        incl = ((s_bits[:, None] & uniq[None, :]) != 0).astype(float)
        n_in, b_in = incl @ n_c, incl @ b_c
        with np.errstate(invalid="ignore", divide="ignore"):
            pocket = np.nan_to_num(100 * (incl @ g_c.T).max(axis=1) / n_in)
        self.out_sets = [()] + [c for k in range(1, max_outs + 1) for c in combinations(range(len(self.outs)), k)]
        n_out = np.zeros(len(self.out_sets)); b_out = np.zeros(len(self.out_sets))
        for i, s in enumerate(self.out_sets):
            if s:
                m = np.logical_or.reduce([MO[j] for j in s])
                n_out[i], b_out[i] = m.sum(), y[m].sum()

        n_new = nb - n_out[None, :] + n_in[:, None]
        bads = base_bads - b_out[None, :] + b_in[:, None]
        br0 = base_bads / nb
        self.growth = 100 * (n_new / nb - 1)
        self.delta = 100 * (bads / n_new - br0)
        self.stressed = 100 * ((bads + self.st["swap_in_shock_pp"] / 100 * n_in[:, None]) / n_new - br0)
        self.turnover = 100 * (n_in[:, None] + n_out[None, :]) / nb
        self.pocket = np.broadcast_to(pocket[:, None], self.growth.shape)
        self.n_plans = self.growth.size

    def within_appetite(self) -> np.ndarray:
        return ((self.delta <= self.d_max) & (self.stressed <= self.st["max"]) &
                (self.growth <= self.g_max) & (self.pocket <= self.c_max))

    def _plan(self, i, j) -> dict:
        return {"ins": [self.ins[k] for k in self.in_sets[i]], "outs": [self.outs[k] for k in self.out_sets[j]],
                "growth_pct": round(float(self.growth[i, j]), 3), "delta_bad_pp": round(float(self.delta[i, j]), 3),
                "stressed_pp": round(float(self.stressed[i, j]), 3),
                "turnover_pct": round(float(self.turnover[i, j]), 2)}

    def key(self, min_growth: float, max_delta: float | None) -> dict:
        """Answer key for a goal: feasible?, best growth, leanest qualifying plan, need for swap-outs."""
        ok = self.within_appetite()
        if max_delta is not None:
            ok = ok & (self.delta <= max_delta)
        best = np.unravel_index(np.argmax(np.where(ok, self.growth, -np.inf)), ok.shape)
        meets = ok & (self.growth >= min_growth)
        out = {"feasible": bool(meets.any()), "max_growth_under_goal_constraints": self._plan(*best) if ok.any() else None}
        if meets.any():
            lean = np.unravel_index(np.argmin(np.where(meets, self.turnover, np.inf)), ok.shape)
            out["leanest_qualifying_plan"] = self._plan(*lean)
            out["needs_swap_outs"] = not bool(meets[:, 0].any())     # column 0 = no swap-outs
        appetite_best = np.unravel_index(np.argmax(np.where(self.within_appetite(), self.growth, -np.inf)), ok.shape)
        out["max_growth_within_appetite"] = self._plan(*appetite_best)
        return out
