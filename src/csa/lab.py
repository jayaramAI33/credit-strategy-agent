"""StrategyLab: the deterministic engine behind the agent's tools.

Loads the phase 1 population once, then answers: baseline, marginal approved bad rate,
segment catalogue, and swap-set simulation with limit checks and escalation band.
All numbers come from here; the agent never computes a number itself.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import config, policy, prep, segments, treatment
from .segments import Condition


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def plain(obj):
    """Frozen config -> plain dict/list (for JSON and for the agent)."""
    if hasattr(obj, "items"):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [plain(v) for v in obj]
    return obj


class StrategyLab:
    def __init__(self):
        p1 = config.path("phase1_out")
        df = prep.add_derived(pd.read_parquet(config.path("processed_file")))
        ttable = pd.read_csv(p1 / "treatment_table.csv")
        self.specs = json.loads((p1 / "binning_spec.json").read_text())
        self.top_features = json.loads((p1 / "phase1_metrics.json").read_text())["top_features"]
        self.cond_features = [*self.top_features, "fico_mid"]
        self.t = treatment.apply_from_table(df, ttable)          # treated values (segments)
        self.df = df                                             # raw values (policy)
        self.approved = policy.approve(df).to_numpy()
        self.group = segments.policy_group(df, pd.Series(self.approved, index=df.index)).to_numpy()
        self.bad = df["bad"].to_numpy().astype(float)
        self.split = {"build": (df.split == "build").to_numpy(), "oot": (df.split == "oot").to_numpy()}
        self.appetite = config.load("risk_appetite")
        self._cells_rank = self._policy_cell_rank()
        self.marginal = {s: self._marginal_bad_rate(s) for s in self.split}
        self.catalogue: pd.DataFrame | None = None
        self._masks: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- baseline
    def baseline(self, split: str = "oot") -> dict:
        m = self.split[split]
        a = self.approved & m
        return {"split": split, "applications": int(m.sum()), "approved": int(a.sum()),
                "approval_rate_pct": _r(100 * a.sum() / m.sum()),
                "approved_bad_rate_pct": _r(100 * self.bad[a].mean()),
                "declined_bad_rate_pct": _r(100 * self.bad[m & ~self.approved].mean()),
                "marginal_approved_bad_rate_pct": _r(100 * self.marginal[split]),
                "policy": plain(config.load("strategy")["current_policy"])}

    def _policy_cell_rank(self) -> pd.Series:
        """Build bad rate of each approved policy cell (grade x FICO bin)."""
        fb = np.searchsorted(np.asarray(self.specs["fico_mid"]["edges"]), self.t["fico_mid"].to_numpy(),
                             side="left")
        self._cell = pd.Series(self.df["grade"].to_numpy() + "_F" + fb.astype(str))
        b = self.split["build"] & self.approved
        return pd.Series(self.bad[b]).groupby(self._cell[b].to_numpy()).mean()

    def _marginal_bad_rate(self, split: str) -> float:
        share = self.appetite["definitions"]["marginal_approved_bad_rate"]["riskiest_share_of_approvals_pct"]
        a = self.approved & self.split[split]
        d = pd.DataFrame({"rank": self._cell[a].map(self._cells_rank).to_numpy(),
                          "id": self.df["id"].to_numpy()[a], "bad": self.bad[a]})
        d = d.sort_values(["rank", "id"], ascending=[False, True])
        k = int(math.ceil(share / 100 * len(d)))
        return float(d["bad"].iloc[:k].mean())

    # ---------------------------------------------------------------- segments
    def _mask(self, group: str, conds: tuple[Condition, ...]) -> np.ndarray:
        m = self.group == group
        for c in conds:
            m = m & c.mask(self.t[c.feature])
        return m

    def build_catalogue(self) -> pd.DataFrame:
        lim = self.appetite["limits"]
        min_vol = lim["segment_min_volume_pct"]["min"]
        nb, no = self.split["build"].sum(), self.split["oot"].sum()
        mb, mo = self.marginal["build"], self.marginal["oot"]
        rows = []
        for kind, groups, side in (("swap_in", segments.DECLINED_GROUPS, "good"),
                                   ("swap_out", segments.APPROVED_GROUPS, "bad")):
            rules = segments.candidate_rules(self.cond_features, self.specs, side)
            for g in groups:
                for conds in rules:
                    m = self._mask(g, conds)
                    b, o = m & self.split["build"], m & self.split["oot"]
                    n_b, n_o = int(b.sum()), int(o.sum())
                    if n_b == 0:
                        continue
                    rb, ro = self.bad[b].mean(), (self.bad[o].mean() if n_o else np.nan)
                    rows.append({"kind": kind, "group": g, "conds": conds,
                                 "n_build": n_b, "vol_build_pct": 100 * n_b / nb,
                                 "bads_build": int(self.bad[b].sum()), "bad_build_pct": 100 * rb,
                                 "n_oot": n_o, "vol_oot_pct": 100 * n_o / no, "bad_oot_pct": 100 * ro})
        c = pd.DataFrame(rows)
        isin = c.kind == "swap_in"
        size_ok = (c.vol_build_pct >= min_vol) & (c.bads_build >= 50)
        c["found_build"] = size_ok & np.where(isin, c.bad_build_pct <= 100 * mb, c.bad_build_pct >= 100 * mb)
        c["confirmed_oot"] = c.found_build & (c.vol_oot_pct >= min_vol) & np.where(
            isin, c.bad_oot_pct <= 100 * mo, c.bad_oot_pct >= 100 * mo)
        c["benefit"] = c.n_build * np.abs(100 * mb - c.bad_build_pct)
        c = c.sort_values(["kind", "benefit"], ascending=[True, False]).reset_index(drop=True)
        # De-duplicate near-identical confirmed segments (Jaccard > 0.8 on build loans).
        keep = []
        for kind in ("swap_in", "swap_out"):
            kept_masks = []
            for i in c.index[(c.kind == kind) & c.confirmed_oot]:
                m = self._mask(c.at[i, "group"], c.at[i, "conds"]) & self.split["build"]
                if all((m & k).sum() / (m | k).sum() <= 0.8 for k in kept_masks):
                    kept_masks.append(m); keep.append(i)
        c["listed"] = c.index.isin(keep)
        c["id"] = None
        for kind, prefix in (("swap_in", "IN"), ("swap_out", "OUT")):
            idx = c.index[(c.kind == kind) & c.listed]
            c.loc[idx, "id"] = [f"{prefix}-{j + 1:02d}" for j in range(len(idx))]
        for i in c.index[c.listed]:
            self._masks[c.at[i, "id"]] = self._mask(c.at[i, "group"], c.at[i, "conds"])
        c["rule"] = [self.rule_text(g, cs) for g, cs in zip(c.group, c.conds)]
        self.catalogue = c
        return c

    @staticmethod
    def rule_text(group: str, conds) -> str:
        base = {**segments.DECLINED_GROUPS, **segments.APPROVED_GROUPS}[group]
        return base + "".join(f" AND {c.text()}" for c in conds)

    def find_segments(self, kind: str, top_n: int = 15) -> list[dict]:
        c = self.catalogue
        c = c[(c.kind == kind) & c.listed].head(top_n)
        return [{"segment_id": r.id, "rule": r.rule, "policy_group": r.group,
                 "build_volume_pct": _r(r.vol_build_pct, 2), "build_bad_rate_pct": _r(r.bad_build_pct, 2),
                 "oot_volume_pct": _r(r.vol_oot_pct, 2), "oot_bad_rate_pct": _r(r.bad_oot_pct, 2)}
                for r in c.itertuples()]

    # ---------------------------------------------------------------- simulation
    def simulate(self, swap_ins: list[str], swap_outs: list[str], split: str = "oot",
                 with_ci: bool = True, _concentration_override: dict | None = None) -> dict:
        """_concentration_override is for analysis scripts only; agent tools never pass it."""
        unknown = [s for s in [*swap_ins, *swap_outs] if s not in self._masks]
        if unknown:
            raise ValueError(f"unknown segment ids: {unknown}")
        if any(s.startswith("OUT") for s in swap_ins) or any(s.startswith("IN") for s in swap_outs):
            raise ValueError("swap_ins must be IN-xx ids and swap_outs OUT-xx ids")
        lim, esc = self.appetite["limits"], self.appetite["escalation"]
        sp = self.split[split]
        N = sp.sum()
        m_in = np.zeros(len(sp), bool); m_out = np.zeros(len(sp), bool)
        for s in swap_ins:
            m_in |= self._masks[s]
        for s in swap_outs:
            m_out |= self._masks[s]
        m_in &= ~self.approved & sp
        m_out &= self.approved & sp
        base = self.approved & sp
        new = (base & ~m_out) | m_in
        br_base, br_new = self.bad[base].mean(), self.bad[new].mean()
        d_bad = 100 * (br_new - br_base)
        marg = 100 * self.marginal[split]

        seg_rows, checks = [], []
        n_in = m_in.sum()
        conc = dict(_concentration_override or lim["swap_in_concentration_pct"])
        rules = self.catalogue.set_index("id")["rule"]
        for s in [*swap_ins, *swap_outs]:
            m = self._masks[s] & sp
            n = int(m.sum())
            r = {"segment_id": s, "rule": rules[s],
                 "loans": n, "volume_pct": _r(100 * n / N, 2), "bad_rate_pct": _r(100 * self.bad[m].mean(), 2)}
            if s.startswith("IN"):
                r["share_of_new_approvals_pct"] = _r(100 * (m & m_in).sum() / n_in, 1) if n_in else 0.0
                checks.append(self._check(f"{s} bad rate vs marginal approved", r["bad_rate_pct"], "max", marg,
                                          "swap_in_segment_bad_rate"))
                if conc["basis"] == "segment":
                    checks.append(self._check(f"{s} concentration", r["share_of_new_approvals_pct"], "max",
                                              conc["max"], "swap_in_concentration_pct"))
            checks.append(self._check(f"{s} volume", r["volume_pct"], "min",
                                      lim["segment_min_volume_pct"]["min"], "segment_min_volume_pct"))
            seg_rows.append(r)
        group_share = {}
        if n_in:
            group_share = (pd.Series(self.group[m_in]).value_counts(normalize=True) * 100).round(1).to_dict()
            if conc["basis"] == "policy_group":
                for g, v in group_share.items():
                    checks.append(self._check(f"{g} share of new approvals", v, "max", conc["max"],
                                              "swap_in_concentration_pct"))
        port = self._check("portfolio bad rate delta (pp)", d_bad, "max",
                           lim["portfolio_bad_rate_delta_pp"]["max"], "portfolio_bad_rate_delta_pp")
        ci = None
        if with_ci and (swap_ins or swap_outs):
            ci = self._bootstrap_delta(base, new, sp)
            if port["status"] != "breach" and ci[1] > lim["portfolio_bad_rate_delta_pp"]["max"]:
                port["status"], port["note"] = "point_estimate_only", "upper 95% bound exceeds the limit"
        checks.insert(0, port)
        st = lim["stressed_portfolio_bad_rate_delta_pp"]
        stressed = 100 * ((self.bad[new].sum() + st["swap_in_shock_pp"] / 100 * n_in) / new.sum() - br_base)
        checks.insert(1, self._check(f"stressed bad rate delta (pp), swap-ins +{st['swap_in_shock_pp']} pp",
                                     stressed, "max", st["max"], "stressed_portfolio_bad_rate_delta_pp"))
        growth = 100 * (new.sum() / base.sum() - 1)
        checks.insert(2, self._check("net approval growth (%)", growth, "max",
                                     lim["net_approval_growth_pct"]["max"], "net_approval_growth_pct"))
        nb, n_out = int(base.sum()), int(m_out.sum())
        volumes = {
            "current_approved": nb, "swap_in": int(n_in), "swap_out": n_out,
            "net_change": int(new.sum()) - nb, "new_approved": int(new.sum()),
            "swap_in_pct_of_current": _r(100 * n_in / nb, 2),
            "swap_out_pct_of_current": _r(100 * n_out / nb, 2),
            "net_change_pct_of_current": _r(growth, 2),
            "turnover_pct_of_current": _r(100 * (n_in + n_out) / nb, 2),
            "swaps_per_net_new_approval": _r((n_in + n_out) / (new.sum() - nb), 2) if new.sum() > nb else None,
        }

        statuses = {c["status"] for c in checks}
        band = "red" if "breach" in statuses else (
            "amber" if statuses & {"near", "point_estimate_only"} else "green")
        return {
            "split": split, "swap_ins": list(swap_ins), "swap_outs": list(swap_outs),
            "baseline": {"approval_rate_pct": _r(100 * base.sum() / N), "approved_bad_rate_pct": _r(100 * br_base)},
            "new": {"approval_rate_pct": _r(100 * new.sum() / N), "approved_bad_rate_pct": _r(100 * br_new)},
            "delta_approval_rate_pp": _r(100 * (new.sum() - base.sum()) / N),
            "approval_growth_pct": _r(100 * (new.sum() / base.sum() - 1), 2),
            "delta_bad_rate_pp": _r(d_bad),
            "delta_bad_rate_ci95_pp": [_r(ci[0]), _r(ci[1])] if ci else None,
            "stressed_delta_bad_rate_pp": _r(stressed),
            "swapped_in_loans": int(n_in), "swapped_out_loans": int(m_out.sum()),
            "volumes": volumes,
            "swap_in_bad_rate_pct": _r(100 * self.bad[m_in].mean()) if n_in else None,
            "swap_out_bad_rate_pct": _r(100 * self.bad[m_out].mean()) if m_out.any() else None,
            "marginal_approved_bad_rate_pct": _r(marg),
            "new_approvals_by_policy_group_pct": group_share,
            "concentration_basis": conc["basis"],
            "segments": seg_rows, "limit_checks": checks,
            "band": band, "route": esc["bands"][band]["route"],
        }

    def _check(self, name, value, kind, limit, limit_id) -> dict:
        near = self.appetite["escalation"]["near_limit_pct"] / 100
        if kind == "max":
            status = "breach" if value > limit else ("near" if value > limit - near * abs(limit) else "pass")
        else:
            status = "breach" if value < limit else ("near" if value < limit * (1 + near) else "pass")
        return {"check": name, "limit_id": limit_id, "value": _r(value), "limit": _r(limit),
                "type": kind, "status": status}

    def _bootstrap_delta(self, base, new, sp) -> tuple[float, float]:
        esc = self.appetite["escalation"]
        rng = np.random.default_rng(esc["bootstrap_seed"])
        idx = np.flatnonzero(sp)
        b, n_, y = base[idx].astype(float), new[idx].astype(float), self.bad[idx]
        out = []
        for _ in range(esc["bootstrap_samples"]):
            w = rng.poisson(1.0, len(idx))
            out.append(100 * ((w * n_ * y).sum() / (w * n_).sum() - (w * b * y).sum() / (w * b).sum()))
        a = (1 - esc["confidence_level"]) / 2
        return float(np.quantile(out, a)), float(np.quantile(out, 1 - a))
