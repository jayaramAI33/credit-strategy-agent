"""Exhaustive small-combination search over listed segments (the "optimizer" benchmark).

Not the agent. Answers: which lean plans pass every limit, and what is the most growth the
appetite allows? Used to score the agent in phase 4. Takes ~4-5 minutes.
Usage: python scripts/benchmark_search.py [max_ins=4] [max_outs=3] [top_outs=12]
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from csa.lab import StrategyLab  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "phase2"


def main(max_ins=4, max_outs=3, top_outs=12):
    t0 = time.time()
    lab = StrategyLab(); lab.build_catalogue()
    lim = lab.appetite["limits"]
    d_max = lim["portfolio_bad_rate_delta_pp"]["max"]
    st = lim["stressed_portfolio_bad_rate_delta_pp"]
    conc = lim["swap_in_concentration_pct"]
    assert conc["basis"] == "policy_group", "benchmark assumes pocket-based concentration"
    oot = lab.split["oot"]; A = lab.approved[oot]; y = lab.bad[oot]; grp = lab.group[oot]
    nb, base_bads = A.sum(), y[A].sum(); br0 = base_bads / nb
    cat = lab.catalogue
    ins = list(cat[(cat.kind == "swap_in") & cat.listed].id)
    outs = list(cat[(cat.kind == "swap_out") & cat.listed].id)[:top_outs]
    MI = {s: lab._masks[s][oot] & ~A for s in ins}
    MO = {s: lab._masks[s][oot] & A for s in outs}
    groups = sorted(set(grp))
    out_sets = [()] + [c for k in range(1, max_outs + 1) for c in combinations(outs, k)]
    out_m = {o: np.logical_or.reduce([MO[s] for s in o]) if o else np.zeros(len(y), bool) for o in out_sets}
    rows = []
    for k in range(1, max_ins + 1):
        for ci in combinations(ins, k):
            mi = np.logical_or.reduce([MI[s] for s in ci])
            n_in, b_in = mi.sum(), y[mi].sum()
            pocket = max(100 * (mi & (grp == g)).sum() / n_in for g in groups)
            for o in out_sets:
                mo = out_m[o]
                n_new = nb - mo.sum() + n_in
                bads = base_bads - y[mo].sum() + b_in
                rows.append({"ins": list(ci), "outs": list(o),
                             "growth_pct": 100 * (n_new / nb - 1),
                             "delta_bad_pp": 100 * (bads / n_new - br0),
                             "stressed_pp": 100 * ((bads + st["swap_in_shock_pp"] / 100 * n_in) / n_new - br0),
                             "pocket_pct": pocket, "turnover_pct": 100 * (n_in + mo.sum()) / nb})
    within = [r for r in rows if r["delta_bad_pp"] <= d_max and r["stressed_pp"] <= st["max"]
              and r["pocket_pct"] <= conc["max"]]
    best = max(within, key=lambda r: r["growth_pct"])
    goal_a = sorted([r for r in within if r["growth_pct"] >= 3 and r["delta_bad_pp"] <= 0],
                    key=lambda r: r["turnover_pct"])
    summary = {"evaluated": len(rows), "within_appetite": len(within),
               "max_growth_within_appetite": best,
               "goal_a_feasible": len(goal_a), "goal_a_leanest": goal_a[:10],
               "runtime_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark_search.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("evaluated", "within_appetite", "goal_a_feasible", "runtime_s")}))
    print("max growth:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})
    print("leanest goal A:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in goal_a[0].items()})


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
