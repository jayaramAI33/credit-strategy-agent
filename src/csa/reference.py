"""Reference (non-agent) greedy swap-set builder.

Not used by the agent. It gives phase 2 an example swap set, gives the tests known
answers, and gives the deck an honest "an optimizer could do this part" comparison.
"""
from __future__ import annotations

from .lab import StrategyLab


def greedy(lab: StrategyLab, target_growth_pct: float, max_delta_bad_pp: float) -> dict:
    cat = lab.catalogue
    ins_q = list(cat[(cat.kind == "swap_in") & cat.listed].id)
    outs_q = list(cat[(cat.kind == "swap_out") & cat.listed].id)
    ins, outs, history = [], [], []
    best_feasible = None
    for _ in range(len(ins_q) + len(outs_q) + 1):
        sim = lab.simulate(ins, outs, with_ci=False)
        breaches = [c for c in sim["limit_checks"] if c["status"] == "breach"]
        conc_breach = any(c["limit_id"] == "swap_in_concentration_pct" for c in breaches)
        feasible = not breaches and sim["delta_bad_rate_pp"] <= max_delta_bad_pp
        history.append({"ins": list(ins), "outs": list(outs), "growth_pct": sim["approval_growth_pct"],
                        "delta_bad_pp": sim["delta_bad_rate_pp"], "feasible": feasible})
        if feasible and (best_feasible is None or sim["approval_growth_pct"] > best_feasible["growth_pct"]):
            best_feasible = history[-1]
        if sim["delta_bad_rate_pp"] > max_delta_bad_pp:
            if not outs_q:
                break
            outs.append(outs_q.pop(0))
        elif sim["approval_growth_pct"] < target_growth_pct or conc_breach:
            if not ins_q:
                break
            ins.append(ins_q.pop(0))
        else:
            return {"met": True, "ins": ins, "outs": outs, "history": history, "best_feasible": best_feasible}
    return {"met": False, "ins": ins, "outs": outs, "history": history, "best_feasible": best_feasible}
