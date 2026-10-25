"""Test proposal and deployable strategy export, built only from a simulation result.

Every number here is copied from (or arithmetically derived from) a simulate() result.
The proposal is a TEST CELL, never a rollout, and is always PENDING HUMAN APPROVAL.
"""
from __future__ import annotations

import math

from . import config
from .lab import StrategyLab, plain

REJECT_INFERENCE_CAVEAT = (
    "Swap-in bad rates are measured here because every Lending Club loan was approved and its "
    "outcome is known. In production, declined applicants' outcomes are never observed, so these "
    "rates would be estimates (reject inference). That is why this is a small randomised test "
    "cell with a stop rule, not a rollout.")

_GROUP_RULES = {
    "KO_DTI": [("dti", ">", 35)],
    "KO_LTI": [("dti", "<=", 35), ("loan_to_income", ">", 0.40)],
    "C_LOW_FICO": [("dti", "<=", 35), ("loan_to_income", "<=", 0.40), ("grade", "==", "C"), ("fico_mid", "<", 677)],
    "D_LOW_FICO": [("dti", "<=", 35), ("loan_to_income", "<=", 0.40), ("grade", "==", "D"), ("fico_mid", "<", 697)],
    "E": [("dti", "<=", 35), ("loan_to_income", "<=", 0.40), ("grade", "==", "E")],
    "FG": [("dti", "<=", 35), ("loan_to_income", "<=", 0.40), ("grade", "in", ["F", "G"])],
    "A": [("dti", "<=", 35), ("loan_to_income", "<=", 0.40), ("grade", "==", "A")],
    "B": [("dti", "<=", 35), ("loan_to_income", "<=", 0.40), ("grade", "==", "B")],
    "C": [("dti", "<=", 35), ("loan_to_income", "<=", 0.40), ("grade", "==", "C"), ("fico_mid", ">=", 677)],
    "D": [("dti", "<=", 35), ("loan_to_income", "<=", 0.40), ("grade", "==", "D"), ("fico_mid", ">=", 697)],
}


def _ci_half_width_pp(p_pct: float, n: int) -> float | None:
    if not n or p_pct is None:
        return None
    p = p_pct / 100
    return round(100 * 1.96 * math.sqrt(p * (1 - p) / n), 2)


def draft_test_proposal(lab: StrategyLab, sim: dict, goal: str) -> dict:
    appetite = lab.appetite
    share = appetite["limits"]["test_cell_share_pct"]["max"]
    marg = sim["marginal_approved_bad_rate_pct"]
    cells = []
    for seg in sim["segments"]:
        kind = "swap_in" if seg["segment_id"].startswith("IN") else "swap_out"
        n_test = int(round(seg["loans"] * share / 100))
        cells.append({
            "cell_id": f"TC-{seg['segment_id']}", "segment_id": seg["segment_id"], "kind": kind,
            "rule": seg["rule"],
            "action_in_test_cell": "approve (currently declined)" if kind == "swap_in"
                                   else "decline (currently approved)",
            "segment_loans_per_year": seg["loans"], "test_share_pct": share,
            "test_loans_per_year": n_test,
            "expected_bad_rate_pct": seg["bad_rate_pct"],
            "test_bad_rate_ci95_half_width_pp": _ci_half_width_pp(seg["bad_rate_pct"], n_test),
            "stop_rule": (f"Pause the cell if its observed bad rate exceeds {marg}% "
                          f"(the swap-in limit: marginal approved bad rate)") if kind == "swap_in" else
                         "Stop if declined volume differs from the expected count by more than 20%",
        })
    return {
        "status": "PENDING HUMAN APPROVAL",
        "goal": goal,
        "recommendation": "Randomised test cell, not a rollout",
        "band": sim["band"], "route": sim["route"],
        "expected_impact_if_rolled_out": {k: sim[k] for k in (
            "delta_approval_rate_pp", "approval_growth_pct", "delta_bad_rate_pp", "delta_bad_rate_ci95_pp",
            "stressed_delta_bad_rate_pp", "swap_in_bad_rate_pct", "swap_out_bad_rate_pct")},
        "baseline": sim["baseline"], "new": sim["new"],
        "volumes": sim["volumes"],
        "test_cells": cells,
        "success_criteria": [
            f"Swap-in test cells: bad rate at or below {marg}% (marginal approved bad rate)",
            f"Combined effect: Δ bad rate within the +{appetite['limits']['portfolio_bad_rate_delta_pp']['max']} pp limit",
            "Approval uplift in the test cells in line with the simulated volume",
        ],
        "duration": {"early_read_months": 6, "decision_months": 12, "full_term_months": 36,
                     "early_indicator": "30+ days past due at 6 months on book, vs current marginal approvals"},
        "monitoring": ["Monthly test-cell volume and mix", "Early delinquency vs control",
                       "PSI of the top-5 features vs build (alert above 0.10)", "Stop-rule check every month"],
        "limit_checks": sim["limit_checks"],
        "reject_inference_caveat": REJECT_INFERENCE_CAVEAT,
        "source": {"simulation_split": sim["split"], "swap_ins": sim["swap_ins"], "swap_outs": sim["swap_outs"]},
    }


def export_strategy(lab: StrategyLab, proposal: dict) -> dict:
    """Decision table a rules engine could consume. NOT written to strategy.yaml."""
    cat = lab.catalogue.set_index("id")
    cells = []
    for c in proposal["test_cells"]:
        row = cat.loc[c["segment_id"]]
        conds = [list(x) for x in _GROUP_RULES[row.group]] + \
                [[k.feature, k.op, k.value] for k in row.conds]
        cells.append({"cell_id": c["cell_id"], "conditions_all": conds,
                      "action": "approve" if c["kind"] == "swap_in" else "decline",
                      "allocation": {"method": "random", "share_pct": c["test_share_pct"]},
                      "stop_rule": c["stop_rule"]})
    return {"status": "PROPOSED - requires human approval before any change to strategy.yaml",
            "current_policy": plain(config.load("strategy")["current_policy"]),
            "test_cells": cells,
            "value_notes": "Feature values as treated in phase 1 (missing -> median, outliers capped)."}


def proposal_markdown(p: dict) -> str:
    e = p["expected_impact_if_rolled_out"]
    lines = [f"# Test proposal — {p['status']}", "", f"**Goal:** {p['goal']}",
             f"**Band:** {p['band']} → route to **{p['route']}**", "",
             "## Expected impact if rolled out (2015 out-of-time data)", "",
             "| Metric | Current policy | With swap set | Change |", "|---|---|---|---|",
             f"| Approval rate | {p['baseline']['approval_rate_pct']}% | {p['new']['approval_rate_pct']}% | "
             f"{e['delta_approval_rate_pp']:+} pp ({e['approval_growth_pct']:+}% approvals) |",
             f"| Approved bad rate | {p['baseline']['approved_bad_rate_pct']}% | {p['new']['approved_bad_rate_pct']}% | "
             f"{e['delta_bad_rate_pp']:+} pp (95% CI {e['delta_bad_rate_ci95_pp']}) |", "",
             f"Swap-ins bad rate {e['swap_in_bad_rate_pct']}%, swap-outs bad rate {e['swap_out_bad_rate_pct']}%.", "",
             f"Reject-inference stress (every swap-in 2 pp worse than measured): Δ bad rate "
             f"{e['stressed_delta_bad_rate_pp']:+} pp.", "",
             "## Volumes (2015, loans per year)", "", "| | Loans | % of current approvals |", "|---|---|---|",
             f"| Current approvals | {p['volumes']['current_approved']:,} | 100% |",
             f"| Swap-ins | +{p['volumes']['swap_in']:,} | {p['volumes']['swap_in_pct_of_current']}% |",
             f"| Swap-outs | -{p['volumes']['swap_out']:,} | {p['volumes']['swap_out_pct_of_current']}% |",
             f"| **Net change** | **{p['volumes']['net_change']:+,}** | **{p['volumes']['net_change_pct_of_current']:+}%** |",
             f"| Turnover (in + out) | {p['volumes']['swap_in'] + p['volumes']['swap_out']:,} | "
             f"{p['volumes']['turnover_pct_of_current']}% |", "",
             f"Swaps per net new approval: {p['volumes']['swaps_per_net_new_approval']}.", "",
             "## Test cells", "", "| Cell | Action | Rule | Loans/yr | Test loans/yr | Bad rate | ±95% at test size |",
             "|---|---|---|---|---|---|---|"]
    for c in p["test_cells"]:
        lines.append(f"| {c['cell_id']} | {c['action_in_test_cell']} | {c['rule']} | {c['segment_loans_per_year']:,} | "
                     f"{c['test_loans_per_year']:,} | {c['expected_bad_rate_pct']}% | ±{c['test_bad_rate_ci95_half_width_pp']} pp |")
    lines += ["", "## Limit checks", "", "| Check | Value | Limit | Status |", "|---|---|---|---|"]
    lines += [f"| {c['check']} | {c['value']} | {c['limit']} | {c['status']} |" for c in p["limit_checks"]]
    lines += ["", "## Success criteria", ""] + [f"- {s}" for s in p["success_criteria"]]
    d = p["duration"]
    lines += ["", "## Duration and monitoring", "",
              f"- Early read at {d['early_read_months']} months ({d['early_indicator']}); decision at "
              f"{d['decision_months']} months; full term {d['full_term_months']} months."]
    lines += [f"- {m}" for m in p["monitoring"]]
    lines += ["", "## Reject-inference caveat", "", p["reject_inference_caveat"]]
    return "\n".join(lines)
