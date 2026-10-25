"""The agent's tools: the ONLY functions Claude can call.

Every result gets an ID (R1, R2, ...) and is stored, so proposals and the number
verifier can trace each figure back to a tool output. There is deliberately no tool
that edits the risk appetite, edits strategy.yaml, deploys, or enables a test cell.
Two tools end a run: submit_for_approval and report_refusal.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import proposal as proposal_mod
from . import verify
from .audit import AuditLog
from .lab import StrategyLab, plain


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


_IDS = {"type": "array", "items": {"type": "string"}}

TOOL_DEFS = [
    {"name": "get_risk_appetite",
     "description": "Read the risk appetite set by the risk function: limits, escalation bands and definitions. "
                    "Read-only; nothing can change it. Call this first.",
     "input_schema": _schema({}, [])},
    {"name": "get_baseline",
     "description": "Current-policy baseline on the 2015 out-of-time year: applications, approvals, approval rate, "
                    "approved and declined bad rates, marginal approved bad rate, and the current policy rules.",
     "input_schema": _schema({}, [])},
    {"name": "find_segments",
     "description": "List candidate segments. kind='swap_in': currently declined segments with bad rate at or below "
                    "the marginal approved bad rate (found on 2012Q4-2014 data, confirmed on 2015). "
                    "kind='swap_out': currently approved segments with bad rate at or above it. Segments in the same "
                    "policy group (pocket) can overlap.",
     "input_schema": _schema({"kind": {"type": "string", "enum": ["swap_in", "swap_out"]},
                              "top_n": {"type": "integer", "description": "Max segments to return (1-30)."}},
                             ["kind", "top_n"])},
    {"name": "simulate_swap_set",
     "description": "Simulate a swap set on 2015 data versus the current policy: approval and bad-rate deltas with a "
                    "95% bootstrap interval, the reject-inference stress result, volumes and turnover, every limit "
                    "check, the escalation band (green/amber/red) and the approver route. Use IN-xx ids for swap-ins "
                    "and OUT-xx ids for swap-outs; either list may be empty.",
     "input_schema": _schema({"swap_in_ids": _IDS, "swap_out_ids": _IDS}, ["swap_in_ids", "swap_out_ids"])},
    {"name": "draft_test_proposal",
     "description": "Build the randomised test-cell proposal (cells, test sizes, stop rules, success criteria, "
                    "duration, monitoring, reject-inference caveat) from a previous simulate_swap_set result.",
     "input_schema": _schema({"simulation_result_id": {"type": "string"}, "goal": {"type": "string"}},
                             ["simulation_result_id", "goal"])},
    {"name": "export_strategy",
     "description": "Produce the decision table (rules -> action, test allocation, stop rule) for a drafted proposal, "
                    "in a form a decision engine could consume. Saved to the run folder; does NOT change the live "
                    "strategy.",
     "input_schema": _schema({"proposal_result_id": {"type": "string"}}, ["proposal_result_id"])},
    {"name": "submit_for_approval",
     "description": "Submit a drafted proposal for human approval and END the run. Rejected if the proposal's band is "
                    "red or if the recommendation contains a number that no tool result supports.",
     "input_schema": _schema({"proposal_result_id": {"type": "string"},
                              "recommendation": {"type": "string",
                                                 "description": "Note for the credit committee, at most 250 words."}},
                             ["proposal_result_id", "recommendation"])},
    {"name": "report_refusal",
     "description": "Decline a goal that cannot be met inside the risk appetite and END the run. Name the binding "
                    "limit ids, cite the simulations that show it, and give a counter-offer simulation (the most "
                    "growth found that is not red), or an empty string if none exists. The standard "
                    "reject-inference caveat is attached automatically; do not restate it.",
     "input_schema": _schema({"goal": {"type": "string"},
                              "binding_limit_ids": _IDS,
                              "evidence_simulation_ids": _IDS,
                              "counter_offer_simulation_id": {"type": "string"},
                              "explanation": {"type": "string",
                                              "description": "Explanation for the business, at most 200 words."}},
                             ["goal", "binding_limit_ids", "evidence_simulation_ids",
                              "counter_offer_simulation_id", "explanation"])},
]
for _t in TOOL_DEFS:
    _t["strict"] = True

TERMINAL = {"submit_for_approval", "report_refusal"}


class ToolError(Exception):
    """Returned to the agent as an is_error tool result."""


class Toolbox:
    def __init__(self, lab: StrategyLab, audit: AuditLog, run_dir: Path):
        self.lab, self.audit, self.run_dir = lab, audit, Path(run_dir)
        self.results: dict[str, dict] = {}
        self.terminal: dict | None = None

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            raise ToolError(f"unknown tool {name}")
        out = fn(**args)
        rid = f"R{len(self.results) + 1}"
        self.results[rid] = {"tool": name, "args": args, "result": out}
        return {"result_id": rid, **out}

    def _get(self, rid: str, tool: str) -> dict:
        r = self.results.get(rid)
        if r is None or r["tool"] != tool:
            raise ToolError(f"{rid} is not a {tool} result")
        return r["result"]

    def _untraceable(self, text: str, goal: str) -> list[str]:
        return verify.unmatched_cited(text, {k: v["result"] for k, v in self.results.items()}, goal)

    # ------------------------------------------------------------------ tools
    def _t_get_risk_appetite(self):
        return {"read_only": True, "appetite": plain(self.lab.appetite)}

    def _t_get_baseline(self):
        return self.lab.baseline("oot")

    def _t_find_segments(self, kind, top_n):
        top_n = max(1, min(int(top_n), 30))
        segs = self.lab.find_segments(kind, top_n)
        return {"kind": kind, "count": len(segs), "segments": segs}

    def _t_simulate_swap_set(self, swap_in_ids, swap_out_ids):
        try:
            return self.lab.simulate(list(swap_in_ids), list(swap_out_ids))
        except ValueError as e:
            raise ToolError(str(e))

    def _t_draft_test_proposal(self, simulation_result_id, goal):
        sim = self._get(simulation_result_id, "simulate_swap_set")
        p = proposal_mod.draft_test_proposal(self.lab, sim, goal)
        p["source"]["simulation_result_id"] = simulation_result_id
        return p

    def _t_export_strategy(self, proposal_result_id):
        p = self._get(proposal_result_id, "draft_test_proposal")
        exp = proposal_mod.export_strategy(self.lab, p)
        path = self.run_dir / f"strategy_export_{proposal_result_id}.json"
        path.write_text(json.dumps(exp, indent=1))
        return {"saved_to": path.name, "cells": len(exp["test_cells"]), "status": exp["status"]}

    def _t_submit_for_approval(self, proposal_result_id, recommendation):
        p = self._get(proposal_result_id, "draft_test_proposal")
        if p["band"] == "red":
            raise ToolError("Cannot submit a red-band proposal: it breaches a limit. Revise the swap set, "
                            "or use report_refusal.")
        bad = self._untraceable(recommendation, p["goal"])
        if bad:
            raise ToolError(f"Recommendation contains figures not found in the results cited in the same "
                            f"sentence: {bad}. Copy each figure exactly from a tool result and cite it, like [R5].")
        if len(recommendation.split()) > 300:
            raise ToolError("Recommendation is too long; keep it to 250 words.")
        exports = [r for r in self.results.values() if r["tool"] == "export_strategy"
                   and r["args"]["proposal_result_id"] == proposal_result_id]
        pending = {"status": "PENDING HUMAN APPROVAL", "proposal_result_id": proposal_result_id,
                   "band": p["band"], "route": p["route"], "recommendation": recommendation,
                   "proposal": p, "export_file": exports[-1]["result"]["saved_to"] if exports else None}
        (self.run_dir / "pending_approval.json").write_text(json.dumps(pending, indent=1))
        (self.run_dir / "proposal.md").write_text(
            proposal_mod.proposal_markdown(p) + "\n\n## Agent recommendation\n\n" + recommendation,
            encoding="utf-8")
        self.terminal = {"type": "submitted", **{k: pending[k] for k in ("band", "route", "proposal_result_id")}}
        return {"status": "PENDING HUMAN APPROVAL", "route": p["route"],
                "message": "Submitted. The run stops here; a human must approve or reject."}

    def _t_report_refusal(self, goal, binding_limit_ids, evidence_simulation_ids,
                          counter_offer_simulation_id, explanation):
        limit_ids = set(self.lab.appetite["limits"].keys())
        unknown = [l for l in binding_limit_ids if l not in limit_ids]
        if unknown or not binding_limit_ids:
            raise ToolError(f"binding_limit_ids must be limit ids from the appetite; unknown: {unknown}")
        evidence = [self._get(s, "simulate_swap_set") for s in evidence_simulation_ids]
        if not evidence:
            raise ToolError("cite at least one simulation as evidence")
        counter = None
        if counter_offer_simulation_id:
            counter = self._get(counter_offer_simulation_id, "simulate_swap_set")
            if counter["band"] == "red":
                raise ToolError("The counter-offer simulation is red; a counter-offer must be inside the appetite.")
        bad = self._untraceable(explanation, goal)
        if bad:
            raise ToolError(f"Explanation contains figures not found in the results cited in the same "
                            f"sentence: {bad}. Copy each figure exactly and cite it, like [R5].")
        refusal = {"status": "REFUSED", "goal": goal, "binding_limit_ids": binding_limit_ids,
                   "evidence_simulation_ids": evidence_simulation_ids,
                   "counter_offer_simulation_id": counter_offer_simulation_id or None,
                   "counter_offer": {k: counter[k] for k in ("approval_growth_pct", "delta_bad_rate_pp",
                                                              "stressed_delta_bad_rate_pp", "band", "swap_ins",
                                                              "swap_outs")} if counter else None,
                   "explanation": explanation,
                   "reject_inference_caveat": proposal_mod.REJECT_INFERENCE_CAVEAT}
        (self.run_dir / "refusal.json").write_text(json.dumps(refusal, indent=1))
        self.terminal = {"type": "refused", "binding_limit_ids": binding_limit_ids,
                         "counter_offer_simulation_id": counter_offer_simulation_id or None}
        return {"status": "REFUSED", "message": "Refusal recorded. The run stops here."}
