"""Credit Strategy Agent: Claude plans, calls deterministic tools, writes the proposal, stops.

Manual tool-use loop on the Anthropic SDK. The conversation is append-only (full
response content, thinking blocks included, is passed back unchanged).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config
from .audit import AuditLog
from .lab import StrategyLab
from .tools import TERMINAL, TOOL_DEFS, Toolbox, ToolError

SYSTEM_PROMPT = """\
You are a credit strategy agent for the credit risk team of an unsecured instalment-loan (pay-later style) portfolio. \
This is a personal prototype on public Lending Club data (36-month loans issued 2012Q4-2015). All results are relative \
to the current policy; absolute rates are not comparable to any real lender.

Your job: given a business goal, design a swap set - swap-ins (currently declined segments to approve) and swap-outs \
(currently approved segments to decline) - that meets the goal inside the risk appetite. Then propose a small \
randomised test cell, never a rollout, and stop for human approval. If the goal cannot be met inside the appetite, \
refuse, name the binding limits, and make a counter-offer: the largest growth you can find whose band is not red.

How to work:
- First call get_risk_appetite and get_baseline. Then, before any other tool call, state in a few lines how you read \
the goal as targets (net approval growth, change in bad rate) and your plan.
- Before each later tool call, say in one line what you are testing and why.
- Use find_segments for both kinds, choose candidates deliberately, and simulate. Read every limit check. Revise and \
re-simulate until the band is green if you can; amber is acceptable only if you explain why green is not reachable.
- Prefer lean plans: low turnover (few swaps per net new approval), few segments, swap-ins well below the marginal \
approved bad rate. Overlapping segments in the same pocket add little volume.
- The risk appetite is set by risk and is read-only. You cannot change limits, deploy, or enable test cells, and you \
must never suggest relaxing a limit to meet a goal.
- When you have a plan: draft_test_proposal, export_strategy, then submit_for_approval with a recommendation for the \
credit committee of at most 250 words: what to test, expected impact, the band and who must approve, the \
reject-inference caveat, and whether the test cells are large enough to read (use the plus/minus 95% figures).
- If the goal is infeasible: keep the simulations that show it, search for the largest growth that is not red, and \
call report_refusal. Name only limits that the evidence shows are binding.
- On reject inference, be exact: in this public data every loan's outcome is observed, which is how swap-ins can be \
measured; in production, declined applicants' outcomes would not be observed, so swap-in bad rates would be estimates.

Number rule (strict): every number you write must be copied from a tool result - never compute, re-round, or estimate. \
Cite the result id in the same sentence as each figure, like [R4]; the submission is rejected if a figure is uncited \
or is not in the result you cite. Refer to segments by id (IN-04, OUT-12). If you need a number no tool \
gives you, say so instead. The run ends when you call submit_for_approval or report_refusal.
"""


def _client():
    import anthropic  # imported here so tests can run without a key
    from dotenv import load_dotenv
    load_dotenv(config.ROOT / ".env")
    return anthropic.Anthropic()


def _block_dict(b) -> dict:
    return b.model_dump() if hasattr(b, "model_dump") else dict(vars(b))


def run(goal: str, run_dir: Path, client=None, lab: StrategyLab | None = None, echo=print) -> dict:
    cfg = config.load("settings")["agent"]
    audit = AuditLog(run_dir)
    lab = lab or StrategyLab()
    if lab.catalogue is None:
        lab.build_catalogue()
    tools = Toolbox(lab, audit, run_dir)
    client = client or _client()
    audit.log("run_start", goal=goal, model=cfg["model"], effort=cfg["effort"],
              appetite_version=lab.appetite["version"])
    messages = [{"role": "user", "content": f"Business goal: {goal}"}]
    usage_tot = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0}
    t0, outcome, nudged = time.time(), None, False

    for call_no in range(1, cfg["max_api_calls"] + 1):
        resp = client.beta.messages.create(
            model=cfg["model"], max_tokens=cfg["max_tokens"], system=SYSTEM_PROMPT, tools=TOOL_DEFS,
            messages=messages, thinking={"type": "adaptive", "display": "updates"},
            output_config={"effort": cfg["effort"]}, cache_control={"type": "ephemeral"},
            betas=list(cfg["betas"]), fallbacks="default")
        u = resp.usage
        call_usage = {k: getattr(u, k, 0) or 0 for k in usage_tot}
        for k, v in call_usage.items():
            usage_tot[k] += v
        audit.log("api_call", n=call_no, stop_reason=resp.stop_reason, model=getattr(resp, "model", None),
                  usage=call_usage)
        messages.append({"role": "assistant", "content": resp.content})   # append-only, unchanged

        tool_uses = []
        for b in resp.content:
            if b.type == "thinking" and getattr(b, "thinking", ""):
                audit.log("agent_note", text=b.thinking); echo(f"  · {b.thinking}")
            elif b.type == "text" and b.text.strip():
                audit.log("agent_text", text=b.text); echo(f"\nAGENT: {b.text}\n")
            elif b.type == "tool_use":
                tool_uses.append(b)
            elif b.type == "fallback":
                audit.log("model_fallback", block=_block_dict(b))

        if resp.stop_reason == "refusal":
            audit.log("api_refusal", stop_details=_block_dict(resp.stop_details) if resp.stop_details else None)
            outcome = {"type": "api_refusal"}; break
        if resp.stop_reason == "max_tokens":
            audit.log("error", message="response hit max_tokens"); outcome = {"type": "max_tokens"}; break
        if resp.stop_reason == "pause_turn":
            continue
        if not tool_uses:
            if nudged:                                    # ended twice without the required final tool
                outcome = {"type": "ended_without_submission"}; break
            nudged = True
            messages.append({"role": "user", "content": "Please finish by calling submit_for_approval or "
                                                        "report_refusal."})
            audit.log("harness_nudge", reason="ended turn without a terminal tool")
            continue

        results = []
        for tu in tool_uses:
            args = dict(tu.input) if not isinstance(tu.input, str) else json.loads(tu.input)
            audit.log("tool_call", tool=tu.name, input=args, tool_use_id=tu.id)
            echo(f"  -> {tu.name}({json.dumps(args)[:160]})")
            try:
                out = tools.call(tu.name, args)
                audit.log("tool_result", tool=tu.name, result_id=out["result_id"], result=out)
                echo(f"     {out['result_id']}: {_short(tu.name, out)}")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(out)})
            except ToolError as e:
                audit.log("tool_error", tool=tu.name, error=str(e))
                echo(f"     ERROR: {e}")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": str(e), "is_error": True})
        messages.append({"role": "user", "content": results})
        if tools.terminal:
            outcome = tools.terminal; break
    else:
        outcome = {"type": "max_api_calls"}

    price = cfg["pricing_per_mtok"]
    cost = (usage_tot["input_tokens"] * price["input"] + usage_tot["output_tokens"] * price["output"]
            + usage_tot["cache_read_input_tokens"] * price["cache_read"]
            + usage_tot["cache_creation_input_tokens"] * price["cache_write"]) / 1e6
    summary = {"goal": goal, "outcome": outcome, "api_calls": call_no, "tool_results": len(tools.results),
               "elapsed_s": round(time.time() - t0, 1), "usage": usage_tot, "est_cost_usd": round(cost, 3)}
    audit.log("run_end", **summary)
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _short(name: str, out: dict) -> str:
    if name == "simulate_swap_set":
        return (f"growth {out['approval_growth_pct']}%, dBad {out['delta_bad_rate_pp']} pp "
                f"(stress {out['stressed_delta_bad_rate_pp']}), turnover {out['volumes']['turnover_pct_of_current']}%,"
                f" band {out['band']}")
    if name == "find_segments":
        return f"{out['count']} {out['kind']} segments"
    return ", ".join(f"{k}={v}" for k, v in out.items() if isinstance(v, (str, int, float)) and k != "result_id")[:160]
