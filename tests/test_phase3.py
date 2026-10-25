"""Phase 3: tool surface, number verifier, terminal-tool gates, agent loop (fake client), approval gate."""
import json
import shutil
from types import SimpleNamespace as NS

import pytest
import yaml

from csa import agent, config, verify
from csa.approval import ApprovalError, record_decision
from csa.tools import TOOL_DEFS


# ---------------------------------------------------------------- no data needed
def test_tool_surface_has_no_writer_or_deploy_tool():
    names = {t["name"] for t in TOOL_DEFS}
    assert names == {"get_risk_appetite", "get_baseline", "find_segments", "simulate_swap_set",
                     "draft_test_proposal", "export_strategy", "submit_for_approval", "report_refusal"}
    for t in TOOL_DEFS:
        assert t["strict"] is True
        assert t["input_schema"]["additionalProperties"] is False


def test_verifier():
    known = verify.numbers_in({"g": 3.197, "d": -0.007, "rule": "annual_inc > 60,000", "n": 201133})
    assert verify.unmatched("Growth +3.2% [R5], bad rate fell 0.007 pp, IN-04 and OUT-12.", known) == []
    assert verify.unmatched("Approvals 201,133; income above 60,000.", known) == []
    assert verify.unmatched("Growth 3.25%", known) == ["3.25"]          # more precise than the source
    assert verify.unmatched("About 4.1% growth", known) == ["4.1"]      # invented
    assert verify.unmatched("3 swap-ins over 12 months in 2015", known) == ["12"]


def test_verifier_checks_against_cited_results_only():
    results = {"R1": {"growth": 3.2}, "R2": {"growth": 4.43}}
    assert verify.unmatched_cited("Growth 3.2% [R1].", results) == []
    assert verify.unmatched_cited("Growth 4.4% [R1].", results) == ["4.4"]          # true in R2, not in R1
    assert verify.unmatched_cited("Growth 3.2%.", results) == ["3.2 (uncited)"]
    assert verify.unmatched_cited("Test 3 segments.", results) == []                 # small counts allowed
    assert verify.unmatched_cited("Growth 3.2% [R9].", results) == ["unknown citation R9", "3.2 (uncited)"]
    assert verify.unmatched_cited("The 15% goal is not reachable.", results, goal="Grow approvals 15%") == []


# ---------------------------------------------------------------- fake Claude client
def _tool(i, name, **inp):
    return NS(type="tool_use", id=f"tu{i}", name=name, input=inp)


def _resp(*blocks, stop="tool_use"):
    usage = NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return NS(content=list(blocks), stop_reason=stop, usage=usage, model="fake", stop_details=None)


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        step = self.script.pop(0)
        return step(kw) if callable(step) else step


@pytest.fixture(scope="module")
def lab():
    if not config.path("processed_file").exists():
        pytest.skip("run scripts/run_phase1.py first")
    from csa.lab import StrategyLab
    lab = StrategyLab(); lab.build_catalogue()
    return lab


def _normal_script(lab, bad_first=False):
    sim = lab.simulate(["IN-04", "IN-07", "IN-11"], ["OUT-12"])
    good = f"Test IN-04, IN-07, IN-11 and OUT-12: growth {sim['approval_growth_pct']}% [R5]."
    s = [
        _resp(NS(type="text", text="Reading the appetite and baseline."),
              _tool(1, "get_risk_appetite"), _tool(2, "get_baseline")),
        _resp(_tool(3, "find_segments", kind="swap_in", top_n=15), _tool(4, "find_segments", kind="swap_out", top_n=15)),
        _resp(NS(type="thinking", thinking="Lean set with LTI pocket for diversity."),
              _tool(5, "simulate_swap_set", swap_in_ids=["IN-04", "IN-07", "IN-11"], swap_out_ids=["OUT-12"])),
        _resp(_tool(6, "draft_test_proposal", simulation_result_id="R5", goal="grow 3-5%")),
        _resp(_tool(7, "export_strategy", proposal_result_id="R6")),
    ]
    if bad_first:
        s.append(_resp(_tool(8, "submit_for_approval", proposal_result_id="R6",
                             recommendation="Growth of about 4.4% expected.")))
    s.append(_resp(_tool(9, "submit_for_approval", proposal_result_id="R6", recommendation=good)))
    return s


def test_agent_normal_run_stops_at_gate(lab, tmp_path):
    fake = FakeClient(_normal_script(lab, bad_first=True))
    summary = agent.run("grow 3-5%", tmp_path, client=fake, lab=lab, echo=lambda *_: None)
    assert summary["outcome"]["type"] == "submitted"
    assert summary["outcome"]["band"] == "green"
    assert not fake.script, "agent should stop right after the successful submit"
    kw = fake.calls[0]
    assert kw["model"] == config.load("settings")["agent"]["model"]
    assert kw["thinking"]["type"] == "adaptive" and "tool_choice" not in kw
    events = [json.loads(l)["event"] for l in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert "tool_error" in events                    # the untraceable 4.4% was rejected
    assert "agent_note" in events
    pending = json.loads((tmp_path / "pending_approval.json").read_text())
    assert pending["status"] == "PENDING HUMAN APPROVAL" and pending["export_file"]
    # the full assistant content is appended unchanged (append-only conversation)
    assert fake.calls[1]["messages"][1]["content"][0].text == "Reading the appetite and baseline."


def test_red_plan_cannot_be_submitted_and_refusal_needs_valid_counter_offer(lab, tmp_path):
    from csa import reference
    g = reference.greedy(lab, 15.0, 0.2)
    red = lab.simulate(g["ins"], g["outs"])
    assert red["band"] == "red"
    script = [
        _resp(_tool(1, "simulate_swap_set", swap_in_ids=g["ins"], swap_out_ids=g["outs"])),      # R1 red
        _resp(_tool(2, "draft_test_proposal", simulation_result_id="R1", goal="grow 15%")),       # R2
        _resp(_tool(3, "submit_for_approval", proposal_result_id="R2", recommendation="ok")),     # rejected
        _resp(_tool(4, "report_refusal", goal="grow 15%", binding_limit_ids=["portfolio_bad_rate_delta_pp"],
                    evidence_simulation_ids=["R1"], counter_offer_simulation_id="R1", explanation="no")),  # red counter
        _resp(_tool(5, "simulate_swap_set", swap_in_ids=["IN-04", "IN-07", "IN-11"], swap_out_ids=["OUT-12"])),  # R3
        _resp(_tool(6, "report_refusal", goal="grow 15%",
                    binding_limit_ids=["stressed_portfolio_bad_rate_delta_pp"], evidence_simulation_ids=["R1"],
                    counter_offer_simulation_id="R3", explanation="Not feasible inside the appetite.")),
    ]
    fake = FakeClient(script)
    summary = agent.run("grow 15%", tmp_path, client=fake, lab=lab, echo=lambda *_: None)
    assert summary["outcome"]["type"] == "refused"
    errors = [json.loads(l) for l in (tmp_path / "audit.jsonl").read_text().splitlines()
              if json.loads(l)["event"] == "tool_error"]
    assert len(errors) == 2
    assert not (tmp_path / "pending_approval.json").exists()


def test_agent_that_never_submits_is_stopped(lab, tmp_path):
    fake = FakeClient([_resp(NS(type="text", text="done"), stop="end_turn"),
                       _resp(NS(type="text", text="still done"), stop="end_turn")])
    s = agent.run("x", tmp_path, client=fake, lab=lab, echo=lambda *_: None)
    assert s["outcome"]["type"] == "ended_without_submission"


# ---------------------------------------------------------------- approval gate
@pytest.fixture
def pending_run(lab, tmp_path):
    run = tmp_path / "run1"
    fake = FakeClient(_normal_script(lab))
    agent.run("grow 3-5%", run, client=fake, lab=lab, echo=lambda *_: None)
    strat = tmp_path / "strategy.yaml"
    shutil.copy(config.CONFIG_DIR / "strategy.yaml", strat)
    return run, strat


def test_approval_writes_only_the_given_strategy_file(pending_run):
    run, strat = pending_run
    before_real = (config.CONFIG_DIR / "strategy.yaml").read_text()
    n_before = len(yaml.safe_load(strat.read_text()).get("test_cells") or [])
    out = record_decision(run, "approve", "Test Approver", "credit_manager", strategy_path=strat)
    assert out["status"] == "APPROVED"
    s = yaml.safe_load(strat.read_text())
    assert len(s["test_cells"]) == n_before + 1 and s["test_cells"][-1]["cells"]
    assert s["current_policy"] == yaml.safe_load(before_real)["current_policy"]   # policy untouched
    assert (config.CONFIG_DIR / "strategy.yaml").read_text() == before_real       # real file untouched
    with pytest.raises(ApprovalError):
        record_decision(run, "approve", "Again", "senior_risk", strategy_path=strat)


def test_amber_needs_senior_risk_and_reject_changes_nothing(pending_run):
    run, strat = pending_run
    p = json.loads((run / "pending_approval.json").read_text())
    p.update(band="amber", route="senior_risk")
    (run / "pending_approval.json").write_text(json.dumps(p))
    with pytest.raises(ApprovalError):
        record_decision(run, "approve", "Junior", "credit_manager", strategy_path=strat)
    before = strat.read_text()
    out = record_decision(run, "reject", "Senior", "senior_risk", note="not now", strategy_path=strat)
    assert out["status"] == "REJECTED" and strat.read_text() == before
