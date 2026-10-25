"""Phase 4 scorer on a scripted (fake-client) run, and the benchmark's basic invariants."""
import json

import pytest

from csa import agent, config, scoring

KEY = config.ROOT / "outputs" / "phase4" / "answer_key.json"


@pytest.fixture(scope="module")
def lab():
    if not config.path("processed_file").exists():
        pytest.skip("run scripts/run_phase1.py first")
    from csa.lab import StrategyLab
    lab = StrategyLab(); lab.build_catalogue()
    return lab


def test_scorer_passes_good_run_and_flags_unmet_goal(lab, tmp_path):
    if not KEY.exists():
        pytest.skip("run scripts/eval_goals.py --keys first")
    from test_phase3 import FakeClient, _normal_script
    keys = json.loads(KEY.read_text())["goals"]
    goals = {g["id"]: dict(g) for g in config.load("eval_goals")["goals"]}
    run = tmp_path / "r"
    agent.run("x", run, client=FakeClient(_normal_script(lab)), lab=lab, echo=lambda *_: None)
    good = scoring.score_run(goals["F2"], keys["F2"], run)     # +3.2%, delta -0.007: meets F2
    assert good["checks"]["goal_met"] and good["checks"]["outcome_correct"]
    unmet = scoring.score_run(goals["F3"], keys["F3"], run)    # F3 needs >= 4% growth
    assert unmet["checks"]["goal_met"] is False and unmet["passed"] is False
    wrong = scoring.score_run(goals["I1"], keys["I1"], run)    # submitting an infeasible goal = wrong outcome
    assert wrong["checks"]["outcome_correct"] is False


def test_benchmark_key_matches_simulator(lab):
    from csa.benchmark import Benchmark
    bm = Benchmark(lab, max_outs=1)
    k = bm.key(3, 0)
    plan = k["leanest_qualifying_plan"]
    sim = lab.simulate(plan["ins"], plan["outs"], with_ci=False)
    assert abs(sim["approval_growth_pct"] - plan["growth_pct"]) < 0.01
    assert abs(sim["delta_bad_rate_pp"] - plan["delta_bad_pp"]) < 0.001
    assert sim["band"] != "red"
