"""Phase 2: read-only appetite, simulation logic, limit bands, proposal traceability."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csa import config, proposal

SRC = Path(__file__).resolve().parents[1] / "src" / "csa"


# ---------------------------------------------------------------- no data needed
def test_appetite_is_frozen_in_memory():
    ap = config.load("risk_appetite")
    with pytest.raises(TypeError):
        ap["limits"]["portfolio_bad_rate_delta_pp"]["max"] = 99


def test_no_code_writes_the_risk_appetite_or_any_yaml():
    # Phase 3's approval.py will be the ONLY writer, and only of strategy.yaml.
    for f in SRC.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "risk_appetite.yaml" not in text, f"{f.name} references the appetite file directly"
        if f.name != "approval.py":
            assert not re.search(r"yaml\.(safe_)?dump", text), f"{f.name} writes YAML"


# ---------------------------------------------------------------- needs data
@pytest.fixture(scope="module")
def lab():
    if not config.path("processed_file").exists():
        pytest.skip("run scripts/run_phase1.py first")
    from csa.lab import StrategyLab
    lab = StrategyLab()
    lab.build_catalogue()
    return lab


def test_empty_swap_set_is_neutral(lab):
    s = lab.simulate([], [])
    assert s["delta_approval_rate_pp"] == 0 and s["delta_bad_rate_pp"] == 0
    assert s["band"] == "green"


def test_marginal_rate_regression(lab):
    assert abs(100 * lab.marginal["build"] - 16.89) < 0.01
    assert abs(100 * lab.marginal["oot"] - 20.13) < 0.01


def test_swap_ins_only_touch_declined_and_outs_only_approved(lab):
    s = lab.simulate(["IN-01"], [], with_ci=False)
    assert s["swapped_out_loans"] == 0
    n = s["baseline"]  # approval rises by exactly the swapped-in loans
    oot = lab.split["oot"].sum()
    assert abs(s["delta_approval_rate_pp"] - 100 * s["swapped_in_loans"] / oot) < 1e-3
    s2 = lab.simulate([], ["OUT-01"], with_ci=False)
    assert s2["swapped_in_loans"] == 0 and s2["delta_approval_rate_pp"] < 0


def test_overlapping_swap_ins_count_once(lab):
    a = lab.simulate(["IN-01"], [], with_ci=False)
    b = lab.simulate(["IN-01", "IN-03"], [], with_ci=False)    # IN-03 is a subset of IN-01
    assert a["swapped_in_loans"] == b["swapped_in_loans"]


def test_listed_segments_respect_marginal_rule_on_both_splits(lab):
    c = lab.catalogue[lab.catalogue.listed]
    mb, mo = 100 * lab.marginal["build"], 100 * lab.marginal["oot"]
    ins, outs = c[c.kind == "swap_in"], c[c.kind == "swap_out"]
    assert (ins.bad_build_pct <= mb).all() and (ins.bad_oot_pct <= mo).all()
    assert (outs.bad_build_pct >= mb).all() and (outs.bad_oot_pct >= mo).all()
    assert (c.vol_build_pct >= 1).all() and (c.vol_oot_pct >= 1).all()


def test_bad_ids_rejected(lab):
    with pytest.raises(ValueError):
        lab.simulate(["IN-99"], [])
    with pytest.raises(ValueError):
        lab.simulate(["OUT-01"], [])


def test_bootstrap_is_deterministic(lab):
    a = lab.simulate(["IN-01"], ["OUT-01"])
    b = lab.simulate(["IN-01"], ["OUT-01"])
    assert a["delta_bad_rate_ci95_pp"] == b["delta_bad_rate_ci95_pp"]


def test_band_logic(lab):
    chk = lab._check
    assert chk("x", 0.21, "max", 0.2, "l")["status"] == "breach"
    assert chk("x", 0.19, "max", 0.2, "l")["status"] == "near"
    assert chk("x", 0.17, "max", 0.2, "l")["status"] == "pass"
    assert chk("x", 0.9, "min", 1.0, "l")["status"] == "breach"
    assert chk("x", 1.05, "min", 1.0, "l")["status"] == "near"


def test_stress_formula(lab):
    s = lab.simulate(["IN-01"], [], with_ci=False)
    v = s["volumes"]
    expected_gap = 100 * 0.02 * v["swap_in"] / v["new_approved"]
    assert abs((s["stressed_delta_bad_rate_pp"] - s["delta_bad_rate_pp"]) - expected_gap) < 2e-3


def test_volumes_add_up(lab):
    v = lab.simulate(["IN-01", "IN-05"], ["OUT-01"], with_ci=False)["volumes"]
    assert v["new_approved"] == v["current_approved"] + v["swap_in"] - v["swap_out"]
    assert v["net_change"] == v["swap_in"] - v["swap_out"]


def test_benchmark_lean_plan_is_green(lab):
    # Leanest plan found by scripts/benchmark_search.py for goal A (2026-10-22).
    s = lab.simulate(["IN-04", "IN-07", "IN-11"], ["OUT-12"])
    assert s["band"] == "green"
    assert s["approval_growth_pct"] >= 3 and s["delta_bad_rate_pp"] <= 0


def test_high_turnover_plan_fails_stress(lab):
    from csa import reference
    g = reference.greedy(lab, 3.0, 0.0)
    s = lab.simulate(g["ins"], g["outs"], with_ci=False)
    stress = next(c for c in s["limit_checks"] if c["limit_id"] == "stressed_portfolio_bad_rate_delta_pp")
    assert stress["status"] == "breach" and s["band"] == "red"


def test_proposal_numbers_trace_to_simulation(lab):
    sim = lab.simulate(["IN-01", "IN-05"], ["OUT-01"])
    p = proposal.draft_test_proposal(lab, sim, "test goal")
    assert p["status"] == "PENDING HUMAN APPROVAL"
    assert p["reject_inference_caveat"]
    for k, v in p["expected_impact_if_rolled_out"].items():
        assert v == sim[k]
    seg = {s["segment_id"]: s for s in sim["segments"]}
    for c in p["test_cells"]:
        assert c["expected_bad_rate_pct"] == seg[c["segment_id"]]["bad_rate_pct"]
        assert c["test_share_pct"] <= config.load("risk_appetite")["limits"]["test_cell_share_pct"]["max"]


def test_export_conditions_reproduce_segments(lab):
    sim = lab.simulate(["IN-01", "IN-05"], ["OUT-01"], with_ci=False)
    exp = proposal.export_strategy(lab, proposal.draft_test_proposal(lab, sim, "g"))
    frame = lab.t.copy()
    for col in ("dti", "loan_to_income", "grade"):
        frame[col] = lab.df[col]
    ops = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}
    for cell in exp["test_cells"]:
        m = np.ones(len(frame), bool)
        for f, op, v in cell["conditions_all"]:
            x = frame[f]
            if op == "==":
                m &= (x == v).to_numpy()
            elif op == "in":
                m &= x.isin(v).to_numpy()
            else:
                m &= ops[op](x.to_numpy(dtype=float), v)
        seg = cell["cell_id"].removeprefix("TC-")
        assert (m == lab._masks[seg]).all(), f"{seg}: export rule does not match the segment"
