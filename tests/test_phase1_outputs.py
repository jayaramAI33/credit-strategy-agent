"""Phase 1 pass conditions, checked on the generated outputs."""
import json

import pandas as pd
import pytest

from csa import config

OUT = config.path("phase1_out")
pytestmark = pytest.mark.skipif(not (OUT / "phase1_metrics.json").exists(),
                                reason="run scripts/run_phase1.py first")

# Expected from an independent stdlib count of 2026-10-22
# (36-month, resolved, issued 2012Q4-2015Q4; build 2012Q4-2014Q4 = 278,173, OOT 2015 = 283,026).
EXPECTED_ROWS = 561_199
EXPECTED_BAD = {"build": 13.17, "oot": 14.89}


def _m():
    return json.loads((OUT / "phase1_metrics.json").read_text())


def test_counts_match_independent_check():
    m = _m()
    assert m["rows"] == EXPECTED_ROWS
    for split, v in EXPECTED_BAD.items():
        assert abs(m["bad_pct"][split] - v) < 0.01


def test_no_blocked_or_role_field_downstream():
    blocked = set(config.blocked_fields())
    roles = set(config.role_fields())
    log = pd.read_csv(OUT / "screening_log.csv")
    iv = pd.read_csv(OUT / "iv_table.csv")
    for name, used in (("screening", set(log.variable)), ("iv", set(iv.variable))):
        assert not used & blocked, f"blocked field in {name}: {used & blocked}"
        assert not used & roles, f"role field in {name}: {used & roles}"


def test_top_features_pass_iv_and_hold_oot():
    iv = pd.read_csv(OUT / "iv_table.csv").set_index("variable")
    top = _m()["top_features"]
    assert len(top) == config.load("prep")["selection"]["top_n"]
    for v in top:
        assert iv.loc[v, "iv_build"] >= 0.02 and iv.loc[v, "iv_oot"] >= 0.02
        assert iv.loc[v, "monotonic_oot"] in (True, "True") or iv.loc[v, "kind"] == "categorical"
        assert v not in config.load("fields")["policy_variables"]


def test_baseline_approval_oot_near_71():
    b = _m()["baseline"]["oot"]
    assert abs(b["approval_pct"] - 71) <= 2
