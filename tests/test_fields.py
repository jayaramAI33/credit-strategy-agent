"""Field register: every raw column has exactly one role; nothing blocked can be a candidate."""
import csv
import gzip
from collections import Counter

import pytest

from csa import config


def _all_listed():
    f = config.load("fields")
    roles = [v for v in f["roles"].values() if v]
    cands = [c for group in f["candidates"].values() for c in group]
    blocked = [c for group in f["blocked"].values() for c in group]
    return roles, cands, blocked


def test_no_field_listed_twice():
    roles, cands, blocked = _all_listed()
    dupes = [k for k, n in Counter(roles + cands + blocked).items() if n > 1]
    assert not dupes, f"listed more than once: {dupes}"


def test_register_matches_raw_header():
    raw = config.path("raw_file")
    if not raw.exists():
        pytest.skip("raw file not present")
    with gzip.open(raw, "rt", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    roles, cands, blocked = _all_listed()
    listed = set(roles + cands + blocked)
    assert set(header) - listed == set(), f"unregistered columns: {set(header) - listed}"
    assert listed - set(header) == set(), f"registered but not in file: {listed - set(header)}"


def test_derived_sources_are_candidates_or_roles():
    f = config.load("fields")
    _, cands, blocked = _all_listed()
    ok = set(cands) | {"issue_d"}
    for name, spec in f["derived"].items():
        for src in spec["from"]:
            assert src in ok, f"{name} uses {src}, which is not an application-time field"
            assert src not in blocked


def test_policy_variables_are_not_blocked():
    _, _, blocked = _all_listed()
    assert not set(config.load("fields")["policy_variables"]) & set(blocked)
