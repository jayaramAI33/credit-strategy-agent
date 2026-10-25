"""Number traceability: every number the agent writes must appear in a tool result.

Rules (deliberately simple and conservative):
- IDs (IN-04, OUT-12, TC-IN-04, R7) and years 2012-2015 are not treated as figures.
- A figure matches a tool-result number if equal after allowing for the figure's own
  rounding (e.g. "3.2" matches 3.197; "3.25" does not match 3.2). Sign is ignored, so
  "fell 0.05 pp" can cite -0.05.
- Integers 0-10 written without decimals are allowed (counts such as "3 swap-ins").
"""
from __future__ import annotations

import re

_ID = re.compile(r"\b(?:TC-)?(?:IN|OUT)-\d+\b|\[?\bR\d+\b\]?")
_NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?")
_ALLOWED = {2012.0, 2013.0, 2014.0, 2015.0, 95.0}


def numbers_in(obj) -> set[float]:
    """All numbers in a nested tool result, including numbers inside strings."""
    out: set[float] = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        out |= {abs(v) for v, _ in _parse(_ID.sub(" ", obj))}
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= numbers_in(v)
    return out


def _parse(text: str):
    for m in _NUM.finditer(text):
        s = m.group().replace("−", "-").replace(",", "").lstrip("+")
        try:
            v = float(s)
        except ValueError:
            continue
        decimals = len(s.split(".")[1]) if "." in s else 0
        yield v, decimals


_SENT = re.compile(r"(?<=[.;!?])\s+|\n+")
_CITE = re.compile(r"\bR\d+\b")


def unmatched_cited(text: str, results: dict[str, object], goal: str = "") -> list[str]:
    """Stricter check: each figure must match a number in the results cited in the SAME
    sentence ([R5]). A figure in a sentence with no citation is rejected as uncited.
    Limitation: this proves a figure exists in the cited result, not that it is used with
    the right meaning; headline figures get a semantic check in the phase 4 evaluation."""
    from_goal = numbers_in(goal)          # figures the user wrote in the goal may be repeated
    bad = []
    for sent in _SENT.split(text):
        cited = _CITE.findall(sent)
        bad += [f"unknown citation {c}" for c in cited if c not in results]
        scope = [c for c in cited if c in results]
        if not scope:
            bad += [f"{x} (uncited)" for x in unmatched(sent, from_goal)]
            continue
        bad += unmatched(sent, numbers_in([results[i] for i in scope]) | from_goal)
    return bad


def unmatched(text: str, known: set[float]) -> list[str]:
    """Figures in text that no tool result supports."""
    bad = []
    for v, d in _parse(_ID.sub(" ", text)):
        a = abs(v)
        if a in _ALLOWED or (d == 0 and a <= 10):
            continue
        tol = 0.5 * 10 ** (-d) + 1e-9
        if not any(abs(a - k) <= tol for k in known):
            bad.append(f"{v:g}" if d == 0 else f"{v:.{d}f}")
    return bad
