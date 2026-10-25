"""Human approval gate: the ONLY code that writes config/strategy.yaml.

Called by a person (scripts/approve.py), never by the agent. The approver's role must
match the band's route: green -> credit_manager or senior_risk; amber -> senior_risk.
Approval enables the TEST CELLS only; the current policy is never changed here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import config
from .audit import AuditLog

ROLE_RANK = {"credit_manager": 1, "senior_risk": 2}
ROUTE_MIN_ROLE = {"credit_manager": 1, "senior_risk": 2}


class ApprovalError(Exception):
    pass


def record_decision(run_dir: Path, decision: str, approver: str, role: str, note: str = "",
                    strategy_path: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    strategy_path = Path(strategy_path or config.CONFIG_DIR / "strategy.yaml")
    pending_file = run_dir / "pending_approval.json"
    if not pending_file.exists():
        raise ApprovalError("no pending proposal in this run")
    pending = json.loads(pending_file.read_text())
    if pending["status"] != "PENDING HUMAN APPROVAL":
        raise ApprovalError(f"proposal is already {pending['status']}")
    if decision not in ("approve", "reject"):
        raise ApprovalError("decision must be approve or reject")
    if role not in ROLE_RANK:
        raise ApprovalError(f"role must be one of {list(ROLE_RANK)}")
    if pending["route"] == "refuse":
        raise ApprovalError("a refused proposal cannot be approved")
    if decision == "approve" and ROLE_RANK[role] < ROUTE_MIN_ROLE[pending["route"]]:
        raise ApprovalError(f"band {pending['band']} needs {pending['route']} approval; you are {role}")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    audit = AuditLog(run_dir)
    if decision == "approve":
        export = json.loads((run_dir / pending["export_file"]).read_text()) if pending.get("export_file") else None
        if export is None:
            raise ApprovalError("proposal has no exported decision table")
        strategy = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))
        strategy.setdefault("test_cells", [])
        if any(tc.get("run") == run_dir.name for tc in strategy["test_cells"]):
            raise ApprovalError("this run's test cells are already in the strategy")
        strategy["test_cells"].append({
            "run": run_dir.name, "approved_by": approver, "role": role, "approved_at": now,
            "band": pending["band"], "status": "enabled (test cells only)", "note": note,
            "cells": export["test_cells"]})
        strategy_path.write_text(yaml.safe_dump(strategy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    pending.update({"status": "APPROVED" if decision == "approve" else "REJECTED",
                    "decided_by": approver, "role": role, "decided_at": now, "note": note})
    pending_file.write_text(json.dumps(pending, indent=1))
    audit.log("human_decision", decision=decision, approver=approver, role=role, note=note,
              band=pending["band"], strategy_updated=decision == "approve")
    return pending
