"""Human approval gate (run by a person, never by the agent).

Usage: python scripts/approve.py <run folder name under outputs/runs>
Shows the proposal and asks for approve / reject, your name and role.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csa.approval import ApprovalError, record_decision  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = ROOT / "outputs" / "runs" / sys.argv[1]
    pending = json.loads((run_dir / "pending_approval.json").read_text())
    print((run_dir / "proposal.md").read_text(encoding="utf-8"))
    print(f"\nStatus: {pending['status']} | band {pending['band']} | needs: {pending['route']}\n")
    def ask(prompt, allowed):
        while True:
            v = input(prompt).strip().lower().replace(" ", "_").replace("-", "_")
            if v in allowed:
                return v
            print(f"  Please type one of: {', '.join(allowed)}")

    decision = ask("Decision (approve / reject): ", ["approve", "reject"])
    approver = input("Your name: ").strip()
    role = ask("Your role (credit manager / senior risk): ", ["credit_manager", "senior_risk"])
    note = input("Note (optional): ").strip()
    try:
        out = record_decision(run_dir, decision, approver, role, note)
    except ApprovalError as e:
        sys.exit(f"Not recorded: {e}")
    print(f"Recorded: {out['status']} by {approver} ({role}).")
    if out["status"] == "APPROVED":
        print("Test cells added to config/strategy.yaml. The current policy is unchanged.")


if __name__ == "__main__":
    main()
