"""Run the Credit Strategy Agent on a business goal. Stops at the human approval gate.

Usage:
  python scripts/run_agent.py "Grow approvals 3-5% without raising the bad rate"
Needs ANTHROPIC_API_KEY in the environment or in a .env file in the project root.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csa import agent  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    goal = sys.argv[1]
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")[:40]
    run_dir = ROOT / "outputs" / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{slug}"
    print(f"Goal: {goal}\nRun folder: {run_dir}\n")
    summary = agent.run(goal, run_dir)
    print("\n" + json.dumps(summary, indent=1))
    if summary["outcome"].get("type") == "submitted":
        print(f"\nPENDING HUMAN APPROVAL ({summary['outcome']['route']}). To decide:\n"
              f'  python scripts/approve.py "{run_dir.name}"')


if __name__ == "__main__":
    main()
