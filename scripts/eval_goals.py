"""Phase 4 evaluation on the goals in config/eval_goals.yaml.

  python scripts/eval_goals.py --keys            # answer keys only (no API calls)
  python scripts/eval_goals.py --run [IDS...]    # run the agent on goals (API cost!)
  python scripts/eval_goals.py --score           # score the latest run of each goal
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csa import config  # noqa: E402

OUT = ROOT / "outputs" / "phase4"
RUNS = ROOT / "outputs" / "runs"


def goals():
    return list(config.load("eval_goals")["goals"])


def compute_keys():
    from csa.benchmark import Benchmark
    from csa.lab import StrategyLab
    t0 = time.time()
    lab = StrategyLab(); lab.build_catalogue()
    bm = Benchmark(lab)
    keys = {"plans_searched": int(bm.n_plans), "plans_within_appetite": int(bm.within_appetite().sum()),
            "goals": {}}
    for g in goals():
        k = bm.key(g["min_growth_pct"], g["max_delta_bad_pp"])
        k["expected_outcome"] = "submit" if k["feasible"] else "refuse"
        keys["goals"][g["id"]] = {"text": g["text"], "group": g["group"], **k}
    keys["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answer_key.json").write_text(json.dumps(keys, indent=1))
    return keys


def run(ids):
    from csa import agent
    from csa.lab import StrategyLab
    lab = StrategyLab(); lab.build_catalogue()
    for g in goals():
        if ids and g["id"] not in ids:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", g["text"].lower()).strip("-")[:30]
        run_dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_eval-{g['id']}_{slug}"
        print(f"\n=== {g['id']}: {g['text']}")
        s = agent.run(g["text"], run_dir, lab=lab, echo=lambda *_: None)
        print(f"    -> {s['outcome'].get('type')} | calls {s['api_calls']} | {s['elapsed_s']} s | ${s['est_cost_usd']}")


if __name__ == "__main__":
    if "--keys" in sys.argv:
        k = compute_keys()
        print(f"plans searched {k['plans_searched']:,}; within appetite {k['plans_within_appetite']:,}; "
              f"{k['runtime_s']} s")
        for gid, v in k["goals"].items():
            best = v["max_growth_under_goal_constraints"]
            lean = v.get("leanest_qualifying_plan")
            print(f"{gid} [{v['group']}] {v['text']}\n    expected: {v['expected_outcome']}"
                  f" | best growth under goal: {best and best['growth_pct']}%"
                  f" | leanest qualifying: {lean and (lean['ins'], lean['outs'], lean['growth_pct'], lean['turnover_pct'])}"
                  f" | needs swap-outs: {v.get('needs_swap_outs')}"
                  f" | max within appetite: {v['max_growth_within_appetite']['growth_pct']}%")
    elif "--run" in sys.argv:
        run([a for a in sys.argv[2:] if not a.startswith("--")])
    elif "--score" in sys.argv:
        from csa.scoring import score_all
        score_all()
    else:
        sys.exit(__doc__)
