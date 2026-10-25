"""Append-only JSONL audit log: every step, tool call, result, decision and token count."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "audit.jsonl"
        self.t0 = time.time()
        self.seq = 0

    def log(self, event: str, **payload) -> dict:
        self.seq += 1
        rec = {"seq": self.seq, "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "elapsed_s": round(time.time() - self.t0, 2), "event": event, **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec


def read(run_dir: Path) -> list[dict]:
    with open(Path(run_dir) / "audit.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
