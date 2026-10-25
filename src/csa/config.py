"""Config loading. Config files are read-only to everything in this package."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


def _freeze(obj):
    """Recursively make config immutable so no code path can edit it in memory."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


@lru_cache(maxsize=None)
def load(name: str):
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return _freeze(yaml.safe_load(f))


def path(key: str) -> Path:
    return ROOT / load("settings")["paths"][key]


# ---- field register helpers ------------------------------------------------

def candidate_fields() -> dict[str, tuple[str, ...]]:
    c = load("fields")["candidates"]
    return {"numeric": c["numeric"], "categorical": c["categorical"], "date": c["date"]}


def role_fields() -> tuple[str, ...]:
    return tuple(v for v in load("fields")["roles"].values() if v)


def blocked_fields() -> dict[str, str]:
    """field -> blocked category"""
    out = {}
    for cat, fields in load("fields")["blocked"].items():
        for f in fields:
            out[f] = cat
    return out
