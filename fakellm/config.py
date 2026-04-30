"""Configuration loading. Reads fakellm.yaml into a Config object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    rules: list[dict[str, Any]] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path = "fakellm.yaml") -> Config:
    """Load config from a YAML file. Returns empty Config if file is missing."""
    p = Path(path)
    if not p.exists():
        return Config()

    with p.open() as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        rules=raw.get("rules", []),
        defaults=raw.get("defaults", {}),
    )
