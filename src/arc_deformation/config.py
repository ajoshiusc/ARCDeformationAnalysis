"""TOML configuration loading."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def config_value(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    return config.get(section, {}).get(key, default)
