"""Safe path localization, atomic output, and table helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or TSV using the filename suffix."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    """Reject missing or duplicate record keys with actionable examples."""
    if column not in frame:
        raise ValueError(f"{label} has no {column!r} column")
    duplicate = frame[column].duplicated(keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, column].astype(str).head(5).tolist()
        raise ValueError(f"{label} contains duplicate {column} values, including {examples}")


def ensure_output_outside_data(output_dir: Path, data_root: Path | None) -> Path:
    """Create an output directory only when it is outside the read-only data root."""
    output = Path(output_dir).expanduser().resolve()
    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
        if output == root or root in output.parents:
            raise ValueError(f"Refusing to write analysis products inside data root: {root}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def atomic_text(path: Path, content: str) -> None:
    """Replace a text file atomically on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and nonfinite values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def localize_arc_path(path: Path | str, arc_root: Path) -> Path:
    """Map an absolute manifest path below an ``ARC`` root to this installation."""
    candidate = Path(str(path))
    if candidate.is_file():
        return candidate
    parts = candidate.parts
    arc_indices = [index for index, value in enumerate(parts) if value == "ARC"]
    if arc_indices:
        relative = Path(*parts[arc_indices[-1] + 1 :])
        root = Path(arc_root).resolve()
        localized = (root / relative).resolve()
        if localized.is_relative_to(root) and localized.is_file():
            return localized
    raise FileNotFoundError(f"Cannot localize ARC path: {candidate}")


def prefix_nonkeys(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = {"case_id", "subject", "session"}
    return frame.rename(
        columns={column: prefix + column for column in frame if column not in keys}
    )


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
