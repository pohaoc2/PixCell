"""Small filesystem helpers shared by task packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_json(payload: Any, output_path: str | Path) -> Path:
    """Write a JSON payload with stable formatting."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
