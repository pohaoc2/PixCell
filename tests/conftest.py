"""Pytest test session configuration."""

from pathlib import Path
import sys

# Ensure repository root is importable in all environments (CI/local).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
