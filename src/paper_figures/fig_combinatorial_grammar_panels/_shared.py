"""Shared helpers for figure 6 panel renderers."""
from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from PIL import Image


STATES: tuple[str, ...] = ("prolif", "nonprolif", "dead")
LEVELS: tuple[str, ...] = ("low", "mid", "high")
MORPHOLOGY_METRICS: tuple[str, ...] = (
    "nuclear_density",
    "eosin_ratio",
    "hematoxylin_ratio",
    "hematoxylin_burden",
    "mean_cell_size",
    "nucleus_area_median",
    "nucleus_area_iqr",
    "glcm_contrast",
    "glcm_homogeneity",
)


def compute_pixel_diff(condition_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """Mean absolute per-channel difference, returned as a 2D float32 map."""
    cond = np.asarray(condition_rgb, dtype=np.float32)
    ref = np.asarray(reference_rgb, dtype=np.float32)
    if cond.shape != ref.shape:
        raise ValueError(f"shape mismatch: cond={cond.shape}, ref={ref.shape}")
    if cond.ndim != 3:
        raise ValueError(f"expected RGB arrays with 3 dimensions, got {cond.ndim}")
    return np.mean(np.abs(cond - ref), axis=-1).astype(np.float32)


def condition_id(state: str, oxygen_label: str, glucose_label: str) -> str:
    return f"{state}_{oxygen_label}_{glucose_label}"


def residual_lookup(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    """Map (state, oxygen_label, glucose_label) to residual metric values."""
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["cell_state"]), str(row["oxygen_label"]), str(row["glucose_label"]))
        parsed: dict[str, float] = {}
        for name, value in row.items():
            if name.startswith("residual_") and value not in (None, ""):
                parsed[name] = float(value)
        out[key] = parsed
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def pick_representative_anchor(signature_rows: list[dict[str, str]]) -> str:
    """Anchor with the most rows in morphological_signatures.csv."""
    counts: dict[str, int] = {}
    for row in signature_rows:
        anchor = str(row["anchor_id"])
        counts[anchor] = counts.get(anchor, 0) + 1
    if not counts:
        raise ValueError("morphological_signatures rows are empty")
    max_count = max(counts.values())
    tied = sorted(anchor for anchor, count in counts.items() if count == max_count)
    return tied[0]


def compute_anchor_sweep_magnitude(signature_rows: list[dict[str, str]]) -> dict[str, float]:
    """Per-anchor sweep response proxy: sum of variance across morphology metrics."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in signature_rows:
        grouped.setdefault(str(row["anchor_id"]), []).append(row)

    out: dict[str, float] = {}
    for anchor, rows in grouped.items():
        total = 0.0
        for metric in MORPHOLOGY_METRICS:
            values = [float(r[metric]) for r in rows if r.get(metric) not in (None, "")]
            if len(values) >= 2:
                total += float(np.var(values, ddof=0))
        out[anchor] = total
    return out


def select_si_anchors(
    signature_rows: list[dict[str, str]],
    *,
    representative_id: str,
    reference_exists_fn: Callable[[str], bool],
) -> list[str]:
    """Return representative plus low/mid/high sweep-magnitude anchors."""
    magnitudes = compute_anchor_sweep_magnitude(signature_rows)
    picks: list[str] = []
    used: set[str] = set()
    if reference_exists_fn(representative_id):
        picks.append(representative_id)
        used.add(representative_id)

    eligible = sorted(
        (
            (anchor_id, magnitude)
            for anchor_id, magnitude in magnitudes.items()
            if anchor_id not in used and reference_exists_fn(anchor_id)
        ),
        key=lambda pair: (pair[1], pair[0]),
    )
    if not eligible:
        return picks

    target_indices = [0, len(eligible) // 2, len(eligible) - 1]
    for idx in target_indices:
        for offset in range(len(eligible)):
            candidate = eligible[(idx + offset) % len(eligible)][0]
            if candidate not in used:
                picks.append(candidate)
                used.add(candidate)
                break
    return picks[:4]

