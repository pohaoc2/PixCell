"""Append full_uni_null rows to null metrics CSVs and re-summarize.

Run after uni_null/generated/ is populated with tme_only.png images.
If CellViT sidecars (.png.json) exist beside tme_only.png, morphology is real.
Without sidecars, target_value = NaN (re-run after importing CellViT).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.a4_uni_probe.edit import _summarize_nulls
from src.a4_uni_probe.metrics import morphology_row_for_image

A4_OUT = ROOT / "inference_output/a1_concat/a4_uni_probe"
UNI_NULL_GENERATED = A4_OUT / "uni_null" / "generated"
NULL_ROOT = A4_OUT / "null"
ATTRS = ["eccentricity_mean", "nuclear_area_mean", "nuclei_density"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def update_attr(attr: str) -> None:
    attr_dir = NULL_ROOT / attr
    metrics_path = attr_dir / "metrics.csv"
    if not metrics_path.is_file():
        print(f"[{attr}] metrics.csv missing — skipping", flush=True)
        return

    rows: list[dict[str, object]] = list(_read_csv(metrics_path))

    # Remove any existing full_uni_null rows so we can rebuild cleanly
    rows = [r for r in rows if r.get("condition") != "full_uni_null"]

    null_tiles = sorted(
        p.name for p in attr_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    added = 0
    missing = 0
    for tile_id in null_tiles:
        tme_only_path = UNI_NULL_GENERATED / tile_id / "tme_only.png"
        if not tme_only_path.is_file():
            missing += 1
            continue
        morph = morphology_row_for_image(tme_only_path)
        rows.append({
            "tile_id": tile_id,
            "condition": "full_uni_null",
            "target_attr": attr,
            "image_path": str(tme_only_path),
            "target_value": float(morph[attr]),
            **{f"morpho.{name}": float(val) for name, val in morph.items()},
        })
        added += 1

    _write_csv(metrics_path, rows)
    print(f"[{attr}] added {added} full_uni_null rows, {missing} tiles missing tme_only.png", flush=True)

    import argparse
    fake_args = argparse.Namespace(attr=attr)
    _summarize_nulls(metrics_path, attr_dir / "null_comparison.json", attr)
    print(f"[{attr}] null_comparison.json updated", flush=True)


def main() -> None:
    for attr in ATTRS:
        update_attr(attr)

    print("\nRe-rendering figures...", flush=True)
    from src.a4_uni_probe.figures import render_all
    render_all(A4_OUT)
    print("Done. Figures in:", A4_OUT / "figures", flush=True)


if __name__ == "__main__":
    main()
