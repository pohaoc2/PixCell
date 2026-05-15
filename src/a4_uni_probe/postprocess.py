"""Post-processing helpers for a4 generated sweep/null outputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.a4_uni_probe.appearance_metrics import appearance_row_for_image
from src.a4_uni_probe.edit import _summarize_nulls, _summarize_slopes
from src.a4_uni_probe.labels import APPEARANCE_ATTR_NAMES
from src.a4_uni_probe.metrics import morphology_row_for_image


def _iter_generated_png_entries(out_dir: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for kind in ("sweep", "null"):
        kind_root = out_dir / kind
        if not kind_root.is_dir():
            continue
        for image_path in sorted(kind_root.glob("*/*/**/*.png")):
            rel = image_path.relative_to(out_dir)
            parts = rel.parts
            if kind == "sweep" and len(parts) != 5:
                continue
            if kind == "null" and len(parts) != 4:
                continue
            attr = parts[1]
            tile_id = parts[2]
            tail = parts[3:]
            flat_name = f"{tile_id}__{'__'.join(parts)}"
            entries.append(
                {
                    "tile_id": tile_id,
                    "attr": attr,
                    "kind": kind,
                    "rel_image_path": rel.as_posix(),
                    "source_path": str(image_path.resolve()),
                    "flat_name": flat_name,
                    "tail": "__".join(tail),
                }
            )
    return entries


def export_generated_cellvit_batch(
    out_dir: Path,
    batch_dir: Path,
    *,
    overwrite: bool = False,
    make_zip: bool = False,
) -> tuple[Path, Path, int]:
    out_dir = out_dir.resolve()
    batch_dir = batch_dir.resolve()
    images_dir = batch_dir / "images"
    if batch_dir.exists() and any(batch_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output dir is not empty: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    entries = _iter_generated_png_entries(out_dir)
    for entry in entries:
        src = Path(entry["source_path"])
        dst = images_dir / entry["flat_name"]
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        entry["flat_path"] = str(dst.resolve())

    manifest = {
        "version": 1,
        "out_dir": str(out_dir),
        "images_dir": str(images_dir),
        "entries": entries,
    }
    manifest_path = batch_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    zip_path = batch_dir.with_suffix(".zip")
    if make_zip:
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(batch_dir), "zip", root_dir=batch_dir)
    return manifest_path, zip_path, len(entries)


def _target_value(attr: str, morph: dict[str, float], appearance: dict[str, float]) -> float:
    if attr in APPEARANCE_ATTR_NAMES:
        return float(appearance[f"appearance.{attr}"])
    return float(morph[attr])


def _refresh_metrics_csv(metrics_path: Path, *, summarize: bool) -> None:
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    if not rows:
        return
    attr = str(rows[0]["target_attr"])
    updated_rows: list[dict[str, object]] = []
    for row in rows:
        image_path = Path(str(row["image_path"]))
        morph = morphology_row_for_image(image_path)
        appearance = appearance_row_for_image(image_path)
        merged: dict[str, object] = dict(row)
        merged["target_value"] = _target_value(attr, morph, appearance)
        for name, value in morph.items():
            merged[f"morpho.{name}"] = float(value)
        for name, value in appearance.items():
            merged[name] = float(value)
        updated_rows.append(merged)

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(updated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(updated_rows)

    if summarize:
        if metrics_path.parent.parent.name == "sweep":
            _summarize_slopes(metrics_path, metrics_path.parent / "slope_summary.json", attr)
        elif metrics_path.parent.parent.name == "null":
            _summarize_nulls(metrics_path, metrics_path.parent / "null_comparison.json", attr)


def refresh_generated_metrics(out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    for metrics_path in sorted(out_dir.glob("sweep/*/metrics.csv")):
        _refresh_metrics_csv(metrics_path, summarize=True)
    for metrics_path in sorted(out_dir.glob("sweep/*/metrics.shard_*of*.csv")):
        _refresh_metrics_csv(metrics_path, summarize=False)
    for metrics_path in sorted(out_dir.glob("null/*/metrics.csv")):
        _refresh_metrics_csv(metrics_path, summarize=True)
    for metrics_path in sorted(out_dir.glob("null/*/metrics.shard_*of*.csv")):
        _refresh_metrics_csv(metrics_path, summarize=False)


def append_full_null_rows(out_dir: Path, full_null_root: Path) -> None:
    out_dir = out_dir.resolve()
    full_null_root = full_null_root.resolve()
    for metrics_path in sorted(out_dir.glob("null/*/metrics.csv")):
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        if not rows:
            continue
        attr = str(rows[0]["target_attr"])
        fieldnames = list(rows[0].keys())
        updated_rows = [row for row in rows if row.get("condition") != "full_uni_null"]
        tile_ids = sorted({str(row["tile_id"]) for row in updated_rows})
        for tile_id in tile_ids:
            image_path = full_null_root / tile_id / "tme_only.png"
            if not image_path.is_file():
                continue
            morph = morphology_row_for_image(image_path)
            appearance = appearance_row_for_image(image_path)
            row = {key: "" for key in fieldnames}
            row["tile_id"] = tile_id
            row["condition"] = "full_uni_null"
            row["target_attr"] = attr
            row["image_path"] = str(image_path)
            row["target_value"] = _target_value(attr, morph, appearance)
            for name, value in morph.items():
                key = f"morpho.{name}"
                if key in row:
                    row[key] = float(value)
            for name, value in appearance.items():
                if name in row:
                    row[name] = float(value)
            updated_rows.append(row)

        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        _summarize_nulls(metrics_path, metrics_path.parent / "null_comparison.json", attr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export-cellvit-batch")
    p_export.add_argument("--out-dir", type=Path, required=True)
    p_export.add_argument("--batch-dir", type=Path, required=True)
    p_export.add_argument("--overwrite", action="store_true")
    p_export.add_argument("--zip", action="store_true")

    p_refresh = sub.add_parser("refresh-metrics")
    p_refresh.add_argument("--out-dir", type=Path, required=True)

    p_full_null = sub.add_parser("append-full-null")
    p_full_null.add_argument("--out-dir", type=Path, required=True)
    p_full_null.add_argument("--full-null-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "export-cellvit-batch":
        manifest_path, zip_path, count = export_generated_cellvit_batch(
            args.out_dir,
            args.batch_dir,
            overwrite=args.overwrite,
            make_zip=args.zip,
        )
        print(f"Exported {count} PNGs")
        print(f"Manifest -> {manifest_path}")
        if args.zip:
            print(f"Zip -> {zip_path}")
    elif args.command == "refresh-metrics":
        refresh_generated_metrics(args.out_dir)
        print(f"Refreshed metrics under {args.out_dir}")
    elif args.command == "append-full-null":
        append_full_null_rows(args.out_dir, args.full_null_root)
        print(f"Appended full-null rows under {args.out_dir} from {args.full_null_root}")
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())