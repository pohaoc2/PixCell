#!/usr/bin/env python3
"""Map flat CellViT outputs back onto the ablation cache as sidecar files."""

from __future__ import annotations

import argparse
from glob import has_magic
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalize_exts(raw: str) -> list[str]:
    exts: list[str] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if not piece.startswith("."):
            piece = f".{piece}"
        exts.append(piece.lower())
    return exts


def _select_best_match(matches: list[Path], preferred_exts: list[str]) -> Path:
    preferred_rank = {ext: idx for idx, ext in enumerate(preferred_exts)}
    return sorted(
        matches,
        key=lambda path: (
            preferred_rank.get(path.suffix.lower(), len(preferred_exts)),
            len(path.name),
            path.name,
        ),
    )[0]


class _ResultIndex:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"CellViT results directory does not exist: {results_dir}. "
                "Replace the placeholder path with the folder containing the exported "
                "CellViT JSON outputs."
            )
        self._by_name: dict[str, list[Path]] = {}
        self._all_files: list[tuple[PurePosixPath, Path]] = []
        self._pattern_cache: dict[str, list[Path]] = {}

        for path in sorted(results_dir.rglob("*")):
            if not path.is_file():
                continue
            rel_path = PurePosixPath(path.relative_to(results_dir).as_posix())
            self._all_files.append((rel_path, path))
            self._by_name.setdefault(path.name, []).append(path)

    def resolve(self, pattern: str) -> list[Path]:
        normalized_pattern = pattern.replace("\\", "/")
        if not has_magic(normalized_pattern) and "/" not in normalized_pattern:
            return list(self._by_name.get(normalized_pattern, []))

        cached = self._pattern_cache.get(normalized_pattern)
        if cached is not None:
            return list(cached)

        matches = [
            path
            for rel_path, path in self._all_files
            if rel_path.match(normalized_pattern)
        ]
        self._pattern_cache[normalized_pattern] = matches
        return list(matches)


def _resolve_result_for_entry(
    result_index: _ResultIndex,
    flat_name: str,
    *,
    result_pattern: str,
    preferred_exts: list[str],
) -> Path:
    flat_path = Path(flat_name)
    stem = flat_path.stem
    pattern = result_pattern.format(name=flat_name, stem=stem)
    matches = result_index.resolve(pattern)
    if not matches:
        raise FileNotFoundError(
            f"no CellViT result matched {pattern!r} for exported file {flat_name!r}"
        )
    return _select_best_match(matches, preferred_exts)


def import_cellvit_results(
    manifest_path: Path,
    results_dir: Path,
    *,
    result_pattern: str = "{stem}.json",
    preferred_exts: list[str] | None = None,
    sidecar_suffix: str = "_cellvit_instances",
    dry_run: bool = False,
) -> tuple[list[dict[str, str]], Path]:
    """Copy matched CellViT outputs next to the original source PNGs."""
    manifest_path = Path(manifest_path).resolve()
    results_dir = Path(results_dir).resolve()
    preferred_exts = preferred_exts or [".json", ".npy", ".png"]
    result_index = _ResultIndex(results_dir)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    imported: list[dict[str, str]] = []

    for entry in entries:
        source_path = Path(entry["source_path"])
        flat_name = str(entry["flat_name"])
        matched = _resolve_result_for_entry(
            result_index,
            flat_name,
            result_pattern=result_pattern,
            preferred_exts=preferred_exts,
        )
        dst = source_path.with_name(f"{source_path.stem}{sidecar_suffix}{matched.suffix.lower()}")
        imported.append(
            {
                "source_path": str(source_path),
                "flat_name": flat_name,
                "matched_result": str(matched),
                "imported_path": str(dst),
            }
        )
        if not dry_run:
            shutil.copy2(matched, dst)

    report = {
        "version": 1,
        "manifest_path": str(manifest_path),
        "results_dir": str(results_dir),
        "sidecar_suffix": sidecar_suffix,
        "entries": imported,
    }
    report_path = manifest_path.parent / "import_report.json"
    if not dry_run:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return imported, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import flat CellViT results back into the ablation cache tree.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest written by tools/cellvit/export_batch.py",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Folder containing CellViT outputs.",
    )
    parser.add_argument(
        "--result-pattern",
        type=str,
        default="{stem}.json",
        help="Glob used under results-dir. Available placeholders: {stem}, {name}.",
    )
    parser.add_argument(
        "--prefer-ext",
        type=str,
        default=".json,.npy,.png",
        help="Preferred result extensions in priority order (default: .json,.npy,.png).",
    )
    parser.add_argument(
        "--sidecar-suffix",
        type=str,
        default="_cellvit_instances",
        help="Suffix added beside each original PNG (default: _cellvit_instances).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve matches without copying files back.",
    )
    args = parser.parse_args()

    imported, report_path = import_cellvit_results(
        args.manifest,
        args.results_dir,
        result_pattern=args.result_pattern,
        preferred_exts=_normalize_exts(args.prefer_ext),
        sidecar_suffix=args.sidecar_suffix,
        dry_run=args.dry_run,
    )
    print(f"Matched {len(imported)} CellViT outputs")
    if args.dry_run:
        print("Dry run only; no files were copied.")
    else:
        print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
