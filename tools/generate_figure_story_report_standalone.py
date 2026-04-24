from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
REPORT_PATH = ROOT / "docs" / "figure_story_report_standalone.html"

START_MARKER = "  <!-- BEGIN GENERATED SRC SUMMARY -->"
END_MARKER = "  <!-- END GENERATED SRC SUMMARY -->"
INSERT_BEFORE = "  <!-- ══ MAIN FIGURES ══ -->"

OUTPUT_DIR_NAMES = {"out", "probe_out"}
IGNORED_DIR_NAMES = OUTPUT_DIR_NAMES | {"__pycache__"}
CODE_SUFFIXES = {".py", ".pyc"}


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _html_text(text: object) -> str:
    return html.escape(str(text), quote=True)


def _iter_output_roots() -> list[Path]:
    roots: list[Path] = []
    for task_dir in sorted(SRC_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        for child_name in sorted(OUTPUT_DIR_NAMES):
            child = task_dir / child_name
            if child.is_dir():
                roots.append(child)
    return roots


def _collect_files(base_dir: Path) -> list[Path]:
    return sorted(path for path in base_dir.rglob("*") if path.is_file())


def _csv_summary(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        row_count = sum(1 for _ in reader)
    return f"{row_count} rows, {len(header)} cols"


def _json_summary(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        parts: list[str] = []
        for key in ("tile_count", "n_targets", "feature_dim", "cv_splits_path"):
            if key in payload:
                parts.append(f"{key}={payload[key]}")
        if "target_names" in payload and isinstance(payload["target_names"], list):
            parts.append(f"target_names={len(payload['target_names'])}")
        if "results" in payload and isinstance(payload["results"], list):
            parts.append(f"results={len(payload['results'])}")
        if parts:
            return ", ".join(parts)
        return f"{len(payload)} top-level keys"
    if isinstance(payload, list):
        return f"{len(payload)} items"
    return type(payload).__name__


def _txt_summary(path: Path) -> str:
    line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
    return f"{line_count} lines"


def _npy_summary(path: Path) -> str:
    array = np.load(path, mmap_mode="r")
    return f"shape={tuple(array.shape)}, dtype={array.dtype}"


def _file_summary(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _npy_summary(path)
    if suffix == ".csv":
        return _csv_summary(path)
    if suffix == ".json":
        return _json_summary(path)
    if suffix == ".txt":
        return _txt_summary(path)
    return _human_bytes(path.stat().st_size)


def _pick_high_signal_files(base_dir: Path, files: list[Path]) -> list[Path]:
    if len(files) <= 12:
        return files

    preferred_tokens = (
        "manifest.json",
        "plan.json",
        "results.csv",
        "results.json",
        "summary.csv",
        "comparison",
        "stats.csv",
        "heatmap.png",
        "bar_chart.png",
        "residuals.csv",
        "signatures.csv",
        "tradeoff_data.csv",
    )
    selected: list[Path] = []
    for token in preferred_tokens:
        for path in files:
            rel = path.relative_to(base_dir).as_posix()
            if token in rel and path not in selected:
                selected.append(path)
    if not selected:
        selected = files[:10]
    return selected[:12]


def _aggregate_line(base_dir: Path, files: list[Path]) -> str:
    total_size = sum(path.stat().st_size for path in files)
    suffix_counts = Counter(path.suffix.lower() or "<none>" for path in files)
    top_suffixes = ", ".join(f"{suffix}={count}" for suffix, count in sorted(suffix_counts.items())[:6])
    return f"{len(files)} files, {_human_bytes(total_size)} total, {top_suffixes}"


def _render_file_rows(base_dir: Path, files: list[Path]) -> str:
    rows: list[str] = []
    for path in _pick_high_signal_files(base_dir, files):
        rel = path.relative_to(base_dir).as_posix()
        rows.append(
            "<tr>"
            f"<td><code>{_html_text(rel)}</code></td>"
            f"<td>{_html_text(_file_summary(path))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_root_card(base_dir: Path) -> str:
    files = _collect_files(base_dir)
    task_name = base_dir.parent.name
    output_name = base_dir.name
    aggregate = _aggregate_line(base_dir, files)
    return (
        "<div class='figure-card'>"
        "<div class='figure-card-header'>"
        f"<span class='figure-number'>{_html_text(task_name)}</span>"
        f"<h2 class='figure-title' style='font-size:1.05rem;'>{_html_text(output_name)}</h2>"
        "</div>"
        f"<p class='figure-desc'><code>{_html_text(base_dir.relative_to(ROOT).as_posix())}</code> · {_html_text(aggregate)}</p>"
        "<table class='supp'>"
        "<thead><tr><th>Artifact</th><th>Summary</th></tr></thead>"
        f"<tbody>{_render_file_rows(base_dir, files)}</tbody>"
        "</table>"
        "</div>"
    )


def _iter_extra_data_files() -> list[Path]:
    extra_files: list[Path] = []
    for path in sorted(SRC_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(SRC_ROOT).parts
        if any(part in IGNORED_DIR_NAMES for part in rel_parts[:-1]):
            continue
        if path.suffix.lower() in CODE_SUFFIXES:
            continue
        extra_files.append(path)
    return extra_files


def _group_extra_files(files: list[Path]) -> list[tuple[str, list[Path]]]:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        rel_parts = path.relative_to(SRC_ROOT).parts
        group_name = rel_parts[0] if rel_parts else "."
        grouped.setdefault(group_name, []).append(path)
    return [(group_name, grouped[group_name]) for group_name in sorted(grouped)]


def _render_extra_card(group_name: str, files: list[Path]) -> str:
    base_dir = SRC_ROOT
    aggregate = _aggregate_line(base_dir, files)
    rows: list[str] = []
    for path in _pick_high_signal_files(base_dir, files):
        rel = path.relative_to(SRC_ROOT).as_posix()
        rows.append(
            "<tr>"
            f"<td><code>{_html_text(rel)}</code></td>"
            f"<td>{_html_text(_file_summary(path))}</td>"
            "</tr>"
        )
    return (
        "<div class='figure-card'>"
        "<div class='figure-card-header'>"
        f"<span class='figure-number'>{_html_text(group_name)}</span>"
        "<h2 class='figure-title' style='font-size:1.05rem;'>src data artifacts</h2>"
        "</div>"
        f"<p class='figure-desc'><code>src/{_html_text(group_name)}</code> · {_html_text(aggregate)}</p>"
        "<table class='supp'>"
        "<thead><tr><th>Artifact</th><th>Summary</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _render_overview(roots: list[Path], extra_files: list[Path]) -> str:
    all_files = [path for root in roots for path in _collect_files(root)]
    combined_files = all_files + extra_files
    total_size = sum(path.stat().st_size for path in combined_files)
    task_count = len({root.parent.name for root in roots} | {path.relative_to(SRC_ROOT).parts[0] for path in extra_files})
    return (
        "<section class='hero' style='margin-top:22px;'>"
        "<h2 style='margin:0 0 8px;font-family:var(--serif);font-size:1.45rem;'>Source Data Inventory</h2>"
        "<p class='subtitle' style='margin-bottom:12px;'>Generated from current artifacts under <code>src/</code></p>"
        "<div class='overview-bar'>"
        f"<span><strong>{task_count} task roots</strong></span>"
        f"<span><strong>{len(roots)} output directories</strong></span>"
        f"<span><strong>{len(combined_files)} data files</strong></span>"
        f"<span><strong>{_html_text(_human_bytes(total_size))}</strong> total on disk</span>"
        "</div>"
        "<p class='logline' style='margin-top:14px;'>"
        "This section is rebuilt from the live contents of <code>src/</code>. "
        "It includes task output directories plus non-code data artifacts that live alongside the source modules. "
        "Small trees list their main artifacts directly; large generation trees are summarized via aggregate counts plus their high-signal summary files."
        "</p>"
        "</section>"
    )


def build_generated_block() -> str:
    roots = _iter_output_roots()
    extra_files = _iter_extra_data_files()
    cards = "".join(_render_root_card(root) for root in roots)
    extra_groups = _group_extra_files(extra_files)
    extra_cards = "".join(_render_extra_card(group_name, files) for group_name, files in extra_groups)
    extra_section = (
        "\n  <div class=\"section-label\">Additional src Data</div>\n\n"
        f"{extra_cards}\n"
        if extra_cards
        else ""
    )
    return (
        f"{START_MARKER}\n"
        f"{_render_overview(roots, extra_files)}\n\n"
        "  <div class=\"section-label\">Task Outputs</div>\n\n"
        f"{cards}\n"
        f"{extra_section}"
        f"{END_MARKER}"
    )


def inject_generated_block(html_text: str, generated_block: str) -> str:
    if START_MARKER in html_text and END_MARKER in html_text:
        start = html_text.index(START_MARKER)
        end = html_text.index(END_MARKER) + len(END_MARKER)
        return html_text[:start] + generated_block + html_text[end:]
    insert_at = html_text.index(INSERT_BEFORE)
    return html_text[:insert_at] + generated_block + "\n\n" + html_text[insert_at:]


def main() -> None:
    html_text = REPORT_PATH.read_text(encoding="utf-8")
    updated = inject_generated_block(html_text, build_generated_block())
    REPORT_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {REPORT_PATH}")


if __name__ == "__main__":
    main()
