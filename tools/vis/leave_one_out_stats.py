#!/usr/bin/env python3
"""Inspect per-tile or dataset-level leave-one-out diff stats.

Examples:
    python tools/vis/leave_one_out_stats.py \
        inference_output/paired_ablation/leave_one_out/512_9728/leave_one_out_diff_stats.json

    python tools/vis/leave_one_out_stats.py \
        inference_output/paired_ablation/leave_one_out

    python tools/vis/leave_one_out_stats.py \
        inference_output/paired_ablation \
        --format json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, _fmt, _mean

STAT_KEYS: tuple[str, ...] = (
    "mean_diff",
    "max_diff",
    "pct_pixels_above_10",
    "delta_e_mean",
    "delta_e_p99",
    "ssim_loss_mean",
    "ssim_loss_p99",
    "causal_inside_mean_dE",
    "causal_outside_mean_dE",
    "causal_ratio",
    "uni_cosine_drop",
)
REPRESENTATIVE_KEYS: tuple[str, ...] = ("mean_diff", "pct_pixels_above_10")
STATS_FILENAME = "leave_one_out_diff_stats.json"


def _pstdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _metric_label(metric_key: str) -> str:
    if metric_key == "mean_diff":
        return "mean"
    if metric_key == "max_diff":
        return "max"
    if metric_key == "pct_pixels_above_10":
        return "pct>10"
    return metric_key


def load_stats_payload(stats_path: Path) -> dict[str, dict[str, float | None]]:
    payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{stats_path} did not contain a JSON object")

    stats: dict[str, dict[str, float | None]] = {}
    for group in FOUR_GROUP_ORDER:
        record = payload.get(group)
        if not isinstance(record, dict):
            continue
        stats[group] = {}
        for key in STAT_KEYS:
            value = record.get(key)
            if value is None:
                if key == "uni_cosine_drop":
                    stats[group][key] = None
                continue
            stats[group][key] = float(value)
    if not stats:
        raise ValueError(f"{stats_path} did not contain any recognized leave-one-out stats")
    return stats


def resolve_stats_paths(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: {path}")

    direct_stats = path / STATS_FILENAME
    if direct_stats.is_file():
        return [direct_stats]

    direct_children = sorted(path.glob(f"*/{STATS_FILENAME}"))
    if direct_children:
        return direct_children

    for child_name in ("leave_one_out", "ablation_results"):
        child_root = path / child_name
        child_stats = sorted(child_root.glob(f"*/{STATS_FILENAME}"))
        if child_stats:
            return child_stats

    raise FileNotFoundError(
        f"No {STATS_FILENAME} files found under {path}. "
        "Expected a stats JSON, a tile directory, a leave_one_out/ root, "
        "an ablation_results/ root, or a dataset root containing one of those."
    )


def resolve_input_mode(path: Path) -> tuple[str, list[Path]]:
    path = Path(path)
    if path.is_file():
        return "single", [path]

    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: {path}")

    direct_stats = path / STATS_FILENAME
    if direct_stats.is_file():
        return "single", [direct_stats]

    return "summary", resolve_stats_paths(path)


def select_representative_tile(tile_vectors: dict[str, dict[str, float]]) -> str | None:
    if not tile_vectors:
        return None

    axes = sorted({axis for vector in tile_vectors.values() for axis in vector})
    axis_means = {
        axis: statistics.mean([vector[axis] for vector in tile_vectors.values() if axis in vector])
        for axis in axes
    }
    axis_stds: dict[str, float] = {}
    for axis in axes:
        values = [vector[axis] for vector in tile_vectors.values() if axis in vector]
        axis_std = statistics.pstdev(values) if len(values) > 1 else 1.0
        axis_stds[axis] = axis_std if axis_std > 0 else 1.0

    best_tile = None
    best_distance = None
    for tile_id, vector in tile_vectors.items():
        distance = 0.0
        for axis in axes:
            if axis not in vector:
                continue
            z_score = (vector[axis] - axis_means[axis]) / axis_stds[axis]
            distance += z_score * z_score
        if best_distance is None or distance < best_distance:
            best_tile = tile_id
            best_distance = distance
    return best_tile


def summarize_stats_paths(stats_paths: list[Path]) -> dict[str, object]:
    grouped: dict[str, dict[str, list[float]]] = {
        group: {key: [] for key in STAT_KEYS}
        for group in FOUR_GROUP_ORDER
    }
    top_tiles: dict[str, dict[str, dict[str, str | float]]] = {
        group: {}
        for group in FOUR_GROUP_ORDER
    }
    tile_vectors: dict[str, dict[str, float]] = {}

    for stats_path in stats_paths:
        tile_id = stats_path.parent.name
        payload = load_stats_payload(stats_path)
        vector: dict[str, float] = {}
        for group in FOUR_GROUP_ORDER:
            record = payload.get(group)
            if not record:
                continue
            for key in STAT_KEYS:
                value = record.get(key)
                if value is None:
                    continue
                grouped[group][key].append(float(value))
                current_best = top_tiles[group].get(key)
                if current_best is None or float(value) > float(current_best["value"]):
                    top_tiles[group][key] = {"tile_id": tile_id, "value": float(value)}
                if key in REPRESENTATIVE_KEYS:
                    vector[f"{group}:{key}"] = float(value)
        if vector:
            tile_vectors[tile_id] = vector

    means: dict[str, dict[str, float]] = {}
    stds: dict[str, dict[str, float]] = {}
    for group in FOUR_GROUP_ORDER:
        means[group] = {}
        stds[group] = {}
        for key in STAT_KEYS:
            values = grouped[group][key]
            mean_value = _mean(values)
            if mean_value is None:
                continue
            means[group][key] = mean_value
            stds[group][key] = _pstdev(values)

    return {
        "tile_count": len(stats_paths),
        "representative_tile": select_representative_tile(tile_vectors),
        "means": means,
        "stds": stds,
        "top_tiles": top_tiles,
    }


def _sort_groups(records: dict[str, dict[str, float]], metric_key: str) -> list[str]:
    return sorted(
        [group for group in FOUR_GROUP_ORDER if records.get(group)],
        key=lambda group: (records[group].get(metric_key, float("-inf")), group),
        reverse=True,
    )


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _format_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    divider = "  ".join("-" * width for width in widths)
    return "\n".join([_format_row(headers), divider, *[_format_row(row) for row in rows]])


def _render_markdown(headers: list[str], rows: list[list[str]]) -> str:
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_single_stats(
    stats_path: Path,
    payload: dict[str, dict[str, float]],
    *,
    metric_key: str,
    output_format: str,
) -> str:
    ordered_groups = _sort_groups(payload, metric_key)
    headers = ["group", "mean", "max", "pct>10", "dE_mean", "dE_p99", "SSIM_loss", "causal"]
    rows = [
        [
            group,
            _fmt(payload[group].get("mean_diff")),
            _fmt(payload[group].get("max_diff")),
            _fmt(payload[group].get("pct_pixels_above_10")),
            _fmt(payload[group].get("delta_e_mean")),
            _fmt(payload[group].get("delta_e_p99")),
            _fmt(payload[group].get("ssim_loss_mean")),
            _fmt(payload[group].get("causal_ratio")),
        ]
        for group in ordered_groups
    ]

    if output_format == "json":
        return json.dumps(
            {
                "mode": "single",
                "path": str(stats_path),
                "rank_metric": metric_key,
                "stats": payload,
            },
            indent=2,
        )

    title = f"Leave-one-out stats: {stats_path}"
    subtitle = f"Ranked by {_metric_label(metric_key)} descending"
    body = _render_markdown(headers, rows) if output_format == "markdown" else _render_table(headers, rows)
    return f"{title}\n{subtitle}\n\n{body}"


def render_batch_summary(
    source_path: Path,
    summary: dict[str, object],
    *,
    metric_key: str,
    output_format: str,
) -> str:
    means = summary["means"]
    stds = summary["stds"]
    top_tiles = summary["top_tiles"]
    ordered_groups = _sort_groups(means, metric_key)
    top_metric_label = f"top_{_metric_label(metric_key)}"
    headers = [
        "group", "mean_avg", "mean_sd", "max_avg", "max_sd", "pct>10_avg", "pct>10_sd",
        "dE_avg", "SSIM_avg", "causal_avg", "top_tile", top_metric_label,
    ]
    rows: list[list[str]] = []
    for group in ordered_groups:
        top_metric = top_tiles.get(group, {}).get(metric_key, {})
        rows.append(
            [
                group,
                _fmt(means[group].get("mean_diff")),
                _fmt(stds[group].get("mean_diff")),
                _fmt(means[group].get("max_diff")),
                _fmt(stds[group].get("max_diff")),
                _fmt(means[group].get("pct_pixels_above_10")),
                _fmt(stds[group].get("pct_pixels_above_10")),
                _fmt(means[group].get("delta_e_mean")),
                _fmt(means[group].get("ssim_loss_mean")),
                _fmt(means[group].get("causal_ratio")),
                str(top_metric.get("tile_id", "-")),
                _fmt(top_metric.get("value") if isinstance(top_metric, dict) else None),
            ]
        )

    if output_format == "json":
        return json.dumps(
            {
                "mode": "summary",
                "path": str(source_path),
                "rank_metric": metric_key,
                **summary,
            },
            indent=2,
        )

    title = f"Leave-one-out summary: {source_path}"
    prefix = [
        title,
        f"Tiles: {summary['tile_count']}",
        f"Representative tile: {summary['representative_tile'] or '-'}",
        f"Ranked by {_metric_label(metric_key)} average descending",
        "",
    ]
    body = _render_markdown(headers, rows) if output_format == "markdown" else _render_table(headers, rows)
    return "\n".join(prefix) + body


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect leave-one-out diff stats from one tile or a dataset root.")
    parser.add_argument(
        "path",
        type=Path,
        help="Stats JSON, tile directory, leave_one_out/ root, ablation_results/ root, or dataset root.",
    )
    parser.add_argument(
        "--metric",
        choices=STAT_KEYS,
        default="pct_pixels_above_10",
        help="Metric used to rank rows (default: pct_pixels_above_10).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Prints to stdout when omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode, stats_paths = resolve_input_mode(args.path)

    if mode == "single":
        text = render_single_stats(
            stats_paths[0],
            load_stats_payload(stats_paths[0]),
            metric_key=args.metric,
            output_format=args.format,
        )
    else:
        text = render_batch_summary(
            args.path,
            summarize_stats_paths(stats_paths),
            metric_key=args.metric,
            output_format=args.format,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote leave-one-out stats summary -> {args.output}")
        return

    print(text)


if __name__ == "__main__":
    main()
