"""Collect and aggregate per-condition tradeoff metrics."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal


CANONICAL_CONDITION_LADDER: tuple[str, ...] = (
    "cell_state",
    "cell_state+cell_types",
    "cell_state+cell_types+microenv",
    "cell_state+cell_types+microenv+vasculature",
)


@dataclass(frozen=True)
class TileConditionRecord:
    """Per-tile tradeoff record for one condition."""

    split: Literal["paired", "unpaired"]
    tile_id: str
    condition: str
    n_groups: int
    aji: float
    pq: float
    realism_key: Literal["fid", "fud"]
    realism: float


@dataclass(frozen=True)
class TradeoffAggregate:
    """Aggregate tradeoff metrics across tiles for one split and condition."""

    split: Literal["paired", "unpaired"]
    condition: str
    n_groups: int
    realism_key: Literal["fid", "fud"]
    aji_mean: float
    aji_sd: float
    pq_mean: float
    pq_sd: float
    realism_mean: float
    realism_sd: float
    n_tiles: int
    is_pareto: bool = False


def _candidate_metric_roots(path: str | Path) -> list[Path]:
    base = Path(path)
    candidates = [base]
    child = base / "ablation_results"
    if child.is_dir():
        candidates.append(child)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if candidate.exists() and resolved not in seen:
            seen.add(resolved)
            ordered.append(candidate)
    return ordered


def collect_tradeoff_records(
    split: Literal["paired", "unpaired"],
    metric_dir: str | Path,
    conditions: Iterable[str] = CANONICAL_CONDITION_LADDER,
) -> list[TileConditionRecord]:
    """Collect per-tile metrics for the canonical four-condition ladder."""
    files: list[Path] = []
    seen: set[Path] = set()
    for root in _candidate_metric_roots(metric_dir):
        for path in sorted(root.glob("*/metrics.json")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)

    records: list[TileConditionRecord] = []
    realism_key: Literal["fid", "fud"] = "fid" if split == "paired" else "fud"
    for metrics_path in files:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_condition = payload.get("per_condition", {})
        if not isinstance(per_condition, dict):
            continue
        tile_id = str(payload.get("tile_id") or metrics_path.parent.name)
        for condition in conditions:
            raw = per_condition.get(condition)
            if not isinstance(raw, dict):
                continue
            aji = raw.get("aji")
            pq = raw.get("pq")
            realism = raw.get(realism_key)
            if aji is None or pq is None or realism is None:
                continue
            records.append(
                TileConditionRecord(
                    split=split,
                    tile_id=tile_id,
                    condition=condition,
                    n_groups=len(condition.split("+")),
                    aji=float(aji),
                    pq=float(pq),
                    realism_key=realism_key,
                    realism=float(realism),
                )
            )
    return records


def aggregate_tradeoff(records: Iterable[TileConditionRecord]) -> list[TradeoffAggregate]:
    """Aggregate metrics per split and per condition."""
    buckets: dict[tuple[str, str], list[TileConditionRecord]] = {}
    for record in records:
        buckets.setdefault((record.split, record.condition), []).append(record)

    rows: list[TradeoffAggregate] = []
    for condition in CANONICAL_CONDITION_LADDER:
        for split in ("paired", "unpaired"):
            group_records = buckets.get((split, condition), [])
            if not group_records:
                continue
            aji_vals = [record.aji for record in group_records]
            pq_vals = [record.pq for record in group_records]
            realism_vals = [record.realism for record in group_records]
            rows.append(
                TradeoffAggregate(
                    split=split,
                    condition=condition,
                    n_groups=group_records[0].n_groups,
                    realism_key=group_records[0].realism_key,
                    aji_mean=float(statistics.fmean(aji_vals)),
                    aji_sd=float(statistics.pstdev(aji_vals)) if len(aji_vals) > 1 else 0.0,
                    pq_mean=float(statistics.fmean(pq_vals)),
                    pq_sd=float(statistics.pstdev(pq_vals)) if len(pq_vals) > 1 else 0.0,
                    realism_mean=float(statistics.fmean(realism_vals)),
                    realism_sd=(
                        float(statistics.pstdev(realism_vals)) if len(realism_vals) > 1 else 0.0
                    ),
                    n_tiles=len(group_records),
                )
            )
    return rows


def mark_pareto_front(rows: Iterable[TradeoffAggregate]) -> list[TradeoffAggregate]:
    """Mark nondominated rows per split using AJI, PQ, and realism objectives."""
    split_rows: dict[str, list[TradeoffAggregate]] = {"paired": [], "unpaired": []}
    for row in rows:
        split_rows[row.split].append(row)

    marked: list[TradeoffAggregate] = []
    for split in ("paired", "unpaired"):
        current = split_rows[split]
        for row in current:
            dominated = False
            for other in current:
                if other.condition == row.condition:
                    continue
                no_worse = (
                    other.aji_mean >= row.aji_mean
                    and other.pq_mean >= row.pq_mean
                    and other.realism_mean <= row.realism_mean
                )
                strictly_better = (
                    other.aji_mean > row.aji_mean
                    or other.pq_mean > row.pq_mean
                    or other.realism_mean < row.realism_mean
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            marked.append(replace(row, is_pareto=not dominated))
    return marked


def write_tradeoff_csv(rows: Iterable[TradeoffAggregate], output_path: str | Path) -> Path:
    """Write the tradeoff summary CSV."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "condition",
        "n_groups",
        "aji_mean",
        "aji_sd",
        "pq_mean",
        "pq_sd",
        "realism_key",
        "realism_mean",
        "realism_sd",
        "is_pareto",
        "n_tiles",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row.split,
                    "condition": row.condition,
                    "n_groups": row.n_groups,
                    "aji_mean": row.aji_mean,
                    "aji_sd": row.aji_sd,
                    "pq_mean": row.pq_mean,
                    "pq_sd": row.pq_sd,
                    "realism_key": row.realism_key,
                    "realism_mean": row.realism_mean,
                    "realism_sd": row.realism_sd,
                    "is_pareto": row.is_pareto,
                    "n_tiles": row.n_tiles,
                }
            )
    return out_path
