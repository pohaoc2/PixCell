"""Collect and aggregate leave-one-out visibility statistics."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, GROUP_SHORT_LABELS


@dataclass(frozen=True)
class TileVisibilityRecord:
    """One per-tile, per-group visibility measurement."""

    split: Literal["paired", "unpaired"]
    tile_id: str
    group: str
    mean_diff: float
    max_diff: float
    pct_pixels_above_10: float
    diff_png: Path | None


@dataclass(frozen=True)
class VisibilityAggregate:
    """Aggregate visibility metrics for one split and one group."""

    split: Literal["paired", "unpaired"]
    group: str
    mean_diff: float
    mean_diff_sd: float
    pct_pixels_above_10: float
    pct_pixels_above_10_sd: float
    n_tiles: int


@dataclass(frozen=True)
class VisibilityRow:
    """Merged paired/unpaired summary row for export and plotting."""

    group: str
    group_label: str
    sort_rank: int
    paired: VisibilityAggregate
    unpaired: VisibilityAggregate


@dataclass(frozen=True)
class InsetTile:
    """Representative diff image selected for the figure inset."""

    tile_id: str
    group: str
    source_path: Path
    label: str
    score: float


def _empty_aggregate(split: Literal["paired", "unpaired"], group: str) -> VisibilityAggregate:
    return VisibilityAggregate(
        split=split,
        group=group,
        mean_diff=math.nan,
        mean_diff_sd=math.nan,
        pct_pixels_above_10=math.nan,
        pct_pixels_above_10_sd=math.nan,
        n_tiles=0,
    )


def _candidate_stats_roots(path: str | Path) -> list[Path]:
    base = Path(path)
    candidates = [base]
    for child_name in ("ablation_results", "leave_one_out"):
        child = base / child_name
        if child.is_dir():
            candidates.append(child)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.exists():
            seen.add(resolved)
            ordered.append(candidate)
    return ordered


def collect_split_records(
    split: Literal["paired", "unpaired"],
    stats_root: str | Path,
) -> list[TileVisibilityRecord]:
    """Collect per-tile visibility records from one split root."""
    files: list[Path] = []
    seen_files: set[Path] = set()
    for root in _candidate_stats_roots(stats_root):
        for path in sorted(root.glob("*/leave_one_out_diff_stats.json")):
            resolved = path.resolve()
            if resolved not in seen_files:
                seen_files.add(resolved)
                files.append(path)

    records: list[TileVisibilityRecord] = []
    for stats_path in files:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        diff_png = stats_path.with_name("leave_one_out_diff.png")
        diff_path = diff_png if diff_png.is_file() else None
        tile_id = stats_path.parent.name
        for group in FOUR_GROUP_ORDER:
            raw = payload.get(group)
            if not isinstance(raw, dict):
                continue
            mean_diff = raw.get("mean_diff")
            pct = raw.get("pct_pixels_above_10")
            if mean_diff is None or pct is None:
                continue
            records.append(
                TileVisibilityRecord(
                    split=split,
                    tile_id=tile_id,
                    group=group,
                    mean_diff=float(mean_diff),
                    max_diff=float(raw.get("max_diff", math.nan)),
                    pct_pixels_above_10=float(pct),
                    diff_png=diff_path,
                )
            )
    return records


def aggregate_visibility(records: Iterable[TileVisibilityRecord]) -> dict[str, VisibilityAggregate]:
    """Aggregate per-group visibility means and standard deviations."""
    by_group: dict[str, list[TileVisibilityRecord]] = {group: [] for group in FOUR_GROUP_ORDER}
    split: Literal["paired", "unpaired"] | None = None
    for record in records:
        split = record.split
        if record.group in by_group:
            by_group[record.group].append(record)

    if split is None:
        return {}

    aggregates: dict[str, VisibilityAggregate] = {}
    for group, group_records in by_group.items():
        if not group_records:
            continue
        mean_diffs = [record.mean_diff for record in group_records]
        pct_values = [record.pct_pixels_above_10 for record in group_records]
        aggregates[group] = VisibilityAggregate(
            split=split,
            group=group,
            mean_diff=float(statistics.fmean(mean_diffs)),
            mean_diff_sd=float(statistics.pstdev(mean_diffs)) if len(mean_diffs) > 1 else 0.0,
            pct_pixels_above_10=float(statistics.fmean(pct_values)),
            pct_pixels_above_10_sd=(
                float(statistics.pstdev(pct_values)) if len(pct_values) > 1 else 0.0
            ),
            n_tiles=len(group_records),
        )
    return aggregates


def build_visibility_rows(
    paired: dict[str, VisibilityAggregate],
    unpaired: dict[str, VisibilityAggregate],
) -> list[VisibilityRow]:
    """Merge paired and unpaired aggregates into plot-ready rows."""
    groups = [group for group in FOUR_GROUP_ORDER if group in paired or group in unpaired]
    groups.sort(
        key=lambda group: paired.get(group, _empty_aggregate("paired", group)).mean_diff,
        reverse=True,
    )
    rows: list[VisibilityRow] = []
    for sort_rank, group in enumerate(groups, start=1):
        rows.append(
            VisibilityRow(
                group=group,
                group_label=GROUP_SHORT_LABELS.get(group, group),
                sort_rank=sort_rank,
                paired=paired.get(group, _empty_aggregate("paired", group)),
                unpaired=unpaired.get(group, _empty_aggregate("unpaired", group)),
            )
        )
    return rows


def select_inset_tiles(records: Iterable[TileVisibilityRecord], max_tiles: int = 6) -> list[InsetTile]:
    """Pick representative inset tiles from the paired split when possible."""
    paired = [record for record in records if record.split == "paired" and record.diff_png is not None]
    pool = paired or [record for record in records if record.diff_png is not None]

    selected: list[InsetTile] = []
    for group in FOUR_GROUP_ORDER:
        candidates = [record for record in pool if record.group == group]
        if not candidates:
            continue
        best = max(candidates, key=lambda record: (record.pct_pixels_above_10, record.mean_diff))
        selected.append(
            InsetTile(
                tile_id=best.tile_id,
                group=group,
                source_path=best.diff_png or Path(),
                label=f"{GROUP_SHORT_LABELS.get(group, group)}: {best.tile_id}",
                score=best.pct_pixels_above_10,
            )
        )
        if len(selected) >= max_tiles:
            break
    return selected


def write_visibility_summary_csv(rows: Iterable[VisibilityRow], output_path: str | Path) -> Path:
    """Write the merged visibility summary CSV."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "group_label",
        "sort_rank",
        "paired_mean_diff",
        "paired_mean_diff_sd",
        "paired_pct_pixels_above_10",
        "paired_pct_pixels_above_10_sd",
        "paired_n_tiles",
        "unpaired_mean_diff",
        "unpaired_mean_diff_sd",
        "unpaired_pct_pixels_above_10",
        "unpaired_pct_pixels_above_10_sd",
        "unpaired_n_tiles",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "group": row.group,
                    "group_label": row.group_label,
                    "sort_rank": row.sort_rank,
                    "paired_mean_diff": row.paired.mean_diff,
                    "paired_mean_diff_sd": row.paired.mean_diff_sd,
                    "paired_pct_pixels_above_10": row.paired.pct_pixels_above_10,
                    "paired_pct_pixels_above_10_sd": row.paired.pct_pixels_above_10_sd,
                    "paired_n_tiles": row.paired.n_tiles,
                    "unpaired_mean_diff": row.unpaired.mean_diff,
                    "unpaired_mean_diff_sd": row.unpaired.mean_diff_sd,
                    "unpaired_pct_pixels_above_10": row.unpaired.pct_pixels_above_10,
                    "unpaired_pct_pixels_above_10_sd": row.unpaired.pct_pixels_above_10_sd,
                    "unpaired_n_tiles": row.unpaired.n_tiles,
                }
            )
    return out_path
