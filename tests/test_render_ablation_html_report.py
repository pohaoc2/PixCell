from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.render_ablation_html_report import (
    DatasetSummary,
    build_leave_one_out_figure,
    build_metric_trends_figure,
    load_dataset_summary,
    render_comparison_table,
    render_report_html,
)
from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER, condition_metric_key


def _write_metrics(tile_dir: Path, payload: dict) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (255, 255, 255)).save(path)


def _write_fid_scores(path: Path, payload: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_cell_mask(path: Path, mask: list[list[int]]) -> None:
    from PIL import Image
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255)).save(path)


def _write_manifest(tile_dir: Path, tile_id: str) -> None:
    payload = {
        "version": 1,
        "tile_id": tile_id,
        "group_names": ["cell_types", "cell_state", "vasculature", "microenv"],
        "sections": [
            {
                "title": "1 active group",
                "subset_size": 1,
                "entries": [
                    {"active_groups": ["cell_types"], "image_path": "singles/01_cell_types.png"},
                    {"active_groups": ["cell_state"], "image_path": "singles/02_cell_state.png"},
                ],
            },
            {
                "title": "2 active groups",
                "subset_size": 2,
                "entries": [
                    {
                        "active_groups": ["cell_types", "cell_state"],
                        "image_path": "pairs/01_cell_types__cell_state.png",
                    },
                ],
            },
            {
                "title": "3 active groups",
                "subset_size": 3,
                "entries": [
                    {
                        "active_groups": ["cell_types", "cell_state", "microenv"],
                        "image_path": "triples/01_cell_types__cell_state__microenv.png",
                    },
                ],
            },
        ],
        "cell_mask_path": "cell_mask.png",
    }
    (tile_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    for rel_path in (
        "singles/01_cell_types.png",
        "singles/02_cell_state.png",
        "pairs/01_cell_types__cell_state.png",
        "triples/01_cell_types__cell_state__microenv.png",
        "all/generated_he.png",
        "cell_mask.png",
    ):
        _write_png(tile_dir / rel_path)


def _make_summary(
    *,
    slug: str,
    title: str,
    metric_keys: list[str],
    condition_stats: dict[str, dict[str, tuple[float, float]]] | None = None,
    by_cardinality_stats: dict[int, dict[str, tuple[float, float]]] | None = None,
    loo_summary: dict[str, dict[str, float]] | None = None,
    loo_stats: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> DatasetSummary:
    return DatasetSummary(
        slug=slug,
        title=title,
        metrics_root=ROOT,
        dataset_root=ROOT,
        tile_count=1,
        metric_keys=metric_keys,
        condition_stats=condition_stats or {},
        by_cardinality={},
        by_cardinality_stats=by_cardinality_stats or {},
        best_worst={},
        added_effects={},
        presence_absence={},
        added_effect_stats={},
        presence_absence_stats={},
        loo_summary=loo_summary or {},
        loo_stats=loo_stats or {},
        representative_tile=None,
        key_takeaways=[],
        ablation_grid_path=None,
        loo_diff_path=None,
    )


def _full_condition_stats(metric_key: str, *, base: float, step: float, std: float) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    index = 0
    for size in range(1, len(FOUR_GROUP_ORDER) + 1):
        for cond in combinations(FOUR_GROUP_ORDER, size):
            out[condition_metric_key(tuple(cond))] = {
                metric_key: (base + index * step, std)
            }
            index += 1
    return out


def test_render_report_html_includes_expected_sections(tmp_path: Path) -> None:
    paired_metrics_root = tmp_path / "paired" / "ablation_results"
    unpaired_metrics_root = tmp_path / "unpaired" / "ablation_results"
    paired_root = tmp_path / "paired"
    unpaired_root = tmp_path / "unpaired"

    _write_metrics(
        paired_metrics_root / "tile_a",
        {
            "version": 2,
            "tile_id": "tile_a",
            "per_condition": {
                "cell_types": {"cosine": 0.55, "lpips": 0.44, "aji": 0.06, "pq": 0.02, "fid": 68.0},
                "cell_state": {"cosine": 0.58, "lpips": 0.42, "aji": 0.20, "pq": 0.14, "fid": 67.0},
                "cell_types+cell_state": {"cosine": 0.60, "lpips": 0.39, "aji": 0.28, "pq": 0.22, "fid": 66.5},
                "cell_types+cell_state+microenv": {"cosine": 0.61, "lpips": 0.38, "aji": 0.36, "pq": 0.31, "fid": 66.2},
                "cell_types+cell_state+microenv+vasculature": {"cosine": 0.62, "lpips": 0.37, "aji": 0.44, "pq": 0.39, "fid": 67.4},
            },
        },
    )
    _write_metrics(
        unpaired_metrics_root / "tile_b",
        {
            "version": 2,
            "tile_id": "tile_b",
            "per_condition": {
                "cell_types": {"aji": 0.04, "pq": 0.01, "fid": 69.0, "style_hed": 0.08},
                "cell_state": {"aji": 0.16, "pq": 0.10, "fid": 68.7, "style_hed": 0.079},
                "cell_types+cell_state": {"aji": 0.22, "pq": 0.15, "fid": 67.8, "style_hed": 0.074},
                "cell_types+cell_state+microenv": {"aji": 0.31, "pq": 0.23, "fid": 64.0, "style_hed": 0.069},
                "cell_types+cell_state+microenv+vasculature": {"aji": 0.39, "pq": 0.32, "fid": 65.4, "style_hed": 0.071},
            },
        },
    )

    for root, tile_id in ((paired_root, "tile_a"), (unpaired_root, "tile_b")):
        stats_path = root / "leave_one_out" / tile_id / "leave_one_out_diff_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(
                {
                    "cell_types": {"mean_diff": 3.0, "max_diff": 10.0, "pct_pixels_above_10": 8.0},
                    "cell_state": {"mean_diff": 12.0, "max_diff": 30.0, "pct_pixels_above_10": 25.0},
                    "vasculature": {"mean_diff": 4.0, "max_diff": 12.0, "pct_pixels_above_10": 9.0},
                    "microenv": {"mean_diff": 11.0, "max_diff": 28.0, "pct_pixels_above_10": 23.0},
                }
            ),
            encoding="utf-8",
        )
        _write_png(root / "ablation_results" / tile_id / "ablation_grid.png")
        _write_png(root / "leave_one_out" / tile_id / "leave_one_out_diff.png")
        _write_manifest(root / "ablation_results" / tile_id, tile_id)

    paired = load_dataset_summary(
        slug="paired",
        title="Paired",
        metrics_root=paired_metrics_root,
        dataset_root=paired_root,
    )
    unpaired = load_dataset_summary(
        slug="unpaired",
        title="Unpaired",
        metrics_root=unpaired_metrics_root,
        dataset_root=unpaired_root,
    )

    report = render_report_html("Ablation report", [paired, unpaired], tmp_path / "report.html")

    assert "Metric Tradeoffs" in report
    assert "Channel Effect Sizes" in report
    assert "Representative evidence" in report
    assert "Paired" in report
    assert "Unpaired" in report
    assert "data:image/png;base64" in report
    assert "../" not in report
    assert "paired/ablation_results/tile_a/ablation_grid.png" in report
    assert "unpaired/leave_one_out/tile_b/leave_one_out_diff.png" in report
    assert "rowspan='5'" in report
    assert "class='metric-text'" in report
    assert "CT / CS / Vas / Env" in report
    assert "class='condition-glyph'" in report
    assert "class='condition-dot is-active'" in report


def test_build_metric_trends_figure_uses_dashed_unpaired_std() -> None:
    paired = _make_summary(
        slug="paired",
        title="Paired",
        metric_keys=["aji"],
        condition_stats=_full_condition_stats("aji", base=0.12, step=0.01, std=0.01),
    )
    unpaired = _make_summary(
        slug="unpaired",
        title="Unpaired",
        metric_keys=["aji"],
        condition_stats=_full_condition_stats("aji", base=0.08, step=0.009, std=0.012),
    )

    fig = build_metric_trends_figure([paired, unpaired])
    try:
        aji_ax = next(ax for ax in fig.axes if ax.get_title().startswith("AJI"))
        dot_ax = next(
            ax for ax in fig.axes
            if ax is not aji_ax and not ax.axison and len(ax.collections) == 60
        )
        paired_line = aji_ax.lines[0]
        unpaired_line = aji_ax.lines[3]
        linestyles = [collection.get_linestyle() for collection in aji_ax.collections]

        assert len(aji_ax.lines) >= 4
        assert len(linestyles) == 2
        assert paired_line.get_linestyle() == "-"
        assert unpaired_line.get_linestyle() == ":"
        assert paired_line.get_markerfacecolor() == "#000000"
        assert unpaired_line.get_markerfacecolor() == "white"
        assert linestyles[0] != linestyles[1]
        assert linestyles[1][0][1]
        assert aji_ax.get_xlim() == (-0.55, 14.55)
        assert len(dot_ax.collections) == 60
    finally:
        fig.clf()


def test_build_leave_one_out_figure_uses_white_paired_bars() -> None:
    loo_summary = {
        "cell_types": {"mean_diff": 3.0, "pct_pixels_above_10": 8.0},
        "cell_state": {"mean_diff": 12.0, "pct_pixels_above_10": 25.0},
        "vasculature": {"mean_diff": 4.0, "pct_pixels_above_10": 9.0},
        "microenv": {"mean_diff": 11.0, "pct_pixels_above_10": 23.0},
    }
    loo_stats = {
        group: {
            "mean_diff": (values["mean_diff"], 0.5),
            "pct_pixels_above_10": (values["pct_pixels_above_10"], 1.0),
        }
        for group, values in loo_summary.items()
    }
    paired = _make_summary(
        slug="paired",
        title="Paired",
        metric_keys=["aji"],
        loo_summary=loo_summary,
        loo_stats=loo_stats,
    )
    unpaired = _make_summary(
        slug="unpaired",
        title="Unpaired",
        metric_keys=["aji"],
        loo_summary=loo_summary,
        loo_stats=loo_stats,
    )

    fig = build_leave_one_out_figure([paired, unpaired])
    try:
        first_panel = fig.axes[0]
        paired_bar = first_panel.patches[0]
        unpaired_bar = first_panel.patches[1]

        assert paired_bar.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert paired_bar.get_hatch() in (None, "")
        assert unpaired_bar.get_hatch() == "//"
    finally:
        fig.clf()


def test_render_comparison_table_omits_unpaired_cosine_and_lpips() -> None:
    paired = _make_summary(
        slug="paired",
        title="Paired",
        metric_keys=["cosine", "lpips", "aji"],
    )
    object.__setattr__(
        paired,
        "best_worst",
        {
            "cosine": {
                "best_condition": "cell_types",
                "best_value": 0.61,
                "worst_condition": "cell_state",
                "worst_value": 0.41,
            },
            "lpips": {
                "best_condition": "cell_types+cell_state",
                "best_value": 0.22,
                "worst_condition": "cell_types",
                "worst_value": 0.44,
            },
            "aji": {
                "best_condition": "cell_types+microenv",
                "best_value": 0.35,
                "worst_condition": "cell_state",
                "worst_value": 0.14,
            },
        },
    )
    unpaired = _make_summary(
        slug="unpaired",
        title="Unpaired",
        metric_keys=["cosine", "lpips", "aji"],
    )
    object.__setattr__(
        unpaired,
        "best_worst",
        {
            "cosine": {
                "best_condition": "cell_types",
                "best_value": 0.55,
                "worst_condition": "cell_state",
                "worst_value": 0.32,
            },
            "lpips": {
                "best_condition": "cell_types+cell_state",
                "best_value": 0.24,
                "worst_condition": "cell_types",
                "worst_value": 0.47,
            },
            "aji": {
                "best_condition": "cell_types+microenv",
                "best_value": 0.28,
                "worst_condition": "cell_state",
                "worst_value": 0.11,
            },
        },
    )

    table_html = render_comparison_table([paired, unpaired])

    assert table_html.count(">Cosine<") == 1
    assert table_html.count(">LPIPS<") == 1
    assert table_html.count(">AJI<") == 2


def test_render_report_html_self_contained_embeds_evidence_images(tmp_path: Path) -> None:
    metrics_root = tmp_path / "paired" / "ablation_results"
    dataset_root = tmp_path / "paired"
    tile_id = "tile_a"

    _write_metrics(
        metrics_root / tile_id,
        {
            "version": 2,
            "tile_id": tile_id,
            "per_condition": {
                "cell_types": {"aji": 0.06, "pq": 0.02, "fid": 68.0},
                "cell_state": {"aji": 0.20, "pq": 0.14, "fid": 67.0},
                "cell_types+cell_state": {"aji": 0.28, "pq": 0.22, "fid": 66.5},
                "cell_types+cell_state+microenv": {"aji": 0.36, "pq": 0.31, "fid": 66.2},
                "cell_types+cell_state+microenv+vasculature": {"aji": 0.44, "pq": 0.39, "fid": 67.4},
            },
        },
    )
    stats_path = dataset_root / "leave_one_out" / tile_id / "leave_one_out_diff_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "cell_types": {"mean_diff": 3.0, "max_diff": 10.0, "pct_pixels_above_10": 8.0},
                "cell_state": {"mean_diff": 12.0, "max_diff": 30.0, "pct_pixels_above_10": 25.0},
                "vasculature": {"mean_diff": 4.0, "max_diff": 12.0, "pct_pixels_above_10": 9.0},
                "microenv": {"mean_diff": 11.0, "max_diff": 28.0, "pct_pixels_above_10": 23.0},
            }
        ),
        encoding="utf-8",
    )
    _write_png(dataset_root / "ablation_results" / tile_id / "ablation_grid.png")
    _write_png(dataset_root / "leave_one_out" / tile_id / "leave_one_out_diff.png")
    _write_manifest(dataset_root / "ablation_results" / tile_id, tile_id)

    summary = load_dataset_summary(
        slug="paired",
        title="Paired",
        metrics_root=metrics_root,
        dataset_root=dataset_root,
    )

    report = render_report_html(
        "Ablation report",
        [summary],
        tmp_path / "report.html",
        self_contained=True,
    )

    assert "data:image/png;base64" in report
    assert "src='../paired/ablation_results/tile_a/ablation_grid.png'" not in report


def test_load_dataset_summary_computes_missing_paired_style_hed(tmp_path: Path) -> None:
    paired_metrics_root = tmp_path / "paired" / "ablation_results"
    paired_root = tmp_path / "paired"
    tile_id = "tile_a"

    _write_metrics(
        paired_metrics_root / tile_id,
        {
            "version": 2,
            "tile_id": tile_id,
            "per_condition": {
                "cell_types": {"cosine": 0.55, "lpips": 0.44, "aji": 0.06, "pq": 0.02, "fid": 68.0},
                "cell_state": {"cosine": 0.58, "lpips": 0.42, "aji": 0.20, "pq": 0.14, "fid": 67.0},
                "cell_types+cell_state": {"cosine": 0.60, "lpips": 0.39, "aji": 0.28, "pq": 0.22, "fid": 66.5},
                "cell_types+cell_state+microenv": {"cosine": 0.61, "lpips": 0.38, "aji": 0.36, "pq": 0.31, "fid": 66.2},
                "cell_types+cell_state+microenv+vasculature": {
                    "cosine": 0.62,
                    "lpips": 0.37,
                    "aji": 0.44,
                    "pq": 0.39,
                    "fid": 67.4,
                },
            },
        },
    )
    _write_manifest(paired_metrics_root / tile_id, tile_id)
    _write_png(paired_root / "data" / "orion-crc33" / "he" / f"{tile_id}.png")

    summary = load_dataset_summary(
        slug="paired",
        title="Paired",
        metrics_root=paired_metrics_root,
        dataset_root=paired_root,
    )

    assert "style_hed" in summary.metric_keys
    assert summary.best_worst["style_hed"]["best_value"] == 0.0
    assert summary.by_cardinality[4]["style_hed"] == 0.0


def test_load_dataset_summary_uses_fid_scores_when_metrics_json_lacks_fid(tmp_path: Path) -> None:
    metrics_root = tmp_path / "paired" / "ablation_results"
    dataset_root = tmp_path / "paired"
    tile_id = "tile_a"

    _write_metrics(
        metrics_root / tile_id,
        {
            "version": 2,
            "tile_id": tile_id,
            "per_condition": {
                "cell_types": {"cosine": 0.55, "lpips": 0.44, "aji": 0.06, "pq": 0.02},
                "cell_state": {"cosine": 0.58, "lpips": 0.42, "aji": 0.20, "pq": 0.14},
                "cell_types+cell_state": {"cosine": 0.60, "lpips": 0.39, "aji": 0.28, "pq": 0.22},
                "cell_types+cell_state+microenv": {"cosine": 0.61, "lpips": 0.38, "aji": 0.36, "pq": 0.31},
                "cell_types+cell_state+microenv+vasculature": {"cosine": 0.62, "lpips": 0.37, "aji": 0.44, "pq": 0.39},
            },
        },
    )
    _write_fid_scores(
        metrics_root / "fid_scores.json",
        {
            "cell_types": 68.0,
            "cell_state": 67.0,
            "cell_types+cell_state": 66.5,
            "cell_types+cell_state+microenv": 66.2,
            "cell_types+cell_state+microenv+vasculature": 67.4,
        },
    )

    summary = load_dataset_summary(
        slug="paired",
        title="Paired",
        metrics_root=metrics_root,
        dataset_root=dataset_root,
        enable_style_hed_backfill=False,
    )

    assert "fid" in summary.metric_keys
    assert summary.condition_stats["cell_types"]["fid"] == (68.0, 0.0)
    assert summary.by_cardinality[4]["fid"] == 67.4


def test_load_dataset_summary_filters_by_min_gt_cells(tmp_path: Path) -> None:
    metrics_root = tmp_path / "paired" / "ablation_results"
    dataset_root = tmp_path / "paired"

    _write_metrics(
        metrics_root / "tile_a",
        {
            "version": 2,
            "tile_id": "tile_a",
            "per_condition": {
                "cell_types": {"aji": 0.10, "pq": 0.05},
                "cell_types+cell_state+microenv+vasculature": {"aji": 0.40, "pq": 0.30},
            },
        },
    )
    _write_metrics(
        metrics_root / "tile_b",
        {
            "version": 2,
            "tile_id": "tile_b",
            "per_condition": {
                "cell_types": {"aji": 0.80, "pq": 0.70},
                "cell_types+cell_state+microenv+vasculature": {"aji": 0.90, "pq": 0.85},
            },
        },
    )
    _write_cell_mask(
        dataset_root / "data" / "orion-crc33" / "exp_channels" / "cell_masks" / "tile_a.png",
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
    )
    _write_cell_mask(
        dataset_root / "data" / "orion-crc33" / "exp_channels" / "cell_masks" / "tile_b.png",
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )

    summary = load_dataset_summary(
        slug="paired",
        title="Paired",
        metrics_root=metrics_root,
        dataset_root=dataset_root,
        enable_style_hed_backfill=False,
        min_gt_cells=2,
    )

    assert summary.tile_count == 1
    assert summary.condition_stats["cell_types"]["aji"] == (0.10, 0.0)
    assert summary.by_cardinality[4]["pq"] == 0.30
