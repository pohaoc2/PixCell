from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.render_ablation_html_report import load_dataset_summary, render_report_html


def _write_metrics(tile_dir: Path, payload: dict) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (255, 255, 255)).save(path)


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
