from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.vis.leave_one_out_stats import (
    load_stats_payload,
    main,
    render_batch_summary,
    render_single_stats,
    resolve_input_mode,
    resolve_stats_paths,
    summarize_stats_paths,
)


def _write_stats(tile_dir: Path, payload: dict[str, dict[str, float]]) -> Path:
    tile_dir.mkdir(parents=True, exist_ok=True)
    stats_path = tile_dir / "leave_one_out_diff_stats.json"
    stats_path.write_text(json.dumps(payload), encoding="utf-8")
    return stats_path


def test_resolve_stats_paths_supports_json_tile_root_and_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "paired_ablation"
    tile_dir = dataset_root / "leave_one_out" / "tile_a"
    stats_path = _write_stats(
        tile_dir,
        {
            "cell_types": {"mean_diff": 1.0, "max_diff": 11.0, "pct_pixels_above_10": 2.0},
            "cell_state": {"mean_diff": 2.0, "max_diff": 12.0, "pct_pixels_above_10": 3.0},
            "vasculature": {"mean_diff": 3.0, "max_diff": 13.0, "pct_pixels_above_10": 4.0},
            "microenv": {"mean_diff": 4.0, "max_diff": 14.0, "pct_pixels_above_10": 5.0},
        },
    )

    assert resolve_stats_paths(stats_path) == [stats_path]
    assert resolve_stats_paths(tile_dir) == [stats_path]
    assert resolve_stats_paths(dataset_root / "leave_one_out") == [stats_path]
    assert resolve_stats_paths(dataset_root) == [stats_path]
    assert resolve_input_mode(stats_path)[0] == "single"
    assert resolve_input_mode(tile_dir)[0] == "single"
    assert resolve_input_mode(dataset_root / "leave_one_out")[0] == "summary"
    assert resolve_input_mode(dataset_root)[0] == "summary"


def test_summarize_stats_paths_returns_means_stds_and_representative_tile(tmp_path: Path) -> None:
    root = tmp_path / "leave_one_out"
    stats_paths = [
        _write_stats(
            root / "tile_a",
            {
                "cell_types": {"mean_diff": 2.0, "max_diff": 10.0, "pct_pixels_above_10": 5.0},
                "cell_state": {"mean_diff": 8.0, "max_diff": 30.0, "pct_pixels_above_10": 20.0},
                "vasculature": {"mean_diff": 3.0, "max_diff": 12.0, "pct_pixels_above_10": 6.0},
                "microenv": {"mean_diff": 7.0, "max_diff": 28.0, "pct_pixels_above_10": 18.0},
            },
        ),
        _write_stats(
            root / "tile_b",
            {
                "cell_types": {"mean_diff": 4.0, "max_diff": 14.0, "pct_pixels_above_10": 10.0},
                "cell_state": {"mean_diff": 16.0, "max_diff": 42.0, "pct_pixels_above_10": 38.0},
                "vasculature": {"mean_diff": 5.0, "max_diff": 16.0, "pct_pixels_above_10": 12.0},
                "microenv": {"mean_diff": 15.0, "max_diff": 40.0, "pct_pixels_above_10": 34.0},
            },
        ),
        _write_stats(
            root / "tile_c",
            {
                "cell_types": {"mean_diff": 6.0, "max_diff": 18.0, "pct_pixels_above_10": 15.0},
                "cell_state": {"mean_diff": 24.0, "max_diff": 54.0, "pct_pixels_above_10": 56.0},
                "vasculature": {"mean_diff": 7.0, "max_diff": 20.0, "pct_pixels_above_10": 16.0},
                "microenv": {"mean_diff": 23.0, "max_diff": 52.0, "pct_pixels_above_10": 50.0},
            },
        ),
    ]

    summary = summarize_stats_paths(stats_paths)

    assert summary["tile_count"] == 3
    assert summary["representative_tile"] == "tile_b"
    assert round(summary["means"]["cell_state"]["mean_diff"], 3) == 16.0
    assert round(summary["stds"]["cell_state"]["mean_diff"], 3) == round((128.0 / 3.0) ** 0.5, 3)
    assert summary["top_tiles"]["microenv"]["pct_pixels_above_10"]["tile_id"] == "tile_c"


def test_render_single_stats_sorts_groups_by_requested_metric(tmp_path: Path) -> None:
    stats_path = _write_stats(
        tmp_path / "tile_001",
        {
            "cell_types": {"mean_diff": 1.0, "max_diff": 11.0, "pct_pixels_above_10": 2.0},
            "cell_state": {"mean_diff": 8.0, "max_diff": 12.0, "pct_pixels_above_10": 25.0},
            "vasculature": {"mean_diff": 3.0, "max_diff": 13.0, "pct_pixels_above_10": 4.0},
            "microenv": {"mean_diff": 6.0, "max_diff": 14.0, "pct_pixels_above_10": 19.0},
        },
    )

    rendered = render_single_stats(
        stats_path,
        load_stats_payload(stats_path),
        metric_key="pct_pixels_above_10",
        output_format="table",
    )

    assert "Ranked by pct>10 descending" in rendered
    assert rendered.index("cell_state") < rendered.index("microenv")
    assert rendered.index("microenv") < rendered.index("vasculature")


def test_load_stats_payload_preserves_new_optional_fields(tmp_path: Path) -> None:
    stats_path = _write_stats(
        tmp_path / "tile_new",
        {
            "cell_types": {
                "mean_diff": 1.0,
                "max_diff": 11.0,
                "pct_pixels_above_10": 2.0,
                "delta_e_mean": 3.5,
                "delta_e_p99": 9.5,
                "ssim_loss_mean": 0.012,
                "ssim_loss_p99": 0.09,
                "causal_inside_mean_dE": 4.0,
                "causal_outside_mean_dE": 2.0,
                "causal_ratio": 2.0,
                "uni_cosine_drop": None,
            }
        },
    )

    payload = load_stats_payload(stats_path)

    assert payload["cell_types"]["delta_e_mean"] == 3.5
    assert payload["cell_types"]["ssim_loss_mean"] == 0.012
    assert payload["cell_types"]["causal_ratio"] == 2.0
    assert payload["cell_types"]["uni_cosine_drop"] is None


def test_render_batch_summary_json_contains_top_tiles(tmp_path: Path) -> None:
    root = tmp_path / "ablation_results"
    stats_paths = [
        _write_stats(
            root / "tile_a",
            {
                "cell_types": {"mean_diff": 1.0, "max_diff": 10.0, "pct_pixels_above_10": 5.0},
                "cell_state": {"mean_diff": 3.0, "max_diff": 30.0, "pct_pixels_above_10": 20.0},
                "vasculature": {"mean_diff": 2.0, "max_diff": 12.0, "pct_pixels_above_10": 6.0},
                "microenv": {"mean_diff": 4.0, "max_diff": 28.0, "pct_pixels_above_10": 18.0},
            },
        ),
        _write_stats(
            root / "tile_b",
            {
                "cell_types": {"mean_diff": 2.0, "max_diff": 11.0, "pct_pixels_above_10": 7.0},
                "cell_state": {"mean_diff": 5.0, "max_diff": 35.0, "pct_pixels_above_10": 26.0},
                "vasculature": {"mean_diff": 3.0, "max_diff": 13.0, "pct_pixels_above_10": 9.0},
                "microenv": {"mean_diff": 6.0, "max_diff": 32.0, "pct_pixels_above_10": 24.0},
            },
        ),
    ]

    rendered = render_batch_summary(
        root,
        summarize_stats_paths(stats_paths),
        metric_key="pct_pixels_above_10",
        output_format="json",
    )
    payload = json.loads(rendered)

    assert payload["mode"] == "summary"
    assert payload["tile_count"] == 2
    assert payload["top_tiles"]["cell_state"]["pct_pixels_above_10"]["tile_id"] == "tile_b"


def test_main_writes_output_file(tmp_path: Path, capsys) -> None:
    dataset_root = tmp_path / "paired_ablation"
    _write_stats(
        dataset_root / "leave_one_out" / "tile_a",
        {
            "cell_types": {"mean_diff": 1.0, "max_diff": 11.0, "pct_pixels_above_10": 2.0},
            "cell_state": {"mean_diff": 8.0, "max_diff": 12.0, "pct_pixels_above_10": 25.0},
            "vasculature": {"mean_diff": 3.0, "max_diff": 13.0, "pct_pixels_above_10": 4.0},
            "microenv": {"mean_diff": 6.0, "max_diff": 14.0, "pct_pixels_above_10": 19.0},
        },
    )
    output_path = tmp_path / "summary.md"

    main([str(dataset_root), "--format", "markdown", "--output", str(output_path)])

    captured = capsys.readouterr()
    assert "Wrote leave-one-out stats summary" in captured.out
    assert output_path.is_file()
    assert "| group | mean_avg | mean_sd |" in output_path.read_text(encoding="utf-8")
