from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.a2_decomposition.metrics import MODE_KEYS, summarize_decomposition_metrics, write_summary_csv
from src.paper_figures.fig_uni_tme_decomposition import save_uni_tme_decomposition_figure


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((32, 32, 3), value, dtype=np.uint8)
    arr[8:24, 8:24, 0] = min(255, value + 40)
    Image.fromarray(arr).save(path)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    generated_root = tmp_path / "generated"
    metrics_root = tmp_path / "metrics"
    orion_root = tmp_path / "orion"
    tile_id = "tile_a"

    for idx, mode_key in enumerate(MODE_KEYS):
        _write_rgb(generated_root / tile_id / f"{mode_key}.png", 80 + idx * 25)

    _write_rgb(orion_root / "he" / f"{tile_id}.png", 120)
    (orion_root / "exp_channels" / "cell_masks").mkdir(parents=True)
    np.save(orion_root / "exp_channels" / "cell_masks" / f"{tile_id}.npy", np.eye(32, dtype=np.float32))

    per_condition = {}
    for idx, mode_key in enumerate(MODE_KEYS):
        per_condition[mode_key] = {
            "lpips": 0.30 + idx * 0.02,
            "pq": 0.40 - idx * 0.03,
            "dice": 0.70 - idx * 0.04,
            "style_hed": 0.05 + idx * 0.01,
        }
    metrics_path = metrics_root / tile_id / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps({"version": 2, "tile_id": tile_id, "per_condition": per_condition}, indent=2),
        encoding="utf-8",
    )
    representative_json = tmp_path / "representative_tile.json"
    representative_json.write_text(json.dumps({"tile_id": tile_id}), encoding="utf-8")

    fud_json = tmp_path / "fud_scores.json"
    fud_json.write_text(
        json.dumps({mode_key: 20.0 + idx * 2.0 for idx, mode_key in enumerate(MODE_KEYS)}),
        encoding="utf-8",
    )
    summary_csv = write_summary_csv(
        summarize_decomposition_metrics(metrics_root=metrics_root, fud_json=fud_json),
        tmp_path / "summary.csv",
    )
    return generated_root, metrics_root, summary_csv, representative_json, orion_root


def test_save_uni_tme_decomposition_figure_renders_fixture(tmp_path: Path):
    generated_root, metrics_root, summary_csv, representative_json, orion_root = _write_fixture(tmp_path)
    out_png = tmp_path / "fig.png"

    result = save_uni_tme_decomposition_figure(
        out_png=out_png,
        generated_root=generated_root,
        metrics_root=metrics_root,
        summary_csv=summary_csv,
        representative_json=representative_json,
        orion_root=orion_root,
        dpi=80,
    )

    assert result == out_png
    assert out_png.is_file()
    with Image.open(out_png) as image:
        assert image.width > 400
        assert image.height > 200
        assert image.width > image.height  # landscape
