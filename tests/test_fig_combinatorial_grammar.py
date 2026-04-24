"""Tests for figure 6 redesign helpers and panel renderers."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.paper_figures.fig_combinatorial_grammar_panels import _shared


def _write_rgb(path: Path, value: int, *, hot_box: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((32, 32, 3), value, dtype=np.uint8)
    if hot_box:
        arr[8:24, 8:24, 0] = min(255, value + 100)
    Image.fromarray(arr).save(path)


def _write_sweep_tile(generated_root: Path, anchor: str, state: str, oxygen: str, glucose: str, value: int) -> None:
    arr = np.full((16, 16, 3), value, dtype=np.uint8)
    out = generated_root / anchor / f"{state}_{oxygen}_{glucose}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out)


def _write_reference(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _populate_anchor_sweep(generated_root: Path, anchor: str) -> None:
    states = ("prolif", "nonprolif", "dead")
    levels = ("low", "mid", "high")
    value = 100
    for state in states:
        for oxygen in levels:
            for glucose in levels:
                _write_sweep_tile(generated_root, anchor, state, oxygen, glucose, value)
                value += 1


def _make_signature_rows(anchors_to_n_conditions: dict[str, int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    states = ("prolif", "nonprolif", "dead")
    levels = ("low", "mid", "high")
    for anchor, n_conds in anchors_to_n_conditions.items():
        i = 0
        for state in states:
            for oxygen in levels:
                for glucose in levels:
                    if i >= n_conds:
                        break
                    rows.append(
                        {
                            "anchor_id": anchor,
                            "cell_state": state,
                            "oxygen_label": oxygen,
                            "glucose_label": glucose,
                            "mean_cell_size": str(10.0 + i + (0 if anchor == "a0" else 5.0)),
                            "nuclear_density": str(0.1 * i),
                            "nucleus_area_median": str(20.0 + i),
                            "nucleus_area_iqr": str(2.0 + i),
                            "hematoxylin_burden": "0.5",
                            "hematoxylin_ratio": "0.5",
                            "eosin_ratio": "0.5",
                            "glcm_contrast": "1.0",
                            "glcm_homogeneity": "0.8",
                        }
                    )
                    i += 1
    return rows


def _make_residual_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    states = ("prolif", "nonprolif", "dead")
    levels = ("low", "mid", "high")
    i = 0
    for state in states:
        for oxygen in levels:
            for glucose in levels:
                rows.append(
                    {
                        "cell_state": state,
                        "oxygen_label": oxygen,
                        "glucose_label": glucose,
                        "residual_l2_norm": str(1.0 + i),
                        "residual_mean_cell_size": str(0.3 - 0.01 * i),
                        "residual_nucleus_area_median": str(0.2 + 0.01 * i),
                        "residual_nucleus_area_iqr": str(-0.05 - 0.005 * i),
                        "residual_nuclear_density": str(0.001 * i),
                        "residual_hematoxylin_burden": "0.001",
                        "residual_hematoxylin_ratio": "0.0005",
                        "residual_eosin_ratio": "-0.001",
                        "residual_glcm_contrast": "0.0001",
                        "residual_glcm_homogeneity": "0.00005",
                    }
                )
                i += 1
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_compute_pixel_diff_shape_and_nonneg() -> None:
    ref = np.full((32, 32, 3), 100, dtype=np.uint8)
    cond = np.full((32, 32, 3), 130, dtype=np.uint8)
    diff = _shared.compute_pixel_diff(cond, ref)
    assert diff.shape == (32, 32)
    assert diff.dtype == np.float32
    assert np.all(diff >= 0)
    assert np.isclose(diff.mean(), 30.0)


def test_condition_id_format() -> None:
    assert _shared.condition_id("prolif", "low", "high") == "prolif_low_high"


def test_residual_lookup_extracts_all_residual_columns() -> None:
    rows = [
        {
            "cell_state": "prolif",
            "oxygen_label": "low",
            "glucose_label": "low",
            "residual_l2_norm": "1.5",
            "residual_mean_cell_size": "-0.4",
            "n_anchors": "20",
        }
    ]
    lookup = _shared.residual_lookup(rows)
    key = ("prolif", "low", "low")
    assert key in lookup
    assert lookup[key]["residual_l2_norm"] == 1.5
    assert lookup[key]["residual_mean_cell_size"] == -0.4
    assert "n_anchors" not in lookup[key]


def test_pick_representative_anchor_returns_max_coverage() -> None:
    rows = _make_signature_rows({"a0": 27, "a1": 20, "a2": 27})
    assert _shared.pick_representative_anchor(rows) == "a0"


def test_compute_anchor_sweep_magnitude_returns_per_anchor_float() -> None:
    rows = _make_signature_rows({"a0": 27, "a1": 27})
    mags = _shared.compute_anchor_sweep_magnitude(rows)
    assert set(mags.keys()) == {"a0", "a1"}
    assert all(isinstance(value, float) for value in mags.values())


def test_select_si_anchors_returns_four_distinct_with_representative_first() -> None:
    rows = _make_signature_rows({f"a{i}": 27 for i in range(20)})
    picks = _shared.select_si_anchors(rows, representative_id="a0", reference_exists_fn=lambda _aid: True)
    assert len(picks) == 4
    assert picks[0] == "a0"
    assert len(set(picks)) == 4


def test_select_si_anchors_skips_anchors_missing_reference() -> None:
    rows = _make_signature_rows({f"a{i}": 27 for i in range(20)})
    blocked = {"a5", "a10"}
    picks = _shared.select_si_anchors(
        rows,
        representative_id="a0",
        reference_exists_fn=lambda anchor_id: anchor_id not in blocked,
    )
    assert all(anchor_id not in blocked for anchor_id in picks)


def test_render_panel_a_creates_axes_for_grid_plus_reference(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.paper_figures.fig_combinatorial_grammar_panels._diff_grid import render_panel_a

    generated_root = tmp_path / "generated"
    ablation_root = tmp_path / "ablation_results"
    anchor = "anchor_x"
    _populate_anchor_sweep(generated_root, anchor)
    ref_path = ablation_root / anchor / "all" / "generated_he.png"
    _write_reference(ref_path, 200)

    fig = plt.figure(figsize=(6, 4))
    outer = fig.add_gridspec(1, 1)
    render_panel_a(fig, outer[0, 0], anchor_id=anchor, generated_root=generated_root, reference_path=ref_path)
    fig.canvas.draw()
    n_axes = len(fig.axes)
    plt.close(fig)
    assert n_axes >= 29


def test_render_panel_b_creates_heatmap_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.paper_figures.fig_combinatorial_grammar_panels._l2_heatmap import render_panel_b

    fig = plt.figure(figsize=(6, 3))
    outer = fig.add_gridspec(1, 1)
    render_panel_b(fig, outer[0, 0], residual_rows=_make_residual_rows())
    fig.canvas.draw()
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_select_case_rows_returns_low_mid_high() -> None:
    from src.paper_figures.fig_combinatorial_grammar_panels._case_studies import select_case_rows

    cases = select_case_rows(_make_residual_rows())
    labels = [label for label, _ in cases]
    assert labels == ["lowest", "median", "highest"]
    l2_values = [float(row["residual_l2_norm"]) for _, row in cases]
    assert l2_values == sorted(l2_values)


def test_render_panel_c_creates_three_subplots() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.paper_figures.fig_combinatorial_grammar_panels._case_studies import render_panel_c

    fig = plt.figure(figsize=(4, 8))
    outer = fig.add_gridspec(1, 1)
    render_panel_c(fig, outer[0, 0], residual_rows=_make_residual_rows())
    fig.canvas.draw()
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_save_combinatorial_grammar_figure_renders_main_png(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from src.paper_figures.fig_combinatorial_grammar import save_combinatorial_grammar_figure

    generated_root = tmp_path / "generated"
    ablation_root = tmp_path / "ablation_results"
    representative = "anchor_0"
    _populate_anchor_sweep(generated_root, representative)
    _write_reference(ablation_root / representative / "all" / "generated_he.png", 200)

    sig_path = tmp_path / "signatures.csv"
    res_path = tmp_path / "residuals.csv"
    _write_csv(sig_path, _make_signature_rows({representative: 27}))
    _write_csv(res_path, _make_residual_rows())

    out_png = tmp_path / "fig.png"
    result = save_combinatorial_grammar_figure(
        out_png=out_png,
        generated_root=generated_root,
        signatures_csv=sig_path,
        residuals_csv=res_path,
        ablation_root=ablation_root,
        dpi=80,
    )
    assert result == out_png
    assert out_png.is_file()
    with Image.open(out_png) as img:
        assert img.width > 400
        assert img.height > 200


def test_save_combinatorial_grammar_si_figure_renders(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from src.paper_figures.fig_combinatorial_grammar_si import save_combinatorial_grammar_si_figure

    generated_root = tmp_path / "generated"
    ablation_root = tmp_path / "ablation_results"
    anchors = [f"a{i}" for i in range(20)]
    for anchor_id in anchors:
        _populate_anchor_sweep(generated_root, anchor_id)
        _write_reference(ablation_root / anchor_id / "all" / "generated_he.png", 200)

    sig_path = tmp_path / "signatures.csv"
    _write_csv(sig_path, _make_signature_rows({anchor_id: 27 for anchor_id in anchors}))

    out_png = tmp_path / "si.png"
    result = save_combinatorial_grammar_si_figure(
        out_png=out_png,
        generated_root=generated_root,
        signatures_csv=sig_path,
        ablation_root=ablation_root,
        dpi=80,
    )
    assert result == out_png
    assert out_png.is_file()
