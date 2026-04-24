from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_t1_csv(path: Path, rows: list[tuple[str, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "r2_mean", "r2_sd", "n_valid_folds"])
        writer.writeheader()
        for target, r2_mean, r2_sd in rows:
            writer.writerow({"target": target, "r2_mean": r2_mean, "r2_sd": r2_sd, "n_valid_folds": 5})


def _write_t2_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "r2_mean", "r2_sd", "n_valid_folds"])
        writer.writeheader()
        for target, r2_mean in rows:
            writer.writerow({"target": target, "r2_mean": r2_mean, "r2_sd": 0.05, "n_valid_folds": 5})


_T1_ROWS = [
    ("cell_density", 0.953, 0.007),
    ("prolif_frac", 0.863, 0.035),
    ("nonprolif_frac", 0.826, 0.023),
    ("glucose_mean", 0.821, 0.020),
    ("oxygen_mean", 0.810, 0.022),
    ("healthy_frac", 0.710, 0.013),
    ("cancer_frac", 0.669, 0.016),
    ("vasculature_frac", 0.509, 0.022),
    ("immune_frac", 0.495, 0.042),
    ("dead_frac", -0.135, 0.107),
]

_T2_ROWS = [
    ("PD-1", 0.364),
    ("E-cadherin", 0.238),
    ("CD45RO", 0.094),
    ("Ki67", 0.050),
    ("CD3e", 0.045),
    ("Pan-CK", 0.033),
    ("CD45", 0.003),
    ("CD4", -0.002),
    ("CD163", -0.035),
    ("CD68", -0.042),
    ("SMA", -0.052),
    ("CD20", -0.142),
    ("CD31", -0.144),
    ("CD8a", -0.191),
    ("FOXP3", -1.060),
    ("Hoechst", 0.355),
    ("AF1", 0.004),
    ("Argo550", -2.7),
    ("PD-L1", -0.076),
]


def test_build_inverse_decoding_figure_returns_figure_with_two_axes(tmp_path: Path) -> None:
    from src.paper_figures.fig_inverse_decoding import build_inverse_decoding_figure

    uni_csv = tmp_path / "uni.csv"
    virchow_csv = tmp_path / "virchow.csv"
    ctranspath_csv = tmp_path / "ctranspath.csv"
    t2_csv = tmp_path / "t2.csv"

    _write_t1_csv(uni_csv, _T1_ROWS)
    _write_t1_csv(virchow_csv, [(target, r2_mean * 0.98, r2_sd) for target, r2_mean, r2_sd in _T1_ROWS])
    _write_t1_csv(ctranspath_csv, [(target, r2_mean * 0.94, r2_sd) for target, r2_mean, r2_sd in _T1_ROWS])
    _write_t2_csv(t2_csv, _T2_ROWS)

    fig = build_inverse_decoding_figure(
        uni_t1_csv=uni_csv,
        virchow_t1_csv=virchow_csv,
        ctranspath_t1_csv=ctranspath_csv,
        t2_mlp_csv=t2_csv,
    )

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_non_typing_markers_excluded(tmp_path: Path) -> None:
    from src.paper_figures.fig_inverse_decoding import load_t2_data

    t2_csv = tmp_path / "t2.csv"
    _write_t2_csv(t2_csv, _T2_ROWS)

    markers = load_t2_data(t2_csv)
    names = [row["marker"] for row in markers]

    assert "Hoechst" not in names
    assert "AF1" not in names
    assert "Argo550" not in names
    assert "PD-L1" not in names
    assert len(markers) == 15


def test_t1_panel_sorted_by_uni_r2_descending(tmp_path: Path) -> None:
    from src.paper_figures.fig_inverse_decoding import load_t1_data

    uni_csv = tmp_path / "uni.csv"
    _write_t1_csv(uni_csv, _T1_ROWS)

    targets = load_t1_data({"UNI-2h": uni_csv})
    values = [row["encoders"]["UNI-2h"]["r2_mean"] for row in targets]
    assert values == sorted(values, reverse=True)
