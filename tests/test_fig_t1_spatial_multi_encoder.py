from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_dummy_csv(path: Path, targets: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "r2_mean",
                "r2_sd",
                "r2_within_mean",
                "r2_within_sd",
                "pearson_r_mean",
                "pearson_r_sd",
                "delta_shuffle",
                "n_valid_folds",
            ],
        )
        writer.writeheader()
        for index, target in enumerate(targets):
            writer.writerow(
                {
                    "target": target,
                    "r2_mean": 0.1 + index * 0.01,
                    "r2_sd": 0.02,
                    "r2_within_mean": 0.05 + index * 0.01,
                    "r2_within_sd": 0.01,
                    "pearson_r_mean": 0.3 + index * 0.02,
                    "pearson_r_sd": 0.05,
                    "delta_shuffle": "nan",
                    "n_valid_folds": 5,
                }
            )


def _write_dummy_t2_spatial_csv(path: Path, markers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "r2_mean",
                "r2_sd",
                "r2_within_mean",
                "r2_within_sd",
                "pearson_r_mean",
                "pearson_r_sd",
                "delta_shuffle",
                "n_valid_folds",
            ],
        )
        writer.writeheader()
        for index, marker in enumerate(markers):
            writer.writerow(
                {
                    "target": marker,
                    "r2_mean": -0.2 - index * 0.05,
                    "r2_sd": 0.03,
                    "r2_within_mean": -0.3 - index * 0.08,
                    "r2_within_sd": 0.04,
                    "pearson_r_mean": 0.1,
                    "pearson_r_sd": 0.02,
                    "delta_shuffle": "nan",
                    "n_valid_folds": 5,
                }
            )


def test_fig_renders_with_two_encoders(tmp_path: Path) -> None:
    from src.paper_figures.fig_t1_spatial_multi_encoder import build_figure

    targets = ["prolif_frac", "cell_density", "oxygen_mean"]
    csvs = {
        "UNI-2h": tmp_path / "uni" / "results.csv",
        "Virchow2": tmp_path / "virchow" / "results.csv",
    }
    t2_spatial_csv = tmp_path / "t2_spatial" / "results.csv"
    for path in csvs.values():
        _write_dummy_csv(path, targets)
    _write_dummy_t2_spatial_csv(t2_spatial_csv, ["CD31", "CD45", "Ki67"])

    fig = build_figure(encoder_csvs=csvs, t2_spatial_csv=t2_spatial_csv)
    out_path = tmp_path / "fig.png"
    fig.savefig(out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == r"Per-patch $R^2$ (within-tile)"
    assert fig.axes[1].get_ylabel() == r"Per-patch $R^2$ (within-tile)"
    plt.close(fig)