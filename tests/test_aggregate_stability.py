"""Stability aggregator: divergence detection and per-seed loss summary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_log(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_divergence_flagged_on_nan_loss(tmp_path):
    from tools.ablation_a3.aggregate_stability import aggregate

    seed_dir = tmp_path / "seed_1"
    _write_log(
        seed_dir / "train_log.jsonl",
        [
            {"step": 100, "loss": 1.0, "grad_norm": 0.5},
            {"step": 200, "loss": float("nan"), "grad_norm": 1.0},
        ],
    )

    summary = aggregate([seed_dir], fixed_step=100, grad_threshold=10.0, fid_diverge_cutoff=None)

    assert summary["per_seed"][0]["diverged"] is True
    assert summary["per_seed"][0]["divergence_reason"] == "nan_loss"


def test_divergence_flagged_on_grad_explosion(tmp_path):
    from tools.ablation_a3.aggregate_stability import aggregate

    seed_dir = tmp_path / "seed_2"
    _write_log(
        seed_dir / "train_log.jsonl",
        [
            {"step": 100, "loss": 1.0, "grad_norm": 0.5},
            {"step": 200, "loss": 1.1, "grad_norm": 50.0},
        ],
    )

    summary = aggregate([seed_dir], fixed_step=100, grad_threshold=10.0, fid_diverge_cutoff=None)

    assert summary["per_seed"][0]["diverged"] is True
    assert summary["per_seed"][0]["divergence_reason"] == "grad_explosion"


def test_loss_at_fixed_step(tmp_path):
    from tools.ablation_a3.aggregate_stability import aggregate

    seed_dir = tmp_path / "seed_3"
    _write_log(
        seed_dir / "train_log.jsonl",
        [
            {"step": 50, "loss": 2.0, "grad_norm": 0.1},
            {"step": 100, "loss": 1.5, "grad_norm": 0.1},
            {"step": 200, "loss": 1.0, "grad_norm": 0.1},
        ],
    )

    summary = aggregate([seed_dir], fixed_step=100, grad_threshold=100.0, fid_diverge_cutoff=None)

    assert summary["per_seed"][0]["loss_at_fixed_step"] == pytest.approx(1.5)
    assert summary["mean_loss_at_fixed_step"] == pytest.approx(1.5)


def test_text_train_log_supported(tmp_path):
    from tools.ablation_a3.aggregate_stability import aggregate

    seed_dir = tmp_path / "seed_4"
    seed_dir.mkdir()
    (seed_dir / "train_log.log").write_text(
        "Epoch [1/20] Step [50/1000] Loss: 2.0000 LR_ctrl: 1e-5\n"
        "Epoch [1/20] Step [100/1000] Loss: 1.2500 LR_ctrl: 1e-5 GradNorm: 0.5\n",
        encoding="utf-8",
    )

    summary = aggregate([seed_dir], fixed_step=100, grad_threshold=10.0, fid_diverge_cutoff=None)

    assert summary["per_seed"][0]["loss_at_fixed_step"] == pytest.approx(1.25)
    assert summary["per_seed"][0]["diverged"] is False
