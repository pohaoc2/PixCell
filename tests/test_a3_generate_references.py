"""Tests for the reference generator that backfills original-TME H&E renders."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.a3_combinatorial_sweep import generate_references as gr


def _write_anchor_list(path: Path, anchors: list[str]) -> None:
    path.write_text("\n".join(anchors) + "\n", encoding="utf-8")


def _make_existing_reference(output_root: Path, anchor_id: str) -> Path:
    target = output_root / anchor_id / "all" / "generated_he.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(target)
    return target


def test_target_path_uses_ablation_results_layout(tmp_path: Path):
    target = gr.target_path(tmp_path, "10240_11008")
    assert target == tmp_path / "10240_11008" / "all" / "generated_he.png"


def test_plan_missing_anchors_skips_existing_references(tmp_path: Path):
    output_root = tmp_path / "ablation_results"
    _make_existing_reference(output_root, "have")
    anchors_path = tmp_path / "anchors.txt"
    _write_anchor_list(anchors_path, ["have", "missing"])

    plan = gr.plan_missing_anchors(anchors_path=anchors_path, output_root=output_root)

    assert plan == ["missing"]


def test_run_invokes_render_only_for_missing_anchors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_root = tmp_path / "ablation_results"
    _make_existing_reference(output_root, "have")
    anchors_path = tmp_path / "anchors.txt"
    _write_anchor_list(anchors_path, ["have", "missing_a", "missing_b"])

    rendered: list[str] = []

    def _fake_render_and_save(anchor_id: str, output_path: Path, **_kwargs) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(output_path)
        rendered.append(anchor_id)
        return output_path

    monkeypatch.setattr(gr, "render_and_save_reference", _fake_render_and_save)

    summary = gr.run(
        anchors_path=anchors_path,
        output_root=output_root,
        config_path=tmp_path / "config.py",
        checkpoint_dir=tmp_path / "ckpt",
        data_root=tmp_path / "data",
        device="cpu",
    )

    assert sorted(rendered) == ["missing_a", "missing_b"]
    assert summary["skipped"] == ["have"]
    assert sorted(summary["generated"]) == ["missing_a", "missing_b"]
    assert summary["failed"] == []


def test_run_logs_failures_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    output_root = tmp_path / "ablation_results"
    anchors_path = tmp_path / "anchors.txt"
    _write_anchor_list(anchors_path, ["a", "b"])

    def _fake_render_and_save(anchor_id: str, output_path: Path, **_kwargs) -> Path:
        if anchor_id == "a":
            raise RuntimeError("boom")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 50, dtype=np.uint8)).save(output_path)
        return output_path

    monkeypatch.setattr(gr, "render_and_save_reference", _fake_render_and_save)

    with caplog.at_level(logging.ERROR, logger=gr.LOGGER.name):
        summary = gr.run(
            anchors_path=anchors_path,
            output_root=output_root,
            config_path=tmp_path / "config.py",
            checkpoint_dir=tmp_path / "ckpt",
            data_root=tmp_path / "data",
            device="cpu",
        )

    assert summary["generated"] == ["b"]
    assert summary["failed"] == ["a"]
    assert "reference generation failed for a" in caplog.text
