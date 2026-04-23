from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src._tasklib.runtime import JobState, RuntimeProbe


CPU_ONLY = RuntimeProbe(
    has_torch=True,
    has_cuda=False,
    has_diffusers=True,
    has_sklearn=True,
    has_matplotlib=True,
    warnings=("torch available but CUDA is unavailable",),
)


def _write_condition_images(out_dir: Path, anchor_id: str) -> None:
    from src.a3_combinatorial_sweep.main import build_condition_id, enumerate_conditions

    anchor_dir = out_dir / "generated" / anchor_id
    anchor_dir.mkdir(parents=True, exist_ok=True)
    state_bias = {"prolif": 15, "nonprolif": 45, "dead": 75}
    oxygen_bias = {"low": 10, "mid": 30, "high": 60}
    glucose_bias = {"low": 5, "mid": 20, "high": 40}

    for condition in enumerate_conditions():
        base_value = (
            state_bias[condition.cell_state]
            + oxygen_bias[condition.oxygen_label]
            + glucose_bias[condition.glucose_label]
        )
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 0] = np.clip(base_value + 20, 0, 255)
        image[..., 1] = np.clip(base_value, 0, 255)
        image[..., 2] = np.clip(220 - base_value, 0, 255)
        Image.fromarray(image).save(anchor_dir / f"{build_condition_id(condition)}.png")


def test_plan_task_includes_complete_worker_commands(tmp_path: Path):
    from src.a3_combinatorial_sweep.main import CombinatorialSweepConfig, enumerate_conditions, plan_task

    config_path = tmp_path / "config.py"
    config_path.write_text("cfg = {}\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    anchors_path = tmp_path / "anchors.txt"
    anchors_path.write_text("0_0\n2048_0\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    for anchor_id in ("0_0", "2048_0"):
        anchor_dir = out_dir / "generated" / anchor_id
        anchor_dir.mkdir(parents=True, exist_ok=True)
        for condition in enumerate_conditions():
            (anchor_dir / f"{condition.cell_state}_{condition.oxygen_label}_{condition.glucose_label}.png").write_bytes(b"png")

    plan = plan_task(
        CombinatorialSweepConfig(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            data_root=data_root,
            out_dir=out_dir,
            anchor_tile_ids_path=anchors_path,
        ),
        runtime=CPU_ONLY,
    )

    assert len(plan.jobs) == 3
    assert plan.jobs[0].state == JobState.SKIPPED
    deferred_plan = plan_task(
        CombinatorialSweepConfig(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            data_root=data_root,
            out_dir=tmp_path / "out_deferred",
            anchor_tile_ids_path=anchors_path,
        ),
        runtime=CPU_ONLY,
    )
    generate_command = deferred_plan.jobs[0].command
    assert generate_command is not None
    assert generate_command.argv == (
        "python",
        "-m",
        "src.a3_combinatorial_sweep.main",
        "--config-path",
        str(config_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--data-root",
        str(data_root),
        "--out-dir",
        str(tmp_path / "out_deferred"),
        "--anchor-tile-ids-path",
        str(anchors_path),
        "--worker",
        "0_0",
    )
    summary_command = plan.jobs[-1].command
    assert summary_command is not None
    assert summary_command.argv[-2:] == ("--worker", "summarize")
    assert "--anchor-tile-ids-path" in summary_command.argv


def test_worker_generates_27_condition_images_with_mocked_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.a3_combinatorial_sweep import main as sweep_main

    config_path = tmp_path / "config.py"
    config_path.write_text("cfg = {}\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out"

    active_channels = [
        "cell_masks",
        "cell_state_prolif",
        "cell_state_nonprolif",
        "cell_state_dead",
        "oxygen",
        "glucose",
    ]
    runtime_config = SimpleNamespace(
        data=SimpleNamespace(active_channels=active_channels),
        image_size=4,
    )
    base_ctrl = torch.zeros((len(active_channels), 4, 4), dtype=torch.float32)
    base_ctrl[0] = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    observed = {}

    def fake_load_generation_runtime(**kwargs):
        return {}, runtime_config, object(), tmp_path / "exp_channels", tmp_path / "features"

    def fake_load_anchor_ctrl(tile_id: str, *, active_channels: list[str], image_size: int, exp_channels_dir: Path):
        assert tile_id == "0_0"
        assert active_channels == runtime_config.data.active_channels
        assert image_size == 4
        return base_ctrl.clone()

    def fake_load_anchor_uni(tile_id: str, *, feat_dir: Path):
        assert tile_id == "0_0"
        return torch.zeros((1, 1, 1, 1536), dtype=torch.float32)

    def fake_make_generation_noise(**kwargs):
        return torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    def fake_render_generated_image(ctrl_full, **kwargs):
        prolif = float(ctrl_full[1].mean().item())
        nonprolif = float(ctrl_full[2].mean().item())
        dead = float(ctrl_full[3].mean().item())
        oxygen = float(ctrl_full[4].mean().item())
        glucose = float(ctrl_full[5].mean().item())
        state_name = "prolif" if prolif > 0.0 else "nonprolif" if nonprolif > 0.0 else "dead"
        observed[(state_name, oxygen, glucose)] = ctrl_full.clone()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[..., 0] = int(round(oxygen * 100))
        image[..., 1] = int(round(glucose * 100))
        image[..., 2] = {"prolif": 32, "nonprolif": 96, "dead": 160}[state_name]
        return image

    monkeypatch.setattr(sweep_main, "_load_generation_runtime", fake_load_generation_runtime)
    monkeypatch.setattr(sweep_main, "_load_anchor_ctrl", fake_load_anchor_ctrl)
    monkeypatch.setattr(sweep_main, "_load_anchor_uni", fake_load_anchor_uni)
    monkeypatch.setattr(sweep_main, "_make_generation_noise", fake_make_generation_noise)
    monkeypatch.setattr(sweep_main, "_render_generated_image", fake_render_generated_image)

    exit_code = sweep_main.main(
        [
            "--config-path",
            str(config_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
            "--worker",
            "0_0",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    generated_paths = sorted((out_dir / "generated" / "0_0").glob("*.png"))
    assert len(generated_paths) == 27

    mask = base_ctrl[0]
    for (state_name, oxygen, glucose), ctrl in observed.items():
        for idx, candidate in enumerate(("prolif", "nonprolif", "dead"), start=1):
            if candidate == state_name:
                assert torch.equal(ctrl[idx], mask)
            else:
                assert torch.count_nonzero(ctrl[idx]) == 0
        assert torch.allclose(ctrl[4], torch.full_like(ctrl[4], oxygen))
        assert torch.allclose(ctrl[5], torch.full_like(ctrl[5], glucose))


def test_summary_worker_writes_csv_outputs_from_synthetic_images(tmp_path: Path):
    from src.a3_combinatorial_sweep import main as sweep_main

    config_path = tmp_path / "config.py"
    config_path.write_text("cfg = {}\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out"
    _write_condition_images(out_dir, "0_0")

    exit_code = sweep_main.main(
        [
            "--config-path",
            str(config_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
            "--anchor-tile-id",
            "0_0",
            "--worker",
            "summarize",
        ]
    )

    assert exit_code == 0
    signatures_path = out_dir / "morphological_signatures.csv"
    residuals_path = out_dir / "additive_model_residuals.csv"
    heatmap_path = out_dir / "interaction_heatmap.png"
    assert signatures_path.is_file()
    assert residuals_path.is_file()
    assert heatmap_path.is_file()

    with signatures_path.open("r", encoding="utf-8", newline="") as handle:
        signature_rows = list(csv.DictReader(handle))
    with residuals_path.open("r", encoding="utf-8", newline="") as handle:
        residual_rows = list(csv.DictReader(handle))

    assert len(signature_rows) == 27
    assert len(residual_rows) == 27
    assert {row["cell_state"] for row in residual_rows} == {"prolif", "nonprolif", "dead"}
    assert "nuclear_density" in signature_rows[0]
    assert "residual_l2_norm" in residual_rows[0]