from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _write_config(path: Path) -> Path:
    path.write_text("cfg = {}\n", encoding="utf-8")
    return path


def _assert_flag_value(argv: tuple[str, ...], flag: str, expected: Path) -> None:
    index = argv.index(flag)
    assert argv[index + 1] == str(expected)


def test_plan_task_generate_command_includes_execution_args(tmp_path: Path):
    from src.a2_decomposition.main import DecompositionConfig, plan_task

    config_path = _write_config(tmp_path / "config.py")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    data_root = tmp_path / "data"
    features_dir = data_root / "features"
    features_dir.mkdir(parents=True)
    (features_dir / "0_0_uni.npy").write_bytes(b"x")
    out_dir = tmp_path / "out"

    plan = plan_task(
        DecompositionConfig(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            data_root=data_root,
            out_dir=out_dir,
            sample_n=1,
        ),
        runtime=CPU_ONLY,
    )

    job = plan.jobs[0]
    assert job.state == JobState.DEFERRED
    assert job.command is not None
    assert job.command.argv[:4] == ("python", "-m", "src.a2_decomposition.main", "--worker")
    _assert_flag_value(job.command.argv, "--config-path", config_path)
    _assert_flag_value(job.command.argv, "--checkpoint-dir", checkpoint_dir)
    _assert_flag_value(job.command.argv, "--data-root", data_root)
    _assert_flag_value(job.command.argv, "--out-dir", out_dir)


def test_plan_task_summary_command_includes_execution_args_when_ready(tmp_path: Path):
    from src.a2_decomposition.main import DEFAULT_MODES, DecompositionConfig, plan_task

    config_path = _write_config(tmp_path / "config.py")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    out_dir = tmp_path / "out"
    generated_dir = out_dir / "generated" / "tile_001"
    generated_dir.mkdir(parents=True)
    for mode in DEFAULT_MODES:
        Image.fromarray(np.full((4, 4, 3), 64, dtype=np.uint8)).save(generated_dir / f"{mode.name}.png")

    plan = plan_task(
        DecompositionConfig(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            data_root=data_root,
            out_dir=out_dir,
            tile_ids=("tile_001",),
        ),
        runtime=CPU_ONLY,
    )

    job = plan.jobs[-1]
    assert job.state == JobState.READY
    assert job.command is not None
    assert job.command.argv[4] == "summarize"
    _assert_flag_value(job.command.argv, "--config-path", config_path)
    _assert_flag_value(job.command.argv, "--checkpoint-dir", checkpoint_dir)
    _assert_flag_value(job.command.argv, "--data-root", data_root)
    _assert_flag_value(job.command.argv, "--out-dir", out_dir)


def test_worker_tile_branch_writes_four_mode_images(tmp_path: Path, monkeypatch):
    import src.a2_decomposition.main as module

    config_path = _write_config(tmp_path / "config.py")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"

    inference_config = SimpleNamespace(
        data=SimpleNamespace(active_channels=["cell_masks", "oxygen"]),
        image_size=4,
    )
    resources = module.WorkerResources(
        inference_config=inference_config,
        models={"stub": True},
        scheduler="scheduler",
        exp_channels_dir=data_root / "exp_channels",
        feat_dir=data_root / "features",
        he_dir=data_root / "he",
        device="cpu",
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(module, "_load_worker_resources", lambda config, device, num_steps: resources)
    monkeypatch.setattr(module, "_load_control_tensor", lambda tile_id, active_channels, image_size, exp_channels_dir: "ctrl")

    def fake_resolve_uni_embedding(tile_id: str, *, feat_dir: Path, null_uni: bool) -> str:
        return "null_uni" if null_uni else "real_uni"

    def fake_generate_from_control(ctrl_full, **kwargs):
        calls.append((kwargs["uni_embeds"], kwargs["active_groups"]))
        fill = 40 * len(calls)
        image = np.full((4, 4, 3), fill, dtype=np.uint8)
        return image, None

    monkeypatch.setattr(module, "_resolve_uni_embedding", fake_resolve_uni_embedding)
    monkeypatch.setattr(module, "_generate_from_control", fake_generate_from_control)

    exit_code = module.main(
        [
            "--worker",
            "tile_001",
            "--config-path",
            str(config_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    generated_dir = out_dir / "generated" / "tile_001"
    assert (generated_dir / "uni_plus_tme.png").is_file()
    assert (generated_dir / "uni_only.png").is_file()
    assert (generated_dir / "tme_only.png").is_file()
    assert (generated_dir / "neither.png").is_file()
    assert calls == [
        ("real_uni", None),
        ("real_uni", ()),
        ("null_uni", None),
        ("null_uni", ()),
    ]


def test_worker_summarize_branch_writes_mode_csvs(tmp_path: Path):
    import src.a2_decomposition.main as module

    config_path = _write_config(tmp_path / "config.py")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    data_root = tmp_path / "data"
    he_dir = data_root / "he"
    he_dir.mkdir(parents=True)
    out_dir = tmp_path / "out"
    generated_dir = out_dir / "generated" / "tile_001"
    generated_dir.mkdir(parents=True)

    Image.fromarray(np.full((6, 6, 3), 96, dtype=np.uint8)).save(he_dir / "tile_001.png")
    for offset, mode in enumerate(module.DEFAULT_MODES):
        image = np.full((6, 6, 3), 80 + offset * 20, dtype=np.uint8)
        Image.fromarray(image).save(generated_dir / f"{mode.name}.png")

    exit_code = module.main(
        [
            "--worker",
            "summarize",
            "--config-path",
            str(config_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    metrics_path = out_dir / "mode_metrics.csv"
    summary_path = out_dir / "mode_summary.csv"
    assert metrics_path.is_file()
    assert summary_path.is_file()

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 4
    assert {row["mode"] for row in metric_rows} == {mode.name for mode in module.DEFAULT_MODES}
    assert {"tile_id", "mode", "image_path", "tissue_fraction", "reference_rgb_mae", "reference_hed_mae"}.issubset(metric_rows[0])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 4
    summary_by_mode = {row["mode"]: row for row in summary_rows}
    assert summary_by_mode["uni_plus_tme"]["n_tiles"] == "1"
    assert summary_by_mode["uni_plus_tme"]["reference_count"] == "1"
    assert summary_by_mode["uni_plus_tme"]["reference_rgb_mae_mean"] != ""