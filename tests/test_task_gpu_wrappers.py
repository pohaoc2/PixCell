from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src._tasklib.runtime import JobState, RuntimeProbe


CPU_ONLY = RuntimeProbe(
    has_torch=True,
    has_cuda=False,
    has_diffusers=True,
    has_sklearn=True,
    has_matplotlib=True,
    warnings=("torch available but CUDA is unavailable",),
)


def test_probe_encoders_plan_skips_or_defers_gpu_work(tmp_path: Path):
    from src.a1_probe_encoders.main import ProbeEncodersConfig, plan_task

    he_dir = tmp_path / "he"
    he_dir.mkdir(parents=True, exist_ok=True)
    targets_path = tmp_path / "targets.npy"
    targets_path.write_bytes(b"x")
    tile_ids_path = tmp_path / "tile_ids.txt"
    tile_ids_path.write_text("0_0\n", encoding="utf-8")
    cv_splits_path = tmp_path / "cv_splits.json"
    cv_splits_path.write_text("{}", encoding="utf-8")

    plan = plan_task(
        ProbeEncodersConfig(
            he_dir=he_dir,
            targets_path=targets_path,
            tile_ids_path=tile_ids_path,
            cv_splits_path=cv_splits_path,
            out_dir=tmp_path / "out",
            virchow_weights=None,
        ),
        runtime=CPU_ONLY,
    )

    assert plan.jobs[0].state == JobState.SKIPPED
    assert plan.jobs[1].state == JobState.DEFERRED
    assert plan.jobs[2].state == JobState.BLOCKED


def test_generated_probe_plan_indexes_generated_tiles(tmp_path: Path):
    from src.a1_generated_probe.main import GeneratedProbeConfig, plan_task

    generated_root = tmp_path / "paired_ablation" / "ablation_results" / "0_0" / "all"
    generated_root.mkdir(parents=True, exist_ok=True)
    (generated_root / "generated_he.png").write_bytes(b"png")
    uni_model_path = tmp_path / "uni"
    uni_model_path.mkdir(parents=True, exist_ok=True)
    targets_path = tmp_path / "targets.npy"
    targets_path.write_bytes(b"x")
    tile_ids_path = tmp_path / "tile_ids.txt"
    tile_ids_path.write_text("0_0\n2048_0\n", encoding="utf-8")
    cv_splits_path = tmp_path / "cv_splits.json"
    cv_splits_path.write_text("{}", encoding="utf-8")

    plan = plan_task(
        GeneratedProbeConfig(
            generated_root=tmp_path / "paired_ablation",
            uni_model_path=uni_model_path,
            targets_path=targets_path,
            tile_ids_path=tile_ids_path,
            cv_splits_path=cv_splits_path,
            out_dir=tmp_path / "out",
        ),
        runtime=CPU_ONLY,
    )

    assert plan.jobs[0].state == JobState.DEFERRED
    assert any("missing generated PNGs" in warning for warning in plan.warnings)


def test_decomposition_plan_enumerates_per_tile_batches(tmp_path: Path):
    from src.a2_decomposition.main import DecompositionConfig, plan_task

    data_root = tmp_path / "data"
    features_dir = data_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    for tile_id in ("0_0", "2048_0"):
        (features_dir / f"{tile_id}_uni.npy").write_bytes(b"x")
    config_path = tmp_path / "config.py"
    config_path.write_text("cfg = {}\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_task(
        DecompositionConfig(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            data_root=data_root,
            out_dir=tmp_path / "out",
            sample_n=2,
        ),
        runtime=CPU_ONLY,
    )

    assert len(plan.jobs) == 3
    assert plan.jobs[0].state == JobState.DEFERRED
    assert plan.jobs[1].state == JobState.DEFERRED
    assert plan.jobs[2].state == JobState.BLOCKED


def test_combinatorial_sweep_conditions_and_explicit_anchors(tmp_path: Path):
    from src.a3_combinatorial_sweep.main import (
        CombinatorialSweepConfig,
        enumerate_conditions,
        plan_task,
    )

    conditions = enumerate_conditions()
    assert len(conditions) == 27

    config_path = tmp_path / "config.py"
    config_path.write_text("cfg = {}\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    plan = plan_task(
        CombinatorialSweepConfig(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            data_root=data_root,
            out_dir=tmp_path / "out",
            anchor_tile_ids=("0_0", "2048_0"),
        ),
        runtime=CPU_ONLY,
    )

    assert len(plan.jobs) == 3
    assert plan.jobs[0].state == JobState.DEFERRED
    assert plan.jobs[1].state == JobState.DEFERRED
    assert plan.jobs[2].state == JobState.BLOCKED