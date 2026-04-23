"""CPU-safe planner for the encoder-comparison task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.runtime import CommandSpec, JobPlan, JobState, TaskPlan, RuntimeProbe, probe_runtime


@dataclass(frozen=True)
class ProbeEncodersConfig:
    """Inputs required to plan the encoder-comparison task."""

    he_dir: Path
    targets_path: Path
    tile_ids_path: Path
    cv_splits_path: Path
    out_dir: Path
    virchow_weights: Path | None = None
    device: str = "cuda"
    skip_existing: bool = True


def _python_job(job_module: str, *args: str) -> CommandSpec:
    return CommandSpec(argv=("python", "-m", job_module, *args), cwd=Path(__file__).resolve().parents[2])


def plan_task(config: ProbeEncodersConfig, runtime: RuntimeProbe | None = None) -> TaskPlan:
    """Plan GPU-sensitive encoder jobs without executing them."""
    runtime = runtime or probe_runtime()
    out_dir = ensure_directory(config.out_dir)
    virchow_out = out_dir / "virchow_embeddings.npy"
    cnn_out = out_dir / "raw_cnn_embeddings.npy"
    comparison_out = out_dir / "encoder_comparison.csv"
    jobs: list[JobPlan] = []

    if config.skip_existing and virchow_out.is_file():
        virchow_state = JobState.SKIPPED
        virchow_reason = "existing_output"
        virchow_command = None
    elif config.virchow_weights is None:
        virchow_state = JobState.SKIPPED
        virchow_reason = "missing_weights"
        virchow_command = None
    elif not runtime.has_cuda:
        virchow_state = JobState.DEFERRED
        virchow_reason = "missing_gpu"
        virchow_command = _python_job("src.a1_probe_encoders.main", "--worker", "virchow")
    else:
        virchow_state = JobState.READY
        virchow_reason = None
        virchow_command = _python_job("src.a1_probe_encoders.main", "--worker", "virchow")
    jobs.append(
        JobPlan(
            job_id="cache_virchow_embeddings",
            state=virchow_state,
            reason=virchow_reason,
            inputs=(config.he_dir, config.tile_ids_path),
            outputs=(virchow_out,),
            command=virchow_command,
        )
    )

    if config.skip_existing and cnn_out.is_file():
        cnn_state = JobState.SKIPPED
        cnn_reason = "existing_output"
        cnn_command = None
    elif not runtime.has_cuda:
        cnn_state = JobState.DEFERRED
        cnn_reason = "missing_gpu"
        cnn_command = _python_job("src.a1_probe_encoders.main", "--worker", "raw_cnn")
    else:
        cnn_state = JobState.READY
        cnn_reason = None
        cnn_command = _python_job("src.a1_probe_encoders.main", "--worker", "raw_cnn")
    jobs.append(
        JobPlan(
            job_id="train_raw_cnn",
            state=cnn_state,
            reason=cnn_reason,
            inputs=(config.he_dir, config.targets_path, config.tile_ids_path, config.cv_splits_path),
            outputs=(cnn_out,),
            command=cnn_command,
        )
    )

    comparison_ready = virchow_out.is_file() or cnn_out.is_file()
    jobs.append(
        JobPlan(
            job_id="fit_probe_heads",
            state=JobState.READY if comparison_ready else JobState.BLOCKED,
            reason=None if comparison_ready else "missing_encoder_embeddings",
            inputs=(virchow_out, cnn_out, config.targets_path, config.cv_splits_path),
            outputs=(comparison_out,),
            command=_python_job("src.a1_probe_encoders.main", "--worker", "compare") if comparison_ready else None,
        )
    )
    return TaskPlan(task_name="a1_probe_encoders", jobs=tuple(jobs), warnings=runtime.warnings)


def main(argv: list[str] | None = None) -> int:
    """Write a plan file or act as a placeholder worker on GPU machines."""
    parser = argparse.ArgumentParser(description="Plan the encoder-comparison task")
    parser.add_argument("--he-dir", default=None)
    parser.add_argument("--targets-path", default=None)
    parser.add_argument("--tile-ids-path", default=None)
    parser.add_argument("--cv-splits-path", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--virchow-weights", default=None)
    parser.add_argument("--worker", default=None)
    args = parser.parse_args(argv)

    if args.worker is not None:
        raise RuntimeError("encoder-comparison workers are intended to run on a GPU machine")

    config = ProbeEncodersConfig(
        he_dir=Path(args.he_dir),
        targets_path=Path(args.targets_path),
        tile_ids_path=Path(args.tile_ids_path),
        cv_splits_path=Path(args.cv_splits_path),
        out_dir=Path(args.out_dir),
        virchow_weights=Path(args.virchow_weights) if args.virchow_weights else None,
    )
    plan = plan_task(config)
    write_json(
        {
            "task_name": plan.task_name,
            "warnings": list(plan.warnings),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "state": job.state.value,
                    "reason": job.reason,
                    "outputs": [str(path) for path in job.outputs],
                }
                for job in plan.jobs
            ],
        },
        config.out_dir / "plan.json",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
