"""CPU-safe planner for the UNI/TME decomposition task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.runtime import CommandSpec, JobPlan, JobState, RuntimeProbe, TaskPlan, probe_runtime
from src._tasklib.tile_ids import list_feature_tile_ids


@dataclass(frozen=True)
class ModeSpec:
    """One generation mode in the 2x2 UNI/TME decomposition."""

    name: str
    use_uni: bool
    use_tme: bool


DEFAULT_MODES = (
    ModeSpec("uni_plus_tme", True, True),
    ModeSpec("uni_only", True, False),
    ModeSpec("tme_only", False, True),
    ModeSpec("neither", False, False),
)


@dataclass(frozen=True)
class DecompositionConfig:
    """Inputs required to plan the decomposition task."""

    config_path: Path
    checkpoint_dir: Path
    data_root: Path
    out_dir: Path
    tile_ids: tuple[str, ...] = ()
    sample_n: int = 500


def discover_tile_ids(data_root: str | Path, sample_n: int) -> list[str]:
    """Discover candidate tile IDs from cached UNI features."""
    feature_dir = Path(data_root) / "features"
    return list_feature_tile_ids(feature_dir)[:sample_n]


def _python_job(job_module: str, *args: str) -> CommandSpec:
    return CommandSpec(argv=("python", "-m", job_module, *args), cwd=Path(__file__).resolve().parents[2])


def _mode_paths(out_dir: Path, tile_id: str) -> tuple[Path, ...]:
    return tuple(out_dir / "generated" / tile_id / f"{mode.name}.png" for mode in DEFAULT_MODES)


def plan_task(config: DecompositionConfig, runtime: RuntimeProbe | None = None) -> TaskPlan:
    """Plan one generation batch per tile plus a summary job."""
    runtime = runtime or probe_runtime()
    out_dir = ensure_directory(config.out_dir)
    tile_ids = list(config.tile_ids) if config.tile_ids else discover_tile_ids(config.data_root, config.sample_n)

    jobs: list[JobPlan] = []
    blocked_reason = None
    if not config.config_path.exists() or not config.checkpoint_dir.exists():
        blocked_reason = "missing_checkpoint"
    for tile_id in tile_ids:
        outputs = _mode_paths(out_dir, tile_id)
        if blocked_reason is not None:
            state = JobState.BLOCKED
            reason = blocked_reason
            command = None
        elif all(path.is_file() for path in outputs):
            state = JobState.SKIPPED
            reason = "existing_output"
            command = None
        elif not runtime.has_cuda:
            state = JobState.DEFERRED
            reason = "missing_gpu"
            command = _python_job("src.a2_decomposition.main", "--worker", tile_id)
        else:
            state = JobState.READY
            reason = None
            command = _python_job("src.a2_decomposition.main", "--worker", tile_id)
        jobs.append(
            JobPlan(
                job_id=f"generate_{tile_id}",
                state=state,
                reason=reason,
                inputs=(config.config_path, config.checkpoint_dir, config.data_root),
                outputs=outputs,
                command=command,
            )
        )

    summary_ready = all(all(path.is_file() for path in _mode_paths(out_dir, tile_id)) for tile_id in tile_ids)
    jobs.append(
        JobPlan(
            job_id="summarize_modes",
            state=JobState.READY if summary_ready else JobState.BLOCKED,
            reason=None if summary_ready else "missing_mode_images",
            inputs=tuple(path for tile_id in tile_ids for path in _mode_paths(out_dir, tile_id)),
            outputs=(out_dir / "mode_summary.csv", out_dir / "mode_metrics.csv"),
            command=_python_job("src.a2_decomposition.main", "--worker", "summarize") if summary_ready else None,
        )
    )
    return TaskPlan(task_name="a2_decomposition", jobs=tuple(jobs), warnings=runtime.warnings)


def main(argv: list[str] | None = None) -> int:
    """Write a plan file or act as a placeholder worker on GPU machines."""
    parser = argparse.ArgumentParser(description="Plan the UNI/TME decomposition task")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--sample-n", type=int, default=500)
    parser.add_argument("--worker", default=None)
    args = parser.parse_args(argv)

    if args.worker is not None:
        raise RuntimeError("decomposition workers are intended to run on a GPU machine")

    config = DecompositionConfig(
        config_path=Path(args.config_path),
        checkpoint_dir=Path(args.checkpoint_dir),
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        sample_n=args.sample_n,
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
