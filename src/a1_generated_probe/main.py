"""CPU-safe planner for the generated-H&E probe task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.runtime import CommandSpec, JobPlan, JobState, RuntimeProbe, TaskPlan, probe_runtime
from src.a1_probe_linear.main import load_tile_ids


@dataclass(frozen=True)
class GeneratedProbeConfig:
    """Inputs required to plan the generated-H&E probe task."""

    generated_root: Path
    uni_model_path: Path
    targets_path: Path
    tile_ids_path: Path
    cv_splits_path: Path
    out_dir: Path
    device: str = "cuda"
    skip_existing: bool = True


def index_generated_tiles(generated_root: str | Path) -> dict[str, Path]:
    """Index available all-groups generated PNGs by tile ID."""
    base = Path(generated_root)
    candidates = [base]
    child = base / "ablation_results"
    if child.is_dir():
        candidates.append(child)
    indexed: dict[str, Path] = {}
    for candidate in candidates:
        for path in candidate.glob("*/all/generated_he.png"):
            indexed[path.parent.parent.name] = path
    return indexed


def _python_job(job_module: str, *args: str) -> CommandSpec:
    return CommandSpec(argv=("python", "-m", job_module, *args), cwd=Path(__file__).resolve().parents[2])


def plan_task(config: GeneratedProbeConfig, runtime: RuntimeProbe | None = None) -> TaskPlan:
    """Plan the generated-image embedding and evaluation jobs."""
    runtime = runtime or probe_runtime()
    out_dir = ensure_directory(config.out_dir)
    indexed = index_generated_tiles(config.generated_root)
    tile_ids = load_tile_ids(config.tile_ids_path)
    available = [tile_id for tile_id in tile_ids if tile_id in indexed]

    embeddings_out = out_dir / "generated_uni_embeddings.npy"
    results_out = out_dir / "real_vs_generated_r2.csv"
    jobs: list[JobPlan] = []

    if not config.uni_model_path.exists():
        embed_state = JobState.BLOCKED
        embed_reason = "missing_weights"
        embed_command = None
    elif config.skip_existing and embeddings_out.is_file():
        embed_state = JobState.SKIPPED
        embed_reason = "existing_output"
        embed_command = None
    elif not runtime.has_cuda:
        embed_state = JobState.DEFERRED
        embed_reason = "missing_gpu"
        embed_command = _python_job("src.a1_generated_probe.main", "--worker", "embed")
    else:
        embed_state = JobState.READY
        embed_reason = None
        embed_command = _python_job("src.a1_generated_probe.main", "--worker", "embed")
    jobs.append(
        JobPlan(
            job_id="embed_generated_uni",
            state=embed_state,
            reason=embed_reason,
            inputs=tuple(indexed[tile_id] for tile_id in available),
            outputs=(embeddings_out,),
            command=embed_command,
        )
    )

    jobs.append(
        JobPlan(
            job_id="probe_generated_embeddings",
            state=JobState.READY if embeddings_out.is_file() else JobState.BLOCKED,
            reason=None if embeddings_out.is_file() else "missing_generated_embeddings",
            inputs=(embeddings_out, config.targets_path, config.cv_splits_path),
            outputs=(results_out,),
            command=_python_job("src.a1_generated_probe.main", "--worker", "probe") if embeddings_out.is_file() else None,
        )
    )
    warnings = list(runtime.warnings)
    missing = len(tile_ids) - len(available)
    if missing:
        warnings.append(f"missing generated PNGs for {missing} tiles")
    return TaskPlan(task_name="a1_generated_probe", jobs=tuple(jobs), warnings=tuple(warnings))


def main(argv: list[str] | None = None) -> int:
    """Write a plan file or act as a placeholder worker on GPU machines."""
    parser = argparse.ArgumentParser(description="Plan the generated-H&E probe task")
    parser.add_argument("--generated-root", default=None)
    parser.add_argument("--uni-model-path", default=None)
    parser.add_argument("--targets-path", default=None)
    parser.add_argument("--tile-ids-path", default=None)
    parser.add_argument("--cv-splits-path", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--worker", default=None)
    args = parser.parse_args(argv)

    if args.worker is not None:
        raise RuntimeError("generated-H&E probe workers are intended to run on a GPU machine")

    config = GeneratedProbeConfig(
        generated_root=Path(args.generated_root),
        uni_model_path=Path(args.uni_model_path),
        targets_path=Path(args.targets_path),
        tile_ids_path=Path(args.tile_ids_path),
        cv_splits_path=Path(args.cv_splits_path),
        out_dir=Path(args.out_dir),
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
