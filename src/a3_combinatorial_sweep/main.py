"""CPU-safe planner for the combinatorial sweep task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.runtime import CommandSpec, JobPlan, JobState, RuntimeProbe, TaskPlan, probe_runtime


@dataclass(frozen=True)
class SweepCondition:
    """One cell-state and microenvironment condition."""

    cell_state: str
    oxygen_label: str
    oxygen_value: float
    glucose_label: str
    glucose_value: float


@dataclass(frozen=True)
class CombinatorialSweepConfig:
    """Inputs required to plan the combinatorial sweep task."""

    config_path: Path
    checkpoint_dir: Path
    data_root: Path
    out_dir: Path
    anchor_tile_ids: tuple[str, ...] = ()
    anchor_tile_ids_path: Path | None = None


def enumerate_conditions() -> list[SweepCondition]:
    """Enumerate the canonical 3x3x3 condition grid."""
    states = ("prolif", "nonprolif", "dead")
    levels = (("low", 0.50), ("mid", 0.75), ("high", 1.0))
    conditions: list[SweepCondition] = []
    for state in states:
        for oxygen_label, oxygen_value in levels:
            for glucose_label, glucose_value in levels:
                conditions.append(
                    SweepCondition(
                        cell_state=state,
                        oxygen_label=oxygen_label,
                        oxygen_value=oxygen_value,
                        glucose_label=glucose_label,
                        glucose_value=glucose_value,
                    )
                )
    return conditions


def build_condition_id(condition: SweepCondition) -> str:
    """Stable identifier used for file and manifest names."""
    return f"{condition.cell_state}_{condition.oxygen_label}_{condition.glucose_label}"


def load_anchor_tile_ids(config: CombinatorialSweepConfig) -> tuple[list[str], str | None]:
    """Resolve explicit anchors or an anchor list file."""
    if config.anchor_tile_ids:
        return list(config.anchor_tile_ids), None
    if config.anchor_tile_ids_path is not None and config.anchor_tile_ids_path.is_file():
        tile_ids = [
            line.strip()
            for line in config.anchor_tile_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return tile_ids, None
    return [], "missing_anchor_selection"


def _python_job(job_module: str, *args: str) -> CommandSpec:
    return CommandSpec(argv=("python", "-m", job_module, *args), cwd=Path(__file__).resolve().parents[2])


def plan_task(config: CombinatorialSweepConfig, runtime: RuntimeProbe | None = None) -> TaskPlan:
    """Plan one sweep batch per anchor plus a summary job."""
    runtime = runtime or probe_runtime()
    out_dir = ensure_directory(config.out_dir)
    anchors, anchor_error = load_anchor_tile_ids(config)
    conditions = enumerate_conditions()
    jobs: list[JobPlan] = []

    for anchor in anchors:
        output_dir = out_dir / "generated" / anchor
        outputs = tuple(output_dir / f"{build_condition_id(condition)}.png" for condition in conditions)
        if not config.config_path.exists() or not config.checkpoint_dir.exists():
            state = JobState.BLOCKED
            reason = "missing_checkpoint"
            command = None
        elif all(path.is_file() for path in outputs):
            state = JobState.SKIPPED
            reason = "existing_output"
            command = None
        elif not runtime.has_cuda:
            state = JobState.DEFERRED
            reason = "missing_gpu"
            command = _python_job("src.a3_combinatorial_sweep.main", "--worker", anchor)
        else:
            state = JobState.READY
            reason = None
            command = _python_job("src.a3_combinatorial_sweep.main", "--worker", anchor)
        jobs.append(
            JobPlan(
                job_id=f"generate_{anchor}",
                state=state,
                reason=reason,
                inputs=(config.config_path, config.checkpoint_dir, config.data_root),
                outputs=outputs,
                command=command,
            )
        )

    summary_outputs = (
        out_dir / "morphological_signatures.csv",
        out_dir / "additive_model_residuals.csv",
        out_dir / "interaction_heatmap.png",
    )
    if anchor_error is not None:
        summary_state = JobState.BLOCKED
        summary_reason = anchor_error
    elif not anchors:
        summary_state = JobState.BLOCKED
        summary_reason = "missing_anchor_selection"
    else:
        all_generated = all(all(path.is_file() for path in job.outputs) for job in jobs)
        summary_state = JobState.READY if all_generated else JobState.BLOCKED
        summary_reason = None if all_generated else "missing_generated_tiles"
    jobs.append(
        JobPlan(
            job_id="summarize_sweep",
            state=summary_state,
            reason=summary_reason,
            inputs=tuple(path for job in jobs for path in job.outputs),
            outputs=summary_outputs,
            command=_python_job("src.a3_combinatorial_sweep.main", "--worker", "summarize") if summary_state == JobState.READY else None,
        )
    )
    return TaskPlan(task_name="a3_combinatorial_sweep", jobs=tuple(jobs), warnings=runtime.warnings)


def main(argv: list[str] | None = None) -> int:
    """Write a plan file or act as a placeholder worker on GPU machines."""
    parser = argparse.ArgumentParser(description="Plan the combinatorial sweep task")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--anchor-tile-ids-path", default=None)
    parser.add_argument("--worker", default=None)
    args = parser.parse_args(argv)

    if args.worker is not None:
        raise RuntimeError("combinatorial-sweep workers are intended to run on a GPU machine")

    config = CombinatorialSweepConfig(
        config_path=Path(args.config_path),
        checkpoint_dir=Path(args.checkpoint_dir),
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        anchor_tile_ids_path=Path(args.anchor_tile_ids_path) if args.anchor_tile_ids_path else None,
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
