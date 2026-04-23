"""CPU-safe planner and worker entry point for the UNI/TME decomposition task."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.runtime import CommandSpec, JobPlan, JobState, RuntimeProbe, TaskPlan, probe_runtime
from src._tasklib.tile_ids import list_feature_tile_ids


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "config_controlnet_exp.py"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "pixcell_controlnet_exp" / "npy_inputs"
DEFAULT_DATA_ROOT = ROOT / "data" / "orion-crc33"
DEFAULT_OUT_DIR = ROOT / "src" / "a2_decomposition" / "out"
DEFAULT_GUIDANCE_SCALE = 2.5
DEFAULT_NUM_STEPS = 20
DEFAULT_SEED = 42
MODE_METRIC_FIELDS = (
    "tissue_fraction",
    "rgb_mean",
    "rgb_std",
    "hematoxylin_mean",
    "hematoxylin_std",
    "eosin_mean",
    "eosin_std",
    "reference_rgb_mae",
    "reference_hed_mae",
)
MODE_SUMMARY_FIELDS = (
    "tissue_fraction",
    "rgb_mean",
    "rgb_std",
    "hematoxylin_mean",
    "eosin_mean",
    "reference_rgb_mae",
    "reference_hed_mae",
)


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


@dataclass(frozen=True)
class WorkerResources:
    """Loaded inference utilities for one worker invocation."""

    inference_config: Any
    models: dict[str, Any]
    scheduler: Any
    exp_channels_dir: Path
    feat_dir: Path
    he_dir: Path
    device: str


def discover_tile_ids(data_root: str | Path, sample_n: int) -> list[str]:
    """Discover candidate tile IDs from cached UNI features."""
    feature_dir = Path(data_root) / "features"
    return list_feature_tile_ids(feature_dir)[:sample_n]


def _python_job(job_module: str, *args: str) -> CommandSpec:
    return CommandSpec(argv=("python", "-m", job_module, *args), cwd=ROOT)


def _worker_job(config: DecompositionConfig, worker: str) -> CommandSpec:
    return _python_job(
        "src.a2_decomposition.main",
        "--worker",
        worker,
        "--config-path",
        str(config.config_path),
        "--checkpoint-dir",
        str(config.checkpoint_dir),
        "--data-root",
        str(config.data_root),
        "--out-dir",
        str(config.out_dir),
    )


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
            command = _worker_job(config, tile_id)
        else:
            state = JobState.READY
            reason = None
            command = _worker_job(config, tile_id)
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
            command=_worker_job(config, "summarize") if summary_ready else None,
        )
    )
    return TaskPlan(task_name="a2_decomposition", jobs=tuple(jobs), warnings=runtime.warnings)


def _default_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_inference_config(config_path: Path):
    from diffusion.utils.misc import read_config

    config = read_config(str(config_path))
    config._filename = str(config_path)
    return config


def _load_models(inference_config: Any, config_path: Path, checkpoint_dir: Path, device: str):
    from tools.stage3.tile_pipeline import load_all_models

    return load_all_models(inference_config, config_path, checkpoint_dir, device)


def _make_scheduler(*, num_steps: int, device: str):
    from tools.stage3.common import make_inference_scheduler

    return make_inference_scheduler(num_steps=num_steps, device=device)


def _resolve_data_layout(data_root: Path) -> tuple[Path, Path, Path]:
    from tools.stage3.tile_pipeline import resolve_data_layout

    return resolve_data_layout(data_root)


def _resolve_uni_embedding(tile_id: str, *, feat_dir: Path, null_uni: bool):
    from tools.stage3.common import resolve_uni_embedding

    return resolve_uni_embedding(tile_id, feat_dir=feat_dir, null_uni=null_uni)


def _load_control_tensor(tile_id: str, active_channels: list[str], image_size: int, exp_channels_dir: Path):
    from tools.stage3.tile_pipeline import load_exp_channels

    return load_exp_channels(tile_id, active_channels, image_size, exp_channels_dir)


def _generate_from_control(ctrl_full, **kwargs):
    from tools.stage3.tile_pipeline import generate_from_ctrl

    return generate_from_ctrl(ctrl_full, **kwargs)


def _load_worker_resources(
    config: DecompositionConfig,
    *,
    device: str,
    num_steps: int,
) -> WorkerResources:
    inference_config = _read_inference_config(config.config_path)
    models = _load_models(inference_config, config.config_path, config.checkpoint_dir, device)
    scheduler = _make_scheduler(num_steps=num_steps, device=device)
    exp_channels_dir, feat_dir, he_dir = _resolve_data_layout(config.data_root)
    return WorkerResources(
        inference_config=inference_config,
        models=models,
        scheduler=scheduler,
        exp_channels_dir=exp_channels_dir,
        feat_dir=feat_dir,
        he_dir=he_dir,
        device=device,
    )


def _save_png(image, output_path: Path) -> Path:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    return output_path


def run_worker_tile(
    config: DecompositionConfig,
    tile_id: str,
    *,
    device: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[Path, ...]:
    resources = _load_worker_resources(config, device=device, num_steps=num_steps)
    ctrl_full = _load_control_tensor(
        tile_id,
        resources.inference_config.data.active_channels,
        resources.inference_config.image_size,
        resources.exp_channels_dir,
    )
    out_dir = ensure_directory(config.out_dir / "generated" / tile_id)
    outputs: list[Path] = []
    for mode in DEFAULT_MODES:
        uni_embeds = _resolve_uni_embedding(
            tile_id,
            feat_dir=resources.feat_dir,
            null_uni=not mode.use_uni,
        )
        active_groups = None if mode.use_tme else ()
        gen_np, _ = _generate_from_control(
            ctrl_full,
            models=resources.models,
            config=resources.inference_config,
            scheduler=resources.scheduler,
            uni_embeds=uni_embeds,
            device=resources.device,
            guidance_scale=guidance_scale,
            seed=seed,
            active_groups=active_groups,
        )
        outputs.append(_save_png(gen_np, out_dir / f"{mode.name}.png"))
    return tuple(outputs)


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _format_csv_value(row.get(name)) for name in fieldnames})
    return path


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _mode_metrics_row(
    *,
    tile_id: str,
    mode_name: str,
    image_path: Path,
    reference_path: Path,
    out_dir: Path,
    data_root: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from tools.stage3.hed_utils import masked_mean_std, rgb_to_hed, tissue_mask_from_rgb

    generated = Image.open(image_path).convert("RGB")
    generated_arr = np.asarray(generated, dtype=np.float32) / 255.0
    tissue_mask = tissue_mask_from_rgb(generated)
    hed = rgb_to_hed(generated)
    h_mean, h_std = masked_mean_std(hed[..., 0], tissue_mask)
    e_mean, e_std = masked_mean_std(hed[..., 1], tissue_mask)

    has_reference = reference_path.is_file()
    record: dict[str, Any] = {
        "tile_id": tile_id,
        "mode": mode_name,
        "image_path": _safe_relative_path(image_path, out_dir),
        "reference_path": _safe_relative_path(reference_path, data_root) if has_reference else "",
        "has_reference": int(has_reference),
        "tissue_fraction": float(tissue_mask.mean()),
        "rgb_mean": float(generated_arr.mean()),
        "rgb_std": float(generated_arr.std()),
        "hematoxylin_mean": h_mean,
        "hematoxylin_std": h_std,
        "eosin_mean": e_mean,
        "eosin_std": e_std,
        "reference_rgb_mae": None,
        "reference_hed_mae": None,
    }
    if not has_reference:
        return record

    reference = Image.open(reference_path).convert("RGB")
    if reference.size != generated.size:
        reference = reference.resize(generated.size, Image.BILINEAR)
    reference_arr = np.asarray(reference, dtype=np.float32) / 255.0
    reference_mask = tissue_mask_from_rgb(reference)
    joint_mask = tissue_mask | reference_mask
    if not joint_mask.any():
        joint_mask = np.ones_like(tissue_mask, dtype=bool)
    reference_hed = rgb_to_hed(reference)
    rgb_delta = np.abs(generated_arr - reference_arr).mean(axis=2)
    hed_delta = np.abs(hed - reference_hed).mean(axis=2)
    record["reference_rgb_mae"] = float(rgb_delta[joint_mask].mean())
    record["reference_hed_mae"] = float(hed_delta[joint_mask].mean())
    return record


def summarize_mode_outputs(config: DecompositionConfig) -> tuple[Path, Path]:
    out_dir = ensure_directory(config.out_dir)
    generated_root = out_dir / "generated"
    he_dir = config.data_root / "he" if (config.data_root / "he").is_dir() else config.data_root

    metric_rows: list[dict[str, Any]] = []
    if generated_root.is_dir():
        for tile_dir in sorted(path for path in generated_root.iterdir() if path.is_dir()):
            tile_id = tile_dir.name
            reference_path = he_dir / f"{tile_id}.png"
            for mode in DEFAULT_MODES:
                image_path = tile_dir / f"{mode.name}.png"
                if not image_path.is_file():
                    continue
                metric_rows.append(
                    _mode_metrics_row(
                        tile_id=tile_id,
                        mode_name=mode.name,
                        image_path=image_path,
                        reference_path=reference_path,
                        out_dir=out_dir,
                        data_root=config.data_root,
                    )
                )

    metrics_fields = (
        "tile_id",
        "mode",
        "image_path",
        "reference_path",
        "has_reference",
        *MODE_METRIC_FIELDS,
    )
    metrics_path = _write_csv_rows(out_dir / "mode_metrics.csv", metrics_fields, metric_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row["mode"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for mode in DEFAULT_MODES:
        rows = grouped.get(mode.name, [])
        summary_row: dict[str, Any] = {
            "mode": mode.name,
            "n_tiles": len(rows),
            "reference_count": sum(int(row["has_reference"]) for row in rows),
        }
        for field in MODE_SUMMARY_FIELDS:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            mean_value, std_value = _mean_std(values)
            summary_row[f"{field}_mean"] = mean_value
            summary_row[f"{field}_std"] = std_value
        summary_rows.append(summary_row)

    summary_fields = (
        "mode",
        "n_tiles",
        "reference_count",
        *(f"{field}_mean" for field in MODE_SUMMARY_FIELDS),
        *(f"{field}_std" for field in MODE_SUMMARY_FIELDS),
    )
    summary_path = _write_csv_rows(out_dir / "mode_summary.csv", summary_fields, summary_rows)
    return metrics_path, summary_path


def main(argv: list[str] | None = None) -> int:
    """Write a plan file or execute one worker branch."""
    parser = argparse.ArgumentParser(description="Plan the UNI/TME decomposition task")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-n", type=int, default=500)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    config = DecompositionConfig(
        config_path=Path(args.config_path),
        checkpoint_dir=Path(args.checkpoint_dir),
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        sample_n=args.sample_n,
    )

    if args.worker is not None:
        device = str(args.device or _default_device())
        if args.worker == "summarize":
            summarize_mode_outputs(config)
        else:
            run_worker_tile(
                config,
                args.worker,
                device=device,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
        return 0

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
