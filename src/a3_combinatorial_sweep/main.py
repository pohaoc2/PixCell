"""CPU-safe planner for the combinatorial sweep task."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.runtime import CommandSpec, JobPlan, JobState, RuntimeProbe, TaskPlan, probe_runtime


DEFAULT_DEVICE = "cuda"
DEFAULT_GUIDANCE_SCALE = 2.5
DEFAULT_NUM_STEPS = 20
DEFAULT_SEED = 42

CELL_STATE_CHANNELS = {
    "prolif": "cell_state_prolif",
    "nonprolif": "cell_state_nonprolif",
    "dead": "cell_state_dead",
}

LEVEL_VALUES = {
    "low": 0.50,
    "mid": 0.75,
    "high": 1.0,
}

MORPHOLOGY_METRICS = (
    "nuclear_density",
    "eosin_ratio",
    "hematoxylin_ratio",
    "mean_cell_size",
    "glcm_contrast",
    "glcm_homogeneity",
)


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


def _summary_output_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    return (
        out_dir / "morphological_signatures.csv",
        out_dir / "additive_model_residuals.csv",
        out_dir / "interaction_heatmap.png",
    )


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


def _config_cli_args(config: CombinatorialSweepConfig) -> tuple[str, ...]:
    argv: list[str] = [
        "--config-path",
        str(config.config_path),
        "--checkpoint-dir",
        str(config.checkpoint_dir),
        "--data-root",
        str(config.data_root),
        "--out-dir",
        str(config.out_dir),
    ]
    if config.anchor_tile_ids_path is not None:
        argv.extend(("--anchor-tile-ids-path", str(config.anchor_tile_ids_path)))
    else:
        for anchor_tile_id in config.anchor_tile_ids:
            argv.extend(("--anchor-tile-id", anchor_tile_id))
    return tuple(argv)


def _python_job(job_module: str, *args: str) -> CommandSpec:
    return CommandSpec(argv=("python", "-m", job_module, *args), cwd=Path(__file__).resolve().parents[2])


def plan_task(config: CombinatorialSweepConfig, runtime: RuntimeProbe | None = None) -> TaskPlan:
    """Plan one sweep batch per anchor plus a summary job."""
    runtime = runtime or probe_runtime()
    out_dir = ensure_directory(config.out_dir)
    anchors, anchor_error = load_anchor_tile_ids(config)
    conditions = enumerate_conditions()
    jobs: list[JobPlan] = []
    common_worker_args = _config_cli_args(config)

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
            command = _python_job("src.a3_combinatorial_sweep.main", *common_worker_args, "--worker", anchor)
        else:
            state = JobState.READY
            reason = None
            command = _python_job("src.a3_combinatorial_sweep.main", *common_worker_args, "--worker", anchor)
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

    summary_outputs = _summary_output_paths(out_dir)
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
            command=_python_job("src.a3_combinatorial_sweep.main", *common_worker_args, "--worker", "summarize")
            if summary_state == JobState.READY
            else None,
        )
    )
    return TaskPlan(task_name="a3_combinatorial_sweep", jobs=tuple(jobs), warnings=runtime.warnings)


def _load_generation_runtime(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    data_root: Path,
    device: str,
    num_steps: int,
) -> tuple[dict[str, Any], Any, Any, Path, Path]:
    from tools.stage3.channel_sweep import load_sweep_models
    from tools.stage3.tile_pipeline import resolve_data_layout

    models, config, scheduler = load_sweep_models(
        config_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        num_steps=num_steps,
    )
    exp_channels_dir, feat_dir, _ = resolve_data_layout(data_root)
    return models, config, scheduler, exp_channels_dir, feat_dir


def _load_anchor_ctrl(tile_id: str, *, active_channels: list[str], image_size: int, exp_channels_dir: Path):
    from tools.stage3.tile_pipeline import load_exp_channels

    return load_exp_channels(tile_id, active_channels, image_size, exp_channels_dir)


def _load_anchor_uni(tile_id: str, *, feat_dir: Path):
    from tools.stage3.common import resolve_uni_embedding

    return resolve_uni_embedding(tile_id, feat_dir=feat_dir, null_uni=False)


def _make_generation_noise(*, config: Any, scheduler: Any, device: str, seed: int):
    from tools.stage3.common import inference_dtype
    from tools.stage3.tile_pipeline import _make_fixed_noise

    dtype = inference_dtype(device)
    return _make_fixed_noise(
        config=config,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        seed=seed,
    )


def _render_generated_image(
    ctrl_full,
    *,
    models: dict[str, Any],
    config: Any,
    scheduler: Any,
    uni_embeds,
    device: str,
    guidance_scale: float,
    fixed_noise,
    seed: int,
):
    from tools.stage3.channel_sweep import generate_from_ctrl

    return generate_from_ctrl(
        ctrl_full,
        models=models,
        config=config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        fixed_noise=fixed_noise,
        seed=seed,
    )


def _resolve_channel_index(active_channels: list[str], *candidate_names: str) -> int:
    for candidate_name in candidate_names:
        if candidate_name in active_channels:
            return active_channels.index(candidate_name)
    raise KeyError(f"missing required channel; expected one of {candidate_names}")


def _build_condition_ctrl(base_ctrl, *, active_channels: list[str], condition: SweepCondition):
    import torch

    ctrl = base_ctrl.clone()
    mask_index = _resolve_channel_index(active_channels, "cell_masks", "cell_mask")
    oxygen_index = _resolve_channel_index(active_channels, "oxygen")
    glucose_index = _resolve_channel_index(active_channels, "glucose")
    state_indices = {
        state_name: _resolve_channel_index(active_channels, channel_name)
        for state_name, channel_name in CELL_STATE_CHANNELS.items()
    }

    mask_plane = base_ctrl[mask_index].clone()
    for state_name, channel_index in state_indices.items():
        if state_name == condition.cell_state:
            ctrl[channel_index] = mask_plane
        else:
            ctrl[channel_index] = torch.zeros_like(mask_plane)
    ctrl[oxygen_index] = torch.full_like(ctrl[oxygen_index], condition.oxygen_value)
    ctrl[glucose_index] = torch.full_like(ctrl[glucose_index], condition.glucose_value)
    return ctrl


def _save_image(image, out_path: Path) -> Path:
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(out_path)
    return out_path


def run_anchor_worker(
    config: CombinatorialSweepConfig,
    anchor_id: str,
    *,
    device: str = DEFAULT_DEVICE,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, ...]:
    models, runtime_config, scheduler, exp_channels_dir, feat_dir = _load_generation_runtime(
        config_path=config.config_path,
        checkpoint_dir=config.checkpoint_dir,
        data_root=config.data_root,
        device=device,
        num_steps=num_steps,
    )
    active_channels = list(runtime_config.data.active_channels)
    base_ctrl = _load_anchor_ctrl(
        anchor_id,
        active_channels=active_channels,
        image_size=runtime_config.image_size,
        exp_channels_dir=exp_channels_dir,
    )
    uni_embeds = _load_anchor_uni(anchor_id, feat_dir=feat_dir)
    fixed_noise = _make_generation_noise(
        config=runtime_config,
        scheduler=scheduler,
        device=device,
        seed=seed,
    )

    outputs: list[Path] = []
    for condition in enumerate_conditions():
        condition_ctrl = _build_condition_ctrl(base_ctrl, active_channels=active_channels, condition=condition)
        generated = _render_generated_image(
            condition_ctrl,
            models=models,
            config=runtime_config,
            scheduler=scheduler,
            uni_embeds=uni_embeds,
            device=device,
            guidance_scale=guidance_scale,
            fixed_noise=fixed_noise,
            seed=seed,
        )
        out_path = config.out_dir / "generated" / anchor_id / f"{build_condition_id(condition)}.png"
        outputs.append(_save_image(generated, out_path))
    return tuple(outputs)


def _discover_summary_anchors(config: CombinatorialSweepConfig) -> list[str]:
    anchors, _ = load_anchor_tile_ids(config)
    if anchors:
        return anchors
    generated_root = config.out_dir / "generated"
    if not generated_root.is_dir():
        return []
    return sorted(path.name for path in generated_root.iterdir() if path.is_dir())


def _condition_lookup() -> dict[str, SweepCondition]:
    return {build_condition_id(condition): condition for condition in enumerate_conditions()}


def _connected_component_sizes(binary_mask) -> list[int]:
    import numpy as np

    binary = np.asarray(binary_mask, dtype=bool)
    try:
        from scipy import ndimage as ndi

        labels, count = ndi.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        return [int(np.sum(labels == label)) for label in range(1, int(count) + 1)]
    except Exception:
        pass

    height, width = binary.shape
    visited = np.zeros((height, width), dtype=bool)
    sizes: list[int] = []
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for row in range(height):
        for col in range(width):
            if not binary[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            size = 0
            while stack:
                current_row, current_col = stack.pop()
                size += 1
                for delta_row, delta_col in neighbors:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                        continue
                    if visited[next_row, next_col] or not binary[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            sizes.append(size)
    return sizes


def _compute_glcm_features(gray_image, tissue_mask, *, levels: int = 8) -> tuple[float, float]:
    import numpy as np

    gray = np.asarray(gray_image, dtype=np.float32)
    mask = np.asarray(tissue_mask, dtype=bool)
    quantized = np.clip(np.floor(gray * levels), 0, levels - 1).astype(np.int32)
    left = quantized[:, :-1]
    right = quantized[:, 1:]
    pair_mask = mask[:, :-1] & mask[:, 1:]
    if np.any(pair_mask):
        left = left[pair_mask]
        right = right[pair_mask]
    else:
        left = left.reshape(-1)
        right = right.reshape(-1)

    matrix = np.zeros((levels, levels), dtype=np.float64)
    for src, dst in zip(left.tolist(), right.tolist(), strict=False):
        matrix[src, dst] += 1.0
        matrix[dst, src] += 1.0
    total = float(matrix.sum())
    if total == 0.0:
        return 0.0, 1.0
    matrix /= total

    indices = np.arange(levels, dtype=np.float64)
    delta = indices[:, None] - indices[None, :]
    contrast = float(np.sum(matrix * (delta ** 2)))
    homogeneity = float(np.sum(matrix / (1.0 + (delta ** 2))))
    return contrast, homogeneity


def _compute_signature(image_path: Path) -> dict[str, float]:
    import numpy as np
    from PIL import Image

    from tools.stage3.hed_utils import rgb_to_hed, tissue_mask_from_rgb

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    tissue_mask = tissue_mask_from_rgb(image)
    if not np.any(tissue_mask):
        tissue_mask = np.ones(rgb.shape[:2], dtype=bool)

    hed = rgb_to_hed(image)
    hematoxylin = np.clip(hed[..., 0], 0.0, None)
    eosin = np.clip(hed[..., 1], 0.0, None)
    masked_h = hematoxylin[tissue_mask]
    masked_e = eosin[tissue_mask]
    h_mean = float(masked_h.mean()) if masked_h.size else 0.0
    e_mean = float(masked_e.mean()) if masked_e.size else 0.0
    stain_total = h_mean + e_mean
    hematoxylin_ratio = h_mean / stain_total if stain_total > 0.0 else 0.0
    eosin_ratio = e_mean / stain_total if stain_total > 0.0 else 0.0

    h_threshold = float(masked_h.mean() + 0.5 * masked_h.std()) if masked_h.size else 0.0
    nuclear_mask = tissue_mask & (hematoxylin >= h_threshold)
    component_sizes = _connected_component_sizes(nuclear_mask)
    tissue_area = int(np.count_nonzero(tissue_mask))
    nuclear_density = float(len(component_sizes) / tissue_area) if tissue_area else 0.0
    mean_cell_size = float(sum(component_sizes) / len(component_sizes)) if component_sizes else 0.0

    gray = np.clip(np.dot(rgb[..., :3], (0.299, 0.587, 0.114)), 0.0, 1.0)
    glcm_contrast, glcm_homogeneity = _compute_glcm_features(gray, tissue_mask)
    return {
        "nuclear_density": nuclear_density,
        "eosin_ratio": eosin_ratio,
        "hematoxylin_ratio": hematoxylin_ratio,
        "mean_cell_size": mean_cell_size,
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
    }


def _iter_signature_rows(config: CombinatorialSweepConfig) -> list[dict[str, Any]]:
    anchors = _discover_summary_anchors(config)
    condition_lookup = _condition_lookup()
    rows: list[dict[str, Any]] = []
    for anchor_id in anchors:
        anchor_dir = config.out_dir / "generated" / anchor_id
        if not anchor_dir.is_dir():
            continue
        for condition_id, condition in condition_lookup.items():
            image_path = anchor_dir / f"{condition_id}.png"
            if not image_path.is_file():
                continue
            row = {
                "anchor_id": anchor_id,
                "cell_state": condition.cell_state,
                "oxygen_label": condition.oxygen_label,
                "oxygen_value": condition.oxygen_value,
                "glucose_label": condition.glucose_label,
                "glucose_value": condition.glucose_value,
                "image_path": str(image_path),
            }
            row.update(_compute_signature(image_path))
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _fit_additive_rows(signature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in signature_rows:
        key = (row["cell_state"], row["oxygen_label"], row["glucose_label"])
        grouped.setdefault(key, []).append(row)

    by_state = {state: [row for row in signature_rows if row["cell_state"] == state] for state in CELL_STATE_CHANNELS}
    by_oxygen = {label: [row for row in signature_rows if row["oxygen_label"] == label] for label in LEVEL_VALUES}
    by_glucose = {label: [row for row in signature_rows if row["glucose_label"] == label] for label in LEVEL_VALUES}

    additive_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        cell_state, oxygen_label, glucose_label = key
        group_rows = grouped[key]
        summary_row: dict[str, Any] = {
            "cell_state": cell_state,
            "oxygen_label": oxygen_label,
            "oxygen_value": LEVEL_VALUES[oxygen_label],
            "glucose_label": glucose_label,
            "glucose_value": LEVEL_VALUES[glucose_label],
            "n_anchors": len(group_rows),
        }
        residual_sum = 0.0
        for metric_name in MORPHOLOGY_METRICS:
            grand_mean = _mean([float(row[metric_name]) for row in signature_rows])
            state_effect = _mean([float(row[metric_name]) for row in by_state[cell_state]]) - grand_mean
            oxygen_effect = _mean([float(row[metric_name]) for row in by_oxygen[oxygen_label]]) - grand_mean
            glucose_effect = _mean([float(row[metric_name]) for row in by_glucose[glucose_label]]) - grand_mean
            actual_value = _mean([float(row[metric_name]) for row in group_rows])
            expected_value = grand_mean + state_effect + oxygen_effect + glucose_effect
            residual_value = actual_value - expected_value
            summary_row[f"actual_{metric_name}"] = actual_value
            summary_row[f"expected_{metric_name}"] = expected_value
            summary_row[f"residual_{metric_name}"] = residual_value
            residual_sum += residual_value ** 2
        summary_row["residual_l2_norm"] = math.sqrt(residual_sum)
        additive_rows.append(summary_row)
    return additive_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_interaction_heatmap(path: Path, additive_rows: list[dict[str, Any]]) -> Path:
    from PIL import Image, ImageDraw

    states = tuple(CELL_STATE_CHANNELS.keys())
    levels = tuple(LEVEL_VALUES.keys())
    width = len(levels) * len(levels)
    height = len(states)
    cell_size = 32
    residual_lookup = {
        (row["cell_state"], row["oxygen_label"], row["glucose_label"]): float(row["residual_l2_norm"])
        for row in additive_rows
    }
    max_residual = max((float(row["residual_l2_norm"]) for row in additive_rows), default=1.0)
    if max_residual <= 0.0:
        max_residual = 1.0

    image = Image.new("RGB", (width * cell_size, height * cell_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    for state_index, state_name in enumerate(states):
        for oxygen_index, oxygen_label in enumerate(levels):
            for glucose_index, glucose_label in enumerate(levels):
                column_index = oxygen_index * len(levels) + glucose_index
                residual = residual_lookup.get((state_name, oxygen_label, glucose_label), 0.0)
                normalized = max(0.0, min(1.0, residual / max_residual))
                red = int(255 * normalized)
                blue = int(255 * (1.0 - normalized))
                x0 = column_index * cell_size
                y0 = state_index * cell_size
                draw.rectangle((x0, y0, x0 + cell_size - 1, y0 + cell_size - 1), fill=(red, 64, blue), outline=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def run_summary_worker(config: CombinatorialSweepConfig) -> tuple[Path, Path, Path]:
    signature_rows = _iter_signature_rows(config)
    if not signature_rows:
        raise FileNotFoundError(f"no generated PNGs found under {config.out_dir / 'generated'}")

    signature_fieldnames = (
        "anchor_id",
        "cell_state",
        "oxygen_label",
        "oxygen_value",
        "glucose_label",
        "glucose_value",
        "image_path",
        *MORPHOLOGY_METRICS,
    )
    additive_rows = _fit_additive_rows(signature_rows)
    additive_fieldnames = (
        "cell_state",
        "oxygen_label",
        "oxygen_value",
        "glucose_label",
        "glucose_value",
        "n_anchors",
        *(f"actual_{metric_name}" for metric_name in MORPHOLOGY_METRICS),
        *(f"expected_{metric_name}" for metric_name in MORPHOLOGY_METRICS),
        *(f"residual_{metric_name}" for metric_name in MORPHOLOGY_METRICS),
        "residual_l2_norm",
    )
    signatures_path, residuals_path, heatmap_path = _summary_output_paths(config.out_dir)
    _write_csv(signatures_path, signature_rows, signature_fieldnames)
    _write_csv(residuals_path, additive_rows, additive_fieldnames)
    _write_interaction_heatmap(heatmap_path, additive_rows)
    return signatures_path, residuals_path, heatmap_path


def _require_args(parser: argparse.ArgumentParser, args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) in (None, [], ())]
    if missing:
        parser.error(f"missing required arguments: {', '.join('--' + name.replace('_', '-') for name in missing)}")


def main(argv: list[str] | None = None) -> int:
    """Write a plan file or execute a sweep worker."""
    parser = argparse.ArgumentParser(description="Plan the combinatorial sweep task")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--anchor-tile-id", action="append", default=None)
    parser.add_argument("--anchor-tile-ids-path", default=None)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    if args.worker is None:
        _require_args(parser, args, ("config_path", "checkpoint_dir", "data_root", "out_dir"))
    elif args.worker == "summarize":
        _require_args(parser, args, ("out_dir",))
    else:
        _require_args(parser, args, ("config_path", "checkpoint_dir", "data_root", "out_dir"))

    config = CombinatorialSweepConfig(
        config_path=Path(args.config_path or "."),
        checkpoint_dir=Path(args.checkpoint_dir or "."),
        data_root=Path(args.data_root or "."),
        out_dir=Path(args.out_dir),
        anchor_tile_ids=tuple(args.anchor_tile_id or ()),
        anchor_tile_ids_path=Path(args.anchor_tile_ids_path) if args.anchor_tile_ids_path else None,
    )

    if args.worker is not None:
        if args.worker == "summarize":
            run_summary_worker(config)
            return 0
        run_anchor_worker(
            config,
            args.worker,
            device=args.device,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
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
