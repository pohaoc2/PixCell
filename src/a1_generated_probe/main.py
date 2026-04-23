"""Planner and workers for the generated-H&E probe task."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.tile_ids import tile_ids_sha1
from src._tasklib.runtime import CommandSpec, JobPlan, JobState, RuntimeProbe, TaskPlan, probe_runtime


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


EMBEDDINGS_NAME = "generated_uni_embeddings.npy"
GENERATED_TILE_IDS_NAME = "generated_tile_ids.txt"
MANIFEST_NAME = "generated_probe_manifest.json"
GENERATED_RESULTS_PREFIX = "generated_probe"
COMPARISON_NAME = "real_vs_generated_r2.csv"


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


def load_tile_ids(tile_ids_path: str | Path) -> list[str]:
    return [line.strip() for line in Path(tile_ids_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_paths(out_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(out_dir)
    return {
        "embeddings": output_dir / EMBEDDINGS_NAME,
        "generated_tile_ids": output_dir / GENERATED_TILE_IDS_NAME,
        "manifest": output_dir / MANIFEST_NAME,
        "generated_results_json": output_dir / f"{GENERATED_RESULTS_PREFIX}_results.json",
        "generated_results_csv": output_dir / f"{GENERATED_RESULTS_PREFIX}_results.csv",
        "comparison_csv": output_dir / COMPARISON_NAME,
    }


def _python_job(config: GeneratedProbeConfig, worker: str) -> CommandSpec:
    return CommandSpec(
        argv=(
            "python",
            "-m",
            "src.a1_generated_probe.main",
            "--worker",
            worker,
            "--generated-root",
            str(config.generated_root),
            "--uni-model-path",
            str(config.uni_model_path),
            "--targets-path",
            str(config.targets_path),
            "--tile-ids-path",
            str(config.tile_ids_path),
            "--cv-splits-path",
            str(config.cv_splits_path),
            "--out-dir",
            str(config.out_dir),
            "--device",
            config.device,
        ),
        cwd=_repo_root(),
    )


def _write_tile_ids(tile_ids: list[str], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.write_text("\n".join(tile_ids) + ("\n" if tile_ids else ""), encoding="utf-8")
    return output_path


def _load_generated_tile_ids(path: str | Path) -> list[str]:
    return load_tile_ids(path)


def _load_uni_extractor_cls():
    from pipeline.extract_features import UNI2hExtractor

    return UNI2hExtractor


def _discover_real_features_dir() -> Path:
    repo_root = _repo_root()
    candidates = (
        repo_root / "data" / "orion-crc33" / "features",
        repo_root / "data" / "test-orion-crc33" / "features",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate real UNI feature directory under data/")


def _default_target_names(n_targets: int) -> list[str]:
    return [f"target_{index}" for index in range(n_targets)]


def _remap_splits(
    splits: list[dict[str, list[int]]],
    selected_indices: list[int],
) -> list[dict[str, list[int]]]:
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
    remapped: list[dict[str, list[int]]] = []
    for split in splits:
        remapped.append(
            {
                "train_idx": [index_map[idx] for idx in split["train_idx"] if idx in index_map],
                "test_idx": [index_map[idx] for idx in split["test_idx"] if idx in index_map],
            }
        )
    return remapped


def _write_comparison_csv(
    rows: list[dict[str, float | str | list[float] | int]],
    real_rows: list[dict[str, float | str | list[float] | int]],
    out_path: str | Path,
) -> Path:
    real_by_target = {str(row["target"]): row for row in real_rows}
    output_path = Path(out_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "real_r2", "generated_r2", "ratio"])
        writer.writeheader()
        for row in rows:
            target = str(row["target"])
            generated_r2 = float(row["r2_mean"])
            real_r2 = float(real_by_target.get(target, {}).get("r2_mean", float("nan")))
            ratio = float("nan")
            if np.isfinite(real_r2) and real_r2 != 0.0:
                ratio = generated_r2 / real_r2
            writer.writerow(
                {
                    "target": target,
                    "real_r2": real_r2,
                    "generated_r2": generated_r2,
                    "ratio": ratio,
                }
            )
    return output_path


def run_embed_worker(
    config: GeneratedProbeConfig,
    *,
    batch_size: int = 16,
) -> dict[str, Path]:
    output_dir = ensure_directory(config.out_dir)
    output_paths = _output_paths(output_dir)
    requested_tile_ids = load_tile_ids(config.tile_ids_path)
    indexed = index_generated_tiles(config.generated_root)
    selected_tile_ids = [tile_id for tile_id in requested_tile_ids if tile_id in indexed]
    missing_tile_ids = [tile_id for tile_id in requested_tile_ids if tile_id not in indexed]
    if not selected_tile_ids:
        raise FileNotFoundError("no generated PNGs matched tile_ids.txt")

    extractor_cls = _load_uni_extractor_cls()
    extractor = extractor_cls(config.uni_model_path, device=config.device)

    embedded_tile_ids: list[str] = []
    failed_tile_ids: list[str] = []
    embedding_batches: list[np.ndarray] = []

    for start in range(0, len(selected_tile_ids), batch_size):
        batch_tile_ids = selected_tile_ids[start : start + batch_size]
        batch_images: list[np.ndarray] = []
        batch_loaded_tile_ids: list[str] = []
        for tile_id in batch_tile_ids:
            try:
                from PIL import Image

                image = Image.open(indexed[tile_id]).convert("RGB")
                batch_images.append(np.asarray(image))
                batch_loaded_tile_ids.append(tile_id)
            except Exception:
                failed_tile_ids.append(tile_id)
        if not batch_images:
            continue
        batch_embeddings = np.asarray(extractor.extract_batch(batch_images), dtype=np.float32)
        if batch_embeddings.ndim == 1:
            batch_embeddings = batch_embeddings[None, :]
        embedding_batches.append(batch_embeddings)
        embedded_tile_ids.extend(batch_loaded_tile_ids)

    if not embedding_batches:
        raise RuntimeError("failed to embed all selected generated PNGs")

    embeddings = np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
    np.save(output_paths["embeddings"], embeddings)
    _write_tile_ids(embedded_tile_ids, output_paths["generated_tile_ids"])
    write_json(
        {
            "version": 1,
            "generated_root": str(Path(config.generated_root).resolve()),
            "requested_tile_count": len(requested_tile_ids),
            "embedded_tile_count": len(embedded_tile_ids),
            "feature_dim": int(embeddings.shape[1]),
            "embedded_tile_ids_sha1": tile_ids_sha1(embedded_tile_ids),
            "missing_generated_tile_ids": missing_tile_ids,
            "failed_tile_ids": failed_tile_ids,
        },
        output_paths["manifest"],
    )
    return {
        "embeddings": output_paths["embeddings"],
        "generated_tile_ids": output_paths["generated_tile_ids"],
        "manifest": output_paths["manifest"],
    }


def run_probe_worker(config: GeneratedProbeConfig) -> dict[str, Path]:
    from src.a1_probe_linear.main import (
        load_cv_splits,
        load_feature_matrix,
        run_cv_regression,
        summarize_probe_results,
        write_probe_results,
    )

    output_dir = ensure_directory(config.out_dir)
    output_paths = _output_paths(output_dir)
    generated_tile_ids = _load_generated_tile_ids(output_paths["generated_tile_ids"])
    if not generated_tile_ids:
        raise ValueError("generated_tile_ids.txt is empty; run --worker embed first")

    generated_embeddings = np.load(output_paths["embeddings"]).astype(np.float32)
    if generated_embeddings.shape[0] != len(generated_tile_ids):
        raise ValueError("generated embeddings row count does not match generated_tile_ids.txt")

    real_features_dir = _discover_real_features_dir()
    real_feature_mask = [
        (real_features_dir / f"{tile_id}_uni.npy").is_file() for tile_id in generated_tile_ids
    ]
    if not any(real_feature_mask):
        raise FileNotFoundError("no matching real UNI feature files found for generated tile IDs")
    if not all(real_feature_mask):
        generated_tile_ids = [
            tile_id for tile_id, keep in zip(generated_tile_ids, real_feature_mask, strict=False) if keep
        ]
        generated_embeddings = generated_embeddings[np.asarray(real_feature_mask, dtype=bool)]

    all_tile_ids = load_tile_ids(config.tile_ids_path)
    tile_index = {tile_id: idx for idx, tile_id in enumerate(all_tile_ids)}
    missing_targets = [tile_id for tile_id in generated_tile_ids if tile_id not in tile_index]
    if missing_targets:
        raise ValueError(f"generated tile IDs missing from tile_ids.txt: {missing_targets[:5]}")

    selected_indices = [tile_index[tile_id] for tile_id in generated_tile_ids]
    targets = np.load(config.targets_path).astype(np.float32)
    if targets.shape[0] != len(all_tile_ids):
        raise ValueError("target matrix row count does not match tile_ids.txt")
    selected_targets = targets[selected_indices]

    splits = load_cv_splits(all_tile_ids, config.cv_splits_path)
    subset_splits = _remap_splits(splits, selected_indices)
    target_names = _default_target_names(selected_targets.shape[1])

    generated_fold_scores, _, _ = run_cv_regression(generated_embeddings, selected_targets, subset_splits)
    generated_rows = summarize_probe_results(generated_fold_scores, target_names)
    generated_result_paths = write_probe_results(generated_rows, output_dir, prefix=GENERATED_RESULTS_PREFIX)

    real_embeddings = load_feature_matrix(real_features_dir, generated_tile_ids)
    real_fold_scores, _, _ = run_cv_regression(real_embeddings, selected_targets, subset_splits)
    real_rows = summarize_probe_results(real_fold_scores, target_names)
    comparison_path = _write_comparison_csv(generated_rows, real_rows, output_paths["comparison_csv"])

    manifest_payload = {}
    if output_paths["manifest"].is_file():
        manifest_payload = json.loads(output_paths["manifest"].read_text(encoding="utf-8"))
    manifest_payload.update(
        {
            "probe_tile_count": len(generated_tile_ids),
            "probe_tile_ids_sha1": tile_ids_sha1(generated_tile_ids),
            "real_features_dir": str(real_features_dir.resolve()),
            "comparison_csv": str(comparison_path),
        }
    )
    write_json(manifest_payload, output_paths["manifest"])
    return {
        **generated_result_paths,
        "comparison_csv": comparison_path,
        "manifest": output_paths["manifest"],
    }


def plan_task(config: GeneratedProbeConfig, runtime: RuntimeProbe | None = None) -> TaskPlan:
    """Plan the generated-image embedding and evaluation jobs."""
    runtime = runtime or probe_runtime()
    out_dir = ensure_directory(config.out_dir)
    output_paths = _output_paths(out_dir)
    indexed = index_generated_tiles(config.generated_root)
    tile_ids = load_tile_ids(config.tile_ids_path)
    available = [tile_id for tile_id in tile_ids if tile_id in indexed]

    jobs: list[JobPlan] = []

    if not available:
        embed_state = JobState.BLOCKED
        embed_reason = "missing_generated_pngs"
        embed_command = None
    elif not config.uni_model_path.exists():
        embed_state = JobState.BLOCKED
        embed_reason = "missing_weights"
        embed_command = None
    elif config.skip_existing and output_paths["embeddings"].is_file() and output_paths["generated_tile_ids"].is_file():
        embed_state = JobState.SKIPPED
        embed_reason = "existing_output"
        embed_command = None
    elif not runtime.has_cuda:
        embed_state = JobState.DEFERRED
        embed_reason = "missing_gpu"
        embed_command = _python_job(config, "embed")
    else:
        embed_state = JobState.READY
        embed_reason = None
        embed_command = _python_job(config, "embed")
    jobs.append(
        JobPlan(
            job_id="embed_generated_uni",
            state=embed_state,
            reason=embed_reason,
            inputs=tuple(indexed[tile_id] for tile_id in available),
            outputs=(output_paths["embeddings"], output_paths["generated_tile_ids"], output_paths["manifest"]),
            command=embed_command,
        )
    )

    probe_ready = output_paths["embeddings"].is_file() and output_paths["generated_tile_ids"].is_file()
    jobs.append(
        JobPlan(
            job_id="probe_generated_embeddings",
            state=JobState.READY if probe_ready else JobState.BLOCKED,
            reason=None if probe_ready else "missing_generated_embeddings",
            inputs=(
                output_paths["embeddings"],
                output_paths["generated_tile_ids"],
                config.targets_path,
                config.tile_ids_path,
                config.cv_splits_path,
            ),
            outputs=(
                output_paths["generated_results_json"],
                output_paths["generated_results_csv"],
                output_paths["comparison_csv"],
            ),
            command=_python_job(config, "probe") if probe_ready else None,
        )
    )
    warnings = list(runtime.warnings)
    missing = len(tile_ids) - len(available)
    if missing:
        warnings.append(f"missing generated PNGs for {missing} tiles")
    return TaskPlan(task_name="a1_generated_probe", jobs=tuple(jobs), warnings=tuple(warnings))


def main(argv: list[str] | None = None) -> int:
    """Write a plan file or execute a worker."""
    parser = argparse.ArgumentParser(description="Plan the generated-H&E probe task")
    parser.add_argument("--generated-root", default=None)
    parser.add_argument("--uni-model-path", default=None)
    parser.add_argument("--targets-path", default=None)
    parser.add_argument("--tile-ids-path", default=None)
    parser.add_argument("--cv-splits-path", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", default=None)
    args = parser.parse_args(argv)

    config = GeneratedProbeConfig(
        generated_root=Path(args.generated_root),
        uni_model_path=Path(args.uni_model_path),
        targets_path=Path(args.targets_path),
        tile_ids_path=Path(args.tile_ids_path),
        cv_splits_path=Path(args.cv_splits_path),
        out_dir=Path(args.out_dir),
        device=args.device,
    )

    if args.worker == "embed":
        run_embed_worker(config)
        return 0
    if args.worker == "probe":
        run_probe_worker(config)
        return 0
    if args.worker is not None:
        raise ValueError(f"unknown worker {args.worker!r}")

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
