"""CPU-safe planner and worker entry point for the encoder-comparison task."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src._tasklib.io import ensure_directory, write_json
from src._tasklib.runtime import CommandSpec, JobPlan, JobState, TaskPlan, RuntimeProbe, probe_runtime
from src._tasklib.tile_ids import tile_ids_sha1


_RAW_CNN_BATCH_SIZE = 8
_RAW_CNN_EPOCHS = 12
_RAW_CNN_IMAGE_SIZE = 64
_RAW_CNN_LEARNING_RATE = 1e-3
_VIRCHOW_BATCH_SIZE = 16
_COMPARISON_FIELDNAMES = ("target", "uni_r2", "virchow_r2", "cnn_r2")
_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def load_tile_ids(tile_ids_path: str | Path) -> list[str]:
    try:
        from src.a1_probe_linear.main import load_tile_ids as imported_load_tile_ids

        return imported_load_tile_ids(tile_ids_path)
    except Exception:
        return [
            line.strip()
            for line in Path(tile_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def load_cv_splits(tile_ids: list[str], cv_splits_path: str | Path) -> list[dict[str, list[int]]]:
    try:
        from src.a1_probe_linear.main import load_cv_splits as imported_load_cv_splits

        return imported_load_cv_splits(tile_ids, cv_splits_path)
    except Exception:
        payload = json.loads(Path(cv_splits_path).read_text(encoding="utf-8"))
        expected_hash = tile_ids_sha1(tile_ids)
        if payload.get("tile_ids_sha1") != expected_hash:
            raise ValueError("tile_ids.txt does not match the saved CV split hash")
        return list(payload["splits"])


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2, dtype=np.float64))
    y_mean = float(np.mean(y_true, dtype=np.float64))
    ss_tot = float(np.sum((y_true - y_mean) ** 2, dtype=np.float64))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _run_cv_regression_fallback(
    X: np.ndarray,
    Y: np.ndarray,
    splits: list[dict[str, list[int]]],
    *,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_targets = Y.shape[1]
    n_features = X.shape[1]
    fold_scores = np.full((len(splits), n_targets), np.nan, dtype=np.float32)
    oof_predictions = np.full_like(Y, np.nan, dtype=np.float32)
    coef_mean = np.zeros((n_targets, n_features), dtype=np.float32)

    for fold_idx, split in enumerate(splits):
        train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)
        X_train = np.asarray(X[train_idx], dtype=np.float64)
        X_test = np.asarray(X[test_idx], dtype=np.float64)
        x_mean = np.mean(X_train, axis=0, dtype=np.float64)
        x_std = np.std(X_train, axis=0, dtype=np.float64)
        x_std[x_std == 0.0] = 1.0
        X_train_scaled = (X_train - x_mean) / x_std
        X_test_scaled = (X_test - x_mean) / x_std

        for target_idx in range(n_targets):
            y_train = np.asarray(Y[train_idx, target_idx], dtype=np.float64)
            y_test = np.asarray(Y[test_idx, target_idx], dtype=np.float64)
            train_mask = np.isfinite(y_train)
            test_mask = np.isfinite(y_test)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            y_train_valid = y_train[train_mask]
            y_mean = float(np.mean(y_train_valid, dtype=np.float64))
            centered = y_train_valid - y_mean
            X_valid = X_train_scaled[train_mask]
            eye = np.eye(X_valid.shape[1], dtype=np.float64)
            coef = np.linalg.solve(X_valid.T @ X_valid + alpha * eye, X_valid.T @ centered)
            coef_mean[target_idx] += coef.astype(np.float32, copy=False)

            preds = X_test_scaled[test_mask] @ coef + y_mean
            preds = preds.astype(np.float32, copy=False)
            oof_predictions[test_idx[test_mask], target_idx] = preds
            fold_scores[fold_idx, target_idx] = float(_r2_score(y_test[test_mask], preds))

    return fold_scores, oof_predictions, coef_mean


def run_cv_regression(
    X: np.ndarray,
    Y: np.ndarray,
    splits: list[dict[str, list[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from src.a1_probe_linear.main import run_cv_regression as imported_run_cv_regression

        return imported_run_cv_regression(X, Y, splits)
    except Exception:
        return _run_cv_regression_fallback(X, Y, splits)


def summarize_probe_results(
    fold_scores: np.ndarray,
    target_names: list[str],
) -> list[dict[str, float | str | list[float] | int]]:
    try:
        from src.a1_probe_linear.main import summarize_probe_results as imported_summarize_probe_results

        return imported_summarize_probe_results(fold_scores, target_names)
    except Exception:
        rows: list[dict[str, float | str | list[float] | int]] = []
        for target_idx, target_name in enumerate(target_names):
            column = fold_scores[:, target_idx]
            finite = np.isfinite(column)
            values = column[finite]
            rows.append(
                {
                    "target": target_name,
                    "r2_mean": float(np.mean(values)) if values.size else float("nan"),
                    "r2_sd": float(np.std(values)) if values.size else float("nan"),
                    "r2_folds": [float(value) for value in values.tolist()],
                    "n_valid_folds": int(values.size),
                }
            )
        return rows


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


def _worker_command(config: ProbeEncodersConfig, worker: str) -> CommandSpec:
    argv = [
        "python",
        "-m",
        "src.a1_probe_encoders.main",
        "--worker",
        worker,
        "--he-dir",
        str(config.he_dir),
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
    ]
    if config.virchow_weights is not None:
        argv.extend(["--virchow-weights", str(config.virchow_weights)])
    return CommandSpec(argv=tuple(argv), cwd=Path(__file__).resolve().parents[2])


def _virchow_skip_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "virchow_SKIPPED.txt"


def _write_virchow_skip_marker(out_dir: str | Path, message: str) -> Path:
    skip_path = _virchow_skip_path(out_dir)
    skip_path.parent.mkdir(parents=True, exist_ok=True)
    skip_path.write_text(message.rstrip() + "\n", encoding="utf-8")
    return skip_path


def _normalize_target_matrix(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim == 1:
        return targets[:, np.newaxis]
    if targets.ndim != 2:
        raise ValueError(f"expected 1D or 2D target matrix, got shape {targets.shape}")
    return targets


def _load_targets(targets_path: str | Path) -> np.ndarray:
    return _normalize_target_matrix(np.load(targets_path))


def _default_target_names(targets: np.ndarray) -> list[str]:
    return [f"target_{index}" for index in range(targets.shape[1])]


def _load_target_names(targets_path: str | Path, cv_splits_path: str | Path) -> list[str]:
    targets = _load_targets(targets_path)
    manifest_path = Path(cv_splits_path).with_name("manifest.json")
    if not manifest_path.is_file():
        return _default_target_names(targets)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_names = payload.get("target_names")
    if not isinstance(target_names, list):
        return _default_target_names(targets)
    if len(target_names) != targets.shape[1]:
        raise ValueError("target_names in manifest.json do not match targets.npy width")
    return [str(name) for name in target_names]


def _load_uni_scores(cv_splits_path: str | Path, target_names: list[str]) -> dict[str, float]:
    results_path = Path(cv_splits_path).with_name("linear_probe_results.json")
    if not results_path.is_file():
        raise FileNotFoundError(
            "expected linear_probe_results.json next to cv_splits.json to populate the uni_r2 column"
        )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("linear_probe_results.json is missing a results list")
    by_target = {str(row["target"]): float(row["r2_mean"]) for row in rows}
    return {target_name: by_target.get(target_name, float("nan")) for target_name in target_names}


def _score_embeddings(
    embeddings_path: str | Path,
    *,
    targets_path: str | Path,
    tile_ids_path: str | Path,
    cv_splits_path: str | Path,
    target_names: list[str],
) -> dict[str, float]:
    tile_ids = load_tile_ids(tile_ids_path)
    targets = _load_targets(targets_path)
    if targets.shape[0] != len(tile_ids):
        raise ValueError("targets.npy row count does not match tile_ids.txt")
    embeddings = np.asarray(np.load(embeddings_path), dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[:, np.newaxis]
    if embeddings.ndim != 2:
        raise ValueError(f"expected 2D embedding matrix, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(tile_ids):
        raise ValueError("embedding row count does not match tile_ids.txt")

    splits = load_cv_splits(tile_ids, cv_splits_path)
    fold_scores, _, _ = run_cv_regression(embeddings, targets, splits)
    rows = summarize_probe_results(fold_scores, target_names)
    return {str(row["target"]): float(row["r2_mean"]) for row in rows}


def _write_encoder_comparison_csv(
    output_path: str | Path,
    *,
    target_names: list[str],
    uni_scores: dict[str, float],
    virchow_scores: dict[str, float],
    cnn_scores: dict[str, float],
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_COMPARISON_FIELDNAMES))
        writer.writeheader()
        for target_name in target_names:
            writer.writerow(
                {
                    "target": target_name,
                    "uni_r2": float(uni_scores.get(target_name, float("nan"))),
                    "virchow_r2": float(virchow_scores.get(target_name, float("nan"))),
                    "cnn_r2": float(cnn_scores.get(target_name, float("nan"))),
                }
            )
    return out_path


def _resolve_device(device: str) -> str:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _seed_torch(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _find_tile_image_path(he_dir: str | Path, tile_id: str) -> Path:
    base = Path(he_dir)
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = base / f"{tile_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing H&E tile for {tile_id!r} under {base}")


def _load_rgb_image(image_path: str | Path, *, image_size: int | None = None) -> Image.Image:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        if image_size is not None:
            rgb = rgb.resize((image_size, image_size), _RESAMPLE_BILINEAR)
        return rgb


def _iter_batches(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _image_to_tensor(image_path: str | Path, *, image_size: int) -> Any:
    import torch

    image = _load_rgb_image(image_path, image_size=image_size)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected RGB image at {image_path}, got shape {array.shape}")
    array = np.transpose(array / 255.0, (2, 0, 1))
    return torch.from_numpy(array)


def _make_raw_cnn_model(n_targets: int) -> Any:
    import torch

    class RawCNNRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            self.head = torch.nn.Linear(256, n_targets)

        def forward(self, inputs):
            embeddings = self.encoder(inputs).flatten(1)
            return self.head(embeddings), embeddings

    return RawCNNRegressor()


def _train_raw_cnn_embeddings_fallback(
    he_dir: str | Path,
    tile_ids_path: str | Path,
    *,
    image_size: int = 16,
) -> np.ndarray:
    tile_ids = load_tile_ids(tile_ids_path)
    if not tile_ids:
        raise ValueError("tile_ids.txt is empty")

    rows: list[np.ndarray] = []
    for tile_id in tile_ids:
        image = _load_rgb_image(_find_tile_image_path(he_dir, tile_id), image_size=image_size).convert("L")
        rows.append((np.asarray(image, dtype=np.float32).reshape(-1) / 255.0).astype(np.float32, copy=False))
    return np.stack(rows, axis=0)


def _train_raw_cnn_embeddings(
    he_dir: str | Path,
    targets_path: str | Path,
    tile_ids_path: str | Path,
    *,
    device: str,
    image_size: int = _RAW_CNN_IMAGE_SIZE,
    batch_size: int = _RAW_CNN_BATCH_SIZE,
    epochs: int = _RAW_CNN_EPOCHS,
    learning_rate: float = _RAW_CNN_LEARNING_RATE,
    seed: int = 0,
) -> np.ndarray:
    try:
        import torch
    except ModuleNotFoundError:
        fallback_size = int(np.sqrt(256))
        return _train_raw_cnn_embeddings_fallback(
            he_dir,
            tile_ids_path,
            image_size=min(image_size, fallback_size),
        )

    tile_ids = load_tile_ids(tile_ids_path)
    targets = _load_targets(targets_path)
    if targets.shape[0] != len(tile_ids):
        raise ValueError("targets.npy row count does not match tile_ids.txt")
    if len(tile_ids) == 0:
        raise ValueError("tile_ids.txt is empty")

    _seed_torch(seed)
    resolved_device = _resolve_device(device)
    model = _make_raw_cnn_model(targets.shape[1]).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    indexed_targets = {tile_id: targets[index] for index, tile_id in enumerate(tile_ids)}
    batches = _iter_batches(tile_ids, max(1, min(batch_size, len(tile_ids))))
    model.train()
    for _ in range(max(1, epochs)):
        order = tile_ids[:]
        random.shuffle(order)
        for batch_tile_ids in _iter_batches(order, max(1, min(batch_size, len(tile_ids)))):
            image_batch = torch.stack(
                [_image_to_tensor(_find_tile_image_path(he_dir, tile_id), image_size=image_size) for tile_id in batch_tile_ids],
                dim=0,
            ).to(resolved_device)
            target_batch = torch.as_tensor(
                np.stack([indexed_targets[tile_id] for tile_id in batch_tile_ids], axis=0),
                dtype=torch.float32,
                device=resolved_device,
            )
            predictions, _ = model(image_batch)
            mask = torch.isfinite(target_batch)
            if not bool(mask.any()):
                continue
            loss = torch.mean((predictions[mask] - target_batch[mask]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_tile_ids in batches:
            image_batch = torch.stack(
                [_image_to_tensor(_find_tile_image_path(he_dir, tile_id), image_size=image_size) for tile_id in batch_tile_ids],
                dim=0,
            ).to(resolved_device)
            _, embeddings = model(image_batch)
            outputs.append(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


class _TorchModuleExtractor:
    def __init__(self, model: Any, *, device: str, image_size: int = 224) -> None:
        self.model = model
        self.device = device
        self.image_size = image_size

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        tensors = []
        for image in images:
            resized = image.convert("RGB").resize((self.image_size, self.image_size), _RESAMPLE_BILINEAR)
            array = np.asarray(resized, dtype=np.float32)
            tensor = torch.from_numpy(np.transpose(array / 255.0, (2, 0, 1)))
            tensors.append(tensor)
        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(batch)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if outputs.ndim == 3 and outputs.shape[1] >= 6:
            class_token = outputs[:, 0]
            patch_tokens = outputs[:, 5:]
            outputs = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
        elif outputs.ndim > 2:
            outputs = outputs.flatten(1)
        return outputs.detach().cpu().numpy().astype(np.float32, copy=False)


def _build_virchow_extractor(weights_path: str | Path, *, device: str) -> Any:
    import torch

    resolved_device = _resolve_device(device)
    model = None
    weights_file = Path(weights_path)
    try:
        model = torch.jit.load(str(weights_file), map_location=resolved_device)
    except Exception:
        payload = torch.load(str(weights_file), map_location=resolved_device, weights_only=False)
        if hasattr(payload, "eval") and callable(payload):
            model = payload
        elif isinstance(payload, dict) and hasattr(payload.get("model"), "eval"):
            model = payload["model"]
    if model is None or not hasattr(model, "eval"):
        raise RuntimeError(
            "Virchow weights must be a TorchScript module, a serialized torch.nn.Module, or a checkpoint dict with a 'model' entry"
        )
    if hasattr(model, "to"):
        model = model.to(resolved_device)
    model.eval()
    return _TorchModuleExtractor(model, device=resolved_device)


def _extract_embeddings_from_images(
    he_dir: str | Path,
    tile_ids_path: str | Path,
    *,
    extractor: Any,
    batch_size: int,
) -> np.ndarray:
    tile_ids = load_tile_ids(tile_ids_path)
    if not tile_ids:
        raise ValueError("tile_ids.txt is empty")
    outputs: list[np.ndarray] = []
    for batch_tile_ids in _iter_batches(tile_ids, max(1, batch_size)):
        images = [_load_rgb_image(_find_tile_image_path(he_dir, tile_id)) for tile_id in batch_tile_ids]
        batch_embeddings = np.asarray(extractor.extract_batch(images), dtype=np.float32)
        if batch_embeddings.ndim == 1:
            batch_embeddings = batch_embeddings[np.newaxis, :]
        if batch_embeddings.shape[0] != len(batch_tile_ids):
            raise ValueError("Virchow extractor returned a different number of rows than input tiles")
        outputs.append(batch_embeddings)
    return np.concatenate(outputs, axis=0)


def run_raw_cnn_worker(config: ProbeEncodersConfig) -> Path:
    output_dir = ensure_directory(config.out_dir)
    embeddings = _train_raw_cnn_embeddings(
        config.he_dir,
        config.targets_path,
        config.tile_ids_path,
        device=config.device,
    )
    output_path = output_dir / "raw_cnn_embeddings.npy"
    np.save(output_path, embeddings.astype(np.float32, copy=False))
    return output_path


def run_virchow_worker(config: ProbeEncodersConfig) -> Path:
    output_dir = ensure_directory(config.out_dir)
    weights_path = config.virchow_weights
    if weights_path is None or not Path(weights_path).is_file():
        return _write_virchow_skip_marker(
            output_dir,
            "Virchow skipped: provide --virchow-weights pointing to a readable model file.",
        )

    extractor = _build_virchow_extractor(weights_path, device=config.device)
    embeddings = _extract_embeddings_from_images(
        config.he_dir,
        config.tile_ids_path,
        extractor=extractor,
        batch_size=_VIRCHOW_BATCH_SIZE,
    )
    output_path = output_dir / "virchow_embeddings.npy"
    np.save(output_path, embeddings.astype(np.float32, copy=False))
    skip_path = _virchow_skip_path(output_dir)
    if skip_path.exists():
        skip_path.unlink()
    return output_path


def run_compare_worker(config: ProbeEncodersConfig) -> Path:
    output_dir = ensure_directory(config.out_dir)
    cnn_embeddings_path = output_dir / "raw_cnn_embeddings.npy"
    if not cnn_embeddings_path.is_file():
        raise FileNotFoundError("raw_cnn_embeddings.npy is required before running --worker compare")

    target_names = _load_target_names(config.targets_path, config.cv_splits_path)
    uni_scores = _load_uni_scores(config.cv_splits_path, target_names)
    cnn_scores = _score_embeddings(
        cnn_embeddings_path,
        targets_path=config.targets_path,
        tile_ids_path=config.tile_ids_path,
        cv_splits_path=config.cv_splits_path,
        target_names=target_names,
    )

    virchow_embeddings_path = output_dir / "virchow_embeddings.npy"
    if virchow_embeddings_path.is_file():
        virchow_scores = _score_embeddings(
            virchow_embeddings_path,
            targets_path=config.targets_path,
            tile_ids_path=config.tile_ids_path,
            cv_splits_path=config.cv_splits_path,
            target_names=target_names,
        )
    elif config.virchow_weights is None or not Path(config.virchow_weights).is_file():
        _write_virchow_skip_marker(
            output_dir,
            "Virchow skipped: provide --virchow-weights pointing to a readable model file.",
        )
        virchow_scores = {target_name: float("nan") for target_name in target_names}
    elif _virchow_skip_path(output_dir).is_file():
        virchow_scores = {target_name: float("nan") for target_name in target_names}
    else:
        raise FileNotFoundError(
            "virchow_embeddings.npy is missing even though --virchow-weights was provided; run --worker virchow first"
        )

    return _write_encoder_comparison_csv(
        output_dir / "encoder_comparison.csv",
        target_names=target_names,
        uni_scores=uni_scores,
        virchow_scores=virchow_scores,
        cnn_scores=cnn_scores,
    )


def plan_task(config: ProbeEncodersConfig, runtime: RuntimeProbe | None = None) -> TaskPlan:
    """Plan GPU-sensitive encoder jobs without executing them."""
    runtime = runtime or probe_runtime()
    out_dir = ensure_directory(config.out_dir)
    virchow_out = out_dir / "virchow_embeddings.npy"
    virchow_skip = _virchow_skip_path(out_dir)
    cnn_out = out_dir / "raw_cnn_embeddings.npy"
    comparison_out = out_dir / "encoder_comparison.csv"
    jobs: list[JobPlan] = []

    if config.skip_existing and virchow_out.is_file():
        virchow_state = JobState.SKIPPED
        virchow_reason = "existing_output"
        virchow_command = None
    elif config.virchow_weights is None or not config.virchow_weights.is_file():
        virchow_state = JobState.SKIPPED
        virchow_reason = "missing_weights"
        virchow_command = None
    elif not runtime.has_cuda:
        virchow_state = JobState.DEFERRED
        virchow_reason = "missing_gpu"
        virchow_command = _worker_command(config, "virchow")
    else:
        virchow_state = JobState.READY
        virchow_reason = None
        virchow_command = _worker_command(config, "virchow")
    jobs.append(
        JobPlan(
            job_id="cache_virchow_embeddings",
            state=virchow_state,
            reason=virchow_reason,
            inputs=(config.he_dir, config.tile_ids_path),
            outputs=(virchow_out, virchow_skip),
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
        cnn_command = _worker_command(config, "raw_cnn")
    else:
        cnn_state = JobState.READY
        cnn_reason = None
        cnn_command = _worker_command(config, "raw_cnn")
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
            command=_worker_command(config, "compare") if comparison_ready else None,
        )
    )
    return TaskPlan(task_name="a1_probe_encoders", jobs=tuple(jobs), warnings=runtime.warnings)


def main(argv: list[str] | None = None) -> int:
    """Write a plan file or execute a local worker entry point."""
    parser = argparse.ArgumentParser(description="Plan the encoder-comparison task")
    parser.add_argument("--he-dir", required=True)
    parser.add_argument("--targets-path", required=True)
    parser.add_argument("--tile-ids-path", required=True)
    parser.add_argument("--cv-splits-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--virchow-weights", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", choices=("raw_cnn", "virchow", "compare"), default=None)
    args = parser.parse_args(argv)

    config = ProbeEncodersConfig(
        he_dir=Path(args.he_dir),
        targets_path=Path(args.targets_path),
        tile_ids_path=Path(args.tile_ids_path),
        cv_splits_path=Path(args.cv_splits_path),
        out_dir=Path(args.out_dir),
        virchow_weights=Path(args.virchow_weights) if args.virchow_weights else None,
        device=args.device,
    )

    if args.worker == "raw_cnn":
        run_raw_cnn_worker(config)
        return 0
    if args.worker == "virchow":
        run_virchow_worker(config)
        return 0
    if args.worker == "compare":
        run_compare_worker(config)
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
