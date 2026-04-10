#!/usr/bin/env python3
"""Compute unified per-condition metrics for one ablation cache directory."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    condition_metric_key,
    default_orion_he_png_path,
    parse_uni_cosine_scores_json,
)
from tools.stage3.ablation_cache import (
    is_per_tile_cache_manifest_dir,
    list_cached_tile_ids,
    load_manifest,
)
from tools.stage3.common import print_progress
from tools.stage3.style_mapping import load_style_mapping

DEFAULT_METRIC_NAMES: tuple[str, ...] = (
    "cosine",
    "lpips",
    "aji",
    "pq",
    "dice",
    "iou",
    "accuracy",
)
OPTIONAL_METRIC_NAMES: tuple[str, ...] = ("style_hed",)
METRIC_NAMES: tuple[str, ...] = DEFAULT_METRIC_NAMES + OPTIONAL_METRIC_NAMES
_SPATIAL_EXTS: tuple[str, ...] = (".png", ".npy", ".jpg", ".jpeg", ".tif", ".tiff")
_METRIC_WORKER_CONFIG: dict[str, Any] | None = None
_METRIC_WORKER_STATE: dict[str, Any] = {}
_RGB_FROM_HED = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float64,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def _empty_metrics_record() -> dict[str, float | None]:
    return {name: None for name in METRIC_NAMES}


def _coerce_metric_value(value: object) -> float | None:
    if value is None:
        return None
    fv = float(value)
    if math.isnan(fv):
        return None
    return fv


def _coerce_metrics_record(raw: object) -> dict[str, float | None]:
    record = _empty_metrics_record()
    if isinstance(raw, dict):
        for metric_name in METRIC_NAMES:
            record[metric_name] = _coerce_metric_value(raw.get(metric_name))
    return record


def _merge_cosine_into_metrics(
    existing: dict[str, dict],
    cosine_scores: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    merged = {
        str(key): _coerce_metrics_record(value)
        for key, value in existing.items()
    }
    for cond_key, score in cosine_scores.items():
        rec = merged.setdefault(str(cond_key), _empty_metrics_record())
        rec["cosine"] = _coerce_metric_value(score)
    return merged


def _tile_id_from_manifest(cache_dir: Path) -> str:
    manifest = load_manifest(cache_dir)
    tile_id = str(manifest.get("tile_id", "")).strip()
    if not tile_id:
        raise ValueError(f"manifest.json must contain tile_id: {cache_dir / 'manifest.json'}")
    return tile_id


def _iter_condition_images(cache_dir: Path) -> dict[str, Path]:
    manifest = load_manifest(cache_dir)
    per_condition: dict[str, Path] = {}
    for section in manifest.get("sections", []):
        for entry in section.get("entries", []):
            key = condition_metric_key(tuple(entry["active_groups"]))
            per_condition[key] = cache_dir / entry["image_path"]

    all4_path = cache_dir / "all" / "generated_he.png"
    if all4_path.is_file():
        per_condition[condition_metric_key(FOUR_GROUP_ORDER)] = all4_path
    return per_condition


def load_or_build_metrics(cache_dir: Path) -> dict[str, dict[str, float | None]]:
    """Load ``metrics.json`` when present; otherwise migrate ``uni_cosine_scores.json`` in-memory."""
    cache_dir = Path(cache_dir)
    metrics_path = cache_dir / "metrics.json"
    if metrics_path.is_file():
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_condition = raw.get("per_condition", {})
        if isinstance(per_condition, dict):
            return {
                str(key): _coerce_metrics_record(value)
                for key, value in per_condition.items()
            }

    cosine_scores, _ = parse_uni_cosine_scores_json(cache_dir)
    return _merge_cosine_into_metrics({}, cosine_scores)


def write_metrics(cache_dir: Path, per_condition: dict[str, dict]) -> Path:
    """Write ``<cache_dir>/metrics.json`` and return its path."""
    cache_dir = Path(cache_dir)
    tile_id = _tile_id_from_manifest(cache_dir)
    normalized = {
        key: _coerce_metrics_record(per_condition[key])
        for key in sorted(per_condition)
    }
    payload = {
        "version": 2,
        "tile_id": tile_id,
        "per_condition": normalized,
    }
    out_path = cache_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def _resolve_metric_selection(raw_metrics: list[str]) -> list[str]:
    requested = [metric.lower() for metric in raw_metrics]
    if "all" in requested:
        return list(METRIC_NAMES)
    invalid = [metric for metric in requested if metric not in METRIC_NAMES]
    if invalid:
        raise ValueError(f"unsupported metrics: {invalid}")
    return requested


def _ensure_cosine_scores(
    cache_dir: Path,
    orion_root: Path,
    *,
    style_mapping: dict[str, str] | None,
    uni_model: Path,
    device: str,
    uni_extractor: Any | None = None,
) -> dict[str, float]:
    scores, _ = parse_uni_cosine_scores_json(cache_dir)
    if scores:
        return scores

    from tools.stage3.compute_ablation_uni_cosine import compute_and_write_uni_cosine_scores

    _, _, scores = compute_and_write_uni_cosine_scores(
        cache_dir,
        orion_root=orion_root,
        style_mapping=style_mapping,
        uni_model=uni_model,
        device=device,
        extractor=uni_extractor,
    )
    return scores


def _load_uni_extractor(uni_model: Path, device: str):
    from tools.stage3.compute_ablation_uni_cosine import load_uni_extractor

    return load_uni_extractor(uni_model=uni_model, device=device)


def _load_rgb_pil(path: Path, *, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return image


def _rgb_to_hed(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64) / 255.0
    arr = np.clip(arr, 1e-6, 1.0)
    optical_density = -np.log(arr)
    return optical_density @ _HED_FROM_RGB.T


def _tissue_mask_from_rgb(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.mean(arr, axis=2) < 0.95


def _masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    data = values[np.asarray(mask, dtype=bool)]
    if data.size == 0:
        data = values.reshape(-1)
    return float(np.mean(data)), float(np.std(data))


def _resolve_lpips_device(requested: str) -> str:
    resolved = str(requested).lower()
    try:
        import torch
    except Exception:
        return "cpu" if resolved == "cuda" else resolved

    if resolved == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return resolved


def _load_lpips_model(device: str):
    try:
        import lpips
    except Exception as exc:
        raise RuntimeError(
            "LPIPS scoring requires the optional 'lpips' and 'torch' packages."
        ) from exc

    resolved_device = _resolve_lpips_device(device)
    loss_fn = lpips.LPIPS(net="alex").to(resolved_device)
    loss_fn.eval()
    return loss_fn, resolved_device


def _init_metrics_worker(config: dict[str, Any]) -> None:
    global _METRIC_WORKER_CONFIG, _METRIC_WORKER_STATE

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _METRIC_WORKER_CONFIG = dict(config)
    _METRIC_WORKER_STATE = {}


def _get_worker_uni_extractor():
    if _METRIC_WORKER_CONFIG is None:
        raise RuntimeError("metric worker not initialized")
    extractor = _METRIC_WORKER_STATE.get("uni_extractor")
    if extractor is None:
        extractor = _load_uni_extractor(
            Path(_METRIC_WORKER_CONFIG["uni_model"]),
            str(_METRIC_WORKER_CONFIG["device"]),
        )
        _METRIC_WORKER_STATE["uni_extractor"] = extractor
    return extractor


def _get_worker_lpips_model():
    if _METRIC_WORKER_CONFIG is None:
        raise RuntimeError("metric worker not initialized")
    loss_fn = _METRIC_WORKER_STATE.get("lpips_loss_fn")
    if loss_fn is None:
        loss_fn, _ = _load_lpips_model(str(_METRIC_WORKER_CONFIG["device"]))
        _METRIC_WORKER_STATE["lpips_loss_fn"] = loss_fn
    return loss_fn


def _compute_metrics_for_cache_dir_job(cache_dir: str) -> tuple[str, str]:
    if _METRIC_WORKER_CONFIG is None:
        raise RuntimeError("metric worker not initialized")

    cache_path = Path(cache_dir)
    metrics_to_compute = list(_METRIC_WORKER_CONFIG["metrics_to_compute"])
    out_path = compute_metrics_for_cache_dir(
        cache_path,
        orion_root=Path(_METRIC_WORKER_CONFIG["orion_root"]),
        style_mapping=load_style_mapping(_METRIC_WORKER_CONFIG.get("style_mapping_json")),
        metrics_to_compute=metrics_to_compute,
        device=str(_METRIC_WORKER_CONFIG["device"]),
        uni_model=Path(_METRIC_WORKER_CONFIG["uni_model"]),
        lpips_loss_fn=_get_worker_lpips_model() if "lpips" in metrics_to_compute else None,
        lpips_batch_size=int(_METRIC_WORKER_CONFIG["lpips_batch_size"]),
        uni_extractor=_get_worker_uni_extractor() if "cosine" in metrics_to_compute else None,
    )
    return cache_path.name, str(out_path)


def _print_progress(completed: int, total: int, *, prefix: str) -> None:
    print_progress(completed, total, prefix=prefix)


def _run_parallel_cache_metrics(
    *,
    cache_parent: Path,
    tile_ids: list[str],
    orion_root: Path,
    style_mapping_json: Path | None,
    metrics_to_compute: list[str],
    device: str,
    uni_model: Path,
    lpips_batch_size: int,
    requested_jobs: int,
) -> list[tuple[str, Path]]:
    worker_count = min(
        max(1, int(requested_jobs)),
        len(tile_ids),
        os.cpu_count() or max(1, int(requested_jobs)),
    )
    print(f"Running {len(tile_ids)} metrics job(s) with {worker_count} worker process(es)")

    worker_config = {
        "orion_root": str(orion_root),
        "style_mapping_json": None if style_mapping_json is None else str(style_mapping_json),
        "metrics_to_compute": list(metrics_to_compute),
        "device": device,
        "uni_model": str(uni_model),
        "lpips_batch_size": int(lpips_batch_size),
    }

    results: list[tuple[str, Path]] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_metrics_worker,
        initargs=(worker_config,),
    ) as executor:
        futures = [
            executor.submit(_compute_metrics_for_cache_dir_job, str(cache_parent / tile_id))
            for tile_id in tile_ids
        ]
        total = len(futures)
        completed = 0
        _print_progress(0, total, prefix="Metrics")
        for future in as_completed(futures):
            tile_id, out_path_str = future.result()
            out_path = Path(out_path_str)
            results.append((tile_id, out_path))
            completed += 1
            _print_progress(completed, total, prefix="Metrics")
            print(f"[{completed}/{total}] Wrote metrics → {out_path}")
    return results


def _image_to_lpips_tensor(image: Image.Image, *, device: str):
    import torch

    arr = np.asarray(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def compute_lpips_scores(
    cache_dir: Path,
    orion_root: Path,
    *,
    style_mapping: dict[str, str] | None = None,
    device: str = "cuda",
    loss_fn=None,
    batch_size: int = 8,
) -> dict[str, float]:
    """Run LPIPS(AlexNet) between the paired real H&E and each generated H&E.

    Generated images are scored in batches for better single-GPU throughput.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "LPIPS scoring requires the optional 'lpips' and 'torch' packages."
        ) from exc

    cache_dir = Path(cache_dir)
    orion_root = Path(orion_root)
    tile_id = _tile_id_from_manifest(cache_dir)
    ref_path = default_orion_he_png_path(orion_root, tile_id, style_mapping=style_mapping)
    if ref_path is None:
        raise FileNotFoundError(f"reference H&E not found for tile {tile_id!r}")

    if loss_fn is None:
        loss_fn, resolved_device = _load_lpips_model(device)
    else:
        resolved_device = _resolve_lpips_device(device)
    batch_size = max(1, int(batch_size))

    ref_img = _load_rgb_pil(ref_path)
    ref_size = ref_img.size
    ref_tensor = _image_to_lpips_tensor(ref_img, device=resolved_device)
    condition_images = list(_iter_condition_images(cache_dir).items())
    scores: dict[str, float] = {}
    with torch.no_grad():
        for start_idx in range(0, len(condition_images), batch_size):
            chunk = condition_images[start_idx:start_idx + batch_size]
            gen_tensors = []
            cond_keys: list[str] = []
            for cond_key, img_path in chunk:
                if not img_path.is_file():
                    raise FileNotFoundError(f"generated image not found: {img_path}")
                gen_tensors.append(
                    _image_to_lpips_tensor(_load_rgb_pil(img_path, size=ref_size), device=resolved_device)
                )
                cond_keys.append(cond_key)

            gen_batch = torch.cat(gen_tensors, dim=0)
            ref_batch = ref_tensor.expand(gen_batch.shape[0], -1, -1, -1)
            chunk_scores = loss_fn(ref_batch, gen_batch).reshape(-1)
            for cond_key, score in zip(cond_keys, chunk_scores.tolist(), strict=True):
                scores[cond_key] = float(score)
    return scores


def compute_style_hed_scores(
    cache_dir: Path,
    orion_root: Path,
    *,
    style_mapping: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compare generated H&E stain/style moments against the reference H&E in HED space.

    Lower is better. This is intended for style similarity, especially in unpaired runs
    where the dataset root's ``he/<tile_id>`` has been remapped to the style tile.
    """
    cache_dir = Path(cache_dir)
    orion_root = Path(orion_root)
    tile_id = _tile_id_from_manifest(cache_dir)
    ref_path = default_orion_he_png_path(orion_root, tile_id, style_mapping=style_mapping)
    if ref_path is None:
        raise FileNotFoundError(f"reference H&E not found for tile {tile_id!r}")

    ref_img = _load_rgb_pil(ref_path)
    ref_hed = _rgb_to_hed(ref_img)
    ref_mask = _tissue_mask_from_rgb(ref_img)
    scores: dict[str, float] = {}

    for cond_key, img_path in _iter_condition_images(cache_dir).items():
        if not img_path.is_file():
            raise FileNotFoundError(f"generated image not found: {img_path}")
        gen_img = _load_rgb_pil(img_path, size=ref_img.size)
        gen_hed = _rgb_to_hed(gen_img)
        gen_mask = _tissue_mask_from_rgb(gen_img)
        tissue_mask = ref_mask | gen_mask

        score = 0.0
        for stain_channel in (0, 1):  # H and E only
            ref_mean, ref_std = _masked_mean_std(ref_hed[..., stain_channel], tissue_mask)
            gen_mean, gen_std = _masked_mean_std(gen_hed[..., stain_channel], tissue_mask)
            score += abs(gen_mean - ref_mean) + abs(gen_std - ref_std)
        scores[cond_key] = float(score)
    return scores


def _find_spatial_path(base_dir: Path, tile_id: str) -> Path:
    for ext in _SPATIAL_EXTS:
        candidate = base_dir / f"{tile_id}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no spatial file for tile {tile_id!r} in {base_dir}")


def _load_spatial_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.asarray(Image.open(path))
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D spatial array from {path}, got {arr.shape}")
    return np.asarray(arr)


def _array_max(arr: np.ndarray) -> float:
    return float(np.max(arr)) if arr.size else 0.0


def _resize_nearest(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(arr)
    resized = image.resize((out_hw[1], out_hw[0]), Image.NEAREST)
    return np.asarray(resized)


def _label_binary_instances(binary_mask: np.ndarray) -> np.ndarray:
    binary_mask = np.asarray(binary_mask, dtype=bool)
    try:
        from scipy import ndimage as ndi

        labeled, _ = ndi.label(binary_mask, structure=np.ones((3, 3), dtype=np.uint8))
        return labeled.astype(np.int32)
    except Exception:
        pass

    h, w = binary_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x] or labels[y, x] != 0:
                continue
            stack = [(y, x)]
            labels[y, x] = next_label
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not binary_mask[ny, nx] or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = next_label
                    stack.append((ny, nx))
            next_label += 1
    return labels


def _resolve_precomputed_cellvit_mask(image_path: Path) -> Path | None:
    candidate = image_path.with_name(f"{image_path.stem}_cellvit_instances.json")
    return candidate if candidate.is_file() else None


def _cellvit_json_to_instance_mask(json_path: Path, *, image_hw: tuple[int, int]) -> np.ndarray:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    height, width = image_hw
    inst = np.zeros((height, width), dtype=np.int32)

    for idx, cell in enumerate(cells, start=1):
        contour = cell.get("contour")
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        polygon: list[tuple[float, float]] = []
        for point in contour:
            if not isinstance(point, list | tuple) or len(point) < 2:
                continue
            x = float(point[0])
            y = float(point[1])
            polygon.append((x, y))
        if len(polygon) < 3:
            continue

        mask_img = Image.new("I", (width, height), 0)
        ImageDraw.Draw(mask_img).polygon(polygon, outline=idx, fill=idx)
        mask_arr = np.asarray(mask_img, dtype=np.int32)
        inst[mask_arr > 0] = idx
    return inst


def run_cellvit(image_path: Path) -> np.ndarray:
    """Load a labeled instance mask from an imported CellViT JSON sidecar."""
    image_path = Path(image_path)

    precomputed = _resolve_precomputed_cellvit_mask(image_path)
    if precomputed is not None:
        image = Image.open(image_path)
        return _cellvit_json_to_instance_mask(
            precomputed,
            image_hw=(image.height, image.width),
        )
    raise RuntimeError(
        "CellViT JSON sidecar not found next to the generated image. Expected "
        f"{image_path.with_name(f'{image_path.stem}_cellvit_instances.json')}"
    )


def _instance_ids(inst_mask: np.ndarray) -> np.ndarray:
    ids = np.unique(np.asarray(inst_mask, dtype=np.int32))
    return ids[ids > 0]


def _intersection_and_iou_matrices(
    gt_inst: np.ndarray,
    pred_inst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_ids = _instance_ids(gt_inst)
    pred_ids = _instance_ids(pred_inst)
    intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    gt_areas = np.array([int(np.count_nonzero(gt_inst == gid)) for gid in gt_ids], dtype=np.int64)
    pred_areas = np.array([int(np.count_nonzero(pred_inst == pid)) for pid in pred_ids], dtype=np.int64)

    if gt_ids.size and pred_ids.size:
        gt_lookup = {int(gid): idx for idx, gid in enumerate(gt_ids)}
        pred_lookup = {int(pid): idx for idx, pid in enumerate(pred_ids)}
        overlap = (gt_inst > 0) & (pred_inst > 0)
        if np.any(overlap):
            pairs = np.stack([gt_inst[overlap], pred_inst[overlap]], axis=1)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            for (gid, pid), count in zip(unique_pairs, counts):
                intersections[gt_lookup[int(gid)], pred_lookup[int(pid)]] = int(count)

    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.zeros_like(intersections, dtype=np.float64)
    valid = unions > 0
    iou[valid] = intersections[valid] / unions[valid]
    return gt_ids, pred_ids, intersections, iou


def _linear_sum_assignment_maximize(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    try:
        from scipy.optimize import linear_sum_assignment

        try:
            return linear_sum_assignment(scores, maximize=True)
        except TypeError:
            return linear_sum_assignment(-scores)
    except Exception:
        rows, cols = scores.shape
        if min(rows, cols) > 8:
            raise RuntimeError("scipy is required for Hungarian matching on larger masks")
        if rows <= cols:
            best_perm: tuple[int, ...] | None = None
            best_score = -float("inf")
            for perm in permutations(range(cols), rows):
                total = float(scores[np.arange(rows), perm].sum())
                if total > best_score:
                    best_score = total
                    best_perm = perm
            assert best_perm is not None
            return np.arange(rows, dtype=np.int64), np.array(best_perm, dtype=np.int64)

        best_rows: tuple[int, ...] | None = None
        best_score = -float("inf")
        for perm in permutations(range(rows), cols):
            total = float(scores[perm, np.arange(cols)].sum())
            if total > best_score:
                best_score = total
                best_rows = perm
        assert best_rows is not None
        return np.array(best_rows, dtype=np.int64), np.arange(cols, dtype=np.int64)


def _compute_aji(gt_inst: np.ndarray, pred_inst: np.ndarray) -> float:
    """Aggregated Jaccard Index between ground-truth and predicted instances."""
    gt_inst = np.asarray(gt_inst, dtype=np.int32)
    pred_inst = np.asarray(pred_inst, dtype=np.int32)
    gt_ids, pred_ids, intersections, iou = _intersection_and_iou_matrices(gt_inst, pred_inst)
    gt_areas = np.array([int(np.count_nonzero(gt_inst == gid)) for gid in gt_ids], dtype=np.int64)
    pred_areas = np.array([int(np.count_nonzero(pred_inst == pid)) for pid in pred_ids], dtype=np.int64)

    if gt_ids.size == 0 and pred_ids.size == 0:
        return 1.0
    if gt_ids.size == 0 or pred_ids.size == 0:
        return 0.0

    row_idx, col_idx = _linear_sum_assignment_maximize(iou)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    intersection_sum = 0
    union_sum = 0
    for r, c in zip(row_idx, col_idx):
        if iou[r, c] <= 0.0:
            continue
        matched_gt.add(int(r))
        matched_pred.add(int(c))
        inter = int(intersections[r, c])
        union = int(gt_areas[r] + pred_areas[c] - inter)
        intersection_sum += inter
        union_sum += union

    for idx, area in enumerate(gt_areas):
        if idx not in matched_gt:
            union_sum += int(area)
    for idx, area in enumerate(pred_areas):
        if idx not in matched_pred:
            union_sum += int(area)

    if union_sum == 0:
        return 0.0
    return float(intersection_sum / union_sum)


def _compute_pq(gt_inst: np.ndarray, pred_inst: np.ndarray) -> tuple[float, float, float]:
    """Return ``(SQ, RQ, PQ)`` using Hungarian matching at IoU >= 0.5."""
    gt_inst = np.asarray(gt_inst, dtype=np.int32)
    pred_inst = np.asarray(pred_inst, dtype=np.int32)
    gt_ids, pred_ids, _, iou = _intersection_and_iou_matrices(gt_inst, pred_inst)

    n_gt = int(gt_ids.size)
    n_pred = int(pred_ids.size)
    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    row_idx, col_idx = _linear_sum_assignment_maximize(iou)
    matched_ious = [float(iou[r, c]) for r, c in zip(row_idx, col_idx) if iou[r, c] >= 0.5]
    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    sq = float(sum(matched_ious) / tp) if tp else 0.0
    rq_denom = tp + 0.5 * fp + 0.5 * fn
    rq = float(tp / rq_denom) if rq_denom > 0 else 0.0
    pq = float(sq * rq)
    return sq, rq, pq


def _compute_binary_segmentation_metrics(
    gt_inst: np.ndarray,
    pred_inst: np.ndarray,
) -> tuple[float, float, float]:
    """Return pixel-level ``(dice, iou, accuracy)`` on foreground-vs-background masks."""
    gt_fg = np.asarray(gt_inst) > 0
    pred_fg = np.asarray(pred_inst) > 0

    tp = int(np.count_nonzero(gt_fg & pred_fg))
    tn = int(np.count_nonzero(~gt_fg & ~pred_fg))
    fp = int(np.count_nonzero(~gt_fg & pred_fg))
    fn = int(np.count_nonzero(gt_fg & ~pred_fg))

    dice_denom = 2 * tp + fp + fn
    dice = float((2 * tp) / dice_denom) if dice_denom > 0 else 1.0

    iou_denom = tp + fp + fn
    iou = float(tp / iou_denom) if iou_denom > 0 else 1.0

    total = tp + tn + fp + fn
    accuracy = float((tp + tn) / total) if total > 0 else 1.0
    return dice, iou, accuracy


def _load_gt_instance_mask(orion_root: Path, tile_id: str, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    exp_dir = Path(orion_root) / "exp_channels"
    for folder_name in ("cell_masks", "cell_mask"):
        channel_dir = exp_dir / folder_name
        if not channel_dir.is_dir():
            continue
        path = _find_spatial_path(channel_dir, tile_id)
        arr = _load_spatial_array(path)
        if shape is not None and arr.shape != shape:
            arr = _resize_nearest(arr.astype(np.uint8), shape)
        return _label_binary_instances(arr > (0.5 if _array_max(arr) <= 1.0 else 0))
    raise FileNotFoundError(
        f"cell_masks channel not found under {exp_dir} for tile {tile_id!r}"
    )


def compute_cell_metrics(
    cache_dir: Path,
    orion_root: Path,
) -> dict[str, dict[str, float]]:
    """Compute cell-mask metrics from CellViT sidecars vs input ``cell_masks``."""
    cache_dir = Path(cache_dir)
    tile_id = _tile_id_from_manifest(cache_dir)
    condition_images = _iter_condition_images(cache_dir)
    per_condition: dict[str, dict[str, float]] = {}

    gt_inst: np.ndarray | None = None
    for cond_key, image_path in condition_images.items():
        pred_inst = run_cellvit(image_path)
        if gt_inst is None:
            gt_inst = _load_gt_instance_mask(orion_root, tile_id, shape=pred_inst.shape)
        elif gt_inst.shape != pred_inst.shape:
            raise ValueError(
                f"GT cell mask shape {gt_inst.shape} does not match prediction {pred_inst.shape} for {image_path}"
            )
        _, _, pq = _compute_pq(gt_inst, pred_inst)
        dice, iou, accuracy = _compute_binary_segmentation_metrics(gt_inst, pred_inst)
        per_condition[cond_key] = {
            "aji": _compute_aji(gt_inst, pred_inst),
            "pq": pq,
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
        }
    return per_condition


def compute_metrics_for_cache_dir(
    cache_dir: Path,
    *,
    orion_root: Path,
    style_mapping: dict[str, str] | None = None,
    metrics_to_compute: list[str],
    device: str,
    uni_model: Path,
    lpips_loss_fn=None,
    lpips_batch_size: int = 8,
    uni_extractor: Any | None = None,
) -> Path:
    """Compute selected metrics for one tile cache and write ``metrics.json``."""
    per_condition = load_or_build_metrics(cache_dir)

    if "cosine" in metrics_to_compute:
        cosine_scores = _ensure_cosine_scores(
            cache_dir,
            orion_root,
            style_mapping=style_mapping,
            uni_model=uni_model,
            device=device,
            uni_extractor=uni_extractor,
        )
        per_condition = _merge_cosine_into_metrics(per_condition, cosine_scores)

    if "lpips" in metrics_to_compute:
        for cond_key, score in compute_lpips_scores(
            cache_dir,
            orion_root,
            style_mapping=style_mapping,
            device=device,
            loss_fn=lpips_loss_fn,
            batch_size=lpips_batch_size,
        ).items():
            rec = per_condition.setdefault(cond_key, _empty_metrics_record())
            rec["lpips"] = score

    if any(metric in metrics_to_compute for metric in ("aji", "pq", "dice", "iou", "accuracy")):
        cell_scores = compute_cell_metrics(cache_dir, orion_root)
        for cond_key, values in cell_scores.items():
            rec = per_condition.setdefault(cond_key, _empty_metrics_record())
            if "aji" in metrics_to_compute:
                rec["aji"] = values["aji"]
            if "pq" in metrics_to_compute:
                rec["pq"] = values["pq"]
            if "dice" in metrics_to_compute:
                rec["dice"] = values["dice"]
            if "iou" in metrics_to_compute:
                rec["iou"] = values["iou"]
            if "accuracy" in metrics_to_compute:
                rec["accuracy"] = values["accuracy"]

    if "style_hed" in metrics_to_compute:
        for cond_key, score in compute_style_hed_scores(
            cache_dir,
            orion_root,
            style_mapping=style_mapping,
        ).items():
            rec = per_condition.setdefault(cond_key, _empty_metrics_record())
            rec["style_hed"] = score

    return write_metrics(cache_dir, per_condition)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute unified ablation metrics and write metrics.json.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--orion-root",
        type=Path,
        default=ROOT / "data/orion-crc33",
        help="Paired dataset root (default: data/orion-crc33)",
    )
    parser.add_argument(
        "--style-mapping-json",
        type=Path,
        default=None,
        help="Optional layout->style mapping JSON for unpaired style reference lookup.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help=(
            "Metrics to compute: cosine lpips aji pq dice iou accuracy style_hed all "
            "(default: all; includes style_hed)"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--lpips-batch-size",
        type=int,
        default=8,
        help="Number of generated images per LPIPS forward pass (default: 8).",
    )
    parser.add_argument(
        "--uni-model",
        type=Path,
        default=ROOT / "pretrained_models/uni-2h",
        help="UNI-2h model path for cosine fallback",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes for parent cache directories (default: 1).",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    cache_dir = args.cache_dir.resolve()
    orion_root = args.orion_root.resolve()
    style_mapping = load_style_mapping(args.style_mapping_json)
    metrics_to_compute = _resolve_metric_selection(args.metrics)

    if is_per_tile_cache_manifest_dir(cache_dir):
        lpips_loss_fn = None
        if "lpips" in metrics_to_compute:
            lpips_loss_fn, _ = _load_lpips_model(args.device)
        out_path = compute_metrics_for_cache_dir(
            cache_dir,
            orion_root=orion_root,
            style_mapping=style_mapping,
            metrics_to_compute=metrics_to_compute,
            device=args.device,
            uni_model=args.uni_model,
            lpips_loss_fn=lpips_loss_fn,
            lpips_batch_size=args.lpips_batch_size,
        )
        print(f"Wrote metrics → {out_path}")
        return

    tile_ids = list_cached_tile_ids(cache_dir)
    if not tile_ids:
        raise SystemExit(
            f"no per-tile caches under {cache_dir} (expected subdirs like <tile_id>/manifest.json)"
        )

    if args.jobs > 1:
        if str(args.device).lower() == "cuda" and any(
            metric in metrics_to_compute for metric in ("cosine", "lpips")
        ):
            print(
                "Note: parallel metric workers on CUDA each load their own model state and may "
                "contend for GPU memory; reduce --jobs or use --device cpu if needed.",
                file=sys.stderr,
            )
        _run_parallel_cache_metrics(
            cache_parent=cache_dir,
            tile_ids=tile_ids,
            orion_root=orion_root,
            style_mapping_json=args.style_mapping_json,
            metrics_to_compute=metrics_to_compute,
            device=args.device,
            uni_model=args.uni_model,
            lpips_batch_size=args.lpips_batch_size,
            requested_jobs=args.jobs,
        )
        return

    lpips_loss_fn = None
    if "lpips" in metrics_to_compute:
        lpips_loss_fn, _ = _load_lpips_model(args.device)
    shared_uni_extractor = None
    if "cosine" in metrics_to_compute:
        shared_uni_extractor = _load_uni_extractor(args.uni_model, args.device)

    for idx, tile_id in enumerate(tile_ids, start=1):
        tile_cache_dir = cache_dir / tile_id
        out_path = compute_metrics_for_cache_dir(
            tile_cache_dir,
            orion_root=orion_root,
            style_mapping=style_mapping,
            metrics_to_compute=metrics_to_compute,
            device=args.device,
            uni_model=args.uni_model,
            lpips_loss_fn=lpips_loss_fn,
            lpips_batch_size=args.lpips_batch_size,
            uni_extractor=shared_uni_extractor,
        )
        print(f"[{idx}/{len(tile_ids)}] Wrote metrics → {out_path}")


if __name__ == "__main__":
    main()
