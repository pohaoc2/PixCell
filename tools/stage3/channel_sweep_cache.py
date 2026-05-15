"""Cache helpers for stage 3 channel sweep experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.stage3.common import load_json, to_uint8_rgb


CACHE_VERSION = 1


def source_labels_from_results(results: dict[str, dict[str, np.ndarray]]) -> list[str]:
    labels = list(results.keys())
    if labels:
        return labels
    return []


def target_labels_from_results(results: dict[str, dict[str, np.ndarray]]) -> list[str]:
    for row in results.values():
        labels = list(row.keys())
        if labels:
            return labels
    return []


def save_rgb_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_rgb(image, value_range="byte")).save(path)


def load_rgb_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def exp1_condition_slug(o2_scale: float, glucose_scale: float) -> str:
    return f"o2_{o2_scale:.2f}__glucose_{glucose_scale:.2f}"


def save_channel_sweep_manifest(cache_dir: Path, manifest: dict[str, Any]) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_channel_sweep_manifest(cache_dir: Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    path = cache_dir / "manifest.json"
    return load_json(path)


def save_channel_sweep_cache(
    *,
    cache_dir: Path,
    manifest: dict[str, Any] | None = None,
    tile_id: str | None = None,
    results: dict[str, Any] | None = None,
    images: dict[str, Any] | None = None,
    out_dir: Path | None = None,
) -> Path:
    del out_dir
    cache_dir = Path(cache_dir)
    payload = dict(manifest or {})
    payload.setdefault("version", CACHE_VERSION)
    if tile_id is not None:
        payload["tile_id"] = tile_id
    if "experiments" not in payload:
        payload["experiments"] = {}
        tree = results or images or {}
        for exp_name, exp_tree in tree.items():
            entries: list[dict[str, Any]] = []
            if isinstance(exp_tree, dict):
                for src_label, row in exp_tree.items():
                    if not isinstance(row, dict):
                        continue
                    for tgt_label, image in row.items():
                        rel = Path(str(exp_name)) / f"{src_label}__{tgt_label}.png"
                        save_rgb_png(np.asarray(image), cache_dir / rel)
                        entries.append(
                            {
                                "source_label": str(src_label),
                                "target_label": str(tgt_label),
                                "image_path": rel.as_posix(),
                            }
                        )
            payload["experiments"][str(exp_name)] = {"entries": entries}
    return save_channel_sweep_manifest(cache_dir, payload)


def load_channel_sweep_cache(*, cache_dir: Path, cache_path: Path | None = None) -> dict[str, Any]:
    return load_channel_sweep_manifest(cache_path or cache_dir)


def save_exp1_microenv_cache(
    *,
    cache_dir: Path,
    tile_id: str,
    tile_class_label: str,
    images_grid: dict[tuple[float, float], np.ndarray],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for (o2_scale, glucose_scale), image in sorted(images_grid.items()):
        rel = (
            Path("exp1_microenv")
            / f"{tile_class_label}_{tile_id}"
            / f"{exp1_condition_slug(o2_scale, glucose_scale)}.png"
        )
        save_rgb_png(image, Path(cache_dir) / rel)
        items.append(
            {
                "o2_scale": float(o2_scale),
                "glucose_scale": float(glucose_scale),
                "image_path": rel.as_posix(),
            }
        )
    return {
        "tile_id": tile_id,
        "tile_class_label": tile_class_label,
        "baseline": {"o2_scale": 1.0, "glucose_scale": 1.0},
        "items": items,
    }


def load_exp1_microenv_cache(
    cache_dir: Path,
    record: dict[str, Any],
) -> dict[tuple[float, float], np.ndarray]:
    grid: dict[tuple[float, float], np.ndarray] = {}
    for item in record.get("items", []):
        key = (float(item["o2_scale"]), float(item["glucose_scale"]))
        grid[key] = load_rgb_png(Path(cache_dir) / item["image_path"])
    return grid


def save_relabeling_cache(
    *,
    cache_dir: Path,
    exp_name: str,
    results: dict[str, dict[str, np.ndarray]],
    tiles: dict[str, str],
    baseline_group_thumbs: dict[str, np.ndarray],
    input_thumbs: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_labels = list(results.keys())
    target_labels = target_labels_from_results(results)
    for src_label in source_labels:
        images: dict[str, str] = {}
        input_thumb_paths: dict[str, str] = {}
        for tgt_label, image in results[src_label].items():
            rel = Path(exp_name) / src_label / f"{tgt_label}.png"
            save_rgb_png(image, Path(cache_dir) / rel)
            images[tgt_label] = rel.as_posix()
            thumb = input_thumbs.get(src_label, {}).get(tgt_label)
            if thumb is not None:
                input_rel = Path(exp_name) / src_label / f"input__{tgt_label}.png"
                save_rgb_png(thumb, Path(cache_dir) / input_rel)
                input_thumb_paths[tgt_label] = input_rel.as_posix()
        thumb_rel = Path(exp_name) / src_label / "baseline_group.png"
        save_rgb_png(baseline_group_thumbs[src_label], Path(cache_dir) / thumb_rel)
        rows.append(
            {
                "source_label": src_label,
                "tile_id": tiles[src_label],
                "baseline_group_thumb_path": thumb_rel.as_posix(),
                "input_thumb_paths": input_thumb_paths,
                "images": images,
            }
        )
    return {
        "labels": source_labels,
        "source_labels": source_labels,
        "target_labels": target_labels,
        "rows": rows,
    }


def load_relabeling_cache(
    cache_dir: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    labels = [str(label) for label in record.get("labels", [])]
    source_labels = [str(label) for label in record.get("source_labels", labels)]
    target_labels = [str(label) for label in record.get("target_labels", labels)]
    results: dict[str, dict[str, np.ndarray]] = {}
    tiles: dict[str, str] = {}
    baseline_group_thumbs: dict[str, np.ndarray] = {}
    input_thumbs: dict[str, dict[str, np.ndarray]] = {}
    for row in record.get("rows", []):
        src_label = str(row["source_label"])
        tiles[src_label] = str(row["tile_id"])
        baseline_group_thumbs[src_label] = load_rgb_png(
            Path(cache_dir) / row["baseline_group_thumb_path"]
        )
        input_thumbs[src_label] = {
            str(tgt_label): load_rgb_png(Path(cache_dir) / rel_path)
            for tgt_label, rel_path in row.get("input_thumb_paths", {}).items()
        }
        results[src_label] = {
            str(tgt_label): load_rgb_png(Path(cache_dir) / rel_path)
            for tgt_label, rel_path in row.get("images", {}).items()
        }
    return {
        "labels": labels,
        "source_labels": source_labels,
        "target_labels": target_labels,
        "tiles": tiles,
        "baseline_group_thumbs": baseline_group_thumbs,
        "input_thumbs": input_thumbs,
        "results": results,
    }


__all__ = [
    "CACHE_VERSION",
    "exp1_condition_slug",
    "load_channel_sweep_cache",
    "load_channel_sweep_manifest",
    "load_exp1_microenv_cache",
    "load_relabeling_cache",
    "load_rgb_png",
    "save_channel_sweep_cache",
    "save_channel_sweep_manifest",
    "save_exp1_microenv_cache",
    "save_relabeling_cache",
    "save_rgb_png",
    "source_labels_from_results",
    "target_labels_from_results",
]
