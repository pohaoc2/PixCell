from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "torch" not in sys.modules:
    import types

    torch_stub = types.ModuleType("torch")

    class _DummyTensor:
        pass

    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.dtype = object
    torch_stub.Tensor = _DummyTensor
    sys.modules["torch"] = torch_stub

if "diffusers" not in sys.modules:
    import types

    diffusers_stub = types.ModuleType("diffusers")

    class _DummyScheduler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_timesteps(self, *args, **kwargs) -> None:
            pass

    diffusers_stub.DDPMScheduler = _DummyScheduler
    sys.modules["diffusers"] = diffusers_stub

import tools.compute_ablation_metrics as compute_ablation_metrics_module
from tools.compute_ablation_metrics import (
    _compute_aji,
    _compute_binary_segmentation_metrics,
    _compute_pq,
    _cellvit_json_to_instance_mask,
    _empty_metrics_record,
    _merge_cosine_into_metrics,
    _resolve_metric_selection,
    run_cellvit,
)
from tools.stage3.ablation_vis_utils import default_orion_he_png_path, default_orion_uni_npy_path


def test_empty_record_has_all_keys():
    record = _empty_metrics_record()
    assert set(record.keys()) == {
        "cosine",
        "lpips",
        "aji",
        "pq",
        "dice",
        "iou",
        "accuracy",
        "style_hed",
    }
    assert all(value is None for value in record.values())


def test_merge_cosine_preserves_existing():
    existing = {
        "cell_types": {
            "cosine": None,
            "lpips": 0.3,
            "aji": None,
            "pq": None,
            "dice": None,
            "iou": None,
            "accuracy": None,
            "style_hed": None,
        }
    }
    cosine_scores = {"cell_types": 0.9946}
    result = _merge_cosine_into_metrics(existing, cosine_scores)
    assert result["cell_types"]["cosine"] == pytest.approx(0.9946)
    assert result["cell_types"]["lpips"] == pytest.approx(0.3)


def test_resolve_metric_selection_all_includes_style_hed():
    assert _resolve_metric_selection(["all"]) == [
        "cosine",
        "lpips",
        "aji",
        "pq",
        "dice",
        "iou",
        "accuracy",
        "style_hed",
    ]


def test_aji_perfect_match():
    gt = np.zeros((64, 64), dtype=np.int32)
    gt[10:20, 10:20] = 1
    pred = gt.copy()
    assert _compute_aji(gt, pred) == pytest.approx(1.0)


def test_pq_no_detections():
    gt = np.zeros((64, 64), dtype=np.int32)
    gt[10:20, 10:20] = 1
    pred = np.zeros_like(gt)
    sq, rq, pq = _compute_pq(gt, pred)
    assert sq == pytest.approx(0.0)
    assert rq == pytest.approx(0.0)
    assert pq == pytest.approx(0.0)


def test_aji_no_overlap():
    gt = np.zeros((64, 64), dtype=np.int32)
    gt[5:15, 5:15] = 1
    pred = np.zeros((64, 64), dtype=np.int32)
    pred[40:50, 40:50] = 1
    assert _compute_aji(gt, pred) == pytest.approx(0.0)


def test_binary_segmentation_metrics_perfect_match():
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[2:8, 3:9] = 1
    pred = gt.copy()

    dice, iou, accuracy = _compute_binary_segmentation_metrics(gt, pred)

    assert dice == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)
    assert accuracy == pytest.approx(1.0)


def test_binary_segmentation_metrics_partial_overlap():
    gt = np.zeros((10, 10), dtype=np.int32)
    pred = np.zeros((10, 10), dtype=np.int32)
    gt[0:4, 0:4] = 1
    pred[2:6, 2:6] = 1

    dice, iou, accuracy = _compute_binary_segmentation_metrics(gt, pred)

    assert dice == pytest.approx(0.25)
    assert iou == pytest.approx(1.0 / 7.0)
    assert accuracy == pytest.approx(76.0 / 100.0)


def test_cellvit_json_to_instance_mask(tmp_path: Path):
    json_path = tmp_path / "cells.json"
    payload = {
        "patch": "dummy.png",
        "cells": [
            {"contour": [[2, 2], [6, 2], [6, 6], [2, 6]]},
            {"contour": [[10, 10], [14, 10], [14, 14], [10, 14]]},
        ],
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    inst = _cellvit_json_to_instance_mask(json_path, image_hw=(20, 20))
    labels = sorted(v for v in np.unique(inst).tolist() if v > 0)
    assert labels == [1, 2]


def test_run_cellvit_reads_json_sidecar(tmp_path: Path):
    image_path = tmp_path / "generated_he.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
    sidecar = tmp_path / "generated_he_cellvit_instances.json"
    sidecar.write_text(
        json.dumps({"patch": image_path.name, "cells": [{"contour": [[1, 1], [4, 1], [4, 4], [1, 4]]}]}),
        encoding="utf-8",
    )
    inst = run_cellvit(image_path)
    assert int(inst.max()) == 1


def test_default_orion_paths_honor_style_mapping(tmp_path: Path):
    orion_root = tmp_path / "orion"
    (orion_root / "he").mkdir(parents=True)
    (orion_root / "features").mkdir(parents=True)
    mapped_tile = "tile_style"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(orion_root / "he" / f"{mapped_tile}.png")
    np.save(orion_root / "features" / f"{mapped_tile}_uni.npy", np.array([1.0], dtype=np.float32))

    style_mapping = {"tile_layout": mapped_tile}

    assert default_orion_he_png_path(orion_root, "tile_layout", style_mapping=style_mapping) == (
        orion_root / "he" / f"{mapped_tile}.png"
    )
    assert default_orion_uni_npy_path(orion_root, "tile_layout", style_mapping=style_mapping) == (
        orion_root / "features" / f"{mapped_tile}_uni.npy"
    )


def _write_cache_manifest(cache_parent: Path, tile_id: str) -> Path:
    cache_dir = cache_parent / tile_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps({"tile_id": tile_id, "sections": []}),
        encoding="utf-8",
    )
    return cache_dir


def test_main_parent_cache_reuses_uni_extractor_once(monkeypatch, tmp_path: Path):
    cache_parent = tmp_path / "cache"
    _write_cache_manifest(cache_parent, "tile_b")
    _write_cache_manifest(cache_parent, "tile_a")

    sentinel = object()
    seen_extractors: list[object | None] = []
    load_calls = {"count": 0}

    def fake_load_uni_extractor(uni_model: Path, device: str):
        assert isinstance(uni_model, Path)
        assert device == "cuda"
        load_calls["count"] += 1
        return sentinel

    def fake_compute_metrics_for_cache_dir(
        cache_dir: Path,
        *,
        orion_root: Path,
        style_mapping=None,
        metrics_to_compute: list[str],
        device: str,
        uni_model: Path,
        lpips_loss_fn=None,
        lpips_batch_size: int = 8,
        uni_extractor=None,
    ) -> Path:
        assert orion_root == compute_ablation_metrics_module.ROOT / "data/orion-crc33"
        assert style_mapping == {}
        assert metrics_to_compute == ["cosine"]
        assert device == "cuda"
        assert lpips_loss_fn is None
        assert lpips_batch_size == 8
        seen_extractors.append(uni_extractor)
        out_path = cache_dir / "metrics.json"
        out_path.write_text("{}", encoding="utf-8")
        return out_path

    monkeypatch.setattr(compute_ablation_metrics_module, "_load_uni_extractor", fake_load_uni_extractor)
    monkeypatch.setattr(
        compute_ablation_metrics_module,
        "compute_metrics_for_cache_dir",
        fake_compute_metrics_for_cache_dir,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_ablation_metrics.py",
            "--cache-dir",
            str(cache_parent),
            "--metrics",
            "cosine",
            "--jobs",
            "1",
        ],
    )

    compute_ablation_metrics_module.main()

    assert load_calls["count"] == 1
    assert seen_extractors == [sentinel, sentinel]


def test_main_parent_cache_parallel_dispatches_jobs(monkeypatch, tmp_path: Path):
    cache_parent = tmp_path / "cache"
    _write_cache_manifest(cache_parent, "tile_b")
    _write_cache_manifest(cache_parent, "tile_a")

    captured: dict[str, object] = {}

    def fake_run_parallel_cache_metrics(**kwargs):
        captured.update(kwargs)
        return [
            ("tile_a", cache_parent / "tile_a" / "metrics.json"),
            ("tile_b", cache_parent / "tile_b" / "metrics.json"),
        ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("main process should not preload UNI in parallel mode")

    monkeypatch.setattr(
        compute_ablation_metrics_module,
        "_run_parallel_cache_metrics",
        fake_run_parallel_cache_metrics,
    )
    monkeypatch.setattr(compute_ablation_metrics_module, "_load_uni_extractor", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_ablation_metrics.py",
            "--cache-dir",
            str(cache_parent),
            "--metrics",
            "cosine",
            "--jobs",
            "2",
        ],
    )

    compute_ablation_metrics_module.main()

    assert captured["cache_parent"] == cache_parent
    assert captured["tile_ids"] == ["tile_a", "tile_b"]
    assert captured["metrics_to_compute"] == ["cosine"]
    assert captured["device"] == "cuda"
    assert captured["requested_jobs"] == 2
