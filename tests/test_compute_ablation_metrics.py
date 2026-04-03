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

from tools.compute_ablation_metrics import (
    _compute_aji,
    _compute_pq,
    _cellvit_json_to_instance_mask,
    _empty_metrics_record,
    _merge_cosine_into_metrics,
    run_cellvit,
)


def test_empty_record_has_all_keys():
    record = _empty_metrics_record()
    assert set(record.keys()) == {"cosine", "lpips", "aji", "pq"}
    assert all(value is None for value in record.values())


def test_merge_cosine_preserves_existing():
    existing = {
        "cell_types": {
            "cosine": None,
            "lpips": 0.3,
            "aji": None,
            "pq": None,
        }
    }
    cosine_scores = {"cell_types": 0.9946}
    result = _merge_cosine_into_metrics(existing, cosine_scores)
    assert result["cell_types"]["cosine"] == pytest.approx(0.9946)
    assert result["cell_types"]["lpips"] == pytest.approx(0.3)


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
