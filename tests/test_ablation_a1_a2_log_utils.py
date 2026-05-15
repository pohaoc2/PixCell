from __future__ import annotations

import json
import math

import pytest

from tools.ablation_a1_a2.log_utils import deserialise_float, extract_run, serialise_float


def test_serialise_inf():
    assert serialise_float(float("inf")) == "inf"
    assert serialise_float(float("-inf")) == "-inf"


def test_serialise_nan_is_none():
    assert serialise_float(float("nan")) is None


def test_serialise_finite():
    assert serialise_float(0.123) == pytest.approx(0.123)


def test_deserialise_inf():
    assert math.isinf(deserialise_float("inf"))
    assert deserialise_float("inf") > 0


def test_deserialise_none_is_nan():
    assert math.isnan(deserialise_float(None))


def test_extract_run_jsonl(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text(
        json.dumps({"step": 50, "loss": 0.12, "grad_norm": float("inf")}) + "\n"
        + json.dumps({"step": 100, "loss": 0.10, "grad_norm": 0.02}) + "\n",
        encoding="utf-8",
    )
    entries = extract_run(log_path)
    assert len(entries) == 2
    assert entries[0]["step"] == 50
    assert entries[0]["grad_norm"] == "inf"
    assert entries[1]["grad_norm"] == pytest.approx(0.02)


def test_extract_run_skips_missing_file(tmp_path):
    assert extract_run(tmp_path / "nonexistent.jsonl") == []
