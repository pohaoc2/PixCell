"""Tests for tools/stage3/common.py pure utility functions."""

from pathlib import Path

import numpy as np
import pytest
from tools.stage3.common import (
    to_uint8_rgb,
    print_progress,
    inference_dtype,
    make_inference_scheduler,
    resolve_uni_embedding,
)


class TestToUint8Rgb:
    def test_grayscale_hw_expands_to_hwx3(self):
        arr = np.zeros((8, 8), dtype=np.uint8)
        out = to_uint8_rgb(arr)
        assert out.shape == (8, 8, 3)
        assert out.dtype == np.uint8

    def test_hwx1_expands_to_hwx3(self):
        arr = np.zeros((4, 4, 1), dtype=np.uint8)
        out = to_uint8_rgb(arr)
        assert out.shape == (4, 4, 3)

    def test_hwx4_rgba_drops_alpha(self):
        arr = np.ones((4, 4, 4), dtype=np.uint8) * 100
        out = to_uint8_rgb(arr)
        assert out.shape == (4, 4, 3)
        assert np.all(out == 100)

    def test_hwx3_uint8_returned_unchanged(self):
        arr = np.ones((4, 4, 3), dtype=np.uint8) * 42
        out = to_uint8_rgb(arr)
        assert out.shape == (4, 4, 3)
        assert np.all(out == 42)

    def test_float_unit_range_explicit(self):
        arr = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = to_uint8_rgb(arr, value_range="unit")
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 0
        assert out[0, 0, 2] == 255

    def test_float_byte_range_explicit(self):
        arr = np.array([[[0.0, 127.0, 255.0]]], dtype=np.float32)
        out = to_uint8_rgb(arr, value_range="byte")
        assert out.dtype == np.uint8
        assert out[0, 0, 2] == 255

    def test_float_auto_detects_unit_range(self):
        arr = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = to_uint8_rgb(arr, value_range="auto")
        assert out[0, 0, 2] == 255

    def test_float_auto_detects_byte_range(self):
        arr = np.array([[[0.0, 100.0, 200.0]]], dtype=np.float32)
        out = to_uint8_rgb(arr, value_range="auto")
        assert out.dtype == np.uint8
        assert out[0, 0, 2] == 200

    def test_invalid_shape_raises_value_error(self):
        arr = np.zeros((4, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="expected"):
            to_uint8_rgb(arr)

    def test_invalid_value_range_raises_value_error(self):
        arr = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        with pytest.raises(ValueError, match="unsupported value_range"):
            to_uint8_rgb(arr, value_range="unknown")

    def test_empty_array_auto_defaults_to_unit(self):
        arr = np.zeros((0, 0), dtype=np.float32)
        out = to_uint8_rgb(arr, value_range="auto")
        assert out.dtype == np.uint8

    def test_grayscale_hw_float_values_match(self):
        arr = np.array([[0.0, 0.25, 0.5, 1.0]], dtype=np.float32)
        out = to_uint8_rgb(arr, value_range="unit")
        assert out.shape == (1, 4, 3)
        assert out[0, 0, 0] == 0
        assert out[0, 3, 0] == 255


class TestPrintProgress:
    def test_writes_to_stderr(self, capsys):
        print_progress(5, 10, prefix="test")
        captured = capsys.readouterr()
        assert "test" in captured.err
        assert "5/10" in captured.err

    def test_last_call_appends_newline(self, capsys):
        print_progress(10, 10, prefix="done")
        captured = capsys.readouterr()
        assert captured.err.endswith("\n")

    def test_intermediate_call_no_trailing_newline(self, capsys):
        print_progress(3, 10, prefix="mid")
        captured = capsys.readouterr()
        assert "3/10" in captured.err

    def test_zero_total_does_not_divide_by_zero(self, capsys):
        print_progress(0, 0, prefix="x")
        captured = capsys.readouterr()
        assert "x" in captured.err

    def test_progress_bar_contains_hash_chars(self, capsys):
        print_progress(14, 28, prefix="p")
        captured = capsys.readouterr()
        assert "#" in captured.err


class TestTorchDependentGuards:
    """In CI torch is not installed; these functions raise ModuleNotFoundError."""

    def test_inference_dtype_raises_without_torch(self):
        with pytest.raises(ModuleNotFoundError, match="torch"):
            inference_dtype("cuda")

    def test_make_inference_scheduler_raises_without_diffusers(self):
        with pytest.raises(ModuleNotFoundError):
            make_inference_scheduler(num_steps=10, device="cpu")

    def test_resolve_uni_embedding_raises_without_torch(self):
        with pytest.raises(ModuleNotFoundError, match="torch"):
            resolve_uni_embedding("tile_001", feat_dir=Path("/tmp"), null_uni=False)
