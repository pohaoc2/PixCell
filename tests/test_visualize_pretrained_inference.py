import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from tools.visualize_pretrained_inference import (
    load_rgb_image,
    make_contour_overlay,
    save_comparison_figure,
)


def _write_mask(mask_path: Path, size: int = 32) -> None:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    Image.fromarray(mask).save(mask_path)


def test_load_rgb_image_resizes_and_returns_rgb_array():
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "input.png"
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[..., 1] = 180
        Image.fromarray(image).save(image_path)

        out = load_rgb_image(image_path, resolution=64)

        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8


def test_make_contour_overlay_adds_yellow_contour_pixels():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_path = Path(tmp_dir) / "mask.png"
        _write_mask(mask_path, size=32)
        generated = np.zeros((32, 32, 3), dtype=np.uint8)

        overlay = make_contour_overlay(generated, mask_path, resolution=32, thickness=1)

        yellow_pixels = np.sum(
            (overlay[..., 0] == 255) & (overlay[..., 1] == 255) & (overlay[..., 2] == 0)
        )
        assert overlay.shape == generated.shape
        assert yellow_pixels > 0


def test_save_comparison_figure_writes_output_without_reference():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        mask_path = root / "mask.png"
        _write_mask(mask_path, size=32)
        generated = np.zeros((32, 32, 3), dtype=np.uint8)
        output_path = root / "comparison.png"

        save_comparison_figure(
            mask_path=mask_path,
            gen_img=generated,
            save_path=output_path,
            reference_he_path=None,
            resolution=32,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_save_comparison_figure_includes_reference_when_present():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        mask_path = root / "mask.png"
        ref_path = root / "reference.png"
        _write_mask(mask_path, size=32)
        Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(ref_path)
        generated = np.zeros((32, 32, 3), dtype=np.uint8)
        output_path = root / "comparison_with_ref.png"

        save_comparison_figure(
            mask_path=mask_path,
            gen_img=generated,
            save_path=output_path,
            reference_he_path=ref_path,
            resolution=32,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_save_comparison_figure_ignores_missing_reference_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        mask_path = root / "mask.png"
        _write_mask(mask_path, size=32)
        generated = np.zeros((32, 32, 3), dtype=np.uint8)
        output_path = root / "comparison_missing_ref.png"

        save_comparison_figure(
            mask_path=mask_path,
            gen_img=generated,
            save_path=output_path,
            reference_he_path=root / "does_not_exist.png",
            resolution=32,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
