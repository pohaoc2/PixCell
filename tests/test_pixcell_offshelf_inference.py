"""Smoke tests for off-the-shelf PixCell ControlNet inference wrapper."""
from __future__ import annotations

import inspect


def test_offshelf_wrapper_imports():
    from tools.baselines.pixcell_offshelf_inference import OffShelfPixCellInference  # noqa: F401


def test_offshelf_wrapper_constructor_signature():
    from tools.baselines.pixcell_offshelf_inference import OffShelfPixCellInference

    expected = {"controlnet_path", "base_model_path", "vae_path", "uni_path", "device"}
    sig = inspect.signature(OffShelfPixCellInference.__init__)
    params = set(sig.parameters) - {"self"}
    missing = expected - params
    assert not missing, f"missing constructor params: {missing}"


def test_offshelf_run_signature():
    from tools.baselines.pixcell_offshelf_inference import OffShelfPixCellInference

    sig = inspect.signature(OffShelfPixCellInference.run_on_tile)
    params = set(sig.parameters) - {"self"}
    expected = {"tile_id", "cell_mask", "uni_embedding", "out_dir", "num_steps", "guidance_scale"}
    missing = expected - params
    assert not missing, f"missing run_on_tile params: {missing}"
