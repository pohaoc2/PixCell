# tests/test_validate_sim_to_exp.py
import torch
import pytest


def test_null_uni_embed_shape():
    from train_scripts.inference_controlnet import null_uni_embed
    emb = null_uni_embed(device='cpu', dtype=torch.float32)
    assert emb.shape == (1, 1, 1, 1536), f"Got {emb.shape}"


def test_null_uni_embed_is_zeros():
    from train_scripts.inference_controlnet import null_uni_embed
    emb = null_uni_embed(device='cpu', dtype=torch.float32)
    assert torch.all(emb == 0.0)


def test_null_uni_embed_dtype():
    from train_scripts.inference_controlnet import null_uni_embed
    emb = null_uni_embed(device='cpu', dtype=torch.float16)
    assert emb.dtype == torch.float16


def test_cosine_similarity_range():
    """cosine_similarity values must be in [-1, 1]."""
    from validate_sim_to_exp import cosine_similarity_matrix
    a = torch.randn(10, 1536)
    b = torch.randn(10, 1536)
    sims = cosine_similarity_matrix(a, b)
    assert sims.shape == (10,), f"Got {sims.shape}"
    assert torch.all(sims >= -1.0) and torch.all(sims <= 1.0)


def test_cosine_similarity_identical():
    """Identical vectors should give similarity 1.0."""
    from validate_sim_to_exp import cosine_similarity_matrix
    a = torch.randn(5, 1536)
    sims = cosine_similarity_matrix(a, a)
    assert torch.allclose(sims, torch.ones(5), atol=1e-5)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should give similarity 0.0."""
    from validate_sim_to_exp import cosine_similarity_matrix
    a = torch.zeros(1, 4)
    b = torch.zeros(1, 4)
    a[0, 0] = 1.0
    b[0, 1] = 1.0
    sims = cosine_similarity_matrix(a, b)
    assert torch.allclose(sims, torch.zeros(1), atol=1e-6)


def test_load_sim_ctrl_tensor(tmp_path):
    """load_sim_ctrl_tensor returns a [C, H, W] tensor with the correct channel count."""
    import cv2
    import numpy as np
    from validate_sim_to_exp import load_sim_ctrl_tensor

    active_channels = ["cell_mask", "oxygen"]
    sim_id = "snap_0001"
    for ch in active_channels:
        ch_dir = tmp_path / "sim_channels" / ch
        ch_dir.mkdir(parents=True)
        arr = np.zeros((256, 256), dtype=np.uint8)
        arr[10:50, 10:50] = 200
        cv2.imwrite(str(ch_dir / f"{sim_id}.png"), arr)

    ctrl = load_sim_ctrl_tensor(tmp_path, sim_id, active_channels, resolution=256)
    assert ctrl.shape == (len(active_channels), 256, 256), f"Got {ctrl.shape}"
    assert ctrl.dtype == torch.float32
    # cell_mask channel must be thresholded to binary {0, 1}
    assert set(ctrl[0].unique().tolist()).issubset({0.0, 1.0})
