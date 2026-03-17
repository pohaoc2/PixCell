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
