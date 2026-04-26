"""Bypass probe correctness: TME output is zeroed at inference."""
from __future__ import annotations

import torch


def test_zero_tme_yields_mask_latent_only():
    from tools.ablation_a2.run_bypass_probe import compute_bypass_conditioning

    mask_latent = torch.randn(1, 16, 32, 32)
    tme_output = torch.zeros_like(mask_latent)

    cond = compute_bypass_conditioning(mask_latent=mask_latent, tme_output=tme_output)

    assert torch.allclose(cond, mask_latent, atol=1e-7)


def test_full_tme_produces_additive_conditioning():
    from tools.ablation_a2.run_bypass_probe import compute_bypass_conditioning

    mask_latent = torch.randn(1, 16, 32, 32)
    tme_output = torch.randn_like(mask_latent)

    cond = compute_bypass_conditioning(mask_latent=mask_latent, tme_output=tme_output)

    assert torch.allclose(cond, mask_latent + tme_output, atol=1e-7)
