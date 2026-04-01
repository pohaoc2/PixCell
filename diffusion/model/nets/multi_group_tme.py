"""
multi_group_tme.py — Per-group TME conditioning with additive residuals.

Each channel group (cell_types, cell_state, vasculature, microenv) gets:
  1. Its own TMEEncoder CNN: [B, n_ch, 256, 256] → [B, 16, 32, 32]
  2. Its own CrossAttentionWithWeights: Q=mask_latent, KV=group_latent → Δ_group

Fusion: fused = mask_latent + Σ(Δ_group) for all active groups.
Output projections use small normal init (std=0.02) to avoid gradient death with large datasets.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusion.model.builder import MODELS
from diffusion.model.nets.tme_encoder import TMEEncoder
from diffusion.model.nets.cross_attention_with_weights import CrossAttentionWithWeights


class _GroupBlock(nn.Module):
    """One group's encoder + cross-attention + layer norm."""

    def __init__(self, n_channels: int, base_ch: int, latent_ch: int, num_heads: int):
        super().__init__()
        self.encoder = TMEEncoder(n_channels, base_ch, latent_ch)
        self.norm_kv = nn.LayerNorm(latent_ch)
        self.cross_attn = CrossAttentionWithWeights(d_model=latent_ch, num_heads=num_heads)
        nn.init.normal_(self.cross_attn.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cross_attn.proj.bias)

    def forward(self, q_tokens, tme_input, return_attn_weights=False):
        group_latent = self.encoder(tme_input.to(q_tokens.dtype))
        kv_tokens = self.norm_kv(group_latent.flatten(2).transpose(1, 2))
        if return_attn_weights:
            delta, attn_weights = self.cross_attn(
                q_tokens, kv_tokens, mask=None, return_attn_weights=True
            )
            return delta, attn_weights
        return self.cross_attn(q_tokens, kv_tokens), None


@MODELS.register_module()
class MultiGroupTMEModule(ModelMixin, ConfigMixin):
    def __init__(
        self,
        channel_groups: list[dict],
        base_ch: int = 32,
        latent_ch: int = 16,
        num_heads: int = 4,
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.group_names = [g["name"] for g in channel_groups]
        self.norm_q = nn.LayerNorm(latent_ch)
        self.groups = nn.ModuleDict()
        for g in channel_groups:
            self.groups[g["name"]] = _GroupBlock(
                n_channels=g["n_channels"],
                base_ch=base_ch,
                latent_ch=latent_ch,
                num_heads=num_heads,
            )

    def forward(
        self,
        mask_latent: torch.Tensor,
        tme_channel_dict: dict[str, torch.Tensor],
        active_groups: set[str] | None = None,
        return_residuals: bool = False,
        return_attn_weights: bool = False,
    ):
        B, C, H, W = mask_latent.shape
        q_tokens = self.norm_q(mask_latent.flatten(2).transpose(1, 2))

        residual_sum = torch.zeros_like(mask_latent)
        residuals = {}
        attn_maps = {}

        for name in self.group_names:
            if active_groups is not None and name not in active_groups:
                continue
            if name not in tme_channel_dict:
                continue

            group_block = self.groups[name]
            delta_tokens, attn_w = group_block(
                q_tokens, tme_channel_dict[name],
                return_attn_weights=return_attn_weights,
            )
            delta_spatial = delta_tokens.transpose(1, 2).reshape(B, C, H, W)
            residual_sum = residual_sum + delta_spatial

            if return_residuals:
                residuals[name] = delta_spatial
            if return_attn_weights and attn_w is not None:
                attn_maps[name] = attn_w

        fused = mask_latent + residual_sum

        if return_residuals and return_attn_weights:
            return fused, residuals, attn_maps
        if return_residuals:
            return fused, residuals
        return fused

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
