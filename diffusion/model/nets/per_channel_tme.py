"""A1 per-channel TME conditioning modules."""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from diffusion.model.builder import MODELS
from diffusion.model.nets.cross_attention_with_weights import CrossAttentionWithWeights
from diffusion.model.nets.tme_encoder import (
    TMEEncoder,
    TMEInputNormalizer,
    continuous_channel_indices,
)


class _ChannelBlock(nn.Module):
    def __init__(self, base_ch: int, latent_ch: int, num_heads: int, channel_name: str):
        super().__init__()
        self.input_normalizer = TMEInputNormalizer(
            1,
            continuous_channel_indices([channel_name]),
        )
        self.encoder = TMEEncoder(1, base_ch, latent_ch)
        self.norm_kv = nn.LayerNorm(latent_ch)
        self.cross_attn = CrossAttentionWithWeights(d_model=latent_ch, num_heads=num_heads)
        nn.init.normal_(self.cross_attn.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cross_attn.proj.bias)

    def forward(self, q_tokens, channel_input, return_attn_weights=False):
        channel_latent = self.encoder(
            self.input_normalizer(channel_input.to(q_tokens.dtype))
        )
        kv_tokens = self.norm_kv(channel_latent.flatten(2).transpose(1, 2))
        if return_attn_weights:
            delta, attn_weights = self.cross_attn(
                q_tokens, kv_tokens, mask=None, return_attn_weights=True
            )
            return delta, attn_weights
        return self.cross_attn(q_tokens, kv_tokens), None


@MODELS.register_module()
class PerChannelTMEModule(ModelMixin, ConfigMixin):
    def __init__(
        self,
        active_channels: list[str],
        base_ch: int = 16,
        latent_ch: int = 16,
        num_heads: int = 4,
    ):
        super().__init__()
        self.active_channels = list(active_channels)
        self.latent_ch = latent_ch
        self.norm_q = nn.LayerNorm(latent_ch)
        self.channels = nn.ModuleDict(
            {
                name: _ChannelBlock(
                    base_ch=base_ch,
                    latent_ch=latent_ch,
                    num_heads=num_heads,
                    channel_name=name,
                )
                for name in self.active_channels
            }
        )

    def forward(
        self,
        mask_latent: torch.Tensor,
        control_input: torch.Tensor,
        active_groups: set[str] | None = None,
        return_residuals: bool = False,
        return_attn_weights: bool = False,
    ):
        batch_size, channels, _, _ = control_input.shape
        if channels != len(self.active_channels):
            raise ValueError(
                f"Expected {len(self.active_channels)} control channels, got {channels}"
            )

        batch, latent_ch, latent_h, latent_w = mask_latent.shape
        if batch != batch_size:
            raise ValueError("mask_latent and control_input batch size must match")

        q_tokens = self.norm_q(mask_latent.flatten(2).transpose(1, 2))
        residual_sum = torch.zeros_like(mask_latent)
        residuals = {}
        attn_maps = {}
        active_names = set(self.active_channels) if active_groups is None else set(active_groups)

        for index, name in enumerate(self.active_channels):
            if name not in active_names:
                continue
            channel_input = control_input[:, index:index + 1]
            delta_tokens, attn_w = self.channels[name](
                q_tokens,
                channel_input,
                return_attn_weights=return_attn_weights,
            )
            delta_spatial = delta_tokens.transpose(1, 2).reshape(batch, latent_ch, latent_h, latent_w)
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


@MODELS.register_module()
class RawConditioningPassthrough(ModelMixin, ConfigMixin):
    """Pass-through shim so raw control tensors can reuse the existing TME plumbing."""

    def __init__(self, active_channels: list[str] | None = None, **kwargs):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.active_channels = list(active_channels or [])

    def forward(
        self,
        mask_latent: torch.Tensor,
        control_input: torch.Tensor,
        active_groups: set[str] | None = None,
        return_residuals: bool = False,
        return_attn_weights: bool = False,
    ):
        del mask_latent, return_attn_weights
        passthrough = control_input.to(dtype=self.dummy.dtype)
        if active_groups is not None and self.active_channels:
            keep = set(active_groups)
            masked = torch.zeros_like(passthrough)
            for index, name in enumerate(self.active_channels):
                if name in keep:
                    masked[:, index:index + 1] = passthrough[:, index:index + 1]
            passthrough = masked
        if return_residuals:
            return passthrough, {}
        return passthrough
