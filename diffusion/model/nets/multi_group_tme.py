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
from diffusion.model.nets.tme_encoder import (
    TMEEncoder,
    TMEInputNormalizer,
    continuous_channel_indices,
)
from diffusion.model.nets.cross_attention_with_weights import CrossAttentionWithWeights


def _default_continuous_indices(group_name: str, n_channels: int) -> list[int]:
    if group_name in {"vasculature", "microenv"}:
        return list(range(n_channels))
    return []


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


class _GroupBlock(nn.Module):
    """One group's encoder + cross-attention + layer norm."""

    def __init__(
        self,
        n_channels: int,
        base_ch: int,
        latent_ch: int,
        num_heads: int,
        group_name: str,
        channel_names: list[str] | None = None,
    ):
        super().__init__()
        indices = continuous_channel_indices(channel_names)
        if channel_names is None:
            indices = _default_continuous_indices(group_name, n_channels)
        self.input_normalizer = TMEInputNormalizer(n_channels, indices)
        self.encoder = TMEEncoder(n_channels, base_ch, latent_ch)
        self.norm_kv = nn.LayerNorm(latent_ch)
        self.cross_attn = CrossAttentionWithWeights(d_model=latent_ch, num_heads=num_heads)
        nn.init.normal_(self.cross_attn.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.cross_attn.proj.bias)

    def forward(self, q_tokens, tme_input, return_attn_weights=False, return_probe=False):
        normalized_input = self.input_normalizer(tme_input.to(q_tokens.dtype))
        if return_probe:
            group_latent, activations = self.encoder(
                normalized_input, return_activations=True
            )
        else:
            group_latent = self.encoder(normalized_input)
            activations = None
        kv_tokens = self.norm_kv(group_latent.flatten(2).transpose(1, 2))
        if return_attn_weights:
            delta, attn_weights = self.cross_attn(
                q_tokens, kv_tokens, mask=None, return_attn_weights=True
            )
        else:
            delta, attn_weights = self.cross_attn(q_tokens, kv_tokens), None

        probe = None
        if return_probe:
            probe = {
                "tme_channels": _tensor_stats(tme_input),
                "post_stem": _tensor_stats(activations["post_stem"]),
                "post_down3": _tensor_stats(activations["post_down3"]),
            }
        return delta, attn_weights, probe


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
        self._debug_tme_probe_enabled = False
        self._debug_tme_probe_logger = None
        for g in channel_groups:
            self.groups[g["name"]] = _GroupBlock(
                n_channels=g["n_channels"],
                base_ch=base_ch,
                latent_ch=latent_ch,
                num_heads=num_heads,
                group_name=g["name"],
                channel_names=g.get("channels"),
            )

    def enable_debug_tme_probe(self, logger=None) -> None:
        self._debug_tme_probe_enabled = True
        self._debug_tme_probe_logger = logger

    def _log_debug_tme_probe(self, stats: dict[str, dict[str, dict[str, float]]]) -> None:
        logger = self._debug_tme_probe_logger
        log = logger.info if logger is not None else print
        for group_name, group_stats in stats.items():
            for label, values in group_stats.items():
                log(
                    "[debug_tme_probe] "
                    f"{group_name}.{label}: "
                    f"min={values['min']:.4e} max={values['max']:.4e} "
                    f"mean={values['mean']:.4e} std={values['std']:.4e}"
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
        probe_enabled = self._debug_tme_probe_enabled
        probe_stats = {}

        for name in self.group_names:
            if active_groups is not None and name not in active_groups:
                continue
            if name not in tme_channel_dict:
                continue

            group_block = self.groups[name]
            delta_tokens, attn_w, probe = group_block(
                q_tokens, tme_channel_dict[name],
                return_attn_weights=return_attn_weights,
                return_probe=probe_enabled,
            )
            delta_spatial = delta_tokens.transpose(1, 2).reshape(B, C, H, W)
            residual_sum = residual_sum + delta_spatial
            if probe is not None:
                probe["post_mhca_fused"] = _tensor_stats(mask_latent + delta_spatial)
                probe_stats[name] = probe

            if return_residuals:
                residuals[name] = delta_spatial
            if return_attn_weights and attn_w is not None:
                attn_maps[name] = attn_w

        fused = mask_latent + residual_sum
        if probe_enabled:
            self._debug_tme_probe_enabled = False
            self._log_debug_tme_probe(probe_stats)

        if return_residuals and return_attn_weights:
            return fused, residuals, attn_maps
        if return_residuals:
            return fused, residuals
        return fused

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
