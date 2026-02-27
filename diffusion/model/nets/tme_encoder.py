"""
tme_encoder.py - TME Encoder for PixCell-256 ControlNet.

Architecture:
    TME channels [B, C_tme, 256, 256] → ResNet CNN → [B, 16, 32, 32]
                                                             ↓ flatten → [B, 1024, 16]  (key/value)
    VAE mask latent [B, 16, 32, 32]   → flatten     → [B, 1024, 16]  (query)
                                                             ↓
                                          MultiHeadCrossAttention (from PixArt_blocks)
                                                             ↓
                                          reshape → [B, 16, 32, 32]  fused conditioning

Uses your existing MultiHeadCrossAttention (xformers-backed) instead of a
custom attention implementation. Zero-init on the output projection ensures
fusion is identity at step 0.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusion.model.builder import MODELS

# ── ResNet building blocks ────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act  = nn.SiLU()
        self.res  = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.act(self.norm(self.down(x))))


# ── TME CNN Encoder ───────────────────────────────────────────────────────────

class TMEEncoder(nn.Module):
    """
    Lightweight ResNet encoder: [B, C_tme, 256, 256] → [B, latent_ch, 32, 32].

    Three strided downsampling stages (256→128→64→32), each followed by a
    residual block. Output spatial resolution and channel count match the
    SD3 VAE latent so the two can be fused directly.

    Args:
        n_tme_channels: Number of TME input channels (e.g. 1=O2, 3=O2+glc+TGF).
        base_ch:        Base feature width. Default 32 (~300k params).
                        Increase to 64 (~1.1M) if GPU budget allows.
        latent_ch:      Output channel count. Must equal VAE latent channels (16).
    """
    def __init__(self, n_tme_channels: int, base_ch: int = 32, latent_ch: int = 16):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.stem  = nn.Sequential(
            nn.Conv2d(n_tme_channels, c1, 3, padding=1, bias=False),
            nn.GroupNorm(8, c1), nn.SiLU(),
        )
        self.down1 = DownBlock(c1, c2)    # 256 → 128
        self.res1  = ResBlock(c2)
        self.down2 = DownBlock(c2, c3)    # 128 → 64
        self.res2  = ResBlock(c3)
        self.down3 = DownBlock(c3, latent_ch)  # 64 → 32
        self.res3  = ResBlock(latent_ch)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, tme: torch.Tensor) -> torch.Tensor:
        """[B, C_tme, 256, 256] → [B, 16, 32, 32]"""
        x = self.stem(tme)
        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))
        return x


# ── Public module ─────────────────────────────────────────────────────────────
@MODELS.register_module()
class TMEConditioningModule(ModelMixin, ConfigMixin):
    """
    Full TME conditioning module.  Single object to instantiate, optimize,
    checkpoint, and pass to accelerator.prepare().

    Uses your existing MultiHeadCrossAttention from PixArt_blocks (xformers-backed).
    The output projection inside MultiHeadCrossAttention is zero-initialized so
    fusion starts as identity at step 0 — TME contribution grows during training.

    Args:
        n_tme_channels: len(active_channels) - 1   (all channels except cell_mask)
            active_channels=["cell_mask","oxygen"]                → n_tme_channels=1
            active_channels=["cell_mask","oxygen","glucose","tgf"] → n_tme_channels=3
        base_ch:    CNN base channel width. Default 32.
        latent_ch:  Must be 16 (SD3 VAE latent channels).
        num_heads:  Attention heads for cross-attention. Default 4 (head_dim=4).
                    latent_ch must be divisible by num_heads.

    Forward:
        mask_latent  [B, 16, 32, 32]         pre-scaled SD3 VAE cell-mask latent
        tme_channels [B, n_tme_ch, 256, 256]  raw TME channel images
        → fused      [B, 16, 32, 32]          drop-in replacement for vae_mask
    """

    def __init__(
        self,
        n_tme_channels: int,
        base_ch: int = 32,
        latent_ch: int = 16,
        num_heads: int = 4,
    ):
        super().__init__()
        if n_tme_channels < 1:
            raise ValueError(
                f"n_tme_channels must be >= 1. "
                f"active_channels must include at least one TME channel beyond cell_mask."
            )
        assert latent_ch % num_heads == 0, \
            f"latent_ch ({latent_ch}) must be divisible by num_heads ({num_heads})"

        self.latent_ch = latent_ch

        self.tme_encoder = TMEEncoder(n_tme_channels, base_ch, latent_ch)

        # MultiHeadCrossAttention: query=mask tokens, key/value=TME tokens
        # d_model = latent_ch (16) — operates in the latent channel space
        self.cross_attn = MultiHeadCrossAttention(
            d_model=latent_ch,
            num_heads=num_heads,
        )

        # LayerNorm on query (mask latent tokens) before attention
        self.norm_q  = nn.LayerNorm(latent_ch)
        self.norm_kv = nn.LayerNorm(latent_ch)

        # Zero-init the cross-attention output projection → identity at init
        nn.init.zeros_(self.cross_attn.proj.weight)
        nn.init.zeros_(self.cross_attn.proj.bias)

    def forward(
        self,
        mask_latent:  torch.Tensor,   # [B, 16, 32, 32]
        tme_channels: torch.Tensor,   # [B, n_tme_ch, 256, 256]
    ) -> torch.Tensor:
        """Returns fused conditioning [B, 16, 32, 32]."""
        B, C, H, W = mask_latent.shape

        # Encode TME → [B, 16, 32, 32]
        tme_latent = self.tme_encoder(tme_channels.to(mask_latent.dtype))

        # Flatten spatial → token sequences: [B, 16, H, W] → [B, H*W, 16]
        q_tokens  = self.norm_q( mask_latent.flatten(2).transpose(1, 2))   # [B, N, C]
        kv_tokens = self.norm_kv(tme_latent.flatten(2).transpose(1, 2))    # [B, N, C]

        # Cross-attention: mask latent (Q) attends to TME latent (K, V)
        attn_out = self.cross_attn(q_tokens, kv_tokens)   # [B, N, C]

        # Reshape back to spatial + residual connection
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return mask_latent + attn_out   # residual: TME is additive correction

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)