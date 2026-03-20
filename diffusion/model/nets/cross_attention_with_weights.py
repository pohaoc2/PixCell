"""CrossAttention subclass that optionally returns attention weights for interpretability."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention


class CrossAttentionWithWeights(MultiHeadCrossAttention):
    """
    Extends MultiHeadCrossAttention with optional attention weight return.

    When return_attn_weights=False (default), delegates to the parent class
    (xformers-backed, fast, no weights). When True, uses standard PyTorch
    attention to capture the [B, heads, N_q, N_kv] weight matrix.
    Used only at analysis time — not during training.
    """

    def forward(self, x, cond, mask=None, return_attn_weights=False):
        if not return_attn_weights:
            return super().forward(x, cond, mask)

        B, N, C = x.shape
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, attn_weights
