# Multi-Group TME Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic TME encoder with per-group CNN encoders + per-group cross-attention, enabling disentangled, independently droppable TME channel conditioning with interpretable per-group contribution analysis.

**Architecture:** Each of 4 channel groups (cell_identity, cell_state, vasculature, microenv) gets its own TMEEncoder CNN + CrossAttentionWithWeights module. All groups produce additive zero-init residuals to the shared VAE mask latent. Groups are independently droppable at both training (group dropout) and inference time.

**Tech Stack:** PyTorch, xformers, diffusers (ModelMixin/ConfigMixin), matplotlib (visualization), existing TMEEncoder from `tme_encoder.py`, existing MultiHeadCrossAttention from `PixArt_blocks.py`. Tests use `unittest` (pytest not available in the pixcell conda env). All Python commands must use `conda run -n pixcell python`.

**Spec:** `docs/superpowers/specs/multi-group-tme-architecture.md`

---

## Task 1: Channel Group Utilities

**Files:**
- Create: `tools/channel_group_utils.py`
- Test: `tests/test_channel_group_utils.py`

### Step 1: Write failing tests

- [ ] **Step 1.1: Write tests for `split_channels_to_groups`**

Create `tests/test_channel_group_utils.py`:

```python
import unittest
import torch

class TestSplitChannelsToGroups(unittest.TestCase):
    def setUp(self):
        self.active_channels = [
            "cell_mask",
            "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
            "cell_state_prolif", "cell_state_nonprolif", "cell_state_dead",
            "vasculature", "oxygen", "glucose",
        ]
        self.channel_groups = [
            dict(name="cell_identity", channels=["cell_type_healthy", "cell_type_cancer", "cell_type_immune"]),
            dict(name="cell_state", channels=["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]),
            dict(name="vasculature", channels=["vasculature"]),
            dict(name="microenv", channels=["oxygen", "glucose"]),
        ]
        # [B=2, C=10, H=4, W=4] — small spatial for fast tests
        self.control_input = torch.randn(2, 10, 4, 4)

    def test_split_returns_all_groups(self):
        from tools.channel_group_utils import split_channels_to_groups
        result = split_channels_to_groups(self.control_input, self.active_channels, self.channel_groups)
        self.assertEqual(set(result.keys()), {"cell_identity", "cell_state", "vasculature", "microenv"})

    def test_split_shapes(self):
        from tools.channel_group_utils import split_channels_to_groups
        result = split_channels_to_groups(self.control_input, self.active_channels, self.channel_groups)
        self.assertEqual(result["cell_identity"].shape, (2, 3, 4, 4))
        self.assertEqual(result["cell_state"].shape, (2, 3, 4, 4))
        self.assertEqual(result["vasculature"].shape, (2, 1, 4, 4))
        self.assertEqual(result["microenv"].shape, (2, 2, 4, 4))

    def test_split_values_correct(self):
        from tools.channel_group_utils import split_channels_to_groups
        result = split_channels_to_groups(self.control_input, self.active_channels, self.channel_groups)
        # cell_type_healthy is index 1 in active_channels
        torch.testing.assert_close(result["cell_identity"][:, 0], self.control_input[:, 1])
        # oxygen is index 8
        torch.testing.assert_close(result["microenv"][:, 0], self.control_input[:, 8])

    def test_split_excludes_cell_mask(self):
        from tools.channel_group_utils import split_channels_to_groups
        result = split_channels_to_groups(self.control_input, self.active_channels, self.channel_groups)
        self.assertNotIn("cell_mask", result)


class TestApplyGroupDropout(unittest.TestCase):
    def test_returns_set_of_sets(self):
        from tools.channel_group_utils import apply_group_dropout
        group_names = ["cell_identity", "cell_state", "vasculature", "microenv"]
        dropout_probs = dict(cell_identity=0.0, cell_state=0.0, vasculature=0.0, microenv=0.0)
        result = apply_group_dropout(group_names, dropout_probs, batch_size=4)
        # With 0.0 dropout, all groups should be active for all samples
        self.assertEqual(len(result), 4)
        for sample_groups in result:
            self.assertEqual(sample_groups, set(group_names))

    def test_full_dropout(self):
        from tools.channel_group_utils import apply_group_dropout
        group_names = ["cell_identity", "cell_state", "vasculature", "microenv"]
        dropout_probs = dict(cell_identity=1.0, cell_state=1.0, vasculature=1.0, microenv=1.0)
        result = apply_group_dropout(group_names, dropout_probs, batch_size=4)
        for sample_groups in result:
            self.assertEqual(sample_groups, set())

    def test_partial_dropout_returns_correct_length(self):
        from tools.channel_group_utils import apply_group_dropout
        group_names = ["cell_identity", "cell_state"]
        dropout_probs = dict(cell_identity=0.5, cell_state=0.5)
        result = apply_group_dropout(group_names, dropout_probs, batch_size=8)
        self.assertEqual(len(result), 8)
        for sample_groups in result:
            self.assertIsInstance(sample_groups, set)
            self.assertTrue(sample_groups.issubset(set(group_names)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `conda run -n pixcell python -m unittest tests/test_channel_group_utils.py -v`
Expected: ModuleNotFoundError for `tools.channel_group_utils`

### Step 2: Implement channel group utilities

- [ ] **Step 2.1: Create `tools/channel_group_utils.py`**

```python
"""Utilities for splitting TME channels into named groups and applying group-level dropout."""
from __future__ import annotations
import torch


def split_channels_to_groups(
    control_input: torch.Tensor,
    active_channels: list[str],
    channel_groups: list[dict],
) -> dict[str, torch.Tensor]:
    """
    Split a full [B, C, H, W] control tensor into per-group tensors.

    Args:
        control_input:   [B, C, H, W] with channels ordered by active_channels.
        active_channels: Ordered channel name list (cell_mask at index 0).
        channel_groups:  List of dicts with 'name' and 'channels' keys.

    Returns:
        Dict mapping group name → [B, n_group_ch, H, W] tensor.
    """
    ch_to_idx = {name: i for i, name in enumerate(active_channels)}
    result = {}
    for group in channel_groups:
        indices = [ch_to_idx[ch] for ch in group["channels"]]
        result[group["name"]] = control_input[:, indices]
    return result


def apply_group_dropout(
    group_names: list[str],
    dropout_probs: dict[str, float],
    batch_size: int,
) -> list[set[str]]:
    """
    For each sample in the batch, independently decide which groups are active.

    Args:
        group_names:   List of group name strings.
        dropout_probs: Dict mapping group name → dropout probability [0, 1].
        batch_size:    Number of samples in the batch.

    Returns:
        List of length batch_size. Each element is a set of active group names.
    """
    result = []
    for _ in range(batch_size):
        active = set()
        for name in group_names:
            if torch.rand(1).item() >= dropout_probs.get(name, 0.0):
                active.add(name)
        result.append(active)
    return result
```

- [ ] **Step 2.2: Run tests to verify they pass**

Run: `conda run -n pixcell python -m unittest tests/test_channel_group_utils.py -v`
Expected: All 7 tests PASS

- [ ] **Step 2.3: Commit**

```bash
git add tools/channel_group_utils.py tests/test_channel_group_utils.py
git commit -m "feat: add channel group split and dropout utilities"
```

---

## Task 2: CrossAttentionWithWeights

**Files:**
- Create: `diffusion/model/nets/cross_attention_with_weights.py`
- Test: `tests/test_cross_attention_with_weights.py`

This is a thin subclass of `MultiHeadCrossAttention` that adds an optional `return_attn_weights` mode for interpretability analysis. The base class in `PixArt_blocks.py` is NOT modified.

### Step 1: Write failing tests

- [ ] **Step 1.1: Write tests**

Create `tests/test_cross_attention_with_weights.py`:

```python
import unittest
import torch


class TestCrossAttentionWithWeights(unittest.TestCase):
    def setUp(self):
        from diffusion.model.nets.cross_attention_with_weights import CrossAttentionWithWeights
        self.d_model = 16
        self.num_heads = 4
        self.attn = CrossAttentionWithWeights(d_model=self.d_model, num_heads=self.num_heads)
        self.B, self.N = 2, 64
        self.x = torch.randn(self.B, self.N, self.d_model)
        self.cond = torch.randn(self.B, self.N, self.d_model)

    def test_forward_without_weights_shape(self):
        out = self.attn(self.x, self.cond)
        self.assertEqual(out.shape, (self.B, self.N, self.d_model))

    def test_forward_with_weights_returns_tuple(self):
        out, weights = self.attn(self.x, self.cond, return_attn_weights=True)
        self.assertEqual(out.shape, (self.B, self.N, self.d_model))
        self.assertEqual(weights.shape, (self.B, self.num_heads, self.N, self.N))

    def test_weights_sum_to_one(self):
        _, weights = self.attn(self.x, self.cond, return_attn_weights=True)
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)

    def test_backward_compat_with_base_class(self):
        from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention
        self.assertIsInstance(self.attn, MultiHeadCrossAttention)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `conda run -n pixcell python -m unittest tests/test_cross_attention_with_weights.py -v`
Expected: ModuleNotFoundError

### Step 2: Implement CrossAttentionWithWeights

- [ ] **Step 2.1: Create `diffusion/model/nets/cross_attention_with_weights.py`**

```python
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
```

- [ ] **Step 2.2: Run tests to verify they pass**

Run: `conda run -n pixcell python -m unittest tests/test_cross_attention_with_weights.py -v`
Expected: All 4 tests PASS

- [ ] **Step 2.3: Commit**

```bash
git add diffusion/model/nets/cross_attention_with_weights.py tests/test_cross_attention_with_weights.py
git commit -m "feat: add CrossAttentionWithWeights for interpretable attention"
```

---

## Task 3: MultiGroupTMEModule

**Files:**
- Create: `diffusion/model/nets/multi_group_tme.py`
- Test: `tests/test_multi_group_tme.py`
- Modify: `diffusion/model/builder.py` (import to trigger registration)

This is the core new module. It reuses `TMEEncoder` from `tme_encoder.py` and `CrossAttentionWithWeights` from Task 2.

### Step 1: Write failing tests

- [ ] **Step 1.1: Write tests for MultiGroupTMEModule**

Create `tests/test_multi_group_tme.py`:

```python
import unittest
import torch


def _make_module():
    from diffusion.model.nets.multi_group_tme import MultiGroupTMEModule
    channel_groups = [
        dict(name="cell_identity", n_channels=3),
        dict(name="cell_state", n_channels=3),
        dict(name="vasculature", n_channels=1),
        dict(name="microenv", n_channels=2),
    ]
    return MultiGroupTMEModule(channel_groups=channel_groups, base_ch=32, latent_ch=16, num_heads=4)


def _make_inputs(B=2, H=256, W=256):
    mask_latent = torch.randn(B, 16, 32, 32)
    tme_channel_dict = {
        "cell_identity": torch.randn(B, 3, H, W),
        "cell_state": torch.randn(B, 3, H, W),
        "vasculature": torch.randn(B, 1, H, W),
        "microenv": torch.randn(B, 2, H, W),
    }
    return mask_latent, tme_channel_dict


class TestMultiGroupTMEModule(unittest.TestCase):
    def test_forward_shape(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out = module(mask_latent, tme_dict)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_zero_init_identity(self):
        """At initialization, output should equal mask_latent (all residuals are zero)."""
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        with torch.no_grad():
            out = module(mask_latent, tme_dict)
        torch.testing.assert_close(out, mask_latent, atol=1e-5, rtol=1e-5)

    def test_active_groups_subset(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out = module(mask_latent, tme_dict, active_groups={"cell_identity"})
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_empty_active_groups_returns_mask_latent(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        with torch.no_grad():
            out = module(mask_latent, tme_dict, active_groups=set())
        torch.testing.assert_close(out, mask_latent, atol=1e-6, rtol=1e-6)

    def test_return_residuals(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out, residuals = module(mask_latent, tme_dict, return_residuals=True)
        self.assertEqual(set(residuals.keys()), {"cell_identity", "cell_state", "vasculature", "microenv"})
        for name, delta in residuals.items():
            self.assertEqual(delta.shape, (2, 16, 32, 32))

    def test_return_attn_weights(self):
        module = _make_module()
        mask_latent, tme_dict = _make_inputs()
        out, residuals, attn_maps = module(
            mask_latent, tme_dict, return_residuals=True, return_attn_weights=True
        )
        self.assertEqual(set(attn_maps.keys()), {"cell_identity", "cell_state", "vasculature", "microenv"})
        for name, weights in attn_maps.items():
            # [B, num_heads, N_q, N_kv] where N = 32*32 = 1024
            self.assertEqual(weights.shape, (2, 4, 1024, 1024))

    def test_missing_group_in_dict_skipped(self):
        module = _make_module()
        mask_latent, _ = _make_inputs()
        partial_dict = {"cell_identity": torch.randn(2, 3, 256, 256)}
        out = module(mask_latent, partial_dict)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_n_params(self):
        module = _make_module()
        total = sum(p.numel() for p in module.parameters() if p.requires_grad)
        self.assertGreater(total, 1_000_000)
        self.assertLess(total, 2_000_000)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `conda run -n pixcell python -m unittest tests/test_multi_group_tme.py -v`
Expected: ModuleNotFoundError

### Step 2: Implement MultiGroupTMEModule

- [ ] **Step 2.1: Create `diffusion/model/nets/multi_group_tme.py`**

```python
"""
multi_group_tme.py — Per-group TME conditioning with additive residuals.

Each channel group (cell_identity, cell_state, vasculature, microenv) gets:
  1. Its own TMEEncoder CNN: [B, n_ch, 256, 256] → [B, 16, 32, 32]
  2. Its own CrossAttentionWithWeights: Q=mask_latent, KV=group_latent → Δ_group

Fusion: fused = mask_latent + Σ(Δ_group) for all active groups.
All output projections are zero-initialized → identity at step 0.
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
        nn.init.zeros_(self.cross_attn.proj.weight)
        nn.init.zeros_(self.cross_attn.proj.bias)

    def forward(self, q_tokens, tme_input, return_attn_weights=False):
        """
        Args:
            q_tokens:    [B, N, C] — shared query tokens from mask latent.
            tme_input:   [B, n_ch, 256, 256] — raw group channels.
            return_attn_weights: If True, also return attention weights.
        Returns:
            delta: [B, N, C] — additive residual in token space.
            attn_weights: [B, heads, N, N] (only if return_attn_weights=True).
        """
        group_latent = self.encoder(tme_input.to(q_tokens.dtype))
        kv_tokens = self.norm_kv(group_latent.flatten(2).transpose(1, 2))
        if return_attn_weights:
            delta, attn_weights = self.cross_attn(q_tokens, kv_tokens, return_attn_weights=True)
            return delta, attn_weights
        return self.cross_attn(q_tokens, kv_tokens), None


@MODELS.register_module()
class MultiGroupTMEModule(ModelMixin, ConfigMixin):
    """
    Per-group TME conditioning module with additive residuals.

    Args:
        channel_groups: List of dicts with 'name' (str) and 'n_channels' (int).
            Example: [dict(name="cell_identity", n_channels=3), ...]
        base_ch:   CNN base channel width per group encoder. Default 32.
        latent_ch: Output channels, must be 16 (SD3 VAE latent channels).
        num_heads: Attention heads per group cross-attention. Default 4.
    """

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
        """
        Args:
            mask_latent:      [B, 16, 32, 32] — VAE-encoded cell mask.
            tme_channel_dict: {"group_name": [B, n_ch, 256, 256], ...}
            active_groups:    Set of group names to include. None = all.
            return_residuals: If True, return per-group Δ tensors.
            return_attn_weights: If True, return per-group attention weights.

        Returns:
            fused: [B, 16, 32, 32]
            residuals: dict (if return_residuals)
            attn_maps: dict (if return_attn_weights)
        """
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
```

- [ ] **Step 2.2: Register in builder**

Modify `diffusion/model/builder.py` — add import at the end so the `@MODELS.register_module()` decorator fires:

Add this line after `MODELS = Registry('models')` block (after line 5):

```python
import diffusion.model.nets.multi_group_tme  # noqa: F401 — register MultiGroupTMEModule
```

- [ ] **Step 2.3: Run tests to verify they pass**

Run: `conda run -n pixcell python -m unittest tests/test_multi_group_tme.py -v`
Expected: All 8 tests PASS

- [ ] **Step 2.4: Commit**

```bash
git add diffusion/model/nets/multi_group_tme.py diffusion/model/builder.py tests/test_multi_group_tme.py
git commit -m "feat: add MultiGroupTMEModule with per-group encoders and cross-attention"
```

---

## Task 4: Update Config

**Files:**
- Modify: `configs/config_controlnet_exp.py`

### Step 1: Update config

- [ ] **Step 1.1: Modify `configs/config_controlnet_exp.py`**

Changes:
1. Add `channel_groups` list (after `active_channels`)
2. Add `group_dropout_probs` dict (replaces `channel_reliability_weights`)
3. Change `tme_model` from `"TMEConditioningModule"` to `"MultiGroupTMEModule"`
4. Remove `channel_reliability_weights`

In `configs/config_controlnet_exp.py`, replace the `active_channels` + weights section (lines 27-60) with:

```python
    active_channels=[
        "cell_mask",
        "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
        "cell_state_prolif",  "cell_state_nonprolif", "cell_state_dead",
        "vasculature", "oxygen", "glucose",
    ],
```

Keep active_channels as-is. After the `data = dict(...)` block, add:

```python
channel_groups = [
    dict(name="cell_identity", channels=["cell_type_healthy", "cell_type_cancer", "cell_type_immune"]),
    dict(name="cell_state",    channels=["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]),
    dict(name="vasculature",   channels=["vasculature"]),
    dict(name="microenv",      channels=["oxygen", "glucose"]),
]
```

Replace the TME Encoder section:

```python
tme_model   = "MultiGroupTMEModule"
tme_base_ch = 32
tme_lr      = 1e-5
```

Replace the training knobs section:

```python
cfg_dropout_prob = 0.15

group_dropout_probs = dict(
    cell_identity=0.10,
    cell_state=0.10,
    vasculature=0.15,
    microenv=0.20,
)
```

Remove `channel_reliability_weights` entirely.

- [ ] **Step 1.2: Verify config loads without error**

Run: `conda run -n pixcell python -c "from diffusion.utils.misc import read_config; c = read_config('configs/config_controlnet_exp.py'); print('channel_groups:', c.channel_groups); print('tme_model:', c.tme_model)"`
Expected: Prints the channel_groups list and `MultiGroupTMEModule`

- [ ] **Step 1.3: Commit**

```bash
git add configs/config_controlnet_exp.py
git commit -m "config: switch to MultiGroupTMEModule with channel groups and group dropout"
```

---

## Task 5: Update Training Utilities

**Files:**
- Modify: `train_scripts/training_utils.py:23-66` (`_build_tme_module_and_optimizers`)

### Step 1: Update `_build_tme_module_and_optimizers`

- [ ] **Step 1.1: Modify the function**

The function currently takes `active_channels` and computes `n_tme_channels = len(active_channels) - 1` to build a single TMEConditioningModule.

Change it to read `channel_groups` from config and build a `MultiGroupTMEModule`:

```python
def _build_tme_module_and_optimizers(config, controlnet, train_dataloader,
                                     active_channels, logger):
    channel_groups_cfg = getattr(config, "channel_groups", None)

    if channel_groups_cfg is not None:
        # New multi-group path
        group_specs = []
        for g in channel_groups_cfg:
            group_specs.append(dict(name=g["name"], n_channels=len(g["channels"])))
        tme_module = build_model(
            getattr(config, "tme_model", "MultiGroupTMEModule"),
            False, False,
            channel_groups=group_specs,
            base_ch=getattr(config, "tme_base_ch", 32),
        )
    else:
        # Legacy single-encoder path (backward compat)
        n_tme_channels = len(active_channels) - 1
        tme_module = build_model(
            getattr(config, "tme_model", "TMEConditioningModule"),
            False, False,
            n_tme_channels=n_tme_channels,
            base_ch=getattr(config, "tme_base_ch", 32),
        )

    logger.info(
        f"[TME Module: {type(tme_module).__name__}] "
        f"trainable params="
        f"{sum(p.numel() for p in tme_module.parameters() if p.requires_grad):,}"
    )

    # ... rest of optimizer setup unchanged ...
```

- [ ] **Step 1.2: Verify the builder creates the correct module**

Run: `conda run -n pixcell python -c "
from diffusion.utils.misc import read_config
from diffusion.model.builder import build_model
config = read_config('configs/config_controlnet_exp.py')
groups = [dict(name=g['name'], n_channels=len(g['channels'])) for g in config.channel_groups]
m = build_model('MultiGroupTMEModule', False, False, channel_groups=groups, base_ch=32)
print(type(m).__name__, f'{m.n_params:,} params')
print('Groups:', list(m.groups.keys()))
"`
Expected: `MultiGroupTMEModule 1,2XX,XXX params` and `Groups: ['cell_identity', 'cell_state', 'vasculature', 'microenv']`

- [ ] **Step 1.3: Commit**

```bash
git add train_scripts/training_utils.py
git commit -m "feat: update TME builder to support MultiGroupTMEModule from channel_groups config"
```

---

## Task 6: Update Training Loop

**Files:**
- Modify: `train_scripts/train_controlnet_exp.py:150-181` (batch processing section)

### Step 1: Update the training loop

- [ ] **Step 1.1: Modify batch processing in `train_controlnet_exp`**

In `train_controlnet_exp()`, after reading training knobs (around line 108-109), add:

```python
channel_groups = getattr(config, "channel_groups", None)
group_dropout_probs = getattr(config, "group_dropout_probs", {})
use_multi_group = channel_groups is not None
```

Add import at top of file:

```python
from tools.channel_group_utils import split_channels_to_groups, apply_group_dropout
```

Replace lines 170-180 (the TME channel processing section):

Current code:
```python
tme_dtype    = next(tme_module.parameters()).dtype
tme_channels = control_input[:, 1:, :, :].to(dtype=tme_dtype)

if channel_weights is not None:
    w = torch.tensor(
        channel_weights, device=tme_channels.device, dtype=tme_channels.dtype
    ).view(1, -1, 1, 1)
    tme_channels = tme_channels * w

vae_mask = tme_module(vae_mask.to(dtype=tme_dtype), tme_channels)
```

New code:
```python
tme_dtype = next(tme_module.parameters()).dtype

if use_multi_group:
    active_channels = config.data.active_channels
    tme_channel_dict = split_channels_to_groups(
        control_input.to(dtype=tme_dtype), active_channels, channel_groups,
    )
    active_groups_per_sample = apply_group_dropout(
        [g["name"] for g in channel_groups], group_dropout_probs, batch_size=bs,
    )
    # Per-sample dropout: zero out channels for groups dropped in each sample
    for b_idx in range(bs):
        for g in channel_groups:
            gname = g["name"]
            if gname not in active_groups_per_sample[b_idx] and gname in tme_channel_dict:
                tme_channel_dict[gname][b_idx] = 0.0
    vae_mask = tme_module(vae_mask.to(dtype=tme_dtype), tme_channel_dict)
else:
    # Legacy single-encoder path
    tme_channels = control_input[:, 1:, :, :].to(dtype=tme_dtype)
    channel_weights = getattr(config, "channel_reliability_weights", None)
    if channel_weights is not None:
        w = torch.tensor(
            channel_weights, device=tme_channels.device, dtype=tme_channels.dtype
        ).view(1, -1, 1, 1)
        tme_channels = tme_channels * w
    vae_mask = tme_module(vae_mask.to(dtype=tme_dtype), tme_channels)
```

- [ ] **Step 1.2: Verify the training script imports without errors**

Run: `conda run -n pixcell python -c "from train_scripts.train_controlnet_exp import train_controlnet_exp; print('OK')"`
Expected: `OK`

- [ ] **Step 1.3: Commit**

```bash
git add train_scripts/train_controlnet_exp.py
git commit -m "feat: integrate multi-group TME with group dropout into training loop"
```

---

## Task 7: Update Inference Script

**Files:**
- Modify: `stage3_inference.py:116-179` (load_models), `stage3_inference.py:184-259` (generate), `stage3_inference.py:264-299` (CLI args)

### Step 1: Add CLI args and update generate()

- [ ] **Step 1.1: Add `--active-groups` and `--drop-groups` CLI args**

In `stage3_inference.py`, after the `--num-steps` argument (line 297), add:

```python
parser.add_argument("--active-groups", nargs="*", default=None,
                    help="TME groups to include (default: all). "
                         "e.g., --active-groups cell_identity vasculature")
parser.add_argument("--drop-groups", nargs="*", default=None,
                    help="TME groups to exclude. "
                         "e.g., --drop-groups microenv")
```

- [ ] **Step 1.2: Update `load_models()` to handle MultiGroupTMEModule**

In `load_models()` (around line 135), replace the TME module construction:

```python
channel_groups_cfg = getattr(config, "channel_groups", None)
if channel_groups_cfg is not None:
    group_specs = [dict(name=g["name"], n_channels=len(g["channels"])) for g in channel_groups_cfg]
    tme_module = build_model(
        "MultiGroupTMEModule", False, False,
        channel_groups=group_specs,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
else:
    n_tme_channels = len(config.data.active_channels) - 1
    tme_module = build_model(
        "TMEConditioningModule", False, False,
        n_tme_channels=n_tme_channels,
        base_ch=getattr(config, "tme_base_ch", 32),
    )
```

- [ ] **Step 1.3: Update `generate()` to use group-based TME call**

In `generate()` (around lines 234-237), replace the TME fusion section:

```python
# 3. Fuse TME channels through TME module
channel_groups_cfg = getattr(config, "channel_groups", None)
if channel_groups_cfg is not None:
    from tools.channel_group_utils import split_channels_to_groups
    tme_channel_dict = split_channels_to_groups(
        ctrl_full.unsqueeze(0).to(device, dtype=dtype),
        active_channels,
        channel_groups_cfg,
    )
    with torch.no_grad():
        fused_cond = tme_module(vae_mask.to(dtype), tme_channel_dict,
                                active_groups=active_groups)
else:
    tme_channels = ctrl_full[1:].unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        fused_cond = tme_module(vae_mask.to(dtype), tme_channels)
```

Add `active_groups` parameter to `generate()` signature.

- [ ] **Step 1.4: Wire CLI args to active_groups in `main()`**

In `main()`, after loading models, compute `active_groups`:

```python
# Resolve active groups
channel_groups_cfg = getattr(config, "channel_groups", None)
if channel_groups_cfg is not None:
    all_group_names = {g["name"] for g in channel_groups_cfg}
    if args.active_groups is not None:
        unknown = set(args.active_groups) - all_group_names
        if unknown:
            parser.error(f"Unknown groups: {unknown}. Valid: {all_group_names}")
        active_groups = set(args.active_groups)
    elif args.drop_groups is not None:
        unknown = set(args.drop_groups) - all_group_names
        if unknown:
            parser.error(f"Unknown groups: {unknown}. Valid: {all_group_names}")
        active_groups = all_group_names - set(args.drop_groups)
    else:
        active_groups = None  # all groups
else:
    active_groups = None
```

Pass `active_groups` to `generate()` in **both** the single-tile branch and the batch-mode branch:

```python
# Single-tile mode (around line 347)
img = generate(
    ...,
    active_groups=active_groups,
)

# Batch mode (around line 375)
img = generate(
    ...,
    active_groups=active_groups,
)
```

- [ ] **Step 1.5: Verify inference script parses without error**

Run: `conda run -n pixcell python stage3_inference.py --help`
Expected: Help output shows `--active-groups` and `--drop-groups` options

- [ ] **Step 1.6: Commit**

```bash
git add stage3_inference.py
git commit -m "feat: add --active-groups/--drop-groups to inference, support MultiGroupTMEModule"
```

---

## Task 8: Visualization — Group Attention Heatmaps

**Files:**
- Create: `tools/visualize_group_attention.py`

### Step 1: Implement

- [ ] **Step 1.1: Create `tools/visualize_group_attention.py`**

```python
"""Per-group attention heatmap visualization for MultiGroupTMEModule."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def compute_attention_heatmaps(
    attn_maps: dict[str, torch.Tensor],
    spatial_size: tuple[int, int] = (32, 32),
    output_resolution: int = 256,
) -> dict[str, np.ndarray]:
    """
    Convert per-group attention weight matrices to spatial heatmaps.

    Args:
        attn_maps: {"group_name": [B, heads, N_q, N_kv]} from MultiGroupTMEModule.
        spatial_size: Latent spatial dims (H, W) for reshaping tokens.
        output_resolution: Final heatmap pixel resolution.

    Returns:
        Dict of group_name → [output_resolution, output_resolution] numpy arrays (0-1 normalized).
    """
    H, W = spatial_size
    heatmaps = {}
    for name, weights in attn_maps.items():
        # Average over heads and batch: [N_kv]
        avg = weights.mean(dim=(0, 1))  # [N_q, N_kv]
        importance = avg.sum(dim=0)     # sum over queries → [N_kv]
        hmap = importance.reshape(H, W).cpu().numpy()
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        hmap_up = F.interpolate(
            torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0).float(),
            size=(output_resolution, output_resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        heatmaps[name] = hmap_up
    return heatmaps


def save_attention_heatmap_figure(
    mask_image: np.ndarray,
    gen_image: np.ndarray,
    attn_maps: dict[str, torch.Tensor],
    save_path: str | Path,
    spatial_size: tuple[int, int] = (32, 32),
    output_resolution: int = 256,
):
    """
    Save a multi-panel figure: [Cell Mask] [Gen H&E] [Attn per group...].

    Args:
        mask_image:  [H, W, 3] uint8 RGB.
        gen_image:   [H, W, 3] uint8 RGB.
        attn_maps:   {"group_name": [B, heads, N_q, N_kv]} tensors.
        save_path:   Output PNG path.
    """
    heatmaps = compute_attention_heatmaps(attn_maps, spatial_size, output_resolution)
    n_panels = 2 + len(heatmaps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(mask_image)
    axes[0].set_title("Cell Mask", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(gen_image)
    axes[1].set_title("Generated H&E", fontsize=10)
    axes[1].axis("off")

    for i, (name, hmap) in enumerate(heatmaps.items()):
        ax = axes[2 + i]
        ax.imshow(mask_image, alpha=0.3)
        im = ax.imshow(hmap, cmap="jet", alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f"{name} attn", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention heatmaps saved → {save_path}")
```

- [ ] **Step 1.2: Commit**

```bash
git add tools/visualize_group_attention.py
git commit -m "feat: add per-group attention heatmap visualization"
```

---

## Task 9: Visualization — Group Residual Magnitudes

**Files:**
- Create: `tools/visualize_group_residuals.py`

### Step 1: Implement

- [ ] **Step 1.1: Create `tools/visualize_group_residuals.py`**

```python
"""Per-group residual magnitude visualization for MultiGroupTMEModule."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def compute_residual_maps(
    residuals: dict[str, torch.Tensor],
    output_resolution: int = 256,
) -> dict[str, np.ndarray]:
    """
    Convert per-group residual tensors to L2 norm magnitude maps.

    Args:
        residuals: {"group_name": [B, 16, 32, 32]} from MultiGroupTMEModule.
        output_resolution: Final map pixel resolution.

    Returns:
        Dict of group_name → [output_resolution, output_resolution] numpy arrays.
    """
    maps = {}
    for name, delta in residuals.items():
        norm_map = delta[0].norm(dim=0)  # [32, 32] — L2 across channels, first sample
        norm_up = F.interpolate(
            norm_map.unsqueeze(0).unsqueeze(0).float(),
            size=(output_resolution, output_resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()
        maps[name] = norm_up
    return maps


def save_residual_magnitude_figure(
    mask_image: np.ndarray,
    gen_image: np.ndarray,
    residuals: dict[str, torch.Tensor],
    save_path: str | Path,
    output_resolution: int = 256,
):
    """
    Save a multi-panel figure: [Cell Mask] [Gen H&E] [‖Δ‖ per group...].

    Args:
        mask_image: [H, W, 3] uint8 RGB.
        gen_image:  [H, W, 3] uint8 RGB.
        residuals:  {"group_name": [B, 16, 32, 32]} tensors.
        save_path:  Output PNG path.
    """
    res_maps = compute_residual_maps(residuals, output_resolution)

    # Shared color scale across groups for fair comparison
    global_max = max(m.max() for m in res_maps.values()) if res_maps else 1.0

    n_panels = 2 + len(res_maps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(mask_image)
    axes[0].set_title("Cell Mask", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(gen_image)
    axes[1].set_title("Generated H&E", fontsize=10)
    axes[1].axis("off")

    for i, (name, rmap) in enumerate(res_maps.items()):
        ax = axes[2 + i]
        im = ax.imshow(rmap, cmap="inferno", vmin=0, vmax=global_max)
        ax.set_title(f"‖Δ_{name}‖", fontsize=10)
        ax.axis("off")

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Residual magnitude maps saved → {save_path}")
```

- [ ] **Step 1.2: Commit**

```bash
git add tools/visualize_group_residuals.py
git commit -m "feat: add per-group residual magnitude visualization"
```

---

## Task 10: Visualization — Ablation Grid

**Files:**
- Create: `tools/visualize_ablation_grid.py`

### Step 1: Implement

- [ ] **Step 1.1: Create `tools/visualize_ablation_grid.py`**

```python
"""Progressive composition ablation grid for MultiGroupTMEModule."""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_ablation_grid(
    images: list[tuple[str, np.ndarray]],
    save_path: str | Path,
):
    """
    Save a progressive composition grid showing incremental group contributions.

    Args:
        images: List of (label, image_array) tuples in composition order.
                Example: [("Mask only", img0), ("+ Cell ID", img1), ...]
        save_path: Output PNG path.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation grid saved → {save_path}")
```

Note: The actual generation of images for each group combination is done by calling
`generate()` with different `active_groups` sets. This tool only handles the
matplotlib panel layout. The orchestration (calling generate N times with
progressive group sets) is done in the caller — either the training validation
hook or a standalone script.

- [ ] **Step 1.2: Commit**

```bash
git add tools/visualize_ablation_grid.py
git commit -m "feat: add ablation grid visualization for progressive composition"
```

---

## Task 11: Training-Loop Validation Visualization Hook

**Files:**
- Modify: `train_scripts/train_controlnet_exp.py` (add vis hook at checkpoint save)

This task adds the spec §7.4 requirement: at every `save_model_steps` checkpoint, generate
attention heatmaps, residual magnitude maps, and an ablation grid for a fixed validation sample.

### Step 1: Implement validation hook

- [ ] **Step 1.1: Add `generate_validation_visualizations` helper**

At the bottom of `train_scripts/train_controlnet_exp.py` (before `main()`), add:

```python
@torch.no_grad()
def generate_validation_visualizations(
    tme_module, controlnet, base_model, vae, train_diffusion,
    val_control_input, val_vae_mask, val_uni_embeds,
    config, save_dir, device,
):
    """Generate attention heatmaps, residual maps, and ablation grid for one fixed sample."""
    from pathlib import Path
    from tools.channel_group_utils import split_channels_to_groups
    from tools.visualize_group_attention import save_attention_heatmap_figure
    from tools.visualize_group_residuals import save_residual_magnitude_figure
    from tools.visualize_ablation_grid import save_ablation_grid
    from train_scripts.inference_controlnet import denoise
    from diffusers import DDPMScheduler
    import numpy as np

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    channel_groups = config.channel_groups
    active_channels = config.data.active_channels
    dtype = next(tme_module.parameters()).dtype
    vae_scale, vae_shift = config.scale_factor, config.shift_factor

    tme_dict = split_channels_to_groups(
        val_control_input.to(device, dtype=dtype), active_channels, channel_groups,
    )
    mask_latent = val_vae_mask.to(device, dtype=dtype)
    mask_latent_scaled = (mask_latent - vae_shift) * vae_scale

    # Full forward with analysis
    tme_module.eval()
    fused, residuals, attn_maps = tme_module(
        mask_latent_scaled, tme_dict,
        return_residuals=True, return_attn_weights=True,
    )
    tme_module.train()

    # Generate H&E image for the full conditioning
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(20, device=device)
    latent_shape = (1, 16, config.image_size // 8, config.image_size // 8)
    latents = torch.randn(latent_shape, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    controlnet.eval()
    denoised = denoise(
        latents=latents,
        uni_embeds=val_uni_embeds.to(device, dtype=dtype),
        controlnet_input_latent=fused,
        scheduler=scheduler,
        controlnet_model=controlnet,
        pixcell_controlnet_model=base_model,
        guidance_scale=2.5,
        device=device,
    )
    controlnet.train()

    scaled_latents = (denoised.to(dtype) / vae_scale) + vae_shift
    gen_img = vae.decode(scaled_latents, return_dict=False)[0]
    gen_img = (gen_img / 2 + 0.5).clamp(0, 1)
    gen_np = (gen_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    # Mask image for panels
    mask_ch = val_control_input[0, 0].cpu().numpy()
    mask_rgb = np.stack([mask_ch] * 3, axis=-1)
    mask_rgb = (mask_rgb * 255).astype(np.uint8)

    # 1. Attention heatmaps
    save_attention_heatmap_figure(mask_rgb, gen_np, attn_maps, save_dir / "attention_heatmaps.png")

    # 2. Residual magnitudes
    save_residual_magnitude_figure(mask_rgb, gen_np, residuals, save_dir / "residual_magnitudes.png")

    # 3. Ablation grid (simplified: full only — full progressive requires multiple denoise passes)
    save_ablation_grid(
        [("All groups", gen_np)],
        save_dir / "ablation_grid.png",
    )
```

- [ ] **Step 1.2: Wire the hook into the checkpoint save**

In the training loop, after the `save_checkpoint_with_tme(...)` call inside the
`if global_step % config.save_model_steps == 0:` block (around line 230), add:

```python
                if accelerator.is_main_process and use_multi_group:
                    try:
                        generate_validation_visualizations(
                            tme_module=tme_module,
                            controlnet=controlnet,
                            base_model=base_model,
                            vae=vae,
                            train_diffusion=train_diffusion,
                            val_control_input=batch[2][:1],  # first sample from current batch
                            val_vae_mask=batch[3][:1],
                            val_uni_embeds=batch[1][:1],
                            config=config,
                            save_dir=os.path.join(config.work_dir, f"vis/step_{global_step}"),
                            device=accelerator.device,
                        )
                    except Exception as e:
                        logger.warning(f"Validation vis failed at step {global_step}: {e}")
```

- [ ] **Step 1.3: Verify import works**

Run: `conda run -n pixcell python -c "from train_scripts.train_controlnet_exp import train_controlnet_exp; print('OK')"`
Expected: `OK`

- [ ] **Step 1.4: Commit**

```bash
git add train_scripts/train_controlnet_exp.py
git commit -m "feat: add validation visualization hook at checkpoint intervals"
```

---

## Task 12: Integration Test

**Files:**
- Create: `tests/test_multi_group_tme_integration.py`

This test verifies the full pipeline from config → module construction → forward pass with group dropout → residual/attention returns, without requiring GPU or pretrained weights.

### Step 1: Write and run integration test

- [ ] **Step 1.1: Create integration test**

```python
import unittest
import torch
from diffusion.utils.misc import read_config
from diffusion.model.builder import build_model
from tools.channel_group_utils import split_channels_to_groups, apply_group_dropout


class TestMultiGroupIntegration(unittest.TestCase):
    def setUp(self):
        self.config = read_config("configs/config_controlnet_exp.py")
        group_specs = [
            dict(name=g["name"], n_channels=len(g["channels"]))
            for g in self.config.channel_groups
        ]
        self.module = build_model(
            "MultiGroupTMEModule", False, False,
            channel_groups=group_specs, base_ch=32,
        )

    def test_config_to_module_to_forward(self):
        B = 2
        control_input = torch.randn(B, 10, 256, 256)
        mask_latent = torch.randn(B, 16, 32, 32)

        tme_dict = split_channels_to_groups(
            control_input,
            self.config.data.active_channels,
            self.config.channel_groups,
        )

        with torch.no_grad():
            fused = self.module(mask_latent, tme_dict)
        self.assertEqual(fused.shape, (B, 16, 32, 32))

    def test_group_dropout_integration(self):
        B = 4
        control_input = torch.randn(B, 10, 256, 256)
        mask_latent = torch.randn(B, 16, 32, 32)

        tme_dict = split_channels_to_groups(
            control_input,
            self.config.data.active_channels,
            self.config.channel_groups,
        )
        active_per_sample = apply_group_dropout(
            [g["name"] for g in self.config.channel_groups],
            self.config.group_dropout_probs,
            batch_size=B,
        )
        # Zero out dropped groups per sample
        for b in range(B):
            for g in self.config.channel_groups:
                gname = g["name"]
                if gname not in active_per_sample[b] and gname in tme_dict:
                    tme_dict[gname][b] = 0.0

        with torch.no_grad():
            fused = self.module(mask_latent, tme_dict)
        self.assertEqual(fused.shape, (B, 16, 32, 32))

    def test_full_analysis_mode(self):
        B = 1
        control_input = torch.randn(B, 10, 256, 256)
        mask_latent = torch.randn(B, 16, 32, 32)

        tme_dict = split_channels_to_groups(
            control_input,
            self.config.data.active_channels,
            self.config.channel_groups,
        )

        with torch.no_grad():
            fused, residuals, attn_maps = self.module(
                mask_latent, tme_dict,
                return_residuals=True,
                return_attn_weights=True,
            )

        self.assertEqual(fused.shape, (B, 16, 32, 32))
        self.assertEqual(len(residuals), 4)
        self.assertEqual(len(attn_maps), 4)
        for name in ["cell_identity", "cell_state", "vasculature", "microenv"]:
            self.assertEqual(residuals[name].shape, (B, 16, 32, 32))
            self.assertEqual(attn_maps[name].shape, (B, 4, 1024, 1024))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 1.2: Run integration test**

Run: `conda run -n pixcell python -m unittest tests/test_multi_group_tme_integration.py -v`
Expected: All 3 tests PASS

- [ ] **Step 1.3: Commit**

```bash
git add tests/test_multi_group_tme_integration.py
git commit -m "test: add multi-group TME integration test covering config→module→forward pipeline"
```

---

## Task Summary

| Task | Description | New Files | Modified Files |
|------|-------------|-----------|---------------|
| 1 | Channel group utilities | `tools/channel_group_utils.py`, `tests/test_channel_group_utils.py` | — |
| 2 | CrossAttentionWithWeights | `diffusion/model/nets/cross_attention_with_weights.py`, `tests/test_cross_attention_with_weights.py` | — |
| 3 | MultiGroupTMEModule | `diffusion/model/nets/multi_group_tme.py`, `tests/test_multi_group_tme.py` | `diffusion/model/builder.py` |
| 4 | Config update | — | `configs/config_controlnet_exp.py` |
| 5 | Training utilities | — | `train_scripts/training_utils.py` |
| 6 | Training loop (multi-group) | — | `train_scripts/train_controlnet_exp.py` |
| 7 | Inference script | — | `stage3_inference.py` |
| 8 | Attention heatmap vis | `tools/visualize_group_attention.py` | — |
| 9 | Residual magnitude vis | `tools/visualize_group_residuals.py` | — |
| 10 | Ablation grid vis | `tools/visualize_ablation_grid.py` | — |
| 11 | Training-loop validation vis hook | — | `train_scripts/train_controlnet_exp.py` |
| 12 | Integration test | `tests/test_multi_group_tme_integration.py` | — |
