# Why TME `grad_norm` Hits `Infinity`

Date: 2026-04-27
Affected run: `checkpoints/production_retrain/full_seed_42/train_log.jsonl`
Reference implementation path: `diffusion/model/nets/multi_group_tme.py`

## 1. Symptom

- `grad_norm_tme` is `Infinity` from the first logged step onward.
- Loss still looks healthy (`~0.10` to `0.16`).
- Direct probe showed TME gradients around `1e31`, while ControlNet gradients stayed ordinary (`~1e-1`).
- Running in fp32 does not change the failure, so this is not a bf16-only issue.

## 2. Root Cause

The TME gradient explodes because the training-time cross-attention is mixing tokens across the entire batch instead of keeping each sample isolated.

The bug is in `diffusion/model/nets/PixArt_blocks.py`, inside `MultiHeadCrossAttention.forward`:

```python
B, N, C = x.shape

q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
```

`x` and `cond` enter as `[B, N, C]`, but both tensors are reshaped to batch size `1`. That collapses the true batch axis into the token axis:

- expected attention layout: `[B, N, heads, head_dim]`
- actual training layout: `[1, B*N, heads, head_dim]`

As a result, every sample's query tokens attend to every other sample's TME tokens in the minibatch.

## 3. Why That Produces Huge TME Gradients

For the multi-group TME path, each sample contributes `1024` spatial tokens at latent resolution (`32 x 32`). With the current bug, the attention softmax is computed over `B * 1024` keys instead of `1024` keys per sample.

That changes the backward pass in two harmful ways:

1. The TME residual for sample `i` depends on unrelated TME channels from samples `j != i`.
2. The gradient on TME-side projections and norms accumulates across all cross-sample interactions instead of only within one sample.

This creates a dense cross-sample coupling term in every group block. In practice, that pushes some TME parameter gradients to extremely large values (`~1e31` in the observed probe), and the full-module norm overflows fp32 when `clip_grad_norm_` sums squares over the whole TME parameter set.

That is why the logged norm becomes `Infinity`.

## 4. Cheap Discriminating Proof

If attention were batch-correct, changing sample 1 should not affect sample 0 at all.

Probe:

```python
import torch
from diffusion.model.nets.PixArt_blocks import MultiHeadCrossAttention

torch.manual_seed(0)
attn = MultiHeadCrossAttention(d_model=16, num_heads=4)
attn.eval()

x = torch.randn(2, 8, 16)
cond = torch.randn(2, 8, 16)

out1 = attn(x, cond)
cond2 = cond.clone()
cond2[1].mul_(1000)
out2 = attn(x, cond2)
```

Observed result:

```python
{
    'sample0_max_diff': 1103.10400390625,
    'sample1_max_diff': 988.4188842773438,
}
```

`sample0_max_diff` should be approximately zero. It is not. That confirms the training attention path leaks information across batch elements.

## 5. Important Supporting Clue

`diffusion/model/nets/cross_attention_with_weights.py` has a separate code path used when `return_attn_weights=True`, and that path keeps the batch dimension intact:

```python
q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
```

So the analysis path is batch-correct, while the normal training path delegates to the buggy parent implementation. That mismatch explains why the bug is easy to miss by inspection unless the training path itself is probed.

## 6. Why the Loss Still Goes Down

The exploding part is confined to the TME branch.

During training:

```python
grad_norm_tme = accelerator.clip_grad_norm_(tme_module.parameters(), config.gradient_clip)
```

Once the total TME norm is `Infinity`, gradient clipping drives the effective clip coefficient to zero, so the TME update is effectively nulled out for that step. ControlNet gradients remain finite, so the main model can still reduce the diffusion loss even while the TME branch is numerically broken.

So the training run looks superficially healthy, but the TME module is effectively frozen.

## 7. What This Rules Out

- This is not primarily a bf16 overflow issue.
- This is not caused by the MSE loss itself.
- `LayerNorm` and `GroupNorm` may amplify the instability, but they are not the root cause identified by the discriminating probe.
- The small-normal init on `cross_attn.proj` can change how strongly the bug manifests, but it is not the structural failure.

## 8. Minimal Fix Direction

The fix is to preserve the real batch dimension in `MultiHeadCrossAttention.forward`.

The problematic lines should behave like this instead:

```python
q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
kv = self.kv_linear(cond).view(B, -1, 2, self.num_heads, self.head_dim)
```

After that change, each sample attends only to its own TME tokens, which removes the cross-sample coupling that is currently blowing up TME gradients.

## 9. Bottom Line

The TME gradients explode because the training attention implementation collapses batch and sequence together, turning per-sample conditioning into one large cross-sample attention problem. That invalidates isolation between examples, amplifies TME-side gradient accumulation, and drives the module-level gradient norm beyond fp32 range, where `clip_grad_norm_` reports `Infinity`.
