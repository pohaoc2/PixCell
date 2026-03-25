# Issue: TME Proj Gradient Starvation

**Status:** Identified, not yet fixed  
**Symptom:** TME group residuals ~1e-5 after 30 epochs; ablation grid shows no visible difference between groups.

## Root Cause

Each TME group's cross-attention ends with a zero-initialized output projection:

```
Attention(Q, K, V)  →  context  →  proj (zero-init)  →  Δ_group
```

`proj` is the only zero-init layer. After 4890 steps at `tme_lr=1e-5`, its weights are still ~1e-6.  
The encoder CNN and Q/K/V weights are healthy (0.1–0.5 range) — they are learning.  
The bottleneck is proj alone.

**Why it stalls:** the pretrained ControlNet generates acceptable H&E from `mask_latent` without  
any TME contribution. With Δ_group ≈ 0, the loss gradient through proj is proportional to proj  
itself → tiny weights → tiny gradient → tiny update. Self-reinforcing loop.

## Evidence

```
# checkpoint: epoch=30, step=4890
groups.*.cross_attn.proj.weight   max=1e-6   mean=1e-6   ← stuck
groups.*.cross_attn.q_linear.weight  max=0.25  mean=0.12  ← healthy
groups.*.encoder.stem.0.weight    max=0.28   mean=0.07   ← healthy
```

## Fix

In `configs/config_controlnet_exp.py`, increase `tme_lr` from `1e-5` to `3e-4` (already applied).  
Resume training from the epoch-30 checkpoint; proj weights should reach ~1e-3 within ~5k steps,  
making TME group contributions visible in residual maps and ablation grids.

```python
# configs/config_controlnet_exp.py
tme_lr = 3e-4   # was 1e-5; increased 30× to escape zero-init regime
```
