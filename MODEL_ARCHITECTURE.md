# PixCell-ControlNet: Model Architecture

## Overview

PixCell-ControlNet maps tumor microenvironment (TME) simulation outputs to realistic H&E histology images using a conditional latent diffusion model. The key innovation is the **Multi-Group TME Module**, which encodes disentangled spatial biology channels (cell identity, cell state, vasculature, microenvironment) into additive conditioning residuals.

Two latent streams are active throughout the system:

1. **Denoising latent stream**: the current diffusion latent passed as `hidden_states`.
   - Training: `x_t`, obtained by adding noise to the VAE latent of the target H&E image.
   - Inference: starts from random Gaussian noise `x_T`.
2. **Conditioning latent stream**: the latent passed as `conditioning` to ControlNet.
   - Official PixCell ControlNet: VAE-encoded cell-mask latent only.
   - This repo (`zero_mask_latent=True`): mask latent is **zeroed** before the TME module; conditioning = TME residuals only. Cell-type/state channels fully encode cell layout, making the mask VAE latent redundant.

> **Why `zero_mask_latent=True`?**
> With the mask latent active, the pretrained ControlNet produces plausible H&E from mask_latent alone (bypass path), so `∂loss/∂proj ≈ 0` — the TME proj layers never escape near-zero (observed: `proj_wmax ≈ 4e-4` after 30 epochs at `tme_lr=1e-5`).
> Zeroing the mask latent closes the bypass path. Since `controlnet_blocks` (ControlNet output projections) hold non-zero pretrained weights, `∂loss/∂fused_cond ≠ 0` from step 1, and `∂loss/∂proj` is immediately non-zero, breaking the starvation.

---

## System-Level Architecture

### Training

```
H&E Image [256, 256, 3]                     Cell Mask [256, 256, 1]
        │                                            │
        ├──► SD3.5 VAE Encoder (FROZEN)              └──► SD3.5 VAE Encoder (FROZEN)
        │         │                                            │
        │         ▼                                            ▼
        │   clean H&E latent x_0                        mask latent [B, 16, 32, 32]
        │   [B, 16, 32, 32]                                      │
        │         │                                              │
        │         └──► Forward Diffusion                         │
        │               t ~ Uniform(0, 1000)                     │
        │               ε ~ N(0, I)                              │
        │               x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε          │
        │                              │                         │
        ▼                              │                         ▼
UNI-2h Embedder (FROZEN)               │                 Multi-Group TME Module
[B, 1536]                              │                 + raw TME channels [B, 9, 256, 256]
        │                              │                         │
        │                              ▼                         ▼
        │                      noisy latent x_t           conditioning latent
        │                      [B, 16, 32, 32]           [B, 16, 32, 32]
        │                              │                         │
        │                              ├──────────────┬──────────┘
        │                              │              │
        │                              ▼              ▼
        │                      PixCell ControlNet (TRAINABLE, 27 blocks)
        │                      input = x_t + conditioning + UNI + timestep
        │                              │
        │                              ▼
        │                      27 residuals [B, N, 1152]
        │                              │
        │                              ▼
        └──────────────────────► Base PixArt-256 Transformer (FROZEN, 28 blocks)
                                 input = same noisy latent x_t + UNI + timestep
                                         + ControlNet residuals
                                          │
                                          ▼
                                 model output [B, 32, 32, 32]
                                 = ε + variance
                                          │
                                          ▼
                                 Diffusion Loss: ||ε_pred − ε||²
                                 (first 16 output channels used as ε_pred)
```

At inference, the denoising latent stream changes but the conditioning path stays the same:

### Inference

```
Reference H&E [256, 256, 3]             Cell Mask [256, 256, 1]
        │                                        │
        └──► UNI-2h Embedder                     └──► SD3.5 VAE Encoder
             [B, 1536]                                      │
                  │                                          ▼
                  │                                  mask latent [B, 16, 32, 32]
                  │                                          │
                  │                                          ▼
                  │                                  Multi-Group TME Module
                  │                                  + raw TME channels [B, 9, 256, 256]
                  │                                          │
                  │                                          ▼
                  │                                  conditioning latent
                  │                                  [B, 16, 32, 32]
                  │                                          │
                  │                                          │
Random Gaussian latent x_T [B, 16, 32, 32]                  │
                  │                                          │
                  ├──────────────────────────────┬───────────┘
                  │                              │
                  ▼                              ▼
          Denoising loop (20 scheduler steps) with:
            - current latent x_t
            - ControlNet residuals from conditioning + UNI
            - Base transformer cross-attention to UNI
                  │
                  ▼
          final latent x_0_hat [B, 16, 32, 32]
                  │
                  ▼
          SD3.5 VAE Decoder
                  │
                  ▼
          Generated H&E [256, 256, 3]
```

For official PixCell ControlNet, `TMEMOD` is absent and `conditioning latent = mask latent`.

---

## Component 1 — Base PixArt-256 Transformer (Frozen)

```
Input latent x_t  [B, 16, 32, 32]
        │
        ▼
  Patch Embedding ──► 1024 patches [B, 1024, 1152]
  (patch_size=2)       + sinusoidal pos_embed
        │
        │   Timestep t ──► Timestep Embedder ──► [B, 1152]
        │                                              │
        │   UNI-2h emb ──► Caption Projector ──► [B, 1, 1152]
        │   [B, 1536]        (CFG dropout 15%)         │
        │                                              │
        ├──────────────────────────────────────────────┤
        │
        ▼
  ┌─────────────────────────────────┐
  │  PixArtBlock × 28               │
  │  ─────────────────────────────  │
  │  AdaLayerNorm (timestep mod)    │
  │  Self-Attention  [B, 1024, 1152]│
  │  AdaLayerNorm                   │
  │  Cross-Attention (Q=x, KV=cap)  │
  │  AdaLayerNorm                   │
  │  MLP (ratio=4.0)                │
  │  + ControlNet residual (block i)│   ◄── from ControlNet
  └─────────────────────────────────┘
        │
        ▼
  Final Layer + Unpatchify
        │
        ▼
  [B, 32, 32, 32]  (pred_sigma=True: noise + variance)
        │
        └──► first 16 channels used as ε prediction during sampling / loss

Key parameters:
  hidden_size=1152   depth=28   num_heads=16
  patch_size=2       in_channels=16   mlp_ratio=4.0
  caption_channels=1536
```

---

## Component 2 — PixCell ControlNet (Trainable)

```
Conditioning latent [B, 16, 32, 32]
(official PixCell: mask latent only;
 this repo [zero_mask_latent=True]: zeros + TME residuals = TME residuals only)
        │
        ▼
  Cond Embedder  ──►  [B, 1024, 1152]
  (ZERO-INIT)                │
                             │  +  x_embedder(noisy latent x_t)
                             ▼
                      fused patches [B, 1024, 1152]
                             │
                    ┌────────────────────┐
                    │  PixArtBlock × 27  │
                    │  (TRAINABLE copy   │
                    │   of base blocks)  │
                    └────────┬───────────┘
                             │
                    controlnet_blocks × 27
                    Linear(1152 → 1152)
                    (ZERO-INITIALIZED)
                             │
                             ▼
                    27 residuals [B, 1024, 1152]
                    (added to base model blocks 1–27)

Zero initialization ensures ControlNet starts
as identity (no effect), then learns incrementally.

With `zero_mask_latent=True`:
- `cond_pos_embed(0) ≈ 0` initially (pretrained to process cell-mask distribution,
   outputs near-zero when given zeros)
- `controlnet_blocks` weights are non-zero (loaded from pretrained checkpoint)
   → gradient highway from loss → `fused_cond` → TME proj is open from step 1
```

---

## Component 3 — Multi-Group TME Module (Trainable)

The core architectural innovation: each biology channel group gets its own encoder and cross-attention module, producing additive residuals over the mask latent.

With `zero_mask_latent=True`, the mask latent is zeroed before this module, so `fused = 0 + Σ(Δ_g)`. Q tokens derive from zeros → uniform attention weights → cross-attention output ≈ mean(V) × proj. Since proj is zero-init, residuals start near zero but grad is non-zero from the first step.

```
Cell mask latent  [B, 16, 32, 32]
  (zeroed when zero_mask_latent=True)
        │
        ▼
  LayerNorm  ──►  Q tokens  [B, 1024, 16]
        │
        │    TME Channel Groups (per group, independently):
        │
        │    ┌─────────────────────────────────────────────────────────┐
        │    │  GROUP: cell_types   [B, 3, 256, 256]                │
        │    │  (cell_type_healthy, cell_type_cancer, cell_type_immune) │
        │    │          │                                               │
        │    │          ▼                                               │
        │    │    TMEEncoder (CNN)  ──►  [B, 16, 32, 32]               │
        │    │          │                                               │
        │    │          ▼                                               │
        │    │    K, V tokens  [B, 1024, 16]                           │
        │    │          │                                               │
        │    │          ▼                                               │
        │    │  CrossAttentionWithWeights                               │
        │    │    Q (mask) × K,V (group)  ──►  delta [B, 1024, 16]    │
        │    │                                                          │
        │    │  Δ_cell_types  [B, 16, 32, 32]                       │
        │    └─────────────────────────────────────────────────────────┘
        │
        │    ┌─────────────────────────────────────────────────────────┐
        │    │  GROUP: cell_state   [B, 3, 256, 256]                   │
        │    │  (prolif, nonprolif, dead)                               │
        │    │          ▼                                               │
        │    │    TMEEncoder  ──►  CrossAttention  ──►  Δ_cell_state   │
        │    └─────────────────────────────────────────────────────────┘
        │
        │    ┌────────────────────────────────────────────────────────┐
        │    │  GROUP: vasculature  [B, 1, 256, 256]  (CD31)          │
        │    │          ▼                                              │
        │    │    TMEEncoder  ──►  CrossAttention  ──►  Δ_vasculature │
        │    └────────────────────────────────────────────────────────┘
        │
        │    ┌────────────────────────────────────────────────────────┐
        │    │  GROUP: microenv  [B, 2, 256, 256]  (oxygen, glucose)  │
        │    │          ▼                                              │
        │    │    TMEEncoder  ──►  CrossAttention  ──►  Δ_microenv    │
        │    └────────────────────────────────────────────────────────┘
        │
        ▼
  Fusion:
  mask_latent  +  Σ(Δ_g for active groups g)
        │
        ▼
  Fused conditioning  [B, 16, 32, 32]
  ──► ControlNet conditioning input

Group dropout during training (prevents collapse):
  cell_types: 10%   cell_state: 10%
  vasculature:   15%   microenv:   20%
```

---

## Component 4 — TME Encoder (CNN)

One TMEEncoder instance per channel group.

```
Input: [B, n_ch, 256, 256]
        │
        ▼
  Stem: Conv2d(n_ch → 32) + GroupNorm + SiLU
        │
        ▼
  Stage 1: DownBlock(32 → 64, stride=2)        256 → 128
           ResBlock(64)
           Conv + GroupNorm + SiLU + Conv + GroupNorm + skip
        │
        ▼
  Stage 2: DownBlock(64 → 128, stride=2)       128 → 64
           ResBlock(128)
        │
        ▼
  Stage 3: DownBlock(128 → 16, stride=2)       64 → 32
           ResBlock(16)
        │
        ▼
  Output: [B, 16, 32, 32]  (matches VAE latent spatial dims)

base_ch=32   latent_ch=16   Kaiming initialization
```

---

## Component 5 — Frozen Encoders / Latent Sources

```
H&E Image [256, 256, 3]
    ├──► SD3.5 VAE Encoder ──► [B, 16, 32, 32]   (clean latent x_0 for training)
    └──► UNI-2h Embedder   ──► [B, 1536]         (style / morphology reference)

Cell Mask [256, 256, 1]
    └──► SD3.5 VAE Encoder ──► [B, 16, 32, 32]   (ControlNet conditioning base)
```

---

## Training & Inference Modes

### Conditioning Signals

| Signal | Dims | Role | During Inference |
|--------|------|------|-----------------|
| Denoising latent | [B, 16, 32, 32] | Current diffusion state passed as `hidden_states` | Starts from random Gaussian `x_T` |
| Cell mask VAE latent | [B, 16, 32, 32] | Spatial cell layout → ControlNet base | **Zeroed** when `zero_mask_latent=True` (cell type/state channels make it redundant; zeroing closes the bypass path that starved TME gradients) |
| TME channels | [B, 9, 256, 256] | Biology groups → TME Module residuals | Required in this repo, absent in official PixCell ControlNet |
| UNI-2h embedding | [B, 1536] | H&E style / morphology → transformer cross-attn | Usually from a reference H&E image; can be zeroed for unconditional / TME-only runs |
| Timestep | scalar | Diffusion schedule modulation | Required |

### Noise Addition (Training)

Noise is added once per batch, immediately after VAE encoding, before the forward pass:

```
Step 1  t ~ Uniform(0, 1000)          one random timestep per image
Step 2  ε ~ N(0, I)                   sample Gaussian noise, same shape as x_0
Step 3  x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε   forward diffusion (q_sample)
        ↑ this is the only point where noise is injected during training

Step 4  model predicts ε from x_t, t, and all conditioning signals
Step 5  loss = ‖ε_pred − ε‖²
```

Linear beta schedule: β ∈ [0.0001, 0.02], T = 1000.
At small t: x_t ≈ x_0 (mostly clean). At large t: x_t ≈ ε (pure noise).

---

### Sampling Path (Inference)

Inference does **not** start from a reference H&E VAE latent. It starts from random noise in the same latent space:

```
Step 1  x_T ~ N(0, I)                    sample random latent in SD3 latent space
Step 2  build conditioning latent        mask_latent (official) or fused_cond (this repo)
Step 3  extract UNI embedding            from reference H&E, if style conditioning is used
Step 4  for t = T ... 1:                predict ε from current latent + conditioning + UNI
Step 5  scheduler update                 x_t → x_{t-1}
Step 6  decode final latent             SD3.5 VAE Decoder(x_0_hat)
```

So the reference H&E contributes UNI features for style, not the initial denoising latent.

---

### Classifier-Free Guidance

```
Training (cfg_dropout_prob=0.15):
  15% of steps: UNI embedding ← zeros
  → base model learns TME-only generation

Inference (cfg_scale=2.5, 20 DDPM steps):
  x_t → model(x_t, cond_UNI)   ─► ε_cond
  x_t → model(x_t, null_UNI)   ─► ε_uncond
  ε_guided = ε_uncond + scale × (ε_cond − ε_uncond)
```

### Group Ablation at Inference

```bash
# Use only cell identity + vasculature
python stage3_inference.py --active-groups cell_types vasculature

# Drop microenvironment channels
python stage3_inference.py --drop-groups microenv
```

Inactive groups produce identity residuals (Δ = 0), isolating each group's causal contribution to the generated image.

---

## Parameter Summary

| Component | Parameters | Frozen |
|-----------|-----------|--------|
| Base PixArt-256 | ~600M | Yes |
| SD3.5 VAE | ~84M | Yes |
| UNI-2h Embedder | ~307M | Yes |
| PixCell ControlNet | ~600M | No (lr=5e-6) |
| Multi-Group TME Module | ~4M | No (lr=1e-5) |

---

## End-to-End Inference Flow

```
Control stack                                  Reference H&E (optional)
[cell_mask + TME channels]                     [256, 256, 3]
        │                                              │
        ├──► cell_mask ─► SD3.5 VAE Encoder ─► mask_latent [B, 16, 32, 32]
        │                                              │
        ├──► TME groups ───────────────────────────────┘
        │                      │
        │                      ▼
        │              Multi-Group TME Module
        │              fused_cond [B, 16, 32, 32]
        │                      │
        ▼                      ▼
x_T ~ N(0,I) ──► Denoising Loop with Base Transformer + ControlNet ◄── UNI-2h embedding [B, 1536]
                     │
                     │  CFG at each step
                     ▼
              x_0_hat [B, 16, 32, 32]
                     │
               SD3.5 VAE Decoder
                     │
                     ▼
          Generated H&E [256, 256, 3]
```
