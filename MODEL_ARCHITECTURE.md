# PixCell-ControlNet: Model Architecture

## Overview

PixCell-ControlNet maps tumor microenvironment (TME) simulation outputs to realistic H&E histology images using a conditional latent diffusion model. The key innovation is the **Multi-Group TME Module**, which encodes disentangled spatial biology channels (cell identity, cell state, vasculature, microenvironment) into additive conditioning residuals.

---

## System-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING INPUTS                                    │
├────────────────────┬────────────────────────┬──────────────────────────────┤
│   H&E Image        │    Cell Mask           │    TME Channels              │
│  [256, 256, 3]     │   [256, 256, 1]        │   [256, 256, 9]              │
└────────┬───────────┴──────────┬─────────────┴──────────┬───────────────────┘
         │                      │                         │
         ▼                      ▼                         ▼
  ┌──────────────┐      ┌──────────────┐        ┌────────────────────┐
  │  SD3.5 VAE   │      │  SD3.5 VAE   │        │  Multi-Group TME   │
  │   Encoder    │      │   Encoder    │        │     Module         │
  │  (FROZEN)    │      │  (FROZEN)    │        │  (TRAINABLE)       │
  └──────┬───────┘      └──────┬───────┘        └────────┬───────────┘
         │                     │                          │
         ▼                     ▼                          │
  [B, 4, 32, 32]       [B, 16, 32, 32]                   │
  H&E latent x_0       mask latent ──────────────────────►│
         │                     │                          │
         │  ┌──────────────────┘                          │
         │  │  t ~ Uniform(0, 1000)                       │
         │  │  ε ~ N(0, I)                                │
         │  │  x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε          │
         │  │  (noise added here, once per batch)         │
         │  └──────────────────┐                          │
         │                     ▼                          ▼
         │             ┌───────────────┐        [B, 16, 32, 32]
         │             │  ControlNet   │        fused conditioning
         │             │  (TRAINABLE)  │◄───────────────────────
         │             │  27 blocks    │
         │             └──────┬────────┘
         │                    │
         │               27 residuals
         │               [B, N, 1152]
         ▼                    ▼
  ┌─────────────────────────────────────┐
  │     Base PixArt-256 Transformer     │   ◄── Timestep t + UNI-2h embedding
  │         (FROZEN, 28 blocks)         │
  │     input: noisy x_t [B, 4, 32, 32]│
  └────────────────┬────────────────────┘
                   │
                   ▼
          ε_pred  (noise prediction)
                   │
           diffusion loss
          ║ε_pred − ε║²
```

---

## Component 1 — Base PixArt-256 Transformer (Frozen)

```
Input latent x_t  [B, 4, 32, 32]
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
  [B, 8, 32, 32]  (noise prediction)

Key parameters:
  hidden_size=1152   depth=28   num_heads=16
  patch_size=2       in_channels=4   mlp_ratio=4.0
  caption_channels=1536
```

---

## Component 2 — PixCell ControlNet (Trainable)

```
Cell mask VAE latent [B, 16, 32, 32]
        │
        ▼
  Cond Embedder  ──►  [B, 1024, 1152]
  (ZERO-INIT)                │
                             │  +  x_embedder(H&E latent)
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
```

---

## Component 3 — Multi-Group TME Module (Trainable)

The core architectural innovation: each biology channel group gets its own encoder and cross-attention module, producing additive residuals over the mask latent.

```
Cell mask latent  [B, 16, 32, 32]
        │
        ▼
  LayerNorm  ──►  Q tokens  [B, 1024, 16]
        │
        │    TME Channel Groups (per group, independently):
        │
        │    ┌─────────────────────────────────────────────────────────┐
        │    │  GROUP: cell_identity   [B, 3, 256, 256]                │
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
        │    │  Δ_cell_identity  [B, 16, 32, 32]                       │
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
  cell_identity: 10%   cell_state: 10%
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

## Component 5 — Feature Encoders (Frozen, Pre-computed)

```
H&E Image [256, 256, 3]
    ├──► SD3.5 VAE Encoder ──► [B, 4, 32, 32]    (reparameterized at train time)
    └──► UNI-2h Embedder   ──► [B, 1536]         (cached as .npy files)

Cell Mask [256, 256, 1]
    └──► SD3.5 VAE Encoder ──► [B, 16, 32, 32]   (cached as .npy files)
```

---

## Training & Inference Modes

### Conditioning Signals

| Signal | Dims | Role | During Inference |
|--------|------|------|-----------------|
| Cell mask VAE latent | [B, 16, 32, 32] | Spatial cell layout → ControlNet | Required |
| TME channels | [B, 9, 256, 256] | Biology groups → TME Module | Required |
| UNI-2h embedding | [B, 1536] | H&E style → Base model cross-attn | Optional (zero = TME-only) |
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
python stage3_inference.py --active-groups cell_identity vasculature

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
Simulation TME channels                Reference H&E (optional)
[cell_identity, cell_state,            [256, 256, 3]
 vasculature, microenv]                       │
        │                                     ▼
        │                              UNI-2h Embedder
        │                              [1536] style vec
        │                                     │
        ▼                                     │
Multi-Group TME Module                        │
[B, 16, 32, 32] fused                         │
        │                                     │
        ▼                                     │
ControlNet conditioning ──────────────────────┤
                                              │
x_T ~ N(0,I) ──► Denoising Loop (20 steps) ◄─┘
                     │
                     │  CFG at each step
                     ▼
               x_0 [B, 4, 32, 32]
                     │
               SD3.5 VAE Decoder
                     │
                     ▼
          Generated H&E [256, 256, 3]
```
