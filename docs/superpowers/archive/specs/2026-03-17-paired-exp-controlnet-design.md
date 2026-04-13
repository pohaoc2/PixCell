# Paired Experimental ControlNet — Design Spec
**Date**: 2026-03-17
**Branch**: chiu/multi-channel-controlnet

---

## Background

PixCellControlNet currently trains on **unpaired** simulation TME channels + randomly sampled real H&E UNI embeddings. We now have **paired** experimental H&E + CODEX multiplex protein imaging data, enabling the model to learn the true biological correlation between H&E appearance and TME state.

The primary use case is **simulation validation**: map simulation TME outputs into the experimental H&E domain, then compare generated vs. experimental H&E in UNI feature space to quantify similarity.

---

## Goals

1. Train on paired experimental (H&E + CODEX-derived TME channels) to learn H&E ↔ TME correlations
2. Support dual inference modes: style-conditioned (reference H&E) and TME-only (null style)
3. At inference, accept simulation TME channels as a drop-in replacement for CODEX channels
4. Provide a validation pipeline: sim TME → generated H&E → UNI features → similarity to exp target

---

## Data

### Paired Experimental Data

- **H&E**: standard brightfield histology tile
- **CODEX panel** (18 protein markers): Hoechst, CD31, CD45, CD68, CD4, FOXP3, CD8a, CD45RO, CD20, PD-L1, CD3e, CD163, E-cadherin, PD-1, Ki67, Pan-CK, SMA, AF1/Argo550

### Registration Quality

| Channel group | Source | Registration to H&E |
|---|---|---|
| cell_mask | CODEX Hoechst → nuclear segmentation | Pixel-perfect |
| cell_type (healthy/cancer/immune) | CODEX marker classification | Pixel-perfect |
| cell_state (prolif/nonprolif/dead) | CODEX marker classification | Pixel-perfect |
| vasculature | CD31 channel | Approximate |
| oxygen | Ki67 + distance-to-vessel proxy | Approximate |
| glucose | Ki67 + metabolic model | Approximate |

### Channel Representation

All channels are preprocessed to the **same image format as simulation channels** (per-channel PNG/NPY). This makes sim channels a drop-in replacement at inference.

**Cell type and cell state use one-hot encoding** (one binary channel per class) to avoid false ordinal relationships:

| Channel | Classes | Type |
|---|---|---|
| `cell_mask` | background/cell | binary |
| `cell_type_healthy` | present/absent | binary |
| `cell_type_cancer` | present/absent | binary |
| `cell_type_immune` | present/absent | binary |
| `cell_state_prolif` | present/absent | binary |
| `cell_state_nonprolif` | present/absent | binary |
| `cell_state_dead` | present/absent | binary |
| `vasculature` | intensity from CD31 | float [0,1] |
| `oxygen` | proxy from Ki67 + distance | float [0,1] |
| `glucose` | proxy from Ki67 + metabolic model | float [0,1] |

`cell_mask` goes through the VAE separately (`vae_mask`). The remaining 9 channels are `tme_channels` fed into `TMEConditioningModule`.

---

## Architecture

No architectural changes to `TMEConditioningModule` or `PixCellControlNet`. All new behaviour lives in the dataset and training loop.

### Data Flow

```
TRAINING (paired exp)
─────────────────────────────────────────────────────────────────────
  Experimental H&E tile ──► VAE encode ──► vae_feat  [16, 32, 32]  (target)
                        └──► UNI embed ──► ssl_feat   [1, 1, 1152]  (style y)
                                           │
                                    CFG dropout (p=0.15)
                                    → null embed (zeros)  15% of steps

  CODEX channels (preprocessed):
    cell_mask  ──► VAE encode ──► vae_mask             [16, 32, 32]
    cell_type  ┐                                              ↓
    cell_state ├──► ctrl_tensor  [9, 256, 256] ──► TMEEncoder ──► cross-attn
    vasculature│    × channel_weights                          ↕
    O2         │    [1,1,1,1,1,1, 0.5,0.5,0.5]         vae_mask fused
    glucose    ┘                                              ↓
                                                        ControlNet ──► base_model ──► MSE loss

INFERENCE (validation)
─────────────────────────────────────────────────────────────────────
  Sim TME channels ──► ctrl_tensor (same format, no weight attenuation)
  [Mode A] Reference H&E UNI embed ──► y   (style-conditioned)
  [Mode B] zeros(1,1,1,1536)        ──► y   (TME-only)
                                          ↓
                                    Denoising loop
                                          ↓
                                    Generated H&E ──► UNI extractor ──► sim_feat
                                                      compare vs exp_target_feat
```

---

## Training Design

### CFG Dropout

With probability `cfg_dropout_prob=0.15`, replace the UNI embedding `y` with zeros before the forward pass. This enables TME-only inference without needing a separate unconditional model.

```python
if torch.rand(1).item() < config.cfg_dropout_prob:
    y = torch.zeros_like(y)
```

### Channel Reliability Weighting

Approximate channels (vasculature, oxygen, glucose) are attenuated before `TMEConditioningModule` to reflect lower biological reliability of CODEX-derived proxies:

```python
channel_weights = torch.tensor(
    config.channel_reliability_weights,  # [1,1,1,1,1,1, 0.5,0.5,0.5]
    device=tme_channels.device, dtype=tme_channels.dtype
).view(1, -1, 1, 1)
tme_channels = tme_channels * channel_weights
```

Weights are **not applied at inference** with sim channels — simulation outputs are clean, full-range, and internally consistent.

### Fine-tuning

Training resumes from the existing sim ControlNet checkpoint (`resume_from_checkpoint` in config). The sim pretraining provides a strong prior on TME → H&E spatial layout; paired exp fine-tuning refines the appearance correlation.

---

## Inference Modes

| Mode | `uni_embeds` | Use case |
|---|---|---|
| Style-conditioned | UNI embedding from reference H&E patch | Specific staining style |
| TME-only | `torch.zeros(1, 1, 1, 1536)` | Validation pipeline |

Helper added to `inference_controlnet.py`:
```python
def null_uni_embed(device, dtype):
    return torch.zeros(1, 1, 1, 1536, device=device, dtype=dtype)
```

CFG `guidance_scale` controls TME adherence strength in both modes.

---

## Validation Pipeline

```
for each sim snapshot:
    sim_channels → ctrl_tensor [9, 256, 256]   (cell_type/state one-hot + vasculature/O2/glucose)
    null_uni_embed → y
    denoising loop (20–50 steps, guidance_scale=2.5)
    → generated H&E [3, 256, 256]
    → UNI feature extractor → gen_feat [1152]

aggregate over N tiles:
    per-tile:       cosine_sim(gen_feat, exp_target_feat)
    population:     FID(gen_feats, exp_target_feats)
    optional:       MMD(gen_feats, exp_target_feats)
```

`exp_target_feat` vectors are precomputed UNI embeddings from experimental H&E (already in `features/`).

---

## Files

| File | Action |
|---|---|
| `diffusion/data/datasets/paired_exp_controlnet_dataset.py` | New — `PairedExpControlNetData` dataset |
| `train_scripts/train_controlnet_exp.py` | New — training loop with CFG dropout + channel weighting |
| `configs/config_controlnet_exp.py` | New — config for paired exp training |
| `tools/validate_sim_to_exp.py` | New — validation pipeline script |
| `train_scripts/inference_controlnet.py` | Extend — add `null_uni_embed()` helper |
| `diffusion/data/datasets/__init__.py` | Extend — register `PairedExpControlNetData` |

---

## Open Questions

- Exact weight values for approximate channels (default 0.5, tune empirically)
- CFG dropout probability (default 0.15, standard from classifier-free guidance literature)
- Number of fine-tuning epochs / learning rate relative to sim pretraining
