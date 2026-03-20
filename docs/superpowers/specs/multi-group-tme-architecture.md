# Multi-Group TME Architecture

> **Status:** Draft  
> **Created:** 2026-03-19  
> **Goal:** Disentangled, independently droppable TME channel groups with interpretable per-group contribution analysis.

---

## 1. Motivation

The current `TMEConditioningModule` fuses all 9 TME channels through a single ResNet
CNN encoder + single cross-attention with the VAE mask latent. Three problems:

1. **Binary/continuous mixing.** One-hot maps (cell type, cell state) and continuous
   fields (O₂, glucose) have fundamentally different statistics. A shared CNN with
   GroupNorm is dominated by whichever channel type has larger magnitudes.

2. **O₂/glucose are indirect.** They affect cell behavior over time (necrosis,
   stromal changes) but are not visible in a single H&E snapshot. Their influence
   is indirect — through cell density and tissue remodeling — not pixel-level staining.

3. **No disentangled control.** All 9 channels are concatenated into one tensor.
   There is no way to independently include, exclude, or modulate a subset at inference.

## 2. Design Goals

| Priority | Goal |
|----------|------|
| **Primary** | Disentangled control — independently adjust O₂/glucose/cell types at inference |
| **Secondary** | Interpretability — per-group attention heatmaps and residual magnitude maps |
| **Constraint** | Base PixCell transformer (UNI-conditioned) is frozen; ControlNet + TME module retrained from scratch |
| **Constraint** | Trainable on ~10k paired H&E + CODEX samples |

### Use cases enabled

- **Counterfactual sweeps:** Hold cell mask + cell types fixed, vary O₂ levels, observe H&E changes.
- **Progressive composition:** Start from mask only, layer on groups one at a time, see each contribution.
- **Channel dropout for missing data:** Run with only available groups (e.g., mask + cell identity when O₂/glucose are unavailable).
- **Required baseline:** Cell mask + H&E (UNI embedding) are always present. All TME groups are optional.

## 3. Channel Groups

| Group ID | Channels | Input shape | Nature | H&E relationship |
|----------|----------|-------------|--------|-------------------|
| *(always present)* `cell_mask` | `cell_mask` | `[B, 1, 256, 256]` | Binary segmentation | Direct — cell boundaries, nuclear morphology |
| `cell_identity` | `cell_type_healthy`, `cell_type_cancer`, `cell_type_immune` | `[B, 3, 256, 256]` | One-hot categorical (CODEX-derived) | Direct — distinct morphological signatures |
| `cell_state` | `cell_state_prolif`, `cell_state_nonprolif`, `cell_state_dead` | `[B, 3, 256, 256]` | One-hot categorical (CODEX-derived) | Moderate — mitotic figures, necrosis visible |
| `vasculature` | `vasculature` | `[B, 1, 256, 256]` | Continuous (CD31 marker) | Direct — vessel lumens visible in H&E |
| `microenv` | `oxygen`, `glucose` | `[B, 2, 256, 256]` | Continuous (PDE approx from vasculature distance) | Indirect — affects tissue over time |

### Data sources

- **Cell identity/state:** Derived from multi-protein CODEX panels. Pixel-aligned, reliable.
- **Vasculature:** Equal to CD31 marker directly. Reliable.
- **Oxygen/glucose:** PDE approximation of distance to vasculature with different diffusion coefficients. Derived, smooth fields.

### Config representation

```python
channel_groups = [
    dict(name="cell_identity", channels=["cell_type_healthy", "cell_type_cancer", "cell_type_immune"]),
    dict(name="cell_state",    channels=["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]),
    dict(name="vasculature",   channels=["vasculature"]),
    dict(name="microenv",      channels=["oxygen", "glucose"]),
]

group_dropout_probs = dict(
    cell_identity=0.10,
    cell_state=0.10,
    vasculature=0.15,
    microenv=0.20,
)
```

## 4. Architecture: Per-Group Cross-Attention with Additive Residuals

### 4.1 Overview

```
cell_mask ──→ VAE Encode ──→ mask_latent [B,16,32,32] ──→ Q tokens (shared)
                                    │
cell_identity (3ch) ──→ CNN_id  ──→ CrossAttn_id(Q, KV=id_tokens)   ──→ Δ_id
cell_state (3ch)    ──→ CNN_st  ──→ CrossAttn_st(Q, KV=st_tokens)   ──→ Δ_st
vasculature (1ch)   ──→ CNN_vas ──→ CrossAttn_vas(Q, KV=vas_tokens) ──→ Δ_vas
O₂/glucose (2ch)    ──→ CNN_env ──→ CrossAttn_env(Q, KV=env_tokens) ──→ Δ_env
                                                                          │
                                            fused = mask_latent + Σ(Δ_group)
                                                                          │
                                                                     ControlNet

UNI embed ─────────────────────────────────────→ Base Transformer (frozen)
```

### 4.2 Per-group encoder

Each group gets its own `TMEEncoder` instance (reused from `tme_encoder.py`):

```python
TMEEncoder(n_tme_channels=3, base_ch=32, latent_ch=16)  # cell_identity
TMEEncoder(n_tme_channels=3, base_ch=32, latent_ch=16)  # cell_state
TMEEncoder(n_tme_channels=1, base_ch=32, latent_ch=16)  # vasculature
TMEEncoder(n_tme_channels=2, base_ch=32, latent_ch=16)  # microenv
```

Output: `[B, 16, 32, 32]` per group — matching VAE latent spatial dimensions.

### 4.3 Per-group cross-attention

Each group has its own `MultiHeadCrossAttention(d_model=16, num_heads=4)` with:

- **Q** = `LayerNorm(mask_latent)` flattened to tokens `[B, 1024, 16]` — shared, computed once
- **KV** = `LayerNorm(group_encoder_output)` flattened to tokens `[B, 1024, 16]`
- **Output projection** zero-initialized → `Δ_group = 0` at init

### 4.4 Fusion

```python
fused = mask_latent + Σ(Δ_group for each active group)
```

At initialization, all `Δ_group = 0`, so `fused = mask_latent`. Each group's contribution
grows from zero during training.

### 4.5 Module interface

```python
class MultiGroupTMEModule(ModelMixin, ConfigMixin):
    def forward(
        self,
        mask_latent: torch.Tensor,       # [B, 16, 32, 32]
        tme_channel_dict: dict,           # {"cell_identity": [B,3,256,256], ...}
        active_groups: set | None = None, # None = all groups
        return_residuals: bool = False,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple:
        ...
```

- `active_groups=None` → all groups contribute
- `active_groups={"cell_identity"}` → only cell identity contributes
- `return_residuals=True` → returns `(fused, {"group_name": Δ_tensor, ...})`
- `return_attn_weights=True` → returns attention weight matrices for visualization

### 4.6 Attention weight return (interpretability)

A thin subclass `CrossAttentionWithWeights` extends `MultiHeadCrossAttention` with an
optional `return_attn_weights` flag. This avoids modifying the shared `PixArt_blocks.py`.

When `return_attn_weights=True`, falls back to standard PyTorch attention
(instead of xformers) to capture the `[B, heads, N_q, N_kv]` weight matrix.
Used only at analysis time.

### 4.7 Parameter budget

| Component | Params per group | ×4 groups | Total |
|-----------|-----------------|-----------|-------|
| `TMEEncoder` (base_ch=32) | ~300k | 1.2M | |
| `LayerNorm` (KV) | 32 | 128 | |
| `CrossAttentionWithWeights` (d_model=16) | ~1.1k | 4.4k | |
| **Shared `LayerNorm` (Q)** | | | 32 |
| **Grand total** | | | **~1.2M** |

No cross-group parameter sharing. Fully independent encoders guarantee disentanglement.

## 5. Training Strategy

### 5.1 Group-level dropout

Independent per-group dropout during training. For each sample in the batch, each
group's channels are independently zeroed with the group's dropout probability.

```python
group_dropout_probs = dict(
    cell_identity=0.10,
    cell_state=0.10,
    vasculature=0.15,
    microenv=0.20,  # higher for PDE-derived channels
)
```

Combined with existing UNI CFG dropout (`cfg_dropout_prob=0.15`), this produces:

- **Mask-only baseline** (all dropped): ~0.05% of steps
- **TME-only** (UNI dropped, all groups active): ~8.3% of steps
- **Any single group missing**: 10–20% of steps per group

### 5.2 Channel reliability weights removed

The current scalar channel weights `[1.0, ..., 0.5, 0.5, 0.5]` are removed. Replaced by:

- Per-group dropout rates encode reliability (microenv dropped more → model learns it's optional)
- Separate encoders handle different channel statistics naturally
- Zero-init residuals mean unreliable groups contribute less by default

### 5.3 Optimizer setup

| Parameter group | Parameters | Learning rate |
|----------------|------------|---------------|
| ControlNet | Transformer blocks + zero convs | `5e-6` |
| MultiGroupTMEModule (all groups) | All encoders + cross-attentions + norms | `1e-5` |

Single optimizer for the entire TME module (not one per group). Matches existing
`optimizer_tme` pattern.

### 5.4 Training loop changes

```python
# Current:
tme_channels = control_input[:, 1:, :, :]
if channel_weights is not None:
    tme_channels = tme_channels * w
vae_mask = tme_module(vae_mask, tme_channels)

# New:
tme_channel_dict = split_channels_to_groups(control_input, active_channels, channel_groups)
active_groups = apply_group_dropout(group_names, dropout_probs, batch_size=bs)
vae_mask = tme_module(vae_mask, tme_channel_dict, active_groups)
```

### 5.5 Loss function

Unchanged. The existing `training_losses_controlnet` with SNR weighting applies.
The TME module's contribution flows through the ControlNet conditioning input.

### 5.6 Checkpointing

`save_checkpoint_with_tme` / `load_tme_checkpoint` save the entire `MultiGroupTMEModule`
state dict. Keys are structured as:

```
groups.cell_identity.encoder.stem.0.weight
groups.cell_identity.cross_attn.proj.weight
groups.cell_state.encoder.stem.0.weight
...
```

## 6. Inference

### 6.1 CLI changes to `stage3_inference.py`

New arguments:

```
--active-groups    Which TME groups to include (default: all)
                   e.g., --active-groups cell_identity vasculature
--drop-groups      Groups to exclude (alternative syntax)
                   e.g., --drop-groups microenv
```

### 6.2 Inference call

```python
tme_channel_dict = split_channels_to_groups(control_input, active_channels, channel_groups)
fused_cond = tme_module(vae_mask, tme_channel_dict, active_groups=active_groups)
```

Absent groups contribute zero residual. No special masking.

## 7. Validation & Visualization

Three visualization types, all produced at checkpoint intervals during training
and available as standalone tools.

### 7.1 Per-group attention heatmaps

**File:** `tools/visualize_group_attention.py`

Panel layout:
```
[Cell Mask] [Gen H&E] [Cell ID Attn] [Cell State Attn] [Vasc Attn] [μEnv Attn]
```

Each heatmap: average attention weights across heads, sum over query dimension
(how much attention each TME location receives), reshape to spatial map, overlay
on cell mask with `jet` colormap.

### 7.2 Per-group residual magnitude maps

**File:** `tools/visualize_group_residuals.py`

Panel layout:
```
[Cell Mask] [Gen H&E] [‖Δ_identity‖] [‖Δ_state‖] [‖Δ_vasc‖] [‖Δ_μenv‖]
```

Each map: L2 norm of `Δ_group` across channel dimension → `[32, 32]`, upsampled
to `[256, 256]`. Answers "how much does each group change the conditioning at each
spatial location?"

### 7.3 Ablation grid (progressive composition)

**File:** `tools/visualize_ablation_grid.py`

Panel layout:
```
[Mask only] [+Cell ID] [+Cell ID+State] [+ID+State+Vasc] [All groups]
```

Full H&E generation for each group combination. Directly shows what each group
contributes visually.

### 7.4 Training-loop integration

At every `save_model_steps` checkpoint, generate all three visualizations for a
fixed validation sample:

```
vis/step_10000/attention_heatmaps.png
vis/step_10000/residual_magnitudes.png
vis/step_10000/ablation_grid.png
```

Enables tracking how each group's contribution evolves during training.

## 8. File Changes

### New files

| File | Purpose |
|------|---------|
| `diffusion/model/nets/multi_group_tme.py` | `MultiGroupTMEModule` + `CrossAttentionWithWeights` |
| `tools/channel_group_utils.py` | `split_channels_to_groups()`, `apply_group_dropout()` |
| `tools/visualize_group_attention.py` | Per-group attention heatmap panels |
| `tools/visualize_group_residuals.py` | Per-group residual magnitude panels |
| `tools/visualize_ablation_grid.py` | Progressive composition H&E grid |
| `tests/test_multi_group_tme.py` | Unit tests for `MultiGroupTMEModule` |

### Modified files

| File | Changes |
|------|---------|
| `configs/config_controlnet_exp.py` | Add `channel_groups`, `group_dropout_probs`. Remove `channel_reliability_weights`. Update `tme_model`. |
| `train_scripts/train_controlnet_exp.py` | Use `split_channels_to_groups()` + `apply_group_dropout()`. Add validation visualization at checkpoint intervals. |
| `train_scripts/training_utils.py` | Update `_build_tme_module_and_optimizers()` to construct from `channel_groups` config. |
| `stage3_inference.py` | Add `--active-groups` / `--drop-groups` CLI args. Use group-based TME call. |
| `diffusion/model/builder.py` | Register `MultiGroupTMEModule`. |

### Unchanged files

| File | Reason |
|------|--------|
| `diffusion/model/nets/tme_encoder.py` | `TMEEncoder` CNN reused as-is inside each group. `TMEConditioningModule` kept for backward compat. |
| `diffusion/model/nets/PixArtControlNet.py` | Still receives `[B, 16, 32, 32]` fused conditioning. |
| `diffusion/model/nets/PixArt.py` | Frozen base transformer. |
| `diffusion/model/nets/PixArt_blocks.py` | `MultiHeadCrossAttention` reused. Attn weight mode added via subclass in `multi_group_tme.py`. |
| `train_scripts/mapping_weights_helper.py` | Pretrained weight mapping, unrelated. |
| `pipeline/extract_features.py` | UNI/VAE extractors unchanged. |
