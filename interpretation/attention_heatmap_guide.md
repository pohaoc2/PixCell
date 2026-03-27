# Attention Heatmap Interpretation Guide

## What the heatmaps show

Each heatmap corresponds to one TME group (cell_identity, cell_state, vasculature, microenv).
They are generated from the `MultiGroupTMEModule` cross-attention weights at inference time.

### Architecture recap

```
Q  = cell mask latent      [B, 1024 tokens, 16-dim]   (32×32 spatial grid)
KV = TME group latent      [B, 1024 tokens, 16-dim]   (32×32 spatial grid)

attn_weights: [B, heads, Q_len, KV_len]
```

### Current heatmap computation (`visualize_group_attention.py`)

```python
avg        = attn_weights.mean(dim=(0, 1))  # [Q_len, KV_len]
importance = avg.sum(dim=0)                 # sum over Q → [KV_len]
heatmap    = importance.reshape(32, 32)     # KV-space
```

**Reads as:** "Which spatial regions of the TME group's image were most consulted
by the cell mask as a whole?"
→ Heatmap lives in **TME/group coordinate space**.

---

## Per-group biological interpretation

### `cell_identity` (healthy / cancer / immune)
- Attention concentrates where cells exist in the mask — correct behavior.
- Clean, spatially coherent blobs matching cell positions.
- Healthiest signal of all four groups.

### `cell_state` (prolif / nonprolif / dead)
- More diffuse, scattered hotspots.
- Cell state channels have lower SNR; states are intermixed at tile scale.
- Worth monitoring as training progresses.

### `vasculature` (CD31+)
- Noisy, near-uniform attention despite visible dense CD31 cluster.
- **Root cause:** sparse scatter + 8×8 CNN downsampling makes many 32×32 blocks
  look equally weak; the dense cluster does not strongly dominate neighbors.
- The model cannot yet reliably discriminate vessel signal from background scatter.

### `microenv` (oxygen / glucose)
- Distinctive border/frame pattern: high at edges, low in center.
- Likely a boundary artifact in how simulation channels are generated
  (non-zero boundary conditions or constant interior).
- CNN encoder latches onto edge gradients rather than semantic content.

---

## Biological reading: vasculature case study

**What the cell mask *should* seek from CD31 to generate accurate H&E:**

| Region in CD31 | Biological meaning | H&E implication |
|---|---|---|
| Dense coherent cluster | Vessel lumen / wall | Endothelial nuclei + RBC-filled lumen |
| CD31 boundary pixels | Perivascular zone | Pericytes, smooth muscle, perivascular tumor cells |
| CD31-negative regions | Avascular / hypoxic zone | Pyknotic nuclei, pale cytoplasm, necrotic texture |

**What currently happens:** diffuse attention across all CD31 positions because
no single 8×8 block sends a clearly stronger signal than neighboring scatter.

---

## Alternative visualization: inverse heatmap (`sum(dim=1)`)

```python
importance = avg.sum(dim=1)   # sum over KV → [Q_len]
heatmap    = importance.reshape(32, 32)   # Q / mask-space
```

**Reads as:** "Which cell mask positions were most actively seeking information
from this TME group?"
→ Heatmap lives in **cell mask coordinate space** — directly overlayable on
the cell mask and generated H&E.

| | `sum(dim=0)` (current) | `sum(dim=1)` (inverse) |
|---|---|---|
| Space | TME group (CD31) coords | Cell mask coords |
| Question | Which TME regions were consulted? | Which mask positions sought TME info? |
| Overlay | On TME channel image | On cell mask / generated H&E |
| Biological read | Where is vessel signal informative | Which cells are vasculature-aware |

The inverse is more useful for perivascular analysis: it would reveal whether
cells adjacent to the CD31 cluster attend more to vasculature than cells in
the avascular zone.

---

## Summary of heatmap quality signals

| Pattern | Diagnosis |
|---|---|
| Warm blobs matching cell positions | Good — group is spatially grounded |
| Noisy / granular uniform map | Signal too sparse or CNN resolution too coarse |
| Border / frame artifact | Channel has edge gradients, not semantic content |
| Flat / all-blue | Group dropout active, or channel missing |
