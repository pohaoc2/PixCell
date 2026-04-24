# Figure 4 — Inverse Decoding: What H&E Encodes

**Date:** 2026-04-24
**Status:** Approved for implementation

---

## Purpose

Figure 4 answers: *does a frozen pathology encoder already carry the TME information that Figure 2 shows is visibly imprinted?*

It validates that pixel-level changes the generator makes (Figure 2) encode real, linearly-decodable biological signal — and identifies which targets are H&E-invisible, motivating explicit TME conditioning.

Self-consistency data (real vs generated probe R² retention) moves to Supplementary Information.

---

## Layout

Two-panel vertical bar chart. Both panels share the same height (300px SVG).

```
[ (a) T1 targets — 4 encoders ]  [ (b) T2 CODEX markers — UNI MLP ]
```

---

## Panel (a) — T1 mask-fraction targets

### Data sources
| Encoder | Source | Status |
|---|---|---|
| UNI-2h | `src/a1_probe_linear/out/linear_probe_results.csv` | ✅ complete |
| Virchow2 | `src/a1_probe_encoders/out/encoder_comparison.csv` | ✅ mean only |
| REMEDIS | — | ❌ needs probe run |
| ResNet-50 | — | ❌ needs probe run |

### Visual spec
- **Chart type:** Grouped vertical bars, 4 bars per target, sorted by UNI R² descending
- **X-axis:** 10 targets with subscript labels: `density`, `f_{prolif}`, `f_{nonprolif}`, `glucose`, `O₂`, `f_{healthy}`, `f_{cancer}`, `f_{vasc}`, `f_{immune}`, `f_{dead}`. Labels rotated −45°, centered on ticks.
- **Y-axis:** R², range −0.15 to 1.0. Bold zero line. Grid at 0.25/0.50/0.75. Label: "R²".
- **Bar colors:** UNI-2h = `#2c7bb6`, Virchow2 = `#d7191c`, REMEDIS = `#555` (dashed border), ResNet-50 = `#aaa` (dashed border)
- **Error bars:** 95% CI on UNI-2h only (t₄ = 2.776 × SD across 5 folds). All other encoders: mean only until fold scores available.
- **Dead ƒ zone:** All encoders R² < 0. Pink background rectangle. Virchow2 and CI bars capped at plot boundary with overflow arrows.
- **Footnote:** "† REMEDIS and ResNet-50: illustrative — need probe runs. ƒ = fraction of cell mask."

### Target order (UNI R² descending)
| Label | Target | UNI R² | SD |
|---|---|---|---|
| density | cell_density | 0.953 | 0.007 |
| f_{prolif} | prolif_frac | 0.863 | 0.035 |
| f_{nonprolif} | nonprolif_frac | 0.826 | 0.023 |
| glucose | glucose_mean | 0.821 | 0.020 |
| O₂ | oxygen_mean | 0.810 | 0.022 |
| f_{healthy} | healthy_frac | 0.710 | 0.013 |
| f_{cancer} | cancer_frac | 0.669 | 0.016 |
| f_{vasc} | vasculature_frac | 0.509 | 0.022 |
| f_{immune} | immune_frac | 0.495 | 0.042 |
| f_{dead} | dead_frac | −0.135 | 0.107 |

---

## Panel (b) — T2 CODEX marker intensities

### Data sources
- `src/a1_codex_targets/probe_out/t2_mlp/mlp_probe_results.csv`
- Non-typing channels excluded: Hoechst, AF1, Argo550, PD-L1 (per `assign_cells.py:NON_TYPING_MARKERS`)
- 15 markers retained

### Visual spec
- **Chart type:** Vertical diverging bar chart (positive bars up, negative bars down from R²=0)
- **X-axis:** 15 markers, sorted by R² descending. Labels rotated −45°, centered on ticks (`text-anchor="middle"`). No numeric x-axis tick labels.
- **Y-axis:** R², range −0.30 to +0.40. Bold zero line. Grid at ±0.20. Label: "R²".
- **Bar colors by category:**
  - Immune signaling: `#8e44ad` (PD-1)
  - Epithelial: `#e67e22` (E-cadherin, Pan-CK)
  - Proliferation: `#16a085` (Ki67)
  - Immune/structural: `#7f8c8d` (all others)
- **Value labels:** Placed at bar tip — above top of positive bars, below bottom of negative bars. Font 6.5px.
- **FOXP3:** Capped at −0.30 with overflow arrow. True value (−1.06) shown as label with asterisk. Footnote explains.

### Marker data (15 markers, sorted)
| Marker | R² | Category |
|---|---|---|
| PD-1 | 0.364 | immune signaling |
| E-cadherin | 0.238 | epithelial |
| CD45RO | 0.094 | immune/structural |
| Ki67 | 0.050 | proliferation |
| CD3e | 0.045 | immune/structural |
| Pan-CK | 0.033 | epithelial |
| CD45 | 0.003 | immune/structural |
| CD4 | −0.002 | immune/structural |
| CD163 | −0.035 | immune/structural |
| CD68 | −0.042 | immune/structural |
| SMA | −0.052 | immune/structural |
| CD20 | −0.142 | immune/structural |
| CD31 | −0.144 | immune/structural |
| CD8a | −0.191 | immune/structural |
| FOXP3 | −1.060 | immune/structural |

---

## Encoder probe run plan

To replace placeholder bars with real data and enable fold-level CIs for all encoders.

### Encoders to add

| Encoder | Architecture | Pretraining | Why include |
|---|---|---|---|
| ResNet-50 | CNN | ImageNet | General-purpose lower bound |
| REMEDIS | ResNet-50x4 (BiT-L) | JFT-3B → medical fine-tune | Medical-domain ResNet; tests domain adaptation without ViT |

Story arc: **general (ResNet-50) → medical (REMEDIS) → pathology ViT (Virchow2) → pathology ViT (UNI-2h)**

### Steps

#### Step 1 — Feature extraction
For each new encoder, extract tile-level embeddings for all 10,379 tiles in the UNI feature set.

- **ResNet-50:** `torchvision.models.resnet50(weights=IMAGENET1K_V2)`, global average pool → 2048-dim vector. Save as `src/a1_probe_encoders/out/resnet50_embeddings.npy`.
- **REMEDIS:** Load BiT-L ResNet-50x4 weights (confirm public availability; fallback: CTransPath). Extract 2048-dim embedding. Save as `src/a1_probe_encoders/out/remedis_embeddings.npy`.
- Both must be aligned to the same tile ID ordering as `src/a1_probe_linear/out/` (use `src/a1_mask_targets/out/manifest.json` tile list). The tile list has 10,379 entries; embeddings saved as `(10379, D)` float32 arrays in the same row order.

#### Step 2 — Linear probe with 5-fold CV
Run the same CV pipeline used for UNI (`src/a1_probe_linear/main.py`) for each new encoder:
- Reuse `src/a1_probe_linear/out/cv_splits.json` — same fold assignments, no data leakage.
- Output: `resnet50_linear_probe_results.csv`, `remedis_linear_probe_results.csv` (same schema as `linear_probe_results.csv`: `target, r2_mean, r2_sd, n_valid_folds`).

#### Step 3 — Re-run Virchow2 with 5-fold CV
Current Virchow2 data is mean-only. Rerun with same CV splits to get per-fold scores → enables 95% CI on all 4 encoders.
- Virchow2 embeddings: `src/a1_probe_encoders/out/virchow_embeddings.npy` (already exists).
- Output: `virchow2_linear_probe_results.csv`.

#### Step 4 — Update encoder_comparison.csv
Extend schema to include `r2_sd` column for all encoders. Replace illustrative REMEDIS/ResNet bars with real values.

#### Step 5 — Update figure code
`src/paper_figures/fig_inverse_decoding.py` (new file) reads all four encoder CSVs, renders both panels, saves `figures/pngs/07_inverse_decoding.png`.

---

## Output file

`figures/pngs/07_inverse_decoding.png` (renumbered to avoid conflict with existing `05_paired_summary.png`)

300 DPI. Figure size: ~7 × 3.5 inches (two panels side by side).

---

## Out of scope

- Generator self-consistency panel → SI figure (separate spec)
- T3 quantile probe results → not shown in main figure
- MLP vs linear comparison → noted in caption only ("MLP ≈ linear on all targets")
