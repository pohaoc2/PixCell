# Figure 5 - Foundation-Model Decomposition: UNI vs TME

**Date:** 2026-04-24
**Status:** Draft for review

---

## Purpose

Figure 5 tests whether the two conditioning sources in PixCell carry separable information:

- **UNI** provides global H&E style and pathology feature realism.
- **TME** provides local cellular layout and spatial biological structure.
- **UNI + TME** should give the best combined behavior.

The figure uses the same metric family as `figures/pngs/01_metric_tradeoffs.png` for consistency:

| Metric | Direction | Interpretation |
|---|---:|---|
| FUD | lower better | UNI-space distribution realism |
| LPIPS | lower better | paired perceptual similarity |
| PQ | higher better | instance-level cell layout fidelity |
| DICE | higher better | foreground/cell-mask overlap |
| HED | lower better | interpretable H&E stain/style distance |

The current `src/a2_decomposition/out/mode_metrics.csv` fields (`tissue_fraction`, `eosin_mean`, `reference_rgb_mae`, `reference_hed_mae`) are QC-only and should not appear in the main Figure 5 metric panel.

---

## Experimental Conditions

Reuse the four-mode decomposition already implemented in `src/a2_decomposition/main.py`.

| Display label | Existing mode key | UNI | TME | Meaning |
|---|---|---:|---:|---|
| UNI+TME | `uni_plus_tme` | on | on | style plus spatial layout |
| UNI only | `uni_only` | on | off | style without explicit TME layout |
| TME only | `tme_only` | off | on | spatial layout without reference style |
| Neither | `neither` | off | off | null conditioning baseline |

The current output set already contains 500 tiles x 4 modes:

```text
src/a2_decomposition/out/generated/<tile_id>/<mode>.png
```

The metric implementation should preserve these condition names and emit standard per-tile records so Figure 5 can use the same plotting conventions as Figure 1.

---

## Inputs

### Existing generated images

- `src/a2_decomposition/out/generated/*/uni_plus_tme.png`
- `src/a2_decomposition/out/generated/*/uni_only.png`
- `src/a2_decomposition/out/generated/*/tme_only.png`
- `src/a2_decomposition/out/generated/*/neither.png`

### Reference data

- Real H&E references: `data/orion-crc33/he/<tile_id>.png`
- UNI feature/model inputs for FUD: `pretrained_models/uni-2h/`
- TME/ground-truth cell layout inputs: `data/orion-crc33/exp_channels/`
- CellViT-derived or ground-truth instance masks as required by the existing PQ/DICE metric stack

### Existing metric tools to reuse

- `tools/compute_ablation_metrics.py` for LPIPS, PQ, DICE, and HED where possible.
- `src/a2_decomposition/metrics.py --compute-fud` for four-mode UNI-FUD, reusing feature/statistics helpers from `tools/compute_fid.py`. The generic `tools/compute_fid.py` CLI assumes the 15 standard ablation group combinations and should not be called directly for this four-mode figure.
- `tools/ablation_report/shared.py` for metric labels, directionality, and color constants.
- `tools/ablation_report/figures.py` as the visual reference for the Figure 1-style metric panel.

---

## Outputs

New Figure 5 analysis outputs:

```text
src/a2_decomposition/out/decomposition_metrics/
src/a2_decomposition/out/decomposition_metrics/<tile_id>/metrics.json
src/a2_decomposition/out/decomposition_summary.csv
src/a2_decomposition/out/fud_scores.json
figures/pngs/08_uni_tme_decomposition.png
```

The figure number `08` avoids colliding with current outputs from `src/paper_figures/main.py`, where `07_inverse_decoding.png` is already used.

---

## Panel A - Qualitative 2x2 Decomposition

### Layout

Panel A is a 2x2 generated-image grid for one representative tile.

Columns:

```text
TME on | TME off
```

Rows:

```text
UNI on
UNI off
```

Cells:

```text
UNI+TME   | UNI only
TME only  | Neither
```

Include two compact reference thumbnails next to the grid:

- Paired real H&E reference.
- TME composite or cell-mask/TME layout thumbnail.

### Example tile selection

Use a reproducible medoid rule, not manual cherry-picking.

Initial provisional tile:

```text
1792_10496
```

This tile is present in the existing decomposition output and is closest to the current dataset-average behavior when using the available QC metric vectors across all four modes.

Final selection after the new metrics exist:

1. For each tile, build a metric vector from LPIPS, PQ, DICE, and HED across the four modes. FUD is dataset-level and is excluded from tile selection.
2. Orient metrics so higher is better for selection only:
   - `-LPIPS`
   - `PQ`
   - `DICE`
   - `-HED`
3. Z-score each metric dimension across tiles.
4. Pick the tile closest to the median vector.
5. Exclude tiles with blank tissue, edge-only tissue, or obvious generation artifacts.
6. Log the selected tile ID and selection score.

Optional continuity mode:

If visual continuity with stage-2 examples is more important than representativeness, rerun the four-mode decomposition for `14592_5632` and use it as an explicitly illustrative example. It is not currently in the 500-tile decomposition output.

---

## Panel B - Five-Metric Decomposition Panel

Panel B should match the visual language of `01_metric_tradeoffs.png`.

### Metrics

Use five horizontally arranged mini-plots:

```text
FUD | LPIPS | PQ | DICE | HED
```

Each mini-plot has four x-axis positions:

```text
UNI+TME | UNI only | TME only | Neither
```

Use point estimates with uncertainty:

- Per-tile metrics: mean +/- SD or bootstrap 95% CI.
- FUD: one dataset-level value per mode. Prefer bootstrap CI over tiles if computationally feasible; otherwise show point only and mark as dataset-level.

Under each metric axis, show a two-row dot glyph:

```text
UNI  filled filled open open
TME  filled open   filled open
```

This mirrors the condition glyph idea used in Figure 1 while fitting the 2x2 decomposition.

### Directionality

Metric titles should include direction:

- `FUD (down)`
- `LPIPS (down)`
- `PQ (up)`
- `DICE (up)`
- `HED (down)`

Use existing labels from `tools/ablation_report/shared.py`.

---

## Panel C - UNI/TME Effect Decomposition Heatmap

Panel C directly quantifies what each conditioning source contributes.

For each metric, first orient values so higher is better:

| Metric | Oriented score |
|---|---|
| FUD | `-FUD` |
| LPIPS | `-LPIPS` |
| PQ | `PQ` |
| DICE | `DICE` |
| HED | `-HED` |

Then compute three effects:

```text
UNI effect = score(UNI+TME) - score(TME only)
TME effect = score(UNI+TME) - score(UNI only)
Interaction = score(UNI+TME) - score(UNI only) - score(TME only) + score(Neither)
```

Rows:

```text
UNI effect
TME effect
Interaction
```

Columns:

```text
FUD | LPIPS | PQ | DICE | HED
```

Use a diverging color map centered at zero. Positive values mean the component improves the oriented metric.

This panel is the quantitative decomposition argument:

- UNI should improve FUD/HED/LPIPS most strongly.
- TME should improve PQ/DICE most strongly.
- Interaction shows whether the combined model is more than additive.

---

## Panel D - Style/Layout Summary Scatter

Panel D is optional if space is tight, but recommended.

Each point is one mode:

- x-axis: style score, such as normalized `-FUD` or a composite of `-FUD` and `-HED`.
- y-axis: layout score, such as normalized `PQ` or a composite of `PQ` and `DICE`.

Expected interpretation:

| Mode | Expected quadrant |
|---|---|
| UNI+TME | high style, high layout |
| UNI only | high style, lower layout |
| TME only | lower style, high layout |
| Neither | low style, low layout |

If the final figure is too crowded, Panel D can move to supplement.

---

## Recommended Final Layout

Use a 2-row multi-panel figure.

Top row:

```text
Panel A: qualitative 2x2 grid
```

Bottom row:

```text
Panel B: five-metric dot/CI panel
Panel C: effect decomposition heatmap
Panel D: style/layout scatter, optional
```

If journal width becomes tight, keep Panels A, B, and C in the main figure, and move Panel D to SI.

---

## Acceptance Criteria

- All four decomposition modes have metrics for the same tile set.
- Main metric panel uses exactly FUD, LPIPS, PQ, DICE, and HED.
- The qualitative tile is selected by a reproducible rule and the tile ID is recorded.
- Figure 5 uses labels and metric directionality consistent with Figure 1.
- QC metrics are not presented as main evidence.
- The final PNG is written to `figures/pngs/08_uni_tme_decomposition.png`.
- The figure can be rebuilt from source with a single command.

---

## Out of Scope

- New model inference beyond optional rerun of one illustrative tile.
- Adding UNI cosine or AJI to the main Figure 5 panel.
- Reinterpreting Figure 1 metrics or changing existing Figure 1 outputs.
- Training or fine-tuning any model.
