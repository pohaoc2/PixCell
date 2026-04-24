# Figure 5 - UNI/TME Decomposition Implementation Plan

**Goal:** Build `figures/pngs/08_uni_tme_decomposition.png` using the same metric family as Figure 1: FUD, LPIPS, PQ, DICE, and HED.

## Current status

### Done

- Four-mode generation exists under `src/a2_decomposition/out/generated/`.
- There are 2000 generated images: 500 tiles x 4 modes.
- `src/a2_decomposition/out/mode_metrics.csv` and `mode_summary.csv` exist, but they contain QC-style metrics only.
- Existing repo tools already support most required metrics:
  - `tools/compute_ablation_metrics.py` for LPIPS, PQ, DICE, and HED.
  - `src/a2_decomposition/metrics.py --compute-fud` for four-mode UNI-FUD, reusing feature/statistics helpers from `tools/compute_fid.py`.
  - `tools/ablation_report/figures.py` for Figure 1-style metric plotting conventions.

### Main gap

The decomposition output layout does not yet match the manifest/condition layout expected by the existing ablation metric tools. Add a small adapter rather than duplicating metric logic.

## Target files

Expected new or modified files:

```text
src/a2_decomposition/metrics.py
src/paper_figures/fig_uni_tme_decomposition.py
src/paper_figures/main.py
tests/test_a2_decomposition_metrics.py
tests/test_fig_uni_tme_decomposition.py
```

Expected outputs:

```text
src/a2_decomposition/out/decomposition_metrics/<tile_id>/manifest.json
src/a2_decomposition/out/decomposition_metrics/<tile_id>/metrics.json
src/a2_decomposition/out/decomposition_summary.csv
src/a2_decomposition/out/fud_scores.json
figures/pngs/08_uni_tme_decomposition.png
```

## Task 1 - Add a decomposition metric adapter

Create `src/a2_decomposition/metrics.py`.

Responsibilities:

- Discover complete tile directories under `src/a2_decomposition/out/generated/`.
- Convert each tile into a temporary or durable manifest-style metric cache.
- Preserve condition keys:
  - `uni_plus_tme`
  - `uni_only`
  - `tme_only`
  - `neither`
- Write a per-tile `manifest.json` compatible with `tools/compute_ablation_metrics.py` where each condition points to the correct generated PNG.
- Keep the adapter read-only with respect to generated images.

Validation:

```bash
python -m pytest tests/test_a2_decomposition_metrics.py
```

## Task 2 - Compute per-tile metrics

Run LPIPS, PQ, DICE, and HED for all 500 tiles.

Preferred command shape after Task 1:

```bash
conda run -n he-multiplex python -m tools.compute_ablation_metrics \
  --cache-dir src/a2_decomposition/out/decomposition_metrics \
  --orion-root data/orion-crc33 \
  --metrics lpips pq dice style_hed \
  --device cuda \
  --jobs 1
```

Notes:

- Use `--jobs 1` with CUDA to avoid multiple workers loading GPU-heavy models.
- If CellViT-derived masks are missing for PQ/DICE, either reuse existing mask outputs from the paired ablation cache or add a documented pre-step to generate them.
- Do not replace FUD with per-tile cosine; FUD is computed separately as a dataset-level metric.

Expected per-tile output:

```text
src/a2_decomposition/out/decomposition_metrics/<tile_id>/metrics.json
```

## Task 3 - Compute FUD by mode

Use UNI feature Fréchet distance. Use the decomposition adapter because the generic `tools/compute_fid.py` CLI expects the 15 standard ablation group combinations.

Command shape:

```bash
conda run -n he-multiplex python -m src.a2_decomposition.metrics \
  --compute-fud \
  --metrics-root src/a2_decomposition/out/decomposition_metrics \
  --orion-root data/orion-crc33 \
  --uni-model pretrained_models/uni-2h \
  --device cuda \
  --fud-json src/a2_decomposition/out/fud_scores.json
```

Expected output:

```text
src/a2_decomposition/out/fud_scores.json
```

Then backfill FUD into per-tile metric records if needed, matching the pattern already used by `tools/ablation_report/data.py`.

## Task 4 - Aggregate decomposition summary

Add a summary function that reads per-tile `metrics.json` plus `fud_scores.json` and writes:

```text
src/a2_decomposition/out/decomposition_summary.csv
```

Columns:

```text
mode,metric,mean,sd,n,ci95_low,ci95_high,direction
```

For FUD:

- `mean` is the dataset-level FUD score.
- `sd` can be empty unless bootstrap FUD is implemented.
- `n` is the number of tiles used in the mode distribution.

For LPIPS/PQ/DICE/HED:

- `mean`, `sd`, and CI are computed across tiles.

Validation:

- All four modes appear for each of the five metrics.
- Lower/higher direction matches `tools/ablation_report/shared.py`.

## Task 5 - Select Panel A representative tile

Implement deterministic tile selection after metrics exist.

Algorithm:

1. Load per-tile LPIPS, PQ, DICE, and HED for all four modes.
2. Orient metrics so higher is better:
   - `-LPIPS`
   - `PQ`
   - `DICE`
   - `-HED`
3. Z-score each metric dimension.
4. Pick the tile closest to the median vector.
5. Exclude tiles with missing data or obvious blank/edge-only tissue.

Fallback provisional tile:

```text
1792_10496
```

Record the selected tile in:

```text
src/a2_decomposition/out/representative_tile.json
```

## Task 6 - Build the Figure 5 renderer

Create `src/paper_figures/fig_uni_tme_decomposition.py`.

Panels:

- **A:** qualitative 2x2 grid for the selected tile.
- **B:** five mini-plots for FUD, LPIPS, PQ, DICE, and HED, matching Figure 1 style.
- **C:** effect decomposition heatmap with rows `UNI effect`, `TME effect`, `Interaction`.
- **D:** optional style/layout scatter.

Use existing constants from:

```text
tools/ablation_report/shared.py
src/paper_figures/style.py
```

Output:

```text
figures/pngs/08_uni_tme_decomposition.png
```

## Task 7 - Wire into paper figure build

Update `src/paper_figures/main.py`.

Behavior:

- Build Figure 5 only when required inputs exist.
- Print a clear skip message if metrics or generated images are missing.
- Do not break existing figures 01-07.

## Task 8 - Tests

Add focused tests:

```text
tests/test_a2_decomposition_metrics.py
tests/test_fig_uni_tme_decomposition.py
```

Test coverage:

- Adapter discovers only complete four-mode tile directories.
- Manifest output maps each condition to the expected image.
- Summary aggregation handles FUD as dataset-level and other metrics as per-tile.
- Representative tile selection is deterministic.
- Figure builder can render from a tiny synthetic fixture.
- `src/paper_figures/main.py` skips Figure 5 gracefully when inputs are missing.

Run:

```bash
python -m pytest \
  tests/test_a2_decomposition_metrics.py \
  tests/test_fig_uni_tme_decomposition.py
```

## Task 9 - Final metric run and render

After tests pass, run the real metric pipeline on the GPU machine.

Suggested sequence:

```bash
conda run -n he-multiplex python -m src.a2_decomposition.metrics \
  --generated-root src/a2_decomposition/out/generated \
  --out-dir src/a2_decomposition/out/decomposition_metrics

conda run -n he-multiplex python -m tools.compute_ablation_metrics \
  --cache-dir src/a2_decomposition/out/decomposition_metrics \
  --orion-root data/orion-crc33 \
  --metrics lpips pq dice style_hed \
  --device cuda \
  --jobs 1

conda run -n he-multiplex python -m src.a2_decomposition.metrics \
  --compute-fud \
  --metrics-root src/a2_decomposition/out/decomposition_metrics \
  --orion-root data/orion-crc33 \
  --uni-model pretrained_models/uni-2h \
  --device cuda \
  --fud-json src/a2_decomposition/out/fud_scores.json

conda run -n he-multiplex python -m src.a2_decomposition.metrics \
  --summarize \
  --metrics-root src/a2_decomposition/out/decomposition_metrics \
  --fud-json src/a2_decomposition/out/fud_scores.json \
  --out-csv src/a2_decomposition/out/decomposition_summary.csv

conda run -n he-multiplex python -m src.paper_figures.main
```

Expected final output:

```text
figures/pngs/08_uni_tme_decomposition.png
```

## Risks and decisions

- **PQ/DICE dependency:** If required CellViT/mask data are missing, decide whether to reuse paired-ablation segmentation outputs or generate new masks for the four decomposition modes.
- **FUD uncertainty:** Bootstrap CI is preferable but not mandatory for the first figure. If omitted, label FUD as a dataset-level point estimate.
- **Panel D crowding:** Keep Panel D optional. If Figure 5 becomes too dense, move style/layout scatter to supplementary material.
- **Representative tile:** Use `1792_10496` only as a fallback. The final tile should come from the metric medoid rule.

## Acceptance checklist

- [ ] `decomposition_metrics/<tile_id>/metrics.json` exists for all complete tiles.
- [ ] FUD, LPIPS, PQ, DICE, and HED are available for all four modes.
- [ ] `decomposition_summary.csv` includes all five metrics.
- [ ] Representative tile selection is reproducible.
- [ ] `08_uni_tme_decomposition.png` renders without changing Figures 01-07.
- [ ] Tests pass for adapter, aggregation, tile selection, and figure rendering.
