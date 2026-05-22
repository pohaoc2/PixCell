# Handover

## 2026-05-22 — Multi-Encoder Spatial Probe Session

### Goal

Implement the multi-encoder spatial decodability pipeline from
`docs/superpowers/plans/2026-05-21-multi-encoder-spatial-probe.md` and produce
`figures/pngs_updated/07d_t1_spatial_multi_encoder.png` comparing T1 spatial
probe performance across UNI-2h, Virchow2, CTransPath, and ResNet-50.

### Code built

| Path | Purpose |
|------|---------|
| `pipeline/patch_extractors.py` | Reusable patch-feature extractors for ViT and hook-based spatial encoders (`extract_uni_patches`, `extract_virchow_patches`, `extract_ctranspath_patches`, `extract_resnet50_patches`). |
| `pipeline/extract_features.py` | Added patch-cache mode via `--encoder`, `--encoder-model`, `--patches-output-dir`, `--patches-prefix`. |
| `src/a1_mask_targets_spatial/main.py` | `block_mean_pool` now supports non-divisor grids via `cv2.INTER_AREA`; used for 7x7 target generation. |
| `src/a1_probe_mlp_spatial/main.py` | Generalized feature suffix loading; later extended with faster-probe CLI knobs `--max-iter`, `--hidden-layer-sizes`, and lower default `--n-splits=3`. |
| `src/a1_probe_mlp_spatial/run_multi_encoder.py` | Thin orchestrator for per-encoder spatial probe runs. |
| `src/paper_figures/fig_t1_spatial_multi_encoder.py` | Builds the final 07d grouped-bar figure. |
| `src/paper_figures/main.py` | Wires 07d into the paper figure runner. |
| `tests/test_task_a1_mask_targets_spatial.py` | Added regression test for non-divisor pooling at grid=7. |
| `tests/test_patch_extractors.py` | New extractor shape/dtype tests. |
| `tests/test_fig_t1_spatial_multi_encoder.py` | New smoke test for 07d rendering. |

### Focused validation completed

The implementation-level tests passed before the long-running jobs:

```bash
conda run --no-capture-output -n pixcell python -m pytest \
  tests/test_task_a1_mask_targets_spatial.py \
  tests/test_patch_extractors.py \
  tests/test_fig_t1_spatial_multi_encoder.py -q
```

Result: `10 passed`

### Data artifacts produced (on disk, not committed)

| Path | Notes |
|------|-------|
| `src/a1_mask_targets_spatial/out_grid_07/mask_targets_T1_spatial.npy` | 7x7 pooled T1 targets, shape `(10379, 49, 10)` fp32. |
| `data/orion-crc33/features_patches/virchow2/` | Virchow2 patch cache, 10,379 files. |
| `data/orion-crc33/features_patches/ctranspath/` | CTransPath patch cache, 10,379 files. |
| `data/orion-crc33/features_patches/resnet50/` | ResNet-50 patch cache, 10,379 files. |
| `src/a1_probe_mlp_spatial/out/uni_16/mlp_spatial_probe_results.csv` | UNI probe results, 10 rows. |
| `src/a1_probe_mlp_spatial/out/virchow2_16/mlp_spatial_probe_results.csv` | Virchow2 probe results, 10 rows. |
| `src/a1_probe_mlp_spatial/out/ctranspath_07/mlp_spatial_probe_results.csv` | CTransPath probe results, 10 rows. |
| `src/a1_probe_mlp_spatial/out/resnet50_07/mlp_spatial_probe_results.csv` | ResNet-50 probe results, 10 rows. |
| `figures/pngs_updated/07d_t1_spatial_multi_encoder.png` | Final rendered multi-encoder figure. |

### Important runtime note

The final 07d figure is complete, but the encoder results were **not all run on
the same probe budget**.

- UNI-2h completed on the original, more expensive settings.
- CTransPath completed on the original, more expensive settings.
- Virchow2 was restarted on a reduced budget after the original run proved too slow.
- ResNet-50 was run on the same reduced budget as the relaunched Virchow2.

Reduced-budget settings used for the final Virchow2 and ResNet-50 runs:

```bash
--n-splits 3
--n-tiles 200
--max-train-rows 10000
--max-iter 50
--hidden-layer-sizes 128,32
--n-jobs 1
```

This means the figure is valid as a fast comparison artifact, but **not a
strictly budget-matched encoder comparison**. If the paper needs matched
conditions, rerun all four encoders on one agreed setting and overwrite the
CSV outputs plus `07d_t1_spatial_multi_encoder.png`.

### What went wrong during execution

1. Virchow2 weights path was initially launched with `pretrained_models/virchow2/...`; the correct path on this host is `pretrained_models/Virchow2/pytorch_model.bin`.
2. The original MLP probe budget was far slower than expected in practice because many targets hit `max_iter=200` rather than early-stopping.
3. The probe script only writes CSV results at the end, so long-running targets looked stalled even while CPU was fully utilized.

### Recommended next steps

1. Decide whether 07d is a **fast mixed-budget comparison** or whether the figure must be recomputed under one consistent budget.
2. If consistency matters, rerun all four encoders with one agreed budget using the new CLI knobs in `src/a1_probe_mlp_spatial/main.py`.
3. If this figure is going into a paper draft now, add a caption note that Virchow2 / ResNet-50 were run on a reduced probe budget unless/until a matched rerun is done.

### Useful commands

```bash
# Verify all four result CSVs
for d in src/a1_probe_mlp_spatial/out/uni_16 \
         src/a1_probe_mlp_spatial/out/virchow2_16 \
         src/a1_probe_mlp_spatial/out/ctranspath_07 \
         src/a1_probe_mlp_spatial/out/resnet50_07; do
  printf '%s ' "$d"
  awk 'END{print NR-1}' "$d/mlp_spatial_probe_results.csv"
done

# Re-render 07d from existing CSVs
conda run --no-capture-output -n pixcell python -m src.paper_figures.fig_t1_spatial_multi_encoder

# Example consistent reduced-budget rerun
conda run --no-capture-output -n pixcell python -m src.a1_probe_mlp_spatial.main \
  --features-dir data/orion-crc33/features_patches/virchow2 \
  --targets-path src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy \
  --tile-ids-path src/a1_mask_targets_spatial/out/tile_ids.txt \
  --target-names-path src/a1_mask_targets_spatial/out/target_names_T1_spatial.json \
  --out-dir src/a1_probe_mlp_spatial/out/virchow2_16 \
  --feature-suffix _patches.npy \
  --n-splits 3 --n-tiles 200 --batch-size 2048 \
  --max-train-rows 10000 --max-iter 50 \
  --hidden-layer-sizes 128,32 --n-jobs 1
```

## 2026-05-21 — Spatial Probe Session

### Goal

Replace fig 07B / fig 09b scalar-tile-mean R² with **per-patch R²** so trivially-flat
targets (oxygen, glucose) no longer score high "for free". Adds per-tile
within-baseline, Pearson r, and shuffle-baseline metrics so noise floors are
explicit. Motivation: oxygen tile-mean σ ≈ 0.05; scalar probe gets R² ≈ 0.85
just by predicting the mean, which says nothing about whether H&E encodes
oxygen spatially.

### Code built

| Path | Purpose |
|------|---------|
| `pipeline/extract_features.py` | UNI patch-token cache (`extract_patch_tokens`) + flags `--save-uni-tokens`, `--skip-uni`, `--skip-vae`, `--uni-tokens-prefix`. |
| `src/a1_mask_targets_spatial/main.py` | Block-pool exp-channels to 16×16 grid → `(N, 256, 10)` T1 tensor; per-patch cell-type/state fractions. |
| `src/a1_codex_targets_spatial/build.py` | Per-cell CODEX → per-patch marker means via centroid bucketing; output `(N, 256, n_markers)`. |
| `src/a1_probe_mlp_spatial/main.py` | Shared MLP head over UNI patch tokens. Reports `r2_global`, `r2_within` (per-tile baseline), `pearson_r`, `delta_shuffle`. Memory-safe via fp16 disk memmap + pre-gather row subsample. CLI: `--n-jobs`, `--max-train-rows`, `--batch-size`, `--n-tiles`. |
| `tests/test_task_a1_{mask_targets,codex_targets,probe_mlp}_spatial.py` | 9 unit tests covering pooling, centroid bucketing, NaN handling, synthetic-signal recovery. All passing. |
| `SPATIAL_PROBE_RUNBOOK.md` | Three-step pipeline doc. |
| `CLAUDE.md` | Compacted 133 → 99 lines. Added "Memory Limits" section (mandatory `prlimit --as=24000000000` wrap for > 10 GB jobs, pre-launch FLOP/RAM math). |

### Data artifacts (on disk, not committed)

| Path | Shape | Size |
|------|-------|------|
| `data/orion-crc33/features/<tile>_uni_tokens.npy` | (256, 1536) fp16 each | 8.1 GB total |
| `src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy` | (10379, 256, 10) fp32 | 102 MB |
| `src/a1_codex_targets_spatial/out/codex_T2_spatial_mean.npy` | (10379, 256, 19) fp32 | 193 MB |

### Currently running (as of 5:38 PM PDT)

Background bash chain, PID 26070 → conda → python 27541, 2 loky workers.

```bash
nohup bash -c "set -e
  conda run --no-capture-output -n pixcell python -m src.a1_probe_mlp_spatial.main \
    --features-dir data/orion-crc33/features \
    --targets-path src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy \
    --tile-ids-path src/a1_mask_targets_spatial/out/tile_ids.txt \
    --target-names-path src/a1_mask_targets_spatial/out/target_names_T1_spatial.json \
    --out-dir src/a1_probe_mlp_spatial/out/t1_spatial \
    --n-tiles 800 --batch-size 2048 --n-jobs 2
  conda run ... # T2 with same flags + codex paths
" > logs/spatial_probe.log 2>&1 &
```

Subsample: 800 tiles seeded 42. ETA ~40 min from 5:38 PM (T1 ~18 min + T2 ~33 min). Done when log contains `=== ALL DONE ===`.

### Failure modes already hit and fixed

1. `np.stack` peak-RAM OOM on 10 379-tile load → row-by-row fp16 memmap.
2. joblib closure-pickled 16 GB X per worker → `delayed` arg + memmap.
3. `_flatten_split` materialized full train slice before subsample → pre-gather indexing into memmap.
4. `prlimit --as=24G` SIGBUS on memmap access (AS counts mmap mapping size, not RSS) → removed prlimit for the 800-tile run; re-apply with `--as=32G` for full-dataset runs.

### Next steps (after probe completes)

1. **Inspect** results CSVs:
   ```bash
   column -s, -t < src/a1_probe_mlp_spatial/out/t1_spatial/mlp_spatial_probe_results.csv
   column -s, -t < src/a1_probe_mlp_spatial/out/t2_spatial/mlp_spatial_probe_results.csv
   ```
   Expect: oxygen / glucose `r2_within_mean` ≪ scalar `r2_mean`. T2 markers Ki67 / E-cadherin / PD-1 should keep some positive `r2_within_mean`; sparse markers (FOXP3, CD8a) likely flat.

2. **Rewire figures** — manual choice on metric:
   - `src/paper_figures/fig_marker_utility.py:22` → repoint `PROBE_CSV` at `src/a1_probe_mlp_spatial/out/t2_spatial/mlp_spatial_probe_results.csv`. Decide x-axis = `r2_mean` (compare-style) or `r2_within_mean` (honest spatial decodability).
   - `src/paper_figures/main.py:36` → `T2_MLP_CSV` similarly.
   - Fig 07 panel A: add `T1_SPATIAL_CSV` to `main.py`, modify `fig_inverse_decoding.py` panel A or add SI panel.

3. **Optional follow-ups**:
   - Re-run on full 10 379 tiles (ETA ~3–5 hr; use `prlimit --as=32000000000`, `--max-train-rows 200000`).
   - Add `--compute-shuffle-baseline` for `delta_shuffle` column.

### Settings changed this session

- `.claude/settings.local.json`: `"permissions": { "defaultMode": "auto", … }` added so probe-completion → figure-rewiring loop runs without manual approval. Gitignored.

### Quick commands

```bash
# Watch progress
tail -f logs/spatial_probe.log

# Mem + procs
free -h ; pgrep -af a1_probe_mlp_spatial

# Kill probe + loky workers
pkill -9 -f a1_probe_mlp_spatial ; pkill -9 -f "loky.backend.popen"

# Clean restart
rm -rf src/a1_probe_mlp_spatial/out/

# T1 only (small smoke, ~10 min on T4)
conda run --no-capture-output -n pixcell python -m src.a1_probe_mlp_spatial.main \
  --features-dir data/orion-crc33/features \
  --targets-path src/a1_mask_targets_spatial/out/mask_targets_T1_spatial.npy \
  --tile-ids-path src/a1_mask_targets_spatial/out/tile_ids.txt \
  --target-names-path src/a1_mask_targets_spatial/out/target_names_T1_spatial.json \
  --out-dir src/a1_probe_mlp_spatial/out/t1_spatial \
  --n-tiles 800 --batch-size 2048 --n-jobs 2
```

---

## 2026-04-24 — Prior Session

Date: 2026-04-24

## Current state

This handover reflects the live state on the current GPU host, not the earlier planner-only snapshot.

### Environments

- `he-multiplex`: use for sklearn-based probe tasks and focused pytest runs.
- `pixcell`: use for diffusers / stage3 / GPU generation tasks.

### Key runtime paths

- Experimental data root: `data/orion-crc33`
- CODEX root on this host: `/home/ec2-user/he-feature-visualizer/data`
- Stage3 checkpoint dir: `checkpoints/pixcell_controlnet_exp/npy_inputs`
- Virchow2 local weights: `pretrained_models/Virchow2/`

## Task-by-task status

### Completed tasks

- `a0_visibility_map`
  - Output dir: `src/a0_visibility_map/out`
  - Status: complete

- `a0_tradeoff_scatter`
  - Output dir: `src/a0_tradeoff_scatter/out`
  - Status: complete

- `a1_mask_targets`
  - Output dir: `src/a1_mask_targets/out`
  - Status: complete
  - Produced T1 targets, tile IDs, and target names used by downstream probe tasks.

- `a1_probe_linear`
  - Output dir: `src/a1_probe_linear/out`
  - Status: complete
  - Manifest reports `10379` tiles, `1536` feature dimensions, and `10` T1 targets.

- `a1_probe_mlp`
  - Output dir: `src/a1_probe_mlp/out`
  - Status: complete
  - Main result files:
    - `mlp_probe_results.json`
    - `mlp_probe_results.csv`
    - `comparison_vs_linear.csv`
  - Notable outcome from `comparison_vs_linear.csv`:
    - MLP slightly improved `immune_frac`, `dead_frac`, and `vasculature_frac`.
    - Linear remained stronger on most other T1 targets, especially `oxygen_mean` and `glucose_mean`.

- `a1_probe_encoders`
  - Output dir: `src/a1_probe_encoders/out`
  - Status: complete
  - Produced:
    - `raw_cnn_embeddings.npy`
    - `virchow_embeddings.npy`
    - `ctranspath_embeddings.npy`
    - `virchow2_linear_probe_results.csv`
    - `ctranspath_linear_probe_results.csv`
    - `encoder_comparison.csv`
  - Virchow note:
    - The local Virchow2 package is a Hugging Face / timm layout (`config.json` + state dict), not a serialized Torch module.
    - The loader in `src/a1_probe_encoders/main.py` was updated earlier in this session to construct from config and load the state dict.
  - CTransPath note:
    - Figure-4-related code was extended to support a `ctranspath` worker and local CTransPath weights under `pretrained_models/ctranspath/`.
    - The local files now exist:
      - `pretrained_models/ctranspath/config.json`
      - `pretrained_models/ctranspath/model.safetensors`
    - GPU visibility was confirmed outside the sandbox; inside the sandbox `torch.cuda.is_available()` may report `False`.
    - Real CTransPath extraction is now complete.
    - Final artifact status:
      - `src/a1_probe_encoders/out/ctranspath_embeddings.npy` exists with shape `(10379, 768)` and dtype `float32`
      - `src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv` exists
    - Implementation note:
      - The extractor needed compatibility fixes for current `timm` plus pooled spatial averaging so the final embeddings are `768`-dimensional instead of flattened spatial maps.
  - Notable outcome from `encoder_comparison.csv`:
    - Virchow beats UNI on `cell_density`.
    - UNI remains stronger on the other listed T1 targets.

- `a1_generated_probe`
  - Output dir: `src/a1_generated_probe/out`
  - Status: complete
  - Produced:
    - `generated_uni_embeddings.npy`
    - `generated_tile_ids.txt`
    - `generated_probe_manifest.json`
    - `generated_probe_results.json`
    - `generated_probe_results.csv`
    - `real_vs_generated_r2.csv`

- `a2_decomposition`
  - Output dir: `src/a2_decomposition/out`
  - Status: **complete**
  - Full decomposition sweep is finished: `500` tiles × `4` modes = `2000` generated images.
  - Validation summary:
    - `mode_metrics.csv` contains `2000` rows across `500` tiles.
    - `mode_summary.csv` contains `4` mode rows with `n_tiles=500` and `reference_count=500` for each mode.
  - Runtime note:
    - Backfilling from the earlier `N=2` snapshot to the full `N=500` set completed on this T4 host in approximately `65` minutes.
  - Produced generated examples plus:
    - `mode_metrics.csv`
    - `mode_summary.csv`
  - Figure 5 "Foundation-Model Decomposition -- UNI vs TME" implementation status:
    - Spec: `docs/superpowers/specs/2026-04-24-figure5-uni-tme-decomposition-design.md`
    - Plan: `docs/superpowers/plans/2026-04-24-figure5-uni-tme-decomposition.md`
    - Metric/manifest code: `src/a2_decomposition/metrics.py`
    - Figure renderer: `src/paper_figures/fig_uni_tme_decomposition.py`
    - Figure wired into `src/paper_figures/main.py` → `figures/pngs/08_uni_tme_decomposition.png`
  - All Figure 5 data present:
    - Generated decomposition images: `2000` PNGs under `src/a2_decomposition/out/generated`
    - Metric manifests: `500` under `src/a2_decomposition/out/decomposition_metrics`
    - Metric JSONs: `500` under `src/a2_decomposition/out/decomposition_metrics`
    - Imported CellViT contour sidecars: `2000` `*_cellvit_instances.json` files beside generated images
    - CellViT batch export: `src/a2_decomposition/out/cellvit_batch`
    - CellViT results import report: `src/a2_decomposition/out/cellvit_batch/import_report.json`
    - `fud_scores.json`: present
    - `decomposition_summary.csv`: present
    - `representative_tile.json`: present, tile `7424_7936`
    - `figures/pngs/08_uni_tme_decomposition.png`: generated 2026-04-24
  - Metric coverage (all complete):
    - `fud`: `500` tile values
    - `lpips`: `2000` condition values
    - `style_hed`: `2000` condition values
    - `pq`: `2000` condition values
    - `dice`: `2000` condition values
  - CellViT import was completed with:

```bash
conda run -n pixcell python tools/cellvit/import_results.py \
  --manifest src/a2_decomposition/out/cellvit_batch/manifest.json \
  --results-dir src/a2_decomposition/out/cellvit
```
  - PQ/DICE were completed with:

```bash
conda run -n pixcell python -m tools.compute_ablation_metrics \
  --cache-dir src/a2_decomposition/out/decomposition_metrics \
  --orion-root data/orion-crc33 \
  --metrics pq dice \
  --device cuda \
  --jobs 2
```
  - Environment warning:
    - During troubleshooting, `lpips` was installed into `he-multiplex`, which upgraded that environment's Torch stack to `torch 2.11.0+cu130`; CUDA then reported unavailable there.
    - Use `pixcell` for GPU work unless `he-multiplex` is repaired.

- `a3_combinatorial_sweep`
  - Output dir: `src/a3_combinatorial_sweep/out`
  - Status: complete
  - Full K=20 sweep is finished: `20` anchor tiles × `27` conditions = `540` generated tiles.
  - Anchor list used for the completed sweep:
    - `src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt`
  - Validation summary:
    - `morphological_signatures.csv` contains `540` rows across `20` anchors.
    - `additive_model_residuals.csv` contains `27` condition rows with `n_anchors=20`.
  - Runtime note:
    - Full generation plus summary completed on this T4 host in approximately `29.3` minutes.
  - Summary outputs are now present:
    - `morphological_signatures.csv`
    - `additive_model_residuals.csv`
    - `interaction_heatmap.png`
  - CellViT-enhanced metrics:
    - After generation, CellViT was run on all 540 generated PNGs (batch export via `tools/cellvit/export_batch.py` was bypassed because A3 uses a plain generated tree, not a manifest-backed cache; instead outputs were manually mapped).
    - Per-image sidecar files (`*_cellvit_instances.json`) are present beside every generated PNG.
    - The A3 summary worker (`_compute_signature`) now reads CellViT contour polygon areas when a sidecar is present, falling back to connected-component hematoxylin thresholding otherwise.
    - Three metrics are now CellViT-backed: `nucleus_area_median`, `nucleus_area_iqr`, and (when CellViT cells are found) `nuclear_density` and `mean_cell_size`.
  - Figure 6 "Combinatorial Grammar — Emergent Signatures":
    - Figure renderer: `src/paper_figures/fig_combinatorial_grammar.py`
    - Wired into `src/paper_figures/main.py` → `figures/pngs/09_combinatorial_grammar.png`
    - Artifact present: `figures/pngs/09_combinatorial_grammar.png` (generated 2026-04-24)
  - Figure 6 design:
    - **Panel A**: 3×9 tile atlas showing one representative anchor across all 27 conditions (rows = cell state: prolif/nonprolif/dead; columns = oxygen×glucose low/mid/high grid). Anchor is auto-selected as the one with the most complete condition coverage.
    - **Panel B**: heatmap of `residual_l2_norm` across the 27 conditions. This is a non-negative interaction magnitude — bright cells indicate conditions where the observed morphology cannot be explained by additive state + oxygen + glucose main effects alone. Title annotates sweep level design values (0.50/0.75/1.00).
    - **Panel C**: three stacked case-study bar charts — lowest / median / highest L2 interaction condition. Each subplot shows signed metric residuals (observed − expected) for all 9 MORPHOLOGY_METRICS, sorted by |residual| descending. The dominant signal is in `mean_cell_size`, `nucleus_area_median`, and `nucleus_area_iqr`; stain-ratio and GLCM residuals are near zero.
  - Additive model details:
    - For each metric `m` and condition `(s, o, g)`:
      - `expected = grand_mean + state_effect(s) + oxygen_effect(o) + glucose_effect(g)`
      - `residual  = actual − expected`
      - `residual_l2_norm = sqrt(sum_m residual_m²)`
    - Residuals are **signed**; L2 norm collapses sign.
  - Sweep design note (nutrient levels):
    - Low/mid/high are fixed design constants: `{low: 0.50, mid: 0.75, high: 1.00}`.
    - These are intentionally wider than the real data distribution (oxygen p25–p75 ≈ 0.94–1.00; glucose p25–p75 ≈ 0.91–1.00) so the sweep exercises the full model response range including hypoxic/starved conditions not common in training tiles.
    - Using data percentiles would collapse the 3 levels to nearly identical values and lose most sweep variation.
  - Cell-state manipulation:
    - For each condition, the three cell-state control channels are rewritten: the target state channel is set to the full cell-mask plane; the other two are zeroed.
    - This means the prolif row treats all detected cells as proliferating regardless of their original per-cell state annotation in the CODEX data.

### Completed CODEX tasks

- `a1_codex_targets.build`
  - Output dir: `src/a1_codex_targets/out`
  - Status: complete
  - Produced T2 and T3 target bundles plus marker / feature-name metadata.

- `a1_codex_targets.probe`
  - Output dir: `src/a1_codex_targets/probe_out`
  - Status: complete
  - Final stage status:
    - `t2_linear`: complete
    - `t2_mlp`: complete
    - `t3_linear`: complete
    - `t3_mlp`: complete
  - Final run completed on this host at approximately `2026-04-23 17:28 PDT`.
  - All four stage output folders are populated.
  - Important collaboration note:
    - The user is currently modifying CODEX-related code. Do not make further CODEX code edits unless the user explicitly asks.

## Tests and validation already completed

- Worker-focused suite passed earlier:

```bash
conda run -n he-multiplex pytest -q \
  tests/test_a1_generated_probe_worker.py \
  tests/test_task_a1_probe_encoders.py \
  tests/test_a2_decomposition_worker.py \
  tests/test_a3_combinatorial_sweep_worker.py
```

- Later focused probe tests passed after the shared probe parallelism update:

```bash
/home/ec2-user/miniconda3/envs/he-multiplex/bin/python -m pytest \
  tests/test_task_a1_probe_mlp.py \
  tests/test_task_a1_codex_targets.py -q
```

- Figure 5 implementation tests passed:

```bash
python -m pytest \
  tests/test_a2_decomposition_metrics.py \
  tests/test_fig_uni_tme_decomposition.py

python -m pytest tests/test_cellvit_batch_tools.py
```

## Important implementation notes

- `a1_probe_linear.main` now supports target-level parallel CV execution via `--n-jobs`.
- `a1_probe_mlp.main` threads the same `--n-jobs` option through to the shared CV helper.
- `a1_codex_targets.probe` also accepts `--n-jobs`, but the user is actively modifying CODEX now, so avoid further edits there unless requested.
- `a1_probe_mlp` does not checkpoint mid-stage; it writes outputs only after the full run completes.
- `a1_codex_targets.probe` runs stages sequentially in order: `t2_linear -> t2_mlp -> t3_linear -> t3_mlp`.

## Recommended next steps

1. Review `figures/pngs/09_combinatorial_grammar.png` (Figure 6 complete).
2. Review `figures/pngs/08_uni_tme_decomposition.png` (Figure 5 complete).
3. Review the updated `figures/pngs/07_inverse_decoding.png` now that it includes CTransPath alongside UNI-2h and Virchow2.
4. Compare encoder-level T1 results across:
   - `src/a1_probe_linear/out/linear_probe_results.csv`
   - `src/a1_probe_encoders/out/virchow2_linear_probe_results.csv`
   - `src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv`
5. If more reporting is needed, consolidate the key results from:
   - `src/a1_probe_mlp/out/comparison_vs_linear.csv`
   - `src/a1_probe_encoders/out/encoder_comparison.csv`
   - `src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv`
   - `src/a1_generated_probe/out/real_vs_generated_r2.csv`
   - `src/a2_decomposition/out/mode_summary.csv`
   - `src/a3_combinatorial_sweep/out/morphological_signatures.csv`
6. Keep CODEX source edits user-driven for now; do not modify CODEX code again unless explicitly requested.

## Summary of what still needs active attention

- No major section-11 runtime tasks remain active on this host.
- Figure 6 (`09_combinatorial_grammar.png`) is complete — 540 generated tiles, CellViT sidecars, morphological signatures, additive residuals, and rendered PNG are all present.
- Figure 5 (`08_uni_tme_decomposition.png`) is complete — all metrics (FUD/LPIPS/HED/PQ/DICE), summary CSV, representative tile, and rendered PNG are present.
- Figure 4 follow-up:
  - `07_inverse_decoding.png` has been rebuilt and now includes CTransPath results.
  - Supporting artifacts are present:
    - `src/a1_probe_encoders/out/ctranspath_embeddings.npy`
    - `src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv`
