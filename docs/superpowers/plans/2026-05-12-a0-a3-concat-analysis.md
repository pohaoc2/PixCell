# A0–A3 Analysis for A1-Concat Variant

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full paper analysis pipeline (a0–a3 tasks) for the `a1_concat` model variant, saving outputs to `inference_output/a1_concat/` instead of `src/<task>/out/`.

**Architecture:** The concat model already has ~1000 paired ablation tiles at `inference_output/concat_ablation_1000/paired_ablation/ablation_results/` (with per-tile metrics, singles/pairs/triples/all images, and UNI features). Tasks a0 (scatter, visibility) and a1 (probes) can run immediately on those outputs. Tasks a2 (decomposition) and a3 (combinatorial sweep) require generating new images; checkpoint is at `checkpoints/concat_95470_0/checkpoints/step_0002600/`.

**Tech Stack:** Python 3.12, conda `pixcell` env, `src/<task>` CLI modules, `tools/vis/leave_one_out_diff.py`

---

## Output path note (answers Q1)

`src/<task>/out/` exists because `DEFAULT_OUT_DIR` in each task module hardcodes `ROOT / "src" / "<task>" / "out"` and the production runs were invoked without overriding `--out-dir`. No functional reason — pure convention. The production outputs in `src/` are fine to leave as-is; the concat outputs here go to `inference_output/a1_concat/` to establish the correct pattern.

---

## File Map

| File | Action |
|---|---|
| `inference_output/a1_concat/a1_features/{tile_id}_uni.npy` | Create — aggregated UNI features from concat ablation cache |
| `inference_output/a1_concat/a1_mask_targets/` | Create — T1 mask targets for concat tile subset |
| `inference_output/a1_concat/a0_tradeoff_scatter/` | Create — scatter figures + CSV |
| `inference_output/a1_concat/a0_visibility_map/` | Create — visibility chart + CSV |
| `inference_output/a1_concat/a1_probe_linear/` | Create — linear probe results |
| `inference_output/concat_ablation_1000/paired_ablation/ablation_results/*/leave_one_out_diff_stats.json` | Create — computed by leave_one_out_diff tool |
| `inference_output/a1_concat/a2_decomposition/` | Create — pending checkpoint (Task 8) |
| `inference_output/a1_concat/a3_combinatorial_sweep/` | Create — pending checkpoint (Task 9) |

---

### Task 1: Aggregate concat UNI features into flat directory

The probing tasks expect `{tile_id}_uni.npy` in a flat directory. The concat ablation cache stores them at `ablation_results/{tile_id}/features/all/generated_he_uni.npy`.

**Files:**
- Create: `inference_output/a1_concat/a1_features/` (directory)

- [ ] **Step 1: Run aggregation script**

```bash
conda run --no-capture-output -n pixcell python3 - <<'EOF'
from pathlib import Path
import shutil

src = Path("inference_output/concat_ablation_1000/paired_ablation/ablation_results")
dst = Path("inference_output/a1_concat/a1_features")
dst.mkdir(parents=True, exist_ok=True)

count = 0
for tile_dir in sorted(src.iterdir()):
    src_npy = tile_dir / "features" / "all" / "generated_he_uni.npy"
    if src_npy.is_file():
        shutil.copy2(src_npy, dst / f"{tile_dir.name}_uni.npy")
        count += 1

print(f"Copied {count} UNI feature files to {dst}")
EOF
```

Expected: `Copied ~1000 UNI feature files to inference_output/a1_concat/a1_features`

- [ ] **Step 2: Verify**

```bash
ls inference_output/a1_concat/a1_features/ | wc -l
ls inference_output/a1_concat/a1_features/ | head -3
```

Expected: `~1000` files, names like `10240_11008_uni.npy`

- [ ] **Step 3: Commit**

```bash
git add -N inference_output/a1_concat/a1_features/
git commit -m "feat: aggregate concat UNI features for a1 probe"
```

---

### Task 2: Build mask targets for concat tile subset

`a1_mask_targets` reads tile IDs from the features dir and computes T1 targets (cell-type/channel fractions) from ORION exp_channels. Since we only have ~1000 concat tiles, targets will cover only those tiles.

**Files:**
- Create: `inference_output/a1_concat/a1_mask_targets/`

- [ ] **Step 1: Run a1_mask_targets**

```bash
conda run --no-capture-output -n pixcell python -m src.a1_mask_targets.main \
    --features-dir inference_output/a1_concat/a1_features \
    --exp-channels-dir data/orion-crc33/exp_channels \
    --out-dir inference_output/a1_concat/a1_mask_targets
```

Expected: outputs `mask_targets_T1.npy`, `target_names_T1.json`, `tile_ids.txt`, `target_stats.csv`, `manifest.json`

- [ ] **Step 2: Verify tile count matches**

```bash
python3 -c "
import json
m = json.load(open('inference_output/a1_concat/a1_mask_targets/manifest.json'))
print('tile_count:', m['tile_count'])
"
```

Expected: tile count matches `ls inference_output/a1_concat/a1_features/ | wc -l`

- [ ] **Step 3: Commit**

```bash
git add inference_output/a1_concat/a1_mask_targets/
git commit -m "feat: build T1 mask targets for concat tile subset"
```

---

### Task 3: Run a0_tradeoff_scatter

Reads per-tile `metrics.json` files from the concat ablation and `fud_scores.json` (already computed) to produce a tradeoff scatter plot.

**Files:**
- Create: `inference_output/a1_concat/a0_tradeoff_scatter/`

- [ ] **Step 1: Verify inputs exist**

```bash
ls inference_output/concat_ablation_1000/paired_ablation/ablation_results/fud_scores.json
ls inference_output/concat_ablation_1000/unpaired_ablation/ablation_results/fud_scores.json
ls inference_output/concat_ablation_1000/paired_ablation/ablation_results/ | wc -l
```

Expected: both `fud_scores.json` files exist; tile count ~1001 (tiles + the json file)

- [ ] **Step 2: Run a0_tradeoff_scatter**

```bash
conda run --no-capture-output -n pixcell python -m src.a0_tradeoff_scatter.run \
    --paired-metric-dir inference_output/concat_ablation_1000/paired_ablation/ablation_results \
    --unpaired-metric-dir inference_output/concat_ablation_1000/unpaired_ablation/ablation_results \
    --out-dir inference_output/a1_concat/a0_tradeoff_scatter
```

Expected: creates `tradeoff_data.csv`, `tradeoff_scatter_paired.png`, `tradeoff_scatter_unpaired.png`

- [ ] **Step 3: Spot-check output**

```bash
head -5 inference_output/a1_concat/a0_tradeoff_scatter/tradeoff_data.csv
```

Expected: CSV with columns `split,condition,n_groups,aji_mean,...,is_pareto`; ~8 rows (4 conditions × 2 splits)

- [ ] **Step 4: Commit**

```bash
git add inference_output/a1_concat/a0_tradeoff_scatter/
git commit -m "feat: run a0 tradeoff scatter for concat model"
```

---

### Task 4: Compute leave-one-out diffs (prerequisite for a0_visibility_map)

`tools/vis/leave_one_out_diff.py` reads the ablation cache (manifest.json + triples PNG images) to compute per-tile, per-group pixel diffs, saving `leave_one_out_diff_stats.json` in each tile dir.

**Files:**
- Create: `inference_output/concat_ablation_1000/paired_ablation/ablation_results/*/leave_one_out_diff_stats.json`
- Create: `inference_output/concat_ablation_1000/unpaired_ablation/ablation_results/*/leave_one_out_diff_stats.json`

- [ ] **Step 1: Compute paired LOO diffs**

```bash
conda run --no-capture-output -n pixcell python tools/vis/leave_one_out_diff.py \
    --cache-root inference_output/concat_ablation_1000/paired_ablation/ablation_results \
    --orion-root data/orion-crc33
```

Expected: prints `Saved stats -> .../leave_one_out_diff_stats.json` for each tile (~1000 lines)

- [ ] **Step 2: Compute unpaired LOO diffs**

```bash
conda run --no-capture-output -n pixcell python tools/vis/leave_one_out_diff.py \
    --cache-root inference_output/concat_ablation_1000/unpaired_ablation/ablation_results \
    --orion-root data/orion-crc33
```

- [ ] **Step 3: Verify**

```bash
find inference_output/concat_ablation_1000/paired_ablation/ablation_results -name "leave_one_out_diff_stats.json" | wc -l
```

Expected: ~1000

- [ ] **Step 4: Commit**

```bash
git add inference_output/concat_ablation_1000/
git commit -m "feat: compute leave-one-out diff stats for concat ablation"
```

---

### Task 5: Run a0_visibility_map

Aggregates the per-tile LOO diff stats (from Task 4) and renders a visibility bar chart showing how much each TME group visually affects output.

**Files:**
- Create: `inference_output/a1_concat/a0_visibility_map/`

- [ ] **Step 1: Run a0_visibility_map**

```bash
conda run --no-capture-output -n pixcell python -m src.a0_visibility_map.run \
    --paired-stats-root inference_output/concat_ablation_1000/paired_ablation/ablation_results \
    --unpaired-stats-root inference_output/concat_ablation_1000/unpaired_ablation/ablation_results \
    --out-dir inference_output/a1_concat/a0_visibility_map
```

Expected: creates `visibility_summary_table.csv`, `visibility_bar_chart.png`, `inset_tiles/`

- [ ] **Step 2: Spot-check**

```bash
head -5 inference_output/a1_concat/a0_visibility_map/visibility_summary_table.csv
```

Expected: rows for `cell_types`, `cell_state`, `vasculature`, `microenv`; columns `group,mean_diff,...`

- [ ] **Step 3: Commit**

```bash
git add inference_output/a1_concat/a0_visibility_map/
git commit -m "feat: run a0 visibility map for concat model"
```

---

### Task 6: Run a1_probe_linear

Probes whether the concat model's generated-image UNI embeddings (1536-d) linearly predict spatial cell-type fractions (T1 targets).

**Files:**
- Create: `inference_output/a1_concat/a1_probe_linear/`

- [ ] **Step 1: Run a1_probe_linear**

```bash
conda run --no-capture-output -n pixcell python -m src.a1_probe_linear.main \
    --features-dir inference_output/a1_concat/a1_features \
    --targets-path inference_output/a1_concat/a1_mask_targets/mask_targets_T1.npy \
    --tile-ids-path inference_output/a1_concat/a1_mask_targets/tile_ids.txt \
    --target-names-path inference_output/a1_concat/a1_mask_targets/target_names_T1.json \
    --out-dir inference_output/a1_concat/a1_probe_linear \
    --n-splits 5 \
    --n-jobs 4
```

Expected: creates `linear_probe_results.csv`, `linear_probe_results.json`, `linear_probe_fold_scores.npy`, `linear_probe_coef_mean.npy`, `cv_splits.json`, `manifest.json`

- [ ] **Step 2: Spot-check R² scores**

```bash
python3 -c "
import json
r = json.load(open('inference_output/a1_concat/a1_probe_linear/linear_probe_results.json'))
for k, v in list(r.items())[:5]:
    print(k, round(v.get('r2', -99), 3))
"
```

Expected: some targets have positive R² (e.g., cell_density ~0.2–0.5); no crash

- [ ] **Step 3: Compare to production**

```bash
python3 -c "
import json
concat = json.load(open('inference_output/a1_concat/a1_probe_linear/linear_probe_results.json'))
prod = json.load(open('src/a1_probe_linear/out/linear_probe_results.json'))
targets = list(concat.keys())
for t in targets[:5]:
    cr2 = round(concat.get(t, {}).get('r2', -99), 3)
    pr2 = round(prod.get(t, {}).get('r2', -99), 3)
    print(f'{t}: concat={cr2}  prod={pr2}')
"
```

Expected: meaningful comparison showing which model encodes spatial structure better

- [ ] **Step 4: Commit**

```bash
git add inference_output/a1_concat/a1_probe_linear/
git commit -m "feat: run a1 linear probe for concat model"
```

---

### Task 7: Verify checkpoint sync

Checkpoint is syncing to `checkpoints/concat_95470_0/checkpoints/step_0002600/`.

**Files:**
- Exists: `checkpoints/concat_95470_0/checkpoints/step_0002600/controlnet_epoch_20_step_2600.pth`
- Exists: `checkpoints/concat_95470_0/checkpoints/step_0002600/tme_module.pth`

- [ ] **Step 1: Confirm both files present**

```bash
ls checkpoints/concat_95470_0/checkpoints/step_0002600/
```

Expected: `controlnet_epoch_20_step_2600.pth  tme_module.pth`

- [ ] **Step 2: Verify checkpoint loads**

```bash
conda run --no-capture-output -n pixcell python3 - <<'EOF'
from tools.stage3.tile_pipeline import load_all_models
from pathlib import Path
from diffusion.utils.misc import read_config

config_path = "configs/config_controlnet_exp_a1_concat.py"
ckpt_dir = Path("checkpoints/concat_95470_0/checkpoints/step_0002600")
config = read_config(config_path)
models = load_all_models(config, config_path, ckpt_dir, device="cuda")
print("All models loaded OK:", list(models.keys()))
EOF
```

Expected: `All models loaded OK: ['vae', 'controlnet', 'base_model', 'tme_module']`

---

### Task 8: Run a2_decomposition for concat

Generates 2×2 UNI/TME decomposition images (uni+tme, uni-only, tme-only, neither) for a sample of tiles, then computes morphological metrics on each mode.

**Files:**
- Create: `inference_output/a1_concat/a2_decomposition/`

- [ ] **Step 1: Plan the decomposition task (dry run)**

```bash
conda run --no-capture-output -n pixcell python -m src.a2_decomposition.main \
    --config-path configs/config_controlnet_exp_a1_concat.py \
    --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
    --data-root data/orion-crc33 \
    --out-dir inference_output/a1_concat/a2_decomposition \
    --sample-n 500
```

Expected: creates `inference_output/a1_concat/a2_decomposition/plan.json` with generation job list; no images generated yet

- [ ] **Step 2: Run generation worker**

```bash
conda run --no-capture-output -n pixcell python -m src.a2_decomposition.main \
    --config-path configs/config_controlnet_exp_a1_concat.py \
    --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
    --data-root data/orion-crc33 \
    --out-dir inference_output/a1_concat/a2_decomposition \
    --sample-n 500 \
    --worker generate
```

Expected: generates `generated/{tile_id}/{mode}.png` for all modes × 500 tiles

- [ ] **Step 3: Run metrics worker**

```bash
conda run --no-capture-output -n pixcell python -m src.a2_decomposition.main \
    --config-path configs/config_controlnet_exp_a1_concat.py \
    --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
    --data-root data/orion-crc33 \
    --out-dir inference_output/a1_concat/a2_decomposition \
    --sample-n 500 \
    --worker metrics
```

Expected: creates `decomposition_summary.csv`, `mode_metrics.csv`

- [ ] **Step 4: Verify summary**

```bash
head -5 inference_output/a1_concat/a2_decomposition/decomposition_summary.csv
```

Expected: rows for `uni_plus_tme`, `uni_only`, `tme_only`, `neither` with morphological metric columns

- [ ] **Step 5: Commit**

```bash
git add inference_output/a1_concat/a2_decomposition/
git commit -m "feat: run a2 UNI/TME decomposition for concat model"
```

---

### Task 9: Run a3_combinatorial_sweep for concat

Generates images for a 3×3×3 grid of cell states × oxygen × glucose conditions at a set of anchor tiles, then extracts morphological signatures.

**Files:**
- Create: `inference_output/a1_concat/a3_combinatorial_sweep/`

- [ ] **Step 1: Choose anchor tiles**

Reuse the same anchor tiles as production if available, or pick 3–5 representative tiles:

```bash
# Check if production anchors file exists
ls src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt 2>/dev/null || \
    ls inference_output/a1_concat/a1_mask_targets/tile_ids.txt
```

If `anchors_k20_t1_medoid.txt` exists, reuse it. Otherwise, pick a few tile IDs from the concat tile_ids.txt (e.g., `head -5 inference_output/a1_concat/a1_mask_targets/tile_ids.txt`).

- [ ] **Step 2: Plan the sweep (dry run)**

```bash
conda run --no-capture-output -n pixcell python -m src.a3_combinatorial_sweep.main \
    --config-path configs/config_controlnet_exp_a1_concat.py \
    --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
    --data-root data/orion-crc33 \
    --out-dir inference_output/a1_concat/a3_combinatorial_sweep \
    --anchor-tile-ids-path src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt
```

Expected: creates `inference_output/a1_concat/a3_combinatorial_sweep/plan.json`; no images yet

- [ ] **Step 3: Run generation worker**

```bash
conda run --no-capture-output -n pixcell python -m src.a3_combinatorial_sweep.main \
    --config-path configs/config_controlnet_exp_a1_concat.py \
    --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
    --data-root data/orion-crc33 \
    --out-dir inference_output/a1_concat/a3_combinatorial_sweep \
    --anchor-tile-ids-path src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt \
    --worker generate
```

Expected: `generated/{anchor}/{condition}.png` for each anchor × 27 conditions

- [ ] **Step 4: Run analysis worker**

```bash
conda run --no-capture-output -n pixcell python -m src.a3_combinatorial_sweep.main \
    --config-path configs/config_controlnet_exp_a1_concat.py \
    --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
    --data-root data/orion-crc33 \
    --out-dir inference_output/a1_concat/a3_combinatorial_sweep \
    --anchor-tile-ids-path src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt \
    --worker analyze
```

Expected: creates `morphological_signatures.csv`, `additive_model_residuals.csv`, `interaction_heatmap.png`

- [ ] **Step 5: Commit**

```bash
git add inference_output/a1_concat/a3_combinatorial_sweep/
git commit -m "feat: run a3 combinatorial sweep for concat model"
```

---

## Dependency Graph

```
Task 1 (aggregate UNI features)
    └── Task 2 (mask targets) → Task 6 (probe linear)
Task 3 (tradeoff scatter)   ← no deps, runs immediately
Task 4 (LOO diffs) → Task 5 (visibility map)
Task 7 (download checkpoint)
    ├── Task 8 (a2 decomposition)
    └── Task 9 (a3 combinatorial sweep)
```

Tasks 1–6 can all run in parallel except where indicated. Tasks 8–9 are blocked on Task 7.

---

## Self-Review

**Spec coverage check:**
- a0_tradeoff_scatter: Task 3 ✓
- a0_visibility_map: Tasks 4+5 ✓
- a1_probe (mask targets): Tasks 1+2+6 ✓
- a2_decomposition: Task 8 ✓
- a3_combinatorial_sweep: Task 9 ✓
- Output path moved out of src/: all outputs go to `inference_output/a1_concat/` ✓
- Checkpoint download (unresolved blocker): Task 7 ✓ (flagged as manual)

**Potential gaps:**
- `a1_probe_mlp` (MLP probe for CODEX T2/T3 targets) not included — add if needed using the same pattern as Task 6 but with `a1_codex_targets` as targets source
- `a1_probe_encoders` (multi-encoder comparison) not included — separate effort
- The production `src/<task>/out/` outputs are left as-is; fix is to re-run with `--out-dir inference_output/production/<task>/` when convenient
