# Cleanup & Archive Plan — 2026-05-14

Two-part plan. Part 1: code cleanup punch list (read-only audit). Part 2: data/figure/checkpoint archive plan.

---

## Part 1 — Code Cleanup Punch List

Format: `<path>:<line>` — observation — suggested action — risk (low/med/high).

### A. Duplicate code (mechanical)

#### A1. `_load_cellvit_contours` duplicated 3×
- `tools/stage3/ablation_grid_figure.py:467` — returns `list[np.ndarray]`.
- `src/paper_figures/fig_si_a1_a2_unified.py:706` — identical body.
- `src/a3_combinatorial_sweep/main.py:539` — same JSON shape; returns `list[list[tuple[float,float]]]`.
- Action: hoist to `tools/cellvit/contours.py` returning `list[np.ndarray]`; convert at the one call site needing tuples. — **low**

#### A2. Three near-identical CellViT overlay helpers
- `tools/stage3/ablation_grid_figure.py:486,501,514` — `_maybe_overlay_cellvit_contours` (yellow), `_overlay_cellvit_contours_red`, `_overlay_cellvit_contours_gray` differ only in color/linewidth/alpha/zorder.
- `src/paper_figures/fig_si_a1_a2_unified.py:724` — another red copy.
- Action: single `overlay_cellvit_contours(ax, image_path, *, color, linewidth=0.6, alpha=0.85, zorder=4)`. — **low**

#### A3. `_load_rgb_pil` / `load_rgb` reimplemented 6+ times
- `tools/compute_fid.py:356`, `tools/compute_ablation_metrics.py:189`, `tools/ablation_report/data.py:10` (re-export), `tools/stage3/channel_sweep_cache.py:37`, `tools/stage4/figures.py:29`, `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py:60`.
- Plus ~25 inline `np.asarray(Image.open(p).convert("RGB"))` call sites.
- Action: one canonical loader in `tools/stage3/common.py`. — **low**

#### A4. `_save_png` private name reached into
- `src/a2_decomposition/main.py:245` defines `_save_png`.
- `run_a2_uni_null.py:16` imports it as if public.
- Action: expose `save_png` from `src/_tasklib/io.py`. — **low**

#### A5. `MORPHOLOGY_METRICS` duplicated
- `src/a3_combinatorial_sweep/main.py:41` (canonical, used in CSV writers).
- `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py:14` (duplicate).
- Action: import from a3 to prevent silent drift between CSV column order and figure column order. — **low**

#### A6. Per-state / per-level tables duplicated
- `src/a3_combinatorial_sweep/main.py:29` `CELL_STATE_CHANNELS`, `:35` `LEVEL_VALUES`.
- `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py:12-13` `STATES`, `LEVELS`.
- `tools/stage3/channel_sweep.py:63`, `src/a1_mask_targets/main.py:31` — both `_CELL_STATE_CHANNELS`.
- Action: single source in `tools/channel_group_utils.py`. — **low**

#### A7. CellViT sidecar path string duplicated
- `src/a4_uni_probe/metrics.py:13`, `tools/compute_ablation_metrics.py:512`, `src/a3_combinatorial_sweep/main.py:540`, `tools/stage3/ablation_grid_figure.py:469`, `src/paper_figures/fig_si_a1_a2_unified.py:707` all build `f"{stem}_cellvit_instances.json"` sibling path.
- Action: helper `cellvit_sidecar_path(image_path) -> Path` in `tools/cellvit/`. — **low**

#### A8. Two font-size regimes coexist
- `src/paper_figures/style.py:6-17` canonical Nature-Comms font ladder.
- `tools/ablation_report/figures.py:11` imports from it correctly.
- `src/a4_uni_probe/figures.py` + `tools/stage3/figures.py` hardcode `fontsize=8/9/10/14`, never call `apply_style()`.
- Action: import ladder + call `apply_style()`. — **low**

#### A9. Inline `plt.savefig(..., dpi=150, bbox_inches="tight", facecolor="white")`
- `tools/stage3/figures.py:108,285,380,462,652,701` and `tools/stage3/ablation_grid_figure.py:685` repeat ~7×.
- `tools/ablation_report/figures.py:61` has canonical `save_figure_png(fig, path, *, dpi=220)`.
- Action: route stage3 savers through `save_figure_png`. — **low**

#### A11. `OKABE_*` palette leaked as literal
- `src/paper_figures/fig_combinatorial_grammar_panels/_case_studies.py:115` uses `"#4C78A8"` instead of `OKABE_BLUE`.
- Action: replace literal with constant. — **low**

#### A12. `read_tile_ids_file` pattern reimplemented
- `src/a3_combinatorial_sweep/main.py:110`, `tools/stage3/run_evaluation.py`, `tools/stage3/channel_sweep.py` each parse newline-separated tile-ID files inline.
- Action: 4-line helper in `tools/stage3/common.py`. — **low**

### B. Dead / orphan code

#### B1. `run_a2_uni_null.py` — superseded
- `run_a2_uni_null.py:1` — only doc refs in HANDOVER_a4.md:65,162,178,195. No Python imports.
- `src/a4_uni_probe/generate_full_null_shared.py` is the proper home.
- Action: delete; update HANDOVER_a4.md. — **low**

#### B2. `update_null_full_uni.py` — one-shot migration
- One-shot per HANDOVER_a4.md:66,179. No callers.
- Action: delete (or move to `src/a4_uni_probe/scripts/`). — **low**

#### B3. `validate_sim_to_exp.py` — thin shim
- `from pipeline.validate_sim_to_exp import *`. README_stage3.md:446 references the package module directly.
- Action: delete root shim. — **low**

#### B4. `verify_pretrained_inference.py` defaults stale
- `verify_pretrained_inference.py:195-220` defaults `inference_data/test_mask.png` etc.
- Action: verify `inference_data/sample/` still exists; update if not. — **low**

#### B5. `_render_loo_diff_figure_legacy` — legacy LOO renderer
- `tools/vis/leave_one_out_diff.py:728` — only via `legacy_layout=True` (lines 938, 1309, 1418).
- Action: confirm `--legacy-layout` no longer used; if so delete. — **med** (CLI flag)

#### B6. `legacy-paired`/`legacy-unpaired` ablation grid modes
- `tools/stage3/ablation_grid_figure.py:84-85` carry old bar-set definitions; line 774 documents them in `--help`.
- Action: drop if paper uses new bar sets. — **med** (CLI surface)

#### B7. `notebook/`, `paper/`, `previous_data/` directories
- No Python references found.
- Action: confirm not used by Colab notebook workflow; archive or delete. — **low**

#### B8. `tools/pretrained_verify/` — only used by `verify_pretrained_inference.py` (B4)
- Action: tied to B4 fate. — **low**

#### B9. `tools/stage4/` package — orphaned
- `run_archetype_discovery.py`, `run_sim_inference.py`, `run_style_inference.py`, `run_matching.py`, `run_figures.py`, `style_selection.py`, `archetype_discovery.py`, `matching.py`, `figures.py` — only references in `docs/superpowers/archive/plans/2026-04-06-parameter-to-phenotype.md` (archived plan).
- Action: confirm stage4 dormant; move to `archive/` or delete. — **med**

#### B10. Concat checkpoint defaults already correct
- `tools/stage3/run_evaluation.py:153`, `tools/stage3/generate_tile_vis.py:347` — both already on concat. No action.

#### B11. Unused public helpers
- `tools/ablation_report/shared.py:184` `metric_name_cell`, `:188` `metric_plain_text`, `:128` `condition_order_label` — no callers outside `shared.py` / `__init__.py` re-exports.
- Action: drop from public API. — **low**

#### B13. `tools/render_dataset_metrics.py` — wired?
- `tools/render_dataset_metrics.py:13`. Referenced from README_stage3.md, `ablation_cli.md` only. `src/paper_figures/main.py` does not import.
- Action: confirm still produced; otherwise retire. — **low**

#### B14. `tools/generate_figure_story_report_standalone.py`
- No imports outside itself; no doc refs besides itself.
- Action: verify; archive if dead. — **low**

#### B15. Tracked diagnostic file
- `tile_classes.json` (4 MB at repo root). Generated by `tools/stage3/classify_tiles.py`, used by `channel_sweep.py`.
- Action: move under `inference_output/`; gitignore. — **low**

#### B16. Tracked coverage artifacts
- `.coverage`, `coverage.xml` — gitignore + rm. — **low**

### C. Stale checkpoint defaults

#### C1. `tools/vis/visualize_tme_cnn_features.py:489`
- `default="checkpoints/production_retrain_post_fix/full_seed_42/checkpoint/step_0002600"` — stale.
- Action: flip to `checkpoints/concat_95470_0/checkpoints/step_0002600`. — **high** (active CLI)

#### C2. `tools/ablation_a1_a2/build_cache.py:37`
- `production` row → `checkpoints/production_retrain_post_fix/full_seed_42/...`. Line 42 has `a1_concat` correct.
- Action: drop `production` row if paper no longer compares. — **med**

#### C3. `tools/ablation_a1_a2/log_utils.py:14-20`
- `production` log fallback chain dead if production variant retired.
- Action: prune with C2. — **med**

#### C4. Stage scripts at root still in docs
- `stage2_train.py:24,38`, `stage3_inference.py:25,34,42` print examples with `checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX`.
- Action: update docstrings to concat. — **low** (docs only)

#### C5. `configs/config_controlnet_exp.py:128`
- `work_dir = .../pixcell_controlnet_exp` still uses old name.
- Action: confirm with user; flip default if appropriate. — **high** (training output path)

### D. Stage scripts at root — informational

- `stage0_setup.py`, `stage1_extract_features.py`, `stage2_train.py`, `stage3_inference.py` are real entrypoints wired via README. No action.

### E. Duplicate label / constant tables

#### E1. `GROUP_LABELS` vs `GROUP_SHORT_LABELS`
- `tools/stage3/ablation_vis_utils.py:21` short labels (`CT/CS/Vas/NUT`).
- `tools/ablation_report/shared.py:42` long labels (`Cell types/Cell state/Vasculature/Nutrient`).
- Action: add cross-reference comments; resist adding a 3rd. — **low**

#### E2. `_PAPER_GROUP_LABELS` — hardcoded copy
- `tools/stage3/ablation_grid_figure.py:95` `_PAPER_GROUP_LABELS = ("CT","CS","Vas","Nut")` — `Nut` violates the NUC→NUT capitalization fix already landed in `GROUP_SHORT_LABELS`.
- Action: derive from `GROUP_SHORT_LABELS`. — **low**

#### E3. `_METRIC_LABELS` collision
- `src/paper_figures/fig_combinatorial_grammar_panels/_case_studies.py:11` local `_METRIC_LABELS` covers morphology metrics.
- `tools/ablation_report/shared.py:48` `METRIC_LABELS` covers paper metrics.
- Action: rename local to `_MORPHOLOGY_LABELS`. — **low**

#### E4. `_appearance_metric_title` heuristic
- `src/a4_uni_probe/figures.py:15` builds display titles via string replacement.
- Action: replace with `MORPHOLOGY_DISPLAY_LABELS` dict. — **low**

#### E5. Color literals leaked
- `src/a4_uni_probe/figures.py:58` `color="#4a5568"` not in any palette file.
- Action: standardize on `OKABE_GRAY` or document. — **low**

### F. Inconsistent ROOT resolution

#### F1. Two idioms across 30+ files
- `parent.parent.parent`: `tools/stage3/{generate_ablation_subset_cache, run_evaluation, generate_tile_vis, channel_sweep, prepare_unpaired_ablation_dataset, ablation_grid_figure, compute_ablation_uni_cosine}.py`, `tools/ablation_a1_a2/{sensitivity_eval, debug_cache, tsc_eval, metrics_io, build_cache}.py`, `tools/debug/probe_tme_batch_size.py`.
- `parents[N]`: `tools/{compute_fid, compute_ablation_metrics, render_dataset_metrics, summarize_ablation_report, generate_figure_story_report_standalone}.py`, `tools/ablation_report/shared.py`, `tools/stage4/*.py`, `tools/cellvit/*.py`, `tools/vis/leave_one_out_stats.py`, `tools/stage3/common.py`, `src/*/main.py`.
- Action: pick `parents[N]`; import `ROOT` from `tools.ablation_report.shared`. — **low**

#### F3. Function-local ROOT
- `train_scripts/inference_controlnet.py:117` resolves `root = Path(__file__).resolve().parents[1]` inside a function.
- Action: hoist to module-level. — **low**

### G. Imports

#### G1. Repeated function-local imports
- `tools/stage3/ablation_vis_utils.py:373,401,429,442` imports `tools.color_constants` 4× inside functions.
- Action: hoist. — **low**

#### G3. `tools/stage3/channel_sweep.py:79,130,156,187,257,401`
- Six function-local imports of `tools.stage3.tile_pipeline`.
- Action: consolidate. — **low**

#### G5. Unused mid-function import
- `update_null_full_uni.py:77` does `import argparse` mid-function without using it.
- Action: remove (if file survives B2). — **low**

### H. Dataset / inference contract — flag, don't touch

#### H1. Private dataset names imported
- `pipeline/validate_sim_to_exp.py:31,32,66` imports `_load_spatial_file`, `_find_file`, `_BINARY_CHANNELS` from dataset modules.
- Action: promote to public before any dataset refactor. — **med**

#### H2. Legacy sim dataset still on critical path
- `tools/stage3/tile_pipeline.py:15` → `diffusion.data.datasets.sim_controlnet_dataset`. Used by `stage3_inference.py`. Not dead.

#### H3. Training surface
- `train_scripts/training_utils.py`, `train_scripts/train_controlnet_exp.py` — off-limits in sweeping refactors per CLAUDE.md.

### I. Small / cosmetic

- **I1.** `src/paper_figures/main.py:30-37` — 8 `T1_*_CSV` constants + encoder list at 162-165 → single `T1_ENCODER_CSVS: dict[str, Path]` + loop.
- **I2.** `src/paper_figures/main.py:54-58` — `A3_SEEDS_*_LOGS` repeat glob pattern → helper.
- **I3.** Mix of `path.mkdir(parents=True, exist_ok=True)` and `ensure_directory` calls.
- **I8.** `configs/PixArt_xl2_internal.py:68` hard-coded `/cache/pretrained_models/...`. Confirm anything still inherits.
- **I9.** `src/paper_figures/fig_combinatorial_grammar.py:52` hardcodes anchor file path in help text; should reference `DEFAULT_ANCHORS_PATH`.
- **I10.** `stage3_inference.py:80` imports `load_sim_channels as load_sim_channels_shared` then wraps in a 4-line passthrough at line 86. Drop wrapper.

### J. Skip unless explicit opt-in (high-risk)

- `diffusion/` package (model + dataset code; includes `sim_controlnet_dataset.py` still on inference path).
- `train_scripts/train_controlnet_exp.py`, `training_utils.py`, `inference_controlnet.py`.
- `configs/config_controlnet_exp.py` and `_base_` siblings.
- `tools/stage3/tile_pipeline.py` (inference critical path).
- C5 (`work_dir` default), H1 (dataset private→public promotion).
- Stage 0–3 root scripts (D1) — public CLI surface used by README.

### Summary by impact/effort

- **Quick wins (low risk, high yield):** A1, A2, A3, A4, A7 (CellViT consolidation); A5, A6, E2 (state/level/group constants); E4 (a4 metric labels); B1, B2, B3 (root orphan scripts); B15, B16 (untracked artifacts); G1, G3 (inline imports); I1, I2 (paper_figures/main cleanup); F1+F2 (ROOT centralization).
- **Medium-risk (focused pass + test run):** B5, B6 (legacy LOO/grid CLI modes); B9 (stage4 retirement); C2, C3 (production-variant cache rows); H1 (private dataset names).
- **Skip unless user opts in:** C5, J items.

### Key cross-cutting recommendations

1. New `tools/cellvit/contours.py` exposing `cellvit_sidecar_path()`, `load_cellvit_contours()`, `overlay_contours()` — eliminates A1/A2/A7 + prevents NUC→NUT-class drift.
2. Move `ablation_grid_figure.py:95` off hardcoded `_PAPER_GROUP_LABELS` — most likely regression point for NUT capitalization.
3. Delete `run_a2_uni_null.py` + `update_null_full_uni.py` + `validate_sim_to_exp.py` shim in one commit after confirming a4 null cache + CellViT sidecars persisted.
4. Settle on `parents[N]` ROOT style; single import path from `tools.ablation_report.shared`.

---

## Part 2 — Archive Plan

Goal: get `figures/publication_archive/`, `previous_data/`, and old `checkpoints/archive/` out of the active tree without destroying recoverable assets. Move (not delete) to a single repo-root `archive/`.

### Current state (2026-05-14)

| Path | Size | Status |
|------|------|--------|
| `data/previous_data/` (consep, tcga_subset_3660, dummy_sim_data, patches) | 6.7 GB | no python refs |
| `checkpoints/archive/` (grouped_tme, grouped_tme_retrain) | 19 GB | not in concat path |
| `checkpoints/concat/` | 4 KB | empty stub |
| `inference_output/archive/debug_compare_500/` | part of 3.7 GB | **keep in place**, supplies SI sensitivity |
| `figures/publication_archive/` | 52 MB | old publication PNGs, kept for diff |
| `figures/archive/pngs_old/` | 1.2 MB | already archived |
| `tile_classes.json` (4 MB), `.coverage`, `coverage.xml` | small | repo-root debris |

Filesystem: `/dev/nvme0n1p1` 256G total, 109G free, 58% used. Single mount — all moves are cheap `mv` (inode reparent).

### Proposed structure

```
archive/
  data/
    consep/                ← from data/previous_data/consep
    tcga_subset_3660*/     ← from data/previous_data/
    dummy_sim_data/        ← from data/previous_data/
    patches/               ← from data/previous_data/
  checkpoints/
    grouped_tme/           ← from checkpoints/archive/
    grouped_tme_retrain/   ← from checkpoints/archive/
  figures/
    publication/           ← from figures/publication_archive/
    pngs_old/              ← from figures/archive/pngs_old/
  README.md                ← short manifest: source paths, dates, recovery
```

### Move strategy (per item)

1. **Disk-cheap moves first** (figures, small dirs): `mv` directly.
2. **Big moves** (checkpoints, previous_data): `mv` within same FS — free.
3. **`debug_compare_500`**: SI pipeline reads `inference_output/archive/debug_compare_500/cache.json`. Recommend **keep in current location** (option a). Option b: move + symlink back. Option (a) is lower risk.
4. **`checkpoints/concat/`**: empty stub. `rmdir`.
5. **Repo-root debris**:
   - `tile_classes.json` → move to `inference_output/`.
   - `.coverage`, `coverage.xml` → `.gitignore` + `rm`.
6. **Write `archive/README.md`** with source paths, dates moved, recovery instructions.

### What stays put (active paper pipeline)

- `data/orion-crc33/` (8.7 GB) — paired training set.
- `checkpoints/concat_95470_0/` (9.4 GB) — concat checkpoint.
- `checkpoints/debug/` (464 KB) — SI training-curve source.
- `inference_output/concat_ablation_1000/` (5.9 GB) — figs 01–06 source.
- `inference_output/si_a1_a2/` (326 MB), `inference_output/a1_concat/` (767 MB).
- `inference_output/archive/debug_compare_500/` — SI sensitivity source (see step 3 above).
- `figures/pngs_updated/` — current paper figures.

### Pre-move checks

- [ ] Verify `notebook/multichannel_controlnet.ipynb` does not reference `data/previous_data/*` paths.
- [ ] Verify `notebook/stage3_paired_ablation_a100_colab.ipynb` does not reference moved paths.
- [ ] Grep entire repo for string `previous_data` (excluding `previous_data/` itself).
- [ ] Grep entire repo for string `checkpoints/archive` to be sure no script reads from it.
- [ ] Confirm `figures/publication_archive/` has no symlinks pointing in or out.

### Phase 1 — move (preserves recovery)

```bash
ROOT=/home/ec2-user/PixCell
mkdir -p "$ROOT/archive/data" "$ROOT/archive/checkpoints" "$ROOT/archive/figures"

# figures
mv "$ROOT/figures/publication_archive" "$ROOT/archive/figures/publication"
mv "$ROOT/figures/archive/pngs_old"   "$ROOT/archive/figures/pngs_old"
rmdir "$ROOT/figures/archive" 2>/dev/null

# data
mv "$ROOT/data/previous_data"/* "$ROOT/archive/data/"
rmdir "$ROOT/data/previous_data"

# checkpoints
mv "$ROOT/checkpoints/archive"/* "$ROOT/archive/checkpoints/"
rmdir "$ROOT/checkpoints/archive"
rmdir "$ROOT/checkpoints/concat" 2>/dev/null   # empty stub

# debris
mv "$ROOT/tile_classes.json" "$ROOT/inference_output/"
rm "$ROOT/.coverage" "$ROOT/coverage.xml"

# manifest
cat > "$ROOT/archive/README.md" <<EOF
# Archive — 2026-05-14
Moved from active tree to free up clutter. All paths preserved.
- data/         ← from data/previous_data/
- checkpoints/  ← from checkpoints/archive/ (pre-concat grouped_tme runs)
- figures/      ← old publication figures (replaced by figures/pngs_updated/)
Recovery: \`mv archive/<subpath> <original-location>\`
EOF
```

### Phase 2 — reclaim disk (optional, separate confirmation)

Defer at least one paper-revision cycle.

```bash
# Recovers ~19 GB
rm -rf archive/checkpoints/

# Recovers ~6.7 GB
rm -rf archive/data/
```

### Disk impact

- Phase 1: ~26 GB physically moved within same FS (~free); free space unchanged at 109 GB.
- Phase 2: +25.7 GB free.

### `.gitignore` additions

```
.coverage
coverage.xml
tile_classes.json
archive/
```

(Keep `archive/README.md` tracked? Decide separately. If tracked, the `archive/` glob above needs `!archive/README.md`.)

---

## Execution order recommendation

1. Wait for a2/a3 chain (PID 33523 + 40191) to finish → confirms figs 08/09 regenerate cleanly.
2. Execute Phase 1 archive moves (low risk, reversible).
3. Quick-win cleanup pass (A1–A9, B1–B3, B15–B16, F1, E2, I1, I2).
4. Medium-risk pass (B5, B6, B9, C1–C3) — only after a clean commit and a test run.
5. Defer J items + Phase 2 reclaim until next paper revision.
