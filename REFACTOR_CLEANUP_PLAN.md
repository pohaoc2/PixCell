# Refactor & Cleanup Plan — PixCell `tools/`

Generated: 2026-04-12. Covers `tools/`, `tests/`, READMEs, and plan docs.
Default ablation output root: `inference_output/paired_ablation`.

---

## A. Extract Shared Functions into New Modules

### A1. `tools/stage3/hed_utils.py` (new file)

Extract HED color-space matrix and helper functions that are currently duplicated between
`tools/compute_ablation_metrics.py` and `tools/ablation_report/shared.py` + `data.py`.

**What to extract:**
- `_RGB_FROM_HED`, `_HED_FROM_RGB` (3×3 matrix constants)
- `rgb_to_hed(image)`
- `tissue_mask_from_rgb(image)`
- `masked_mean_std(channel, mask)`

No deps beyond `numpy` / `PIL`. After creation, update the three files below to import from
`tools.stage3.hed_utils` and delete their local copies:

| File | Action |
|---|---|
| `tools/compute_ablation_metrics.py` | Delete local matrix + 3 functions; import from `hed_utils`; expose `compute_style_hed_for_pair` as a public importable |
| `tools/ablation_report/shared.py` | Delete inline `_HED_FROM_RGB`; import from `hed_utils` |
| `tools/ablation_report/data.py` | Delete `rgb_to_hed_local`, `tissue_mask_from_rgb_local`, `masked_mean_std_local`, local HED pipeline; import from `hed_utils` and `compute_ablation_metrics` |

### A2. Promote `MetricSpec` / `METRIC_SPEC_BY_KEY` / `DEFAULT_METRIC_ORDER`

Currently defined in `tools/summarize_ablation_report.py` (a script) but consumed by
`tools/ablation_report/shared.py` (a package) — backwards import direction.

**Fix:** Move the three definitions into `ablation_report/shared.py` as the canonical home.
Update `summarize_ablation_report.py` to import them from there instead.

### A3. Add `GROUP_SHORT_LABELS`, `_mean`, `_fmt` to `tools/stage3/ablation_vis_utils.py`

| Symbol | Current duplicates | Action |
|---|---|---|
| `GROUP_SHORT_LABELS` | `ablation_grid_figure.py:94–99` (as `_GROUP_SHORT`) and `ablation_report/shared.py:58–63` | Add to `ablation_vis_utils.py`; both files import from there; delete local copies |
| `_mean`, `_fmt` | `summarize_ablation_report.py:38–45` and `vis/leave_one_out_stats.py:34–47` | Add to `ablation_vis_utils.py`; both scripts import from there; delete local copies |

`ablation_vis_utils.py` is already the shared-utils hub imported by both consumers, making it
the natural home.

---

## B. Remove Dead Code and Thin Wrappers

### B1. Delete thin 14-line CLI shims in `tools/vis/`

Both files contain only a re-export of a `main` function from `tools/stage3/`; they carry no
logic and have zero Python import callers.

| File | Reason to delete |
|---|---|
| `tools/vis/stage3_ablation_grid_figure.py` | Pure shim (`from tools.stage3.ablation_grid_figure import main`). Only references are in `README_stage3.md` and `ablation_cli.md`. |
| `tools/vis/generate_stage3_tile_vis.py` | Pure shim. Real callers already import from `tools.stage3.generate_tile_vis` directly. Confirm with `grep "from tools.vis.generate_stage3_tile_vis"` before deleting. |

**After deleting:** update `README_stage3.md` and `ablation_cli.md` to reference the
`tools/stage3/` implementation paths directly.

### B2. Inline or delete `tools/render_ablation_html_report.py`

A ~100-line mass-re-export facade over `tools.ablation_report`. Its only Python caller is
`tests/test_render_ablation_html_report.py`.

**Steps:**
1. Update the test to import from `tools.ablation_report` directly.
2. Update `README_stage3.md` to use `python tools/ablation_report/cli.py`
   (or `python -m tools.ablation_report`) as the CLI entry point.
3. Delete `tools/render_ablation_html_report.py`.

### B3. Archive or delete one-off operational scripts

| File | Reason | Action |
|---|---|---|
| `tools/verify_stage2_model_loading.py` | One-time GPU verification script; no library callers | Delete + delete companion `configs/config_0epoch_verify.py` |
| `tools/generate_orion_paired_unpaired_batch.py` | One-off batch vis generation; no callers | Delete |
| `tools/debug_fvd.py` | Only consumer is `tests/test_debug_fvd.py` (imports `resolve_condition_key`, `split_records`) | Either extract those two helpers into `tools/compute_fid.py` and keep the test, or delete both script and test together |

### B4. Clean up `.coveragerc` stale omit entries

Eight entries reference files that no longer exist on disk. Remove them:

```
tools/dummy_sim_generator.py
tools/run_evaluation.py
tools/generate_stage3_tile_vis.py
tools/stage3_figures.py
tools/stage3_tile_pipeline.py
tools/visualize_ablation_grid.py
tools/visualize_group_attention.py
tools/visualize_group_residuals.py
```

---

## C. Archive Stale Planning Docs

| File | Status | Action |
|---|---|---|
| `PLAN_batched_ablation.md` | Batched inference shipped | Delete |
| `TME_model_plan.md` | PhysiCell integration never implemented | Delete |
| `docs/physicell_tme_summary_*.md` | Same PhysiCell work | Delete |
| `docs/superpowers/plans/` + `specs/` dated before 2026-04-10 | ~12 files for fully shipped features | Move to `docs/superpowers/archive/` |
| `docs/superpowers/plans/2026-04-10-best-worst-tied-conditions.md` | Active branch work | Keep until branch merges |
| `docs/superpowers/specs/2026-04-10-best-worst-tied-conditions-design.md` | Active branch work | Keep until branch merges |

---

## D. README / Docs Updates

| File | Changes needed |
|---|---|
| `README_stage3.md` | (1) Replace single-tile `--cache-dir` example in "Channel Impact Analysis" section with batch `--cache-root` form, matching `ablation_cli.md` Step 10. (2) Update CLI paths for any deleted thin wrappers in `tools/vis/`. |
| `ablation_cli.md` | (1) Step 6: fix `--metrics all` bundle description — current set is `lpips pq dice style_hed`, not `cosine lpips aji pq style_hed`. (2) Step 7b: add note that `fvd_scores.json` is auto-consumed as a FUD fallback by the report renderer. |
| `README.md` | Add a brief pointer to `ablation_cli.md` for the full ablation workflow. |
| `README_stage1.md` | No changes needed. |
| `README_stage2.md` | No changes needed. |

---

## E. Ordered Migration Steps

Run `pytest tests/` after step 10 and again at step 19 as checkpoints.

1. Create `tools/stage3/hed_utils.py` — HED matrix constant + 3 functions.
2. Update `ablation_report/shared.py`: delete inline matrix, import from `hed_utils`.
3. Update `compute_ablation_metrics.py`: delete local HED code, import from `hed_utils`, expose `compute_style_hed_for_pair` as importable.
4. Update `ablation_report/data.py`: delete 3 local HED functions + local HED pipeline, import from `hed_utils` and `compute_ablation_metrics`.
5. Move `MetricSpec` / `METRIC_SPEC_BY_KEY` / `DEFAULT_METRIC_ORDER` into `ablation_report/shared.py`; flip the import in `summarize_ablation_report.py`.
6. Add `GROUP_SHORT_LABELS`, `_mean`, `_fmt` to `ablation_vis_utils.py`.
7. Update `ablation_grid_figure.py:94–99`: delete `_GROUP_SHORT`, import `GROUP_SHORT_LABELS` from `ablation_vis_utils`.
8. Update `ablation_report/shared.py`: delete local `GROUP_SHORT_LABELS`, import from `ablation_vis_utils`.
9. Update `vis/leave_one_out_stats.py` and `summarize_ablation_report.py`: delete `_mean` / `_fmt`, import from `ablation_vis_utils`.
10. **Run `pytest tests/` — must be green before proceeding.**
11. Delete thin wrappers: `tools/vis/stage3_ablation_grid_figure.py`, `tools/vis/generate_stage3_tile_vis.py`.
12. Update `tests/test_render_ablation_html_report.py` to import from `tools.ablation_report`; delete `tools/render_ablation_html_report.py`.
13. Delete `tools/verify_stage2_model_loading.py` + `configs/config_0epoch_verify.py`.
14. Delete `tools/generate_orion_paired_unpaired_batch.py`.
15. Resolve `debug_fvd.py`: extract 2 helpers into `compute_fid.py` or delete with its test.
16. Clean `.coveragerc`: remove 8 stale omit lines.
17. Delete / archive stale `.md` plan files (see Section C).
18. Update `README_stage3.md`, `ablation_cli.md`, `README.md` (see Section D).
19. **Run `pytest tests/` — final green check.**

---

## F. What is NOT Dead (keep as-is)

| File | Reason to keep |
|---|---|
| `tools/stage3/uni_cosine_similarity.py` | Imported by 5 files including `ablation_grid_figure.py`, `compute_ablation_uni_cosine.py`, `generate_tile_vis.py`, `run_evaluation.py`, and its own test |
| `tools/stage3/style_mapping.py` | Imported by 7+ files across the ablation pipeline |
| `tools/stage3/render_channel_sweep_figures.py` | Legitimate CLI entry point for channel sweep workflow; referenced in `README_stage3.md` and imported by `channel_sweep.py` |
| `tools/vis/leave_one_out_diff.py` | Full implementation (~450 lines), documented public CLI |
| `tools/vis/leave_one_out_stats.py` | Full implementation (~386 lines), documented public CLI |

**Import hygiene:** All modified files on this branch have clean imports — no unused imports found.
