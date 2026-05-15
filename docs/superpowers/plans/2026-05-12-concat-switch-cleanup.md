# Concat-Switch Cleanup & Refactor Plan

Date: 2026-05-12
Context: Primary path switched from grouped `MultiGroupTMEModule` checkpoint to concat ControlNet (`diffusion/model/nets/concat_controlnet.py`). This plan removes dead grouped-path code, extracts duplicated helpers, and reorganizes outputs/checkpoints.

---

## A. Dead Code Removal

### Tier 1 — high confidence, no risk

- **Smoke configs + grad-explosion debug** (~600 lines)
  - `configs/config_controlnet_exp_smoke_*.py` (16 files)
  - `tools/debug/grad_explosion_tme_followup.py`
  - `tools/debug/check_tme_midlength.py`
  - `tests/test_grad_explosion_tme_followup_debug.py`, `tests/test_check_tme_midlength.py`
  - Justification: incident resolved (commit `7dcb171`).
- **`apply_group_dropout` + `group_dropout_probs`** (~50 lines)
  - `tools/channel_group_utils.py`: remove `apply_group_dropout`
  - `train_scripts/train_controlnet_exp.py:304-312` grouped branch
  - `configs/config_controlnet_exp.py` `group_dropout_probs` knob
  - `tests/test_channel_group_utils.py:48-65`
  - Keep `split_channels_to_groups`, `channel_index_map` (concat ablation uses them).
- **Stage3 default checkpoint retargeting** (`pixcell_controlnet_exp` → concat checkpoint)
  - `tools/stage3/run_evaluation.py:153`
  - `tools/stage3/generate_ablation_subset_cache.py:389,644`
  - `tools/stage3/channel_sweep.py:350`
  - `tools/stage3/generate_tile_vis.py:292,347`
  - `tools/stage3/tile_pipeline.py:508` (hardcoded `"MultiGroupTMEModule"` — bug-prone)
  - `tools/vis/visualize_tme_cnn_features.py:489`
  - `tools/ablation_a1_a2/log_utils.py:16`

### Tier 2 — medium confidence, requires paper-lock confirmation

- Legacy sim pipeline (~300+ lines)
  - `pipeline/validate_sim_to_exp.py` + shim `validate_sim_to_exp.py`
  - `tests/test_validate_sim_to_exp.py`
  - `TMEConditioningModule` fallback in `train_scripts/training_utils.py:69-78` and `tools/stage3/tile_pipeline.py:522-530`
- `PerChannelTMEModule` in `diffusion/model/nets/per_channel_tme.py` (keep `RawConditioningPassthrough`) — only if A1-per-channel ablation dropped.
- `configs/config_controlnet_exp_a3_no_zero_init.py` — drop if not in final paper.
- `tools/vis/visualize_tme_cnn_features.py` — grouped-only tool.

### Tier 3 — load-bearing, DO NOT remove

- `diffusion/model/nets/multi_group_tme.py` class — needed to load production checkpoints until paper locked.
- `tools/stage3/figures.py` group-names logic (lines 729, 803, 834, 896).
- `configs/config_controlnet_exp.py` itself — parent of all `a*` configs. Strip grouped knobs only.
- Grouped tests `test_multi_group_tme*.py` — keep until class removed.

### CLAUDE.md update

Lines 75-76 say "prefer paired-exp + multi-group" — **stale**. Update to "prefer paired-exp + concat" in same PR.

---

## B. Sharable Function Extractions

| Helper | New home | Sites | Lines saved |
|---|---|---|---|
| `load_json(path)` | `tools/stage3/common.py` | 14 sites across `tools/stage3/` + `src/paper_figures/` | ~14 |
| `fix_work_dir(config, path, root)` | `tools/stage3/common.py` | `train_scripts/inference_controlnet.py:116-120`, `tools/stage3/generate_ablation_subset_cache.py:394-400` | ~16 |
| Use existing `inference_dtype` | already in `common.py:20` | `train_scripts/inference_controlnet.py:180,272` | ~4 |
| `load_runtime(...)` | `tools/stage3/tile_pipeline.py` | `tools/stage3/generate_ablation_subset_cache.py:368`, `tools/ablation_a1_a2/build_cache.py:105` | ~25 |
| Remove redundant `apply_style()` | n/a (delete) | `src/paper_figures/fig_si_a1_a2_unified.py:202,236,245,254,263` (already called in `main.py:72`) | ~5 |

**Not duplicates:** `generate_ablation_subset_cache.py`, `ablation_cache.py`, `build_cache.py` serve different purposes. Only `_load_runtime` is worth extracting.

---

## C. Output Folder Reorganization

### Path mapping

| Old | New |
|---|---|
| `inference_output/a1_concat/` | `inference_output/concat/eval/` |
| `inference_output/concat_ablation_1000/` | `inference_output/concat/ablation_subset/` |
| `inference_output/production/` (43 GB) | `inference_output/archive/grouped_tme/` then delete after archive |
| `inference_output/debug_compare_500/` | delete |
| `inference_output/debug/` | keep (scratch) |
| `inference_output/si_a1_a2/` | keep until SI A1/A2 figure locked |
| `checkpoints/a1_concat/` | `checkpoints/concat/` |
| `checkpoints/pixcell_controlnet_exp/` | `checkpoints/archive/grouped_tme/` |
| `checkpoints/production_retrain_post_fix/` | `checkpoints/archive/grouped_tme_retrain/` |
| `checkpoints/debug/` (38 GB) | keep only `concat_95470_0/` subdir; delete rest |
| `checkpoints/pixcell_controlnet_exp_{a1_concat,a1_per_channel,a2_bypass}/` (16 KB stubs) | delete |
| `figures/pngs/` | `figures/archive/` |
| `figures/pngs_updated/` | `figures/publication/` |

### Disk freed: ~90 GB

- `inference_output/production`: 43 GB
- `checkpoints/debug` minus `concat_95470_0`: ~28 GB
- `checkpoints/pixcell_controlnet_exp` + `production_retrain_post_fix`: ~19 GB (after archive)

### Hardcoded path update sites

- `tools/ablation_a1_a2/build_cache.py:40-66` (INFERENCE_VARIANTS dict), `:334` (cache-dir default)
- `tools/ablation_a1_a2/log_utils.py:16-37`
- `tools/ablation_a1_a2/debug_cache.py:24-29`
- `tools/stage3/generate_ablation_subset_cache.py:389,644`
- `tools/stage3/generate_tile_vis.py:292,347`
- `tools/stage3/run_evaluation.py:153,156`
- `tools/stage3/channel_sweep.py:350`
- `configs/config_controlnet_exp.py:135` (`work_dir`)
- `configs/config_controlnet_exp_a1_concat.py:34` (`work_dir`)
- `src/paper_figures/fig_si_a1_a2_unified.py:826`
- `src/a4_uni_probe/main.py:15` ← **WIP file, careful**

---

## D. Execution Sequencing

1. **Tier-1 dead code** — smoke configs, grad-explosion debug, `apply_group_dropout`. Pure removal, low blast.
2. **`load_json` extraction** — stage3 first; skip `paper_figures` initial pass. Run `pytest tests/`.
3. **`inference_dtype` + `fix_work_dir` extraction** — read git diff on modified files first; don't clobber WIP.
4. **Checkpoint dir renames + code updates** — ONE atomic commit. Update `src/a4_uni_probe/main.py:15` in same commit; verify `tests/test_a4_*.py` pass.
5. **`inference_output/` renames + figure path constants** — defer `si_a1_a2/` move until paper submitted.
6. **Stage3 default retarget + Tier-2 dead code** — after paper checkpoint requirements confirmed.
7. **`apply_style()` dedup** — cosmetic, last, no functional risk.

---

## E. WIP Guards

| File | Git status | Guard |
|---|---|---|
| `src/paper_figures/fig_si_a1_a2_unified.py` | M | Read diff first; don't clobber SI work |
| `tools/stage3/generate_ablation_subset_cache.py` | M | Checkpoint-path update in separate commit from refactor |
| `tools/stage3/tile_pipeline.py` | M | Read diff before extraction into this file |
| `train_scripts/inference_controlnet.py` | M | 6-line work_dir change committed before step 3 |
| `src/a4_uni_probe/` + `tests/test_a4_*.py` | ?? | Update `main.py:15` + verify tests before any `pixcell_controlnet_exp` rename |

---

## F. Recommended Starting Point

**Tier 1 dead code + `load_json` extraction.** Lowest risk, biggest cleanup-per-line. Delegate to Codex via `codex:codex-rescue`.
