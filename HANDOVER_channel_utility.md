# Channel Utility (Fig 5) Pivot — Handover

**Date:** 2026-05-21
**Status:** Data + code + figure delivered. Paper rewrite pending.

## Story shift

The paper's Section 5 was previously planned as a "combinatorial grammar" use case (Fig 09 grammar sweep on 20 anchors x 27 conditions x 3 seeds). After review, that section was determined to overclaim: data showed mask-driven generation (~95% anchor variance), conditioning shifts marginal, no clear visual trend, and no biological audience landed once unpaired-simulation rendering was set aside as a follow-up paper.

Section 5 is now reframed as a **measurement framework**: cross H&E -> MX decodability with per-sub-channel generative impact to produce a per-channel utility readout for MX panel design. The framework, not ORION-CRC specifically, is the contribution. The unpaired pathway is preserved in the architecture as the enabling capability for the future simulation-to-pathology paper, but the headline of the current paper does not depend on it.

Combinatorial-grammar results are kept intact for revert (no files deleted) — see `HANDOVER_combinatorial_grammar.md` for that pipeline's state.

## What was done

### Per-sub-channel LOO experiment

- Tile cohort: first 300 of `inference_output/concat_ablation_1000/tile_lists/paired_1000_tile_ids.txt`. Split into `shard_a.txt` (150) and `shard_b.txt` (150) under `inference_output/subchannel_loo_n300/_tile_lists/`.
- 9 droppable sub-channels: `cell_type_{healthy,cancer,immune}`, `cell_state_{prolif,nonprolif,dead}`, `vasculature`, `oxygen`, `glucose`. `cell_masks` is always-on at the pipeline level (never knocked out).
- Per tile: 1 freshly-generated baseline (all 9 sub-channels on) + 9 single-sub-channel knockouts = 10 inferences per tile. Total: 3000 inferences across the two shards.
- Diff: per-pixel CIELAB ΔE76 between baseline and each LOO image, plus mean_diff, delta_e_p99, pct_pixels_above_10. Per-tile JSON keyed by sub-channel.
- Wall-clock: dual-shard parallel on single T4 (15 GB), VRAM steady at ~12.2 GB used. ~80 min total (each shard ~78 min after the 90-sec staggered model load).

### Critical baseline finding

Original plan was to reuse the existing `concat_ablation_1000/.../all/generated_he.png` images as baseline. Verification on tile `0_10752` showed ΔE = 10.93 between the cached baseline and a locally-regenerated baseline using identical params (seed=42, num_steps=20, guidance=2.5). This GPU/torch nondeterminism is larger than the LOO signal itself (ΔE 0.3–4.4), so each shard regenerates the baseline locally with `--regenerate-baseline`. Cache reuse is now blocked across sessions — record this for any future ablation runs that try to diff against Fig 3's "all" images.

### Consistency bridge

`group_vs_subchannel_consistency.csv` aggregates the 9 sub-channel ΔE means back to 4 group means:

| Group       | Fig 3 group LOO (ΔE) | Sub-channel agg (ΔE) | Ratio |
|---|---:|---:|---:|
| cell_types  | 3.62 | 0.48 | 7.5x |
| cell_state  | 3.69 | 0.41 | 9.0x |
| vasculature | 4.17 | 4.24 | 1.02x  (single channel — sanity check passes) |
| microenv    | 6.94 | 3.60 | 1.93x |

Strong non-additivity within `cell_types` and `cell_state` (group LOO 7–9x larger than mean of singles). Vasculature is single-channel, so group == sub-channel — the 1.02x match validates the new pipeline against Fig 3's numbers under identical params.

### Figure 5 (new)

`figures/pngs_updated/09_channel_utility.png`. Single-panel scatter, 9 points, color-coded by group, error bars (R² s.d., ΔE SEM). Quadrant shading at R²=0.5 and ΔE=2.0 thresholds.

Result on ORION-CRC: Q2 (Critical) empty; Q1 (Redundant) holds glucose, oxygen, vasculature (vasculature borderline); Q3 (Skip) holds `cell_state_dead` alone (R²=-0.14, ΔE=0.003); Q4 (MX optional) holds the 5 remaining cell-type and cell-state sub-channels.

## Outstanding work (not done)

### Paper edits (the actual remaining lift)

1. Replace Section 5 prose with the per-channel utility framework. Draft text and caption already produced in chat — paste in.
2. Rewrite abstract to lead with the measurement framework, not the generator demo. Draft already produced.
3. Discussion: add the unpaired-mode paragraph framing it as the simulation-bridging capability for future work (Q2 channels are where unpaired mode would diverge from paired; on ORION-CRC, Q2 is empty so paired and unpaired are functionally equivalent).
4. Update title if it referenced "combinatorial grammar" or "in-silico perturbation".

### Optional follow-ups

- Per-CODEX-marker decodability + LOO (raw markers, not derived channels). `src/a1_codex_targets/probe_out/t2_linear/linear_probe_results.csv` has per-marker R² for ~37 markers; per-marker LOO would need a new ablation run. Reviewer payoff if accepted but ~1-2 weeks more compute.
- Vasculature R²=0.51 sits exactly on the Q1/Q2 threshold line. A separate-cohort retrain or richer probe would harden the assignment one way or the other.
- "MX optional" Q4 wording can mislead — those channels still contribute via group non-additivity. Footnote in caption or methods.

## Risks / caveats

1. Baseline regeneration is required for any future LOO experiments diffed against the Fig 3 cache. Treat the existing `concat_ablation_1000/.../all/*.png` images as nonreproducible across sessions.
2. `cell_state_dead` flagged Skip is driven by both broken probe decodability (R² negative) and zero generative impact. Two distinct claims overlap here; if the probe failure is fixable (e.g. dead cells are too rare in the training distribution for ridge to fit), the channel may move out of Skip.
3. Quadrant thresholds (R²=0.5, ΔE=2.0) are heuristic. Sensitivity check at R²=0.4 / ΔE=1.5: same channel placements except `cell_state_nonprolif` (R²=0.83, ΔE=1.02) crosses into Q1.
4. The new framework's reproducibility claim hinges on the LOO measurement being well-defined for any architecture. Architectures that strongly entangle channels (e.g., late-fusion attention TME modules) may show different non-additivity ratios than the a1_concat raw-passthrough used here. Mention in discussion.

## Pointers

- Code:
  - `tools/stage3/run_subchannel_loo.py` — driver
  - `tools/stage3/aggregate_subchannel_loo.py` — CSV aggregation
  - `src/paper_figures/fig_channel_utility.py` — Fig 5 renderer
- Data:
  - `inference_output/subchannel_loo_n300/<tile_id>/<sub_channel>/generated_he.png` — 2700 LOO PNGs
  - `inference_output/subchannel_loo_n300/<tile_id>/all_baseline.png` — 300 regenerated baselines
  - `inference_output/subchannel_loo_n300/<tile_id>/subchannel_loo_diff_stats.json` — per-tile stats
  - `inference_output/subchannel_loo_n300/per_subchannel_summary.csv` — aggregate (input to Fig 5)
  - `inference_output/subchannel_loo_n300/group_vs_subchannel_consistency.csv` — group bridge
  - `inference_output/subchannel_loo_n300/RESULTS_SUMMARY.md` — paper-style summary
- Run logs:
  - `inference_output/subchannel_loo_n300/_logs/shard_a.log`, `shard_b.log`
  - `inference_output/subchannel_loo_n300/_logs/shard_a.pid`, `shard_b.pid` (deleted on completion)
- Reference for prior figure (kept intact):
  - `figures/pngs_updated/09_combinatorial_grammar.png`, `SI_09_combinatorial_grammar_anchors.png`
  - `src/a3_combinatorial_sweep/out/` (all sweep PNGs + CSVs)
  - `HANDOVER_combinatorial_grammar.md`

## How to regenerate the figure

```bash
conda run --no-capture-output -n pixcell python -c \
  "from src.paper_figures.fig_channel_utility import save_channel_utility_figure; save_channel_utility_figure()"
```

## How to re-run the ablation (full 300 tiles, dual shard)

```bash
# Shard A (terminal 1)
conda run --no-capture-output -n pixcell python -m tools.stage3.run_subchannel_loo \
  --tile-list inference_output/subchannel_loo_n300/_tile_lists/shard_a.txt \
  --n-tiles 150 \
  --out-dir inference_output/subchannel_loo_n300 \
  --regenerate-baseline

# Shard B (terminal 2; start ~90 sec after A so model loads do not collide)
conda run --no-capture-output -n pixcell python -m tools.stage3.run_subchannel_loo \
  --tile-list inference_output/subchannel_loo_n300/_tile_lists/shard_b.txt \
  --n-tiles 150 \
  --out-dir inference_output/subchannel_loo_n300 \
  --regenerate-baseline

# After both complete:
conda run --no-capture-output -n pixcell python -m tools.stage3.aggregate_subchannel_loo \
  --out-dir inference_output/subchannel_loo_n300
conda run --no-capture-output -n pixcell python -c \
  "from src.paper_figures.fig_channel_utility import save_channel_utility_figure; save_channel_utility_figure()"
```

The driver is resumable: per-tile-per-sub-channel outputs are skipped if already written.
