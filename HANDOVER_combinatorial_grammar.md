# Combinatorial Grammar Fig 09 — Handover

**Date:** 2026-05-20
**Status:** 16 of 17 tasks done. T17 partial — Change 1 of 4 committed; Changes 2–4 pending.

## Pointers

- Spec: `docs/superpowers/specs/2026-05-20-combinatorial-grammar-design.md`
- Plan: `docs/superpowers/plans/2026-05-20-combinatorial-grammar.md`
- HTML proposal (gitignored): `docs/proposal_combinatorial_grammar_2026-05-20.html`

## Done (12 commits, T1 → T17/Change 1)

| Commit | Subject |
|---|---|
| `8f3de66` | feat: add 3-way ANOVA variance partition for combinatorial sweep |
| `a0c30c3` | refactor(a3): adopt a4 metric schema in _compute_signature |
| `30f74de` | feat(a3): multi-seed render + seed column in signatures |
| `9d67ea4` | test(a3): expect --seed 42 in plan_task worker argv |
| `2114332` | feat(a3): emit variance_partition.csv from summary worker |
| `34cfc2d` | fix(a3): variance_partition handles NaN rows per metric |
| `7ecac5b` | test: align combinatorial grammar fixtures with a4 metric schema |
| `a237d1d` | feat(fig): helper to load variance_partition.csv |
| `3c87153` | feat(fig): variance-partition stacked bar renderer for main fig 09 |
| `9a7b26b` | feat(fig): main fig 09 = variance bars + anchor sweep grid |
| `f6450bc` `ce85542` `d531f89` | SI panel renderers (residual small-multiples, seed CI table, anchor ranking) |
| `5d74c42` | fix(fig): NaN-safe anchor magnitude + residual small-multiples |
| `ee73411` | feat(fig): SI fig 09 adds residual small-multiples + seed CI + anchor ranking |
| `9baf045` | feat(a3): variance_partition supports strip_factor for within-anchor view |

Data:
- `src/a3_combinatorial_sweep/out/generated/{anchor}/*.png` — seed 42 (existing)
- `src/a3_combinatorial_sweep/out/generated_s43/{anchor}/*.png` — 540 PNGs
- `src/a3_combinatorial_sweep/out/generated_s44/{anchor}/*.png` — 540 PNGs
- CellViT sidecars: 1620/1620 (all 3 seeds)
- `morphological_signatures.csv` — 1620 rows, a4 schema, `seed` column
- `variance_partition.csv` — 16 metrics

Figures currently on disk (updated through T16):
- `figures/pngs_updated/09_combinatorial_grammar.png` — Panel A (variance bars, full data) + Panel B (3×9 sweep grid)
- `figures/pngs_updated/SI_09_combinatorial_grammar_anchors.png` — 4-quadrant SI

## Outstanding work (T17 Changes 2–4)

User-approved upgrades not yet applied:

### Change 2 — emit `variance_partition_within.csv`

File: `src/a3_combinatorial_sweep/main.py`
- Extend `_summary_output_paths` to 5-tuple (append `out_dir / "variance_partition_within.csv"`).
- In `run_summary_worker`, after writing `variance_partition.csv`, compute and write within-anchor variant using the new `strip_factor="anchor_id"` kwarg already added in `9baf045`:
  ```python
  within_shares = variance_partition(signature_rows, metrics=MORPHOLOGY_METRICS, strip_factor="anchor_id")
  within_rows = [{"metric": m, **within_shares[m]} for m in MORPHOLOGY_METRICS]
  _write_csv(within_path, within_rows, variance_fieldnames)
  ```
- Update 4-tuple unpack and return statement.
- Run `python -m src.a3_combinatorial_sweep.main --worker summarize --out-dir src/a3_combinatorial_sweep/out` to regenerate.

Verify: `tests/test_a3_combinatorial_sweep_worker.py` may assert old tuple length — update if so.

Commit: `feat(a3): emit variance_partition_within.csv`

### Change 3 — main fig 09: 3-panel + legend below + top-anchor pick

File A: `src/paper_figures/fig_combinatorial_grammar_panels/_variance_bars.py`
- Add `title: str | None = None` kwarg.
- Use `ax.set_title(title or "Variance partition by metric (sorted by interaction share)", fontsize=10)`.
- Replace legend call with:
  ```python
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6, fontsize=7, frameon=False)
  ```

File B: `src/paper_figures/fig_combinatorial_grammar.py::build_combinatorial_grammar_figure`
- `figsize=(7.5, 12.0)`.
- `gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.2], hspace=0.45)`.
- Panel A1 (`gs[0]`): `draw_variance_bars(ax, variance_csv, title="Variance partition (full data)")`.
- Panel A2 (`gs[1]`): same renderer with `variance_partition_within.csv` and title `"Variance partition (within-anchor)"`. Resolve path as `signatures_csv.parent / "variance_partition_within.csv"`.
- Panel B (`gs[2]`): keep existing `_draw_anchor_sweep_grid` call, but change anchor pick to top by sweep magnitude:
  ```python
  panel_b_anchor = sorted(magnitudes.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
  ```

Verify: regen via `from src.paper_figures.fig_combinatorial_grammar import save_combinatorial_grammar_figure; save_combinatorial_grammar_figure()` and run `pytest tests/test_fig_combinatorial_grammar.py -v`.

Commit: `feat(fig): main fig 09 adds within-anchor variance panel, legend below bars`

### Change 4 — regen SI + commit both PNGs

Regen SI fig for consistency:
```
python -c "from src.paper_figures.fig_combinatorial_grammar_si import save_combinatorial_grammar_si_figure; save_combinatorial_grammar_si_figure()"
```

Commit: `fig: regenerate combinatorial-grammar main + SI with within-anchor view`

## Known issues / deferred polish

1. **Anchor variance dominates 90–99% per metric.** Headline finding: model is mask-driven; the (state × O₂ × glucose) conditioning sweep moves morphology only marginally relative to anchor-to-anchor variation. Within-anchor partition (T17 Change 2/3) surfaces grammar signal but does not change the underlying finding.
2. **3 metrics are all-NaN by construction**: `intensity_mean_h`, `intensity_mean_e`, `appearance.stain_vector_angle_deg`. CellViT JSON sidecars don't expose per-nucleus intensity; stain-vector estimator degenerates on uniform inputs. Currently filtered at render time in `_variance_bars.py` and `_residual_small_multiples.py`. Cleaner fix (not done): drop them from `MORPHOLOGY_METRICS` tuple in `src/a3_combinatorial_sweep/main.py`.
3. **Anchor ranking uses raw variance**: dominated by metrics with large absolute scale (`nuclear_area_mean ~340` vs `appearance.h_mean ~0.01`). Would be cleaner with z-scored variance. Deferred per user.
4. **Codex sandbox `.git` read-only**: encountered intermittently during T9, T17. Workaround: dispatch Codex with "no commit" rule; commit from main session.

## Subagent failure note

T17 first attempt (`ae255bb1fb2e2f6a9`) timed out silently after ~11 min. T17 retry committed Change 1 only before hitting the .git read-only block. Subsequent dispatch interrupted by user. The remaining 3 changes are mechanical and well-specified above.

## How to finish

From `/home/ec2-user/PixCell`:

```bash
# Change 2 (manual or via codex)
# edit src/a3_combinatorial_sweep/main.py per spec above
conda run --no-capture-output -n pixcell python -m src.a3_combinatorial_sweep.main \
  --worker summarize --out-dir src/a3_combinatorial_sweep/out
conda run --no-capture-output -n pixcell pytest tests/test_a3_combinatorial_sweep_worker.py -v
git add src/a3_combinatorial_sweep/main.py
git commit -m "feat(a3): emit variance_partition_within.csv"

# Change 3
# edit two fig files per spec above
conda run --no-capture-output -n pixcell python -c \
  "from src.paper_figures.fig_combinatorial_grammar import save_combinatorial_grammar_figure; save_combinatorial_grammar_figure()"
conda run --no-capture-output -n pixcell pytest tests/test_fig_combinatorial_grammar.py -v
git add src/paper_figures/fig_combinatorial_grammar_panels/_variance_bars.py src/paper_figures/fig_combinatorial_grammar.py
git commit -m "feat(fig): main fig 09 adds within-anchor variance panel, legend below bars"

# Change 4
conda run --no-capture-output -n pixcell python -c \
  "from src.paper_figures.fig_combinatorial_grammar_si import save_combinatorial_grammar_si_figure; save_combinatorial_grammar_si_figure()"
git add figures/pngs_updated/09_combinatorial_grammar.png figures/pngs_updated/SI_09_combinatorial_grammar_anchors.png
git commit -m "fig: regenerate combinatorial-grammar main + SI with within-anchor view"
```
