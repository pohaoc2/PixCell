# Change Plan — `leave_one_out_diff.png` Pixel-Impact Upgrade

## Goal
Today's `leave_one_out_diff.png` shows raw mean-RGB |Δ| per pixel, normalized either globally or per-condition (99th-pct cell-masked). Reviewers can't tell:
1. Whether the change is **perceptual** (color-shifted) vs **structural** (texture/morphology).
2. Whether the change is **local to the dropped channel's footprint** (causal) or diffuse (entanglement).
3. How **strong** the per-channel impact is on a comparable, scalar scale.

Plan adds three orthogonal readouts on top of the existing pixel diff and one global summary, while keeping the current API/CLI surface backwards compatible.

---

## Files

| File | Role | Change |
|---|---|---|
| `tools/vis/leave_one_out_diff.py` | rendering + stats | new metric helpers, redesigned figure layout, expanded stats JSON |
| `tools/vis/leave_one_out_stats.py` | aggregator | extend schema parser to include new fields (back-compat read of old stats kept) |
| `tests/test_leave_one_out_diff.py` | unit tests | add cases for new helpers + golden snapshot of new layout shape |
| `tests/test_leave_one_out_stats_cli.py` | CLI tests | extend for new fields |
| `figures/pngs/leave_one_out_diff.png` | regen artifact | regenerate after code lands |

No changes needed: `tools/stage3/ablation_cache.py`, `tools/stage3/ablation_vis_utils.py`, manifest schema.

---

## Metric Additions

### M1. ΔE-CIELAB perceptual diff (replace mean-RGB |Δ| in heatmap)

- Convert `img_all`, `img_loo` from sRGB → CIELAB (D65) via `skimage.color.rgb2lab` (already-pulled dep). Fallback: hand-rolled sRGB→linear→XYZ→Lab if skimage missing (mirrors current SSIM fallback pattern).
- Per-pixel ΔE76: `sqrt((L1−L2)² + (a1−a2)² + (b1−b2)²)`.
- Returns H×W float32. Use as the new "magnitude" channel.

Helper: `delta_e_lab_map(img_all_uint8, img_loo_uint8) -> np.ndarray` next to `ssim_loss_map`.

### M2. Cell-mask causal score (inside vs outside)

For each LOO panel, given the **dropped channel's positive mask** (not the global cell mask):
- `inside_mean = mean(|Δ| inside dropped-channel mask)`
- `outside_mean = mean(|Δ| outside dropped-channel mask, but still inside cell_mask)`
- `causal_ratio = inside_mean / max(outside_mean, eps)`

Source for per-channel mask: read raw exp channels via `tools/stage3/ablation_cache` -> manifest `channel_inputs_path` (or fall back to `data/orion-crc33/exp_channels/<tile>/<channel>.png`). For grouped channels (e.g. `cell_state` = 3 sub-channels), union the binary supports.

Helper: `causal_score(diff_map, channel_mask, cell_mask) -> dict` returning `{inside_mean, outside_mean, causal_ratio, n_inside_pixels}`.

### M3. SSIM structural loss (already implemented)

Reuse `ssim_loss_map`. In the new figure it sits next to ΔE so reviewers can separate **color shift** (high ΔE, low 1−SSIM) from **morphology shift** (high 1−SSIM).

### M4. UNI-cosine drop (scalar, optional)

Per panel: `1 − cosine(UNI(img_all), UNI(img_loo))`.
- Reuses cached UNI features when manifest lists them. If absent, skip silently and don't render the strip.
- Helper: `uni_cosine_drop(cache_dir, manifest) -> dict[group, float]`.

---

## Figure Redesign — `leave_one_out_diff.png`

Current: 2 rows × 5 cols (ref H&E + 5 generated panels, generated row + diff row), single colorbar.

Proposed: 4 rows × 5 cols (one column per condition: All-channels, drop-cell_types, drop-cell_state, drop-vasculature, drop-microenv).

| Row | Content | Norm |
|---|---|---|
| 0 | Generated H&E (with cell-mask contour and dropped-channel-mask outline) | — |
| 1 | ΔE-CIELAB heatmap (cell-masked overlay, H&E grayscale background) | shared row colorbar, 99th-pct global |
| 2 | 1−SSIM map (cell-masked overlay) | shared row colorbar, 99th-pct global |
| 3 | Bar strip: per-condition `inside_mean` vs `outside_mean` ΔE bars + causal_ratio annotation; optional UNI-cos drop dot overlay | per-row absolute units (ΔE, dimensionless) |

Notes:
- All-channels column shows zeros (baseline) for rows 1–3; row 3 strip shows `0` reference bar.
- **Shared per-row colorbar**: fixes the current "auto-scale per panel makes magnitudes incomparable" problem.
- Use ΔE units on row 1 colorbar (e.g. 0–`ΔE_p99`) so values are interpretable, not normalized 0–1.
- Keep dashed-border reference H&E panel on the left (current layout) for style continuity.

Figure size grows from 15×4.45 → 15×9.5. `dpi=150` unchanged.

---

## Stats JSON Schema (`leave_one_out_diff_stats.json`)

Current per-group block:
```json
{ "mean_diff": ..., "max_diff": ..., "pct_pixels_above_10": ... }
```

Extended (adds, never removes — old fields stay for back-compat):
```json
{
  "mean_diff": ...,                   # legacy mean RGB |Δ| (0..255)
  "max_diff": ...,
  "pct_pixels_above_10": ...,
  "delta_e_mean": ...,                # NEW: ΔE76 mean over cell mask
  "delta_e_p99": ...,                 # NEW
  "ssim_loss_mean": ...,              # NEW: 1-SSIM mean over cell mask
  "ssim_loss_p99": ...,               # NEW
  "causal_inside_mean_dE": ...,       # NEW: ΔE inside dropped channel mask
  "causal_outside_mean_dE": ...,      # NEW
  "causal_ratio": ...,                # NEW
  "uni_cosine_drop": ...              # NEW (nullable; null when UNI features absent)
}
```

`save_loo_stats` extended; aggregator in `leave_one_out_stats.py` updated to read new fields with `.get(..., None)` so old runs still parse.

---

## Backwards Compatibility

- Existing `--figure diff|ssim|both` flags preserved; new metrics ride inside the existing `--figure diff` output path.
- Existing legacy `--ssim` alias preserved.
- New optional flag `--no-causal` to skip channel-mask reads when exp channel files are unavailable (e.g. unpaired runs without raw masks).
- New optional flag `--metric {delta_e,rgb}` defaulting to `delta_e`. `--metric rgb` reproduces the current pixel-diff for reproducibility.
- Stats JSON gains fields but keeps old keys: any downstream consumer (paper figure scripts under `src/paper_figures/`) reading `mean_diff` still works.

---

## Implementation Steps (ordered, each independently testable)

1. **Add `delta_e_lab_map` + tests.** Pure function, no figure changes. Verify against synthetic constant-color shift (e.g. ΔL=10 → ΔE=10).
2. **Add `causal_score` + tests.** Synthetic diff + synthetic mask; assert ratio behaves as expected on both signal-in-mask and signal-outside-mask cases.
3. **Extend `save_loo_stats`** to write new fields. Update `tests/test_leave_one_out_stats_cli.py` to cover new keys.
4. **Refactor `render_loo_diff_figure`** to 4-row layout. Keep current 2-row branch reachable behind `--legacy-layout` for one release for safety. Tests assert `(rows, cols)` shape via `fig.axes`.
5. **Wire UNI-cosine drop** when `manifest.get("uni_features_path")` exists. Tests skip when feature unavailable.
6. **Regenerate `figures/pngs/leave_one_out_diff.png`** after code lands. Spot-check tile ID in commit message.
7. **Update `STORYLINE.md`** Section 3 to point at the new fields/rows.

---

## Validation Checklist

- [ ] Unit tests pass (`pytest tests/test_leave_one_out_diff.py tests/test_leave_one_out_stats_cli.py`).
- [ ] CLI `python tools/vis/leave_one_out_diff.py --cache-dir <dir>` produces 4-row figure.
- [ ] `--metric rgb --legacy-layout` reproduces current figure byte-identically (or near-identical).
- [ ] Stats JSON contains all new keys; old consumer scripts still load.
- [ ] Causal-ratio sanity: drop-vasculature has `causal_ratio > 1.5` on tiles with visible vessels; drop-microenv ratio is closer to 1 (diffuse field). Inspect ≥3 tiles.
- [ ] ΔE row reveals expected color shift on drop-cell_state (eosin shift in proliferating regions); SSIM row reveals expected morphology shift on drop-cell_types.

---

## Out of Scope (defer)

- Replacing ΔE76 with ΔE2000 (marginal gain, more code).
- Per-cell instance attribution (would need segmentation, separate effort).
- Animated GIF across LOO conditions (nice-to-have; not for paper figure).

---

## Owner / Execution

Per `CLAUDE.md`, Claude plans only. Implementation delegated to Codex via `codex:codex-rescue` with this plan as the brief. Claude reviews the resulting diff and the regenerated PNG.
