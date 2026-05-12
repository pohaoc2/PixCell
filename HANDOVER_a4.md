# HANDOVER: a4 UNI Probe

## What a4 is for

`a4_uni_probe` exists to answer a question that the earlier ablations could not answer cleanly:

> What biological or morphological information is carried by the UNI appearance prior beyond the spatial TME channels, and does the model causally use that UNI information during generation?

In practice, a4 does three things:

1. `probe`: fit linear probes to compare how well UNI features versus a pooled TME baseline predict per-tile attributes.
2. `sweep`: edit UNI along learned probe directions and test whether generated H&E changes in the expected morphology direction while TME is held fixed.
3. `null`: remove the learned UNI component and compare targeted-null versus random-null degradation.

The intended claim is not "UNI is semantic and TME is spatial." The useful split is:

- UNI = global appearance prior
- TME channels = spatially resolved biological layout

So a4 is the experiment that tries to isolate what morphology/appearance information UNI contributes beyond the explicit channel maps.

## What was implemented

I implemented the `src/a4_uni_probe/` package and its tests.

Key modules:

- `src/a4_uni_probe/main.py`: CLI for `probe`, `sweep`, `null`, `figures`
- `src/a4_uni_probe/probe.py`: Stage 1 probe fitting and ranking
- `src/a4_uni_probe/edit.py`: UNI sweep/null edits, summaries, and run orchestration
- `src/a4_uni_probe/inference.py`: thin wrapper around Stage 3 generation with UNI override
- `src/a4_uni_probe/labels.py`: channel-derived and CellViT-derived tile attributes
- `src/a4_uni_probe/metrics.py`: generated-image morphology sidecar reader
- `src/a4_uni_probe/figures.py`: summary figure generation

Test coverage added under `tests/` for labels, features, probe fitting, edit helpers, and metrics sidecar parsing.

## What was done operationally

The implementation work turned into a full end-to-end concat run under:

- `inference_output/a1_concat/a4_uni_probe`

Major execution steps completed:

1. Ran Stage 1 probe generation and cached labels/features.
2. Prepared/exported real-H&E images for offline CellViT morphology extraction.
3. Patched morphology loading to support the imported CellViT `cells` schema in addition to the expected `nuclei` schema.
4. Switched the active a4 run to the concat checkpoint family, specifically `concat_95470_0`, and wrote outputs into `inference_output/a1_concat/a4_uni_probe`.
5. Optimized generation to reuse a single loaded inference bundle instead of reloading the model stack per image.
6. Added tile sharding so sweep/null could be split across two terminals.
7. Ran sharded sweep with reduced runtime settings:
   - `k-tiles = 30`
   - `alphas = -1 0 1`
   - `top-k-attrs = 4`
   - `num-steps = 20`
8. Patched null execution so the full-UNI-null baseline root is configurable instead of hardcoded to the default `a2` location.
9. Ran sharded null for the concat variant.
10. Exported generated sweep/null PNGs for offline CellViT.
11. Imported the generated-image CellViT outputs back onto the original images as sidecars.
12. Patched generated-image metric loading so `.png.json` sidecars are recognized.
13. Recomputed summaries and regenerated figures after CellViT import.
14. Wrote a short run summary in `inference_output/a1_concat/a4_uni_probe/RESULTS_SUMMARY.md`.

## Important fixes made during the run

These were required to get valid a4 results:

- Channel layout mismatch: the real repo uses channel-major layout `exp_channels/<channel>/<tile>`, not the plan's example layout.
- CellViT schema mismatch: imported CellViT JSONs used `cells` with contours, not only `nuclei`.
- Generated-metric sidecar mismatch: imported generated-image outputs landed as `image.png.json`, so metrics code had to recognize that naming.
- Sweep runtime issue: inference bundle reuse was necessary to avoid paying model-load cost for every image.
- Null baseline mixing risk: concat null runs needed an explicit concat `full_null_root` so outputs were not mixed with another variant.

## Current result summary

Top Stage 1 morphology directions by UNI-over-TME delta $R^2$:

1. `eccentricity_mean`: `0.2309`
2. `nuclear_area_mean`: `0.1594`
3. `nuclei_density`: `0.1094`

Sweep outcomes:

- `eccentricity_mean`: targeted slope `0.0596`, random `0.0030`, pass
- `nuclear_area_mean`: targeted slope `56.05`, random `-18.69`, no pass under the preset rule
- `nuclei_density`: targeted slope `1.615e-4`, random `5.13e-6`, pass

Interpretation:

- The concat checkpoint shows usable directional control for eccentricity and nuclei density.
- Nuclear area shows signal, but not cleanly enough relative to the random-direction control to count as a pass.
- Null effects are modest, which suggests the selected probe axis carries some leverage but morphology is not concentrated in a single isolated UNI direction.

## Main artifacts to know about

Primary output root:

- `inference_output/a1_concat/a4_uni_probe`

Most useful files:

- `inference_output/a1_concat/a4_uni_probe/probe_results.csv`
- `inference_output/a1_concat/a4_uni_probe/probe_results.json`
- `inference_output/a1_concat/a4_uni_probe/sweep/*/slope_summary.json`
- `inference_output/a1_concat/a4_uni_probe/null/*/null_comparison.json`
- `inference_output/a1_concat/a4_uni_probe/figures/panel_a_probe_R2.png`
- `inference_output/a1_concat/a4_uni_probe/figures/panel_b_sweep_slope.png`
- `inference_output/a1_concat/a4_uni_probe/figures/panel_c_null_drop.png`
- `inference_output/a1_concat/a4_uni_probe/RESULTS_SUMMARY.md`

Generated-image CellViT exchange artifacts:

- `inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch`
- `inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch.zip`
- `inference_output/a1_concat/a4_uni_probe/import_generated_cellvit.sh`

## Remaining limitation

`full_uni_null` is still unavailable for this concat run because the sampled concat decomposition cache did not overlap the selected null tiles. That means the targeted-vs-random null comparison is populated and usable, but the stronger full-UNI-null baseline remained `NaN` for this run.

## If someone needs to continue this work

Highest-value next steps:

1. Regenerate concat `a2_decomposition` outputs for the exact null tile set so `full_uni_null` is populated.
2. Re-run `nuclear_area_mean` with either more tiles or another seed to see whether it stabilizes into a clean pass.
3. Keep using the concat-scoped output root for any follow-up so results do not mix with the older default `src/a4_uni_probe/out` path.
4. If new external CellViT outputs are imported, rerun:

   `python -m src.a4_uni_probe.main figures --out-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe`

## Environment notes

- Use `he-multiplex` for sklearn/tests/CPU-side a4 work.
- Use `pixcell` for generation runs.
