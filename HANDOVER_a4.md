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

- `src/a4_uni_probe/main.py`: CLI for `probe`, `sweep`, `null`, `figures`, `appearance`
- `src/a4_uni_probe/probe.py`: Stage 1 probe fitting and ranking
- `src/a4_uni_probe/edit.py`: UNI sweep/null edits, summaries, and run orchestration
- `src/a4_uni_probe/inference.py`: thin wrapper around Stage 3 generation with UNI override
- `src/a4_uni_probe/labels.py`: channel-derived and CellViT-derived tile attributes
- `src/a4_uni_probe/metrics.py`: generated-image morphology sidecar reader
- `src/a4_uni_probe/figures.py`: summary figure generation
- `src/a4_uni_probe/appearance_metrics.py`: stain, stain-distance, and Haralick texture metrics

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
15. Added a concat-scoped `uni_null` generation path for the exact a4 null tile set via `run_a2_uni_null.py`.
16. Added `update_null_full_uni.py` to append `full_uni_null` rows into the null metrics CSVs and regenerate summaries/figures.
17. Generated the exact concat `uni_null` tile set for the a4 null tiles.
18. Ran local CellViT on the `tme_only.png` set via `/home/ec2-user/he-feature-visualizer/stages/run_cellvit_local.py` and copied the JSONs back beside each original image as `tme_only.png.json`.
19. Re-ran `update_null_full_uni.py`, so `full_uni_null` is now populated in the a4 null summaries and figures.
20. Added a post-hoc `appearance` pass that computes stain and texture metrics from the existing generated PNGs without rerunning diffusion generation.
21. Re-scored the concat a4 outputs with the appearance pass and wrote complete appearance summaries for both sweep and null.
22. Extended the figure renderer with full-metric appearance bar charts so the summary does not depend on selectively naming a few metrics in prose.

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
- The full-UNI-null baseline is now available and is lower than targeted/random for all three null-readout attributes, which supports the broader claim that UNI carries morphology information beyond the selected one-axis probe edits.

## Appearance extension

The a4 run now includes a second layer of readouts on the already-generated images:

- H/E stain statistics: `appearance.h_mean`, `appearance.h_std`, `appearance.e_mean`, `appearance.e_std`
- stain-vector distance to the paired real patch: `appearance.stain_vector_angle_deg`
- Haralick texture features on H and E: contrast, homogeneity, energy

These metrics are written by:

- `python -m src.a4_uni_probe.main appearance --out-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe --data-root /home/ec2-user/PixCell/data/orion-crc33`

Output files:

- `inference_output/a1_concat/a4_uni_probe/appearance_sweep_summary.csv`
- `inference_output/a1_concat/a4_uni_probe/appearance_null_summary.csv`
- `inference_output/a1_concat/a4_uni_probe/sweep/*/appearance_summary.csv`
- `inference_output/a1_concat/a4_uni_probe/null/*/appearance_summary.csv`

Important interpretation change:

- The evidence is no longer limited to cell morphology. Full UNI null produces much larger stain-vector shifts than targeted/random null across all three attrs, and it also moves multiple H/E texture metrics in a consistent direction.
- To avoid cherry-picking a few appearance metrics in prose, the full set is now shown as bar-chart panels instead of only being referenced selectively.

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
- `inference_output/a1_concat/a4_uni_probe/figures/panel_d_appearance_sweep_all_metrics.png`
- `inference_output/a1_concat/a4_uni_probe/figures/panel_e_appearance_null_all_metrics.png`
- `inference_output/a1_concat/a4_uni_probe/RESULTS_SUMMARY.md`

Generated-image CellViT exchange artifacts:

- `inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch`
- `inference_output/a1_concat/a4_uni_probe/cellvit_generated_batch.zip`
- `inference_output/a1_concat/a4_uni_probe/import_generated_cellvit.sh`

TME-only CellViT artifacts:

- `inference_output/a1_concat/a4_uni_probe/cellvit_tme_only_batch.zip`
- `inference_output/a1_concat/a4_uni_probe/cellvit_tme_only_results`
- `inference_output/a1_concat/a4_uni_probe/cellvit_tme_only_results.zip`

`full_uni_null` support artifacts:

- `run_a2_uni_null.py`
- `update_null_full_uni.py`
- `inference_output/a1_concat/a4_uni_probe/uni_null/generated`

## `full_uni_null` status

The old limitation was that `full_uni_null` stayed empty because the sampled concat `a2_decomposition` cache did not overlap the selected a4 null tiles.

That is no longer a structural blocker.

There is now a dedicated concat `uni_null` path under:

- `inference_output/a1_concat/a4_uni_probe/uni_null/generated`

and two helper scripts:

- `run_a2_uni_null.py`: generates the exact null-tile `tme_only.png` set using the concat checkpoint while loading the model only once.
- `update_null_full_uni.py`: appends `full_uni_null` rows into each null metrics CSV, rewrites `null_comparison.json`, and re-renders the a4 figures.

This is now populated for the current concat run. The only remaining edge case is tile `21504_17920`, where CellViT returned an empty `cells` list for `tme_only.png`; that leaves `nuclear_area_mean` with `n = 29` instead of `30` for `full_uni_null`.

## If someone needs to continue this work

Highest-value next steps:

1. Inspect or rerun tile `21504_17920` if you want to eliminate the one remaining non-finite `full_uni_null` value for `nuclear_area_mean`.

2. Re-run `nuclear_area_mean` with either more tiles or another seed to see whether it stabilizes into a clean pass.

3. Keep using the concat-scoped output root for any follow-up so results do not mix with the older default `src/a4_uni_probe/out` path.

4. If the `uni_null` workflow needs to be rerun from scratch, the sequence is now:

   `conda activate pixcell && cd /home/ec2-user/PixCell && python run_a2_uni_null.py`

   `conda activate he-multiplex && cd /home/ec2-user/PixCell && python update_null_full_uni.py`

   `python /home/ec2-user/he-feature-visualizer/stages/run_cellvit_local.py --zip <tme_only_batch.zip> --out <cellvit_out_dir> --checkpoint ~/checkpoints/CellViT-256.pth --cellvit-repo ~/CellViT --batch-size 8`

   then copy outputs back as `tme_only.png.json` and rerun `update_null_full_uni.py`.

5. If new generated-image CellViT outputs are imported anywhere in a4, rerun:

   `python -m src.a4_uni_probe.main figures --out-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe`

6. If the appearance summaries need to be regenerated, rerun:

   `conda activate he-multiplex && cd /home/ec2-user/PixCell && python -m src.a4_uni_probe.main appearance --out-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe --data-root /home/ec2-user/PixCell/data/orion-crc33`

   then rerun:

   `python -m src.a4_uni_probe.main figures --out-dir /home/ec2-user/PixCell/inference_output/a1_concat/a4_uni_probe`

## Environment notes

- Use `he-multiplex` for sklearn/tests/CPU-side a4 work.
- Use `pixcell` for generation runs.
