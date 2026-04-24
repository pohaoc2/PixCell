# Handover

Date: 2026-04-23

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
  - Status: complete
  - Full decomposition sweep is finished: `500` tiles × `4` modes = `2000` generated images.
  - Validation summary:
    - `mode_metrics.csv` contains `2000` rows across `500` tiles.
    - `mode_summary.csv` contains `4` mode rows with `n_tiles=500` and `reference_count=500` for each mode.
  - Runtime note:
    - Backfilling from the earlier `N=2` snapshot to the full `N=500` set completed on this T4 host in approximately `65` minutes.
  - Produced generated examples plus:
    - `mode_metrics.csv`
    - `mode_summary.csv`

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

## Important implementation notes

- `a1_probe_linear.main` now supports target-level parallel CV execution via `--n-jobs`.
- `a1_probe_mlp.main` threads the same `--n-jobs` option through to the shared CV helper.
- `a1_codex_targets.probe` also accepts `--n-jobs`, but the user is actively modifying CODEX now, so avoid further edits there unless requested.
- `a1_probe_mlp` does not checkpoint mid-stage; it writes outputs only after the full run completes.
- `a1_codex_targets.probe` runs stages sequentially in order: `t2_linear -> t2_mlp -> t3_linear -> t3_mlp`.

## Recommended next steps

1. Review the updated `figures/pngs/07_inverse_decoding.png` now that it includes CTransPath alongside UNI-2h and Virchow2.
2. Compare encoder-level T1 results across:
   - `src/a1_probe_linear/out/linear_probe_results.csv`
   - `src/a1_probe_encoders/out/virchow2_linear_probe_results.csv`
   - `src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv`
3. If more reporting is needed, consolidate the key results from:
   - `src/a1_probe_mlp/out/comparison_vs_linear.csv`
   - `src/a1_probe_encoders/out/encoder_comparison.csv`
   - `src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv`
   - `src/a1_generated_probe/out/real_vs_generated_r2.csv`
   - `src/a2_decomposition/out/mode_summary.csv`
   - `src/a3_combinatorial_sweep/out/morphological_signatures.csv`
4. Keep CODEX source edits user-driven for now; do not modify CODEX code again unless explicitly requested.

## Summary of what still needs active attention

- No major section-11 runtime tasks remain active on this host.
- All sweep generation and summary work is complete.
- All section-11 tasks relevant to this run are materially complete on this host.
- Figure 4 follow-up:
  - `07_inverse_decoding.png` has been rebuilt and now includes CTransPath results.
  - Supporting artifacts are present:
    - `src/a1_probe_encoders/out/ctranspath_embeddings.npy`
    - `src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv`
