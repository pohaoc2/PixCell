# Plan: Batched Ablation Inference

## Problem

`generate_ablation_subset_cache.py` takes ~20s per tile on A100.  
Root cause: 15 ablation conditions are denoised **sequentially**, each running a full 20-step loop.

## Bottleneck Breakdown

| # | Bottleneck | Impact | Where |
|---|-----------|--------|-------|
| 1 | No batching across conditions | **~80% of wall time** | `tile_pipeline.py:608-622` — sequential loop over 15 conditions |
| 2 | Redundant context prep (4×) | ~5% | `generate_subset_cache_for_tile` calls `_prepare_ablation_context` 4 times for the same tile |
| 3 | Redundant `.to()` in `denoise` | ~1% | `inference_controlnet.py:164` — `controlnet_model.to(device, dtype)` on every call |

## Proposed Changes

### Step 1: Lift context preparation out of per-subset calls

**File:** `tools/stage3/tile_pipeline.py`

- Add a new public function `prepare_tile_context(tile_id, ...)` that calls `_prepare_ablation_context` once and returns the reusable context dict.
- Add a new function `generate_ablation_images_with_context(context, conditions, ...)` that accepts a pre-built context instead of raw tile args.
- `generate_ablation_images` becomes a thin wrapper: build context → delegate to `generate_ablation_images_with_context`.

**File:** `tools/stage3/generate_ablation_subset_cache.py`

- In `generate_subset_cache_for_tile`, call `prepare_tile_context` once, then pass the context to all four subset-size calls.

### Step 2: Batch conditions in the denoising loop

**File:** `train_scripts/inference_controlnet.py`

- Add `denoise_batched(latents_batch, uni_embeds_batch, ctrl_latent_batch, ...)`:
  - `latents_batch`: `(N, C, H, W)` — N conditions sharing the same fixed noise (tiled).
  - CFG doubles the batch to `2N`: `[uncond_1, ..., uncond_N, cond_1, ..., cond_N]`.
  - Single ControlNet pass on `(N, C, H, W)` control input → broadcast residuals for the `2N` base-model pass.
  - Single base-model pass on `(2N, C, H, W)`.
  - CFG split + scheduler step on `(N, C, H, W)`.
  - Returns `(N, C, H, W)` denoised latents.
- Keep the existing `denoise` function unchanged for backward compatibility.

**File:** `tools/stage3/tile_pipeline.py`

- In `generate_ablation_images_with_context`, replace the sequential condition loop with:
  1. Pre-compute all fused conditioning latents (fast, TME forward only).
  2. Stack into `(N, C, H, W)` tensor.
  3. Call `denoise_batched` once (or in chunks of `batch_size`).
  4. VAE-decode all N latents in one batched call.
- Add a `batch_size` parameter (default 8) to cap VRAM. With 4 groups: N=15 fits in one batch on A100 80GB; with `batch_size=8`, it splits into 2 passes.

### Step 3: Batch VAE decode

**File:** `tools/stage3/tile_pipeline.py`

- Replace per-condition `_decode_latents_to_image` call with a single batched VAE decode of `(N, 16, 32, 32)` → `(N, 3, 256, 256)`.
- Split result into individual numpy arrays afterward.

### Step 4: Remove redundant `.to()` in `denoise`

**File:** `train_scripts/inference_controlnet.py:164`

- Guard the `controlnet_model.to(device, dtype)` behind a check: skip if already on correct device/dtype. (Minor, but avoids 15 unnecessary traversals of the model parameter tree.)

## Files Modified

| File | Change |
|------|--------|
| `train_scripts/inference_controlnet.py` | Add `denoise_batched`; guard `.to()` |
| `tools/stage3/tile_pipeline.py` | Add `prepare_tile_context`, `generate_ablation_images_with_context`; batch fuse+denoise+decode |
| `tools/stage3/generate_ablation_subset_cache.py` | Lift context prep; pass context to subset calls |

## Files NOT Modified

- `denoise()` — kept as-is for backward compat (stage3_inference.py, run_evaluation.py, etc.)
- Config files, dataset code, model definitions — no changes needed.
- Tests — existing tests remain valid; new test for `denoise_batched` shape/equivalence added.

## Expected Speedup

| Scenario | Passes (controlnet+base) | Est. time (A100) |
|----------|-------------------------|-------------------|
| Current (sequential) | 15 × 20 = 300 | ~20s |
| Batched (N=15, 1 batch) | 1 × 20 = 20 (batched) | ~3-4s |
| Batched (N=8+7, 2 batches) | 2 × 20 = 40 (batched) | ~4-5s |

~4-6× speedup, bringing 5000 tiles from ~28 hours to ~5-6 hours.

## VRAM Budget (A100 80GB)

- Current per-step: batch=2 (CFG) × 16ch × 32×32 latents ≈ negligible; model weights dominate.
- Batched per-step: batch=30 (15 conditions × 2 CFG) — latent tensors grow to ~30× but are still small vs model weights (~2-3GB total).
- VAE decode of 15 images at once: ~15 × 3 × 256 × 256 × 4 bytes ≈ 12MB — trivial.
- Conservative `batch_size=8` default leaves ample headroom.

## Risks

1. **Numerical equivalence** — batched vs sequential denoising must produce identical outputs. Mitigated by: same fixed noise (tiled), same scheduler state, deterministic `torch.manual_seed`. Add a test asserting `max_abs_diff < 1e-3` between batched and sequential.
2. **OOM on smaller GPUs** — `batch_size` parameter lets users cap memory. Default of 8 is safe for 40GB+ GPUs.
3. **ControlNet batch broadcast** — ControlNet currently expects single-sample input. `denoise_batched` must either run ControlNet on `(N,...)` or run it per-condition and stack residuals. The former is cleaner if the ControlNet supports batch>1 (check `PixArtControlNet.forward`).

## Testing

- Unit test: `denoise_batched` produces same output as N sequential `denoise` calls (tolerance 1e-3).
- Integration test: run `generate_ablation_subset_cache.py` on 1 tile, diff manifest/images against cached baseline.
- Benchmark: time 10 tiles before/after on A100.
