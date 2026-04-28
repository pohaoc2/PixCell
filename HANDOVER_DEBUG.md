# Handover — TME Gradient Explosion Debug

## TL;DR

ControlNet + TME training was producing `Infinity` grad norms on depth=18 smoke runs. After an initial "safe clip" patch the `Infinity` was gone but TME pre-clip grad norm was still ~1e31, dominated by `stem.1.bias` (GroupNorm bias right after the first Conv). With clip coef ≈ 1e-31 the TME branch was effectively receiving zero update — symptom hidden, root cause unfixed.

This change adds **input normalization** for continuous TME channels and fixes **AMP unscale ordering** so clipping is no longer the only thing keeping training stable.

## What changed

| File | Change |
|---|---|
| `diffusion/model/nets/tme_encoder.py` | New `TMEInputNormalizer` (scale-invariant per-sample, per-channel standardization). `TMEEncoder.forward` can return intermediate activations for probe. |
| `diffusion/model/nets/multi_group_tme.py` | Pipe per-group `channels` names → continuous-channel indices → `TMEInputNormalizer` inside each `_GroupBlock`. Debug probe hook (`enable_debug_tme_probe`). |
| `diffusion/model/nets/per_channel_tme.py` | Same normalization pattern per channel. |
| `train_scripts/train_controlnet_exp.py` | `_unscale_gradients_if_available` runs before manual clip (correct AMP ordering). `_grad_health` now accepts `named_parameters` + `top_k` for offender attribution. `_safe_clip_grad_norm_` used for TME params. Debug probe wired to `config.debug_tme_probe`. |
| `configs/config_controlnet_exp.py` | New flag: `debug_tme_probe = False`. |
| `tests/test_multi_group_tme.py` | New test: extreme `oxygen × 1e6` → post-stem activations stay finite and `< 1e3`. |
| `tests/test_train_controlnet_exp.py` | New test: `_safe_clip_grad_norm_` correctly clips 1e30 gradients to unit norm. |

## Normalization design

Per-sample, per-channel standardization (InstanceNorm-like) inside the model, **not** in the dataset — so `stage3_inference.py` keeps working unchanged.

Mechanism (`TMEInputNormalizer.forward`):

1. Center: `centered = x - mean(H,W)`
2. Rescale-then-std (overflow-safe): divide by `max(|centered|)` before computing std, multiply std back. Avoids fp16/fp32 overflow when input range is ~1e6.
3. Normalize: `(centered) / (std + eps)`.

Applied only to continuous channels: `{vasculature, oxygen, glucose}`. Binary channels (`cell_masks`, 3 cell-type maps, 3 cell-state maps) pass through untouched.

## AMP / clip ordering

Previously `accelerator.clip_grad_norm_` and the manual `_safe_clip_grad_norm_` could run on **scaled** gradients. Now `_unscale_gradients_if_available(accelerator)` is called once on `sync_gradients` before any clip. Reported grad-norm logs are now in true (post-unscale) units.

## Verification

Unit tests (run in `pixcell` conda env):

```bash
conda activate pixcell
python -m pytest tests/test_multi_group_tme.py \
                 tests/test_train_controlnet_exp.py \
                 tests/test_paired_exp_dataset.py \
                 tests/test_channel_group_utils.py -q
```

All 42 pass locally.

GPU smoke (recommended next step — disk was full so not run by Claude):

```bash
# Free space first if needed (smoke checkpoints are large).
# Then enable the probe one time to confirm input ranges:
#   set debug_tme_probe = True in the smoke config
python train_scripts/train_controlnet_exp.py \
       --config configs/config_controlnet_exp_smoke_depth18_bs2_grouped_fixed.py
```

**Pass criterion:** TME `max finite grad norm < 1e3` across all four variants (concat, additive, grouped, per-channel) at depth=18, bs=2, max_train_samples=10. Compare against prior baseline (~1e31).

After confirmation, set `debug_tme_probe = False` (default) — probe logs are first-step only but should not be left enabled in long runs.

## Known caveats / follow-ups

- **GroupNorm in TME stem unchanged.** If grad norm is still elevated after step-1 normalization, consider swapping `nn.GroupNorm(8, c1)` → `nn.InstanceNorm2d(c1, affine=True)` in `tme_encoder.py:75`. Sparse channels (vasculature) violate GroupNorm's within-group stationarity assumption.
- **TME LR warmup not added.** Optional follow-up: linear 0→`tme_lr` warmup over first ~500 steps so zero-init projections receive sane updates.
- **Panic-skip on extreme grad.** Optional safety net: if pre-clip TME norm > 1e6, skip optimizer step + log instead of clipping. Prevents silent dead-update mode.
- **Default-fallback indices.** When `channel_names` is not provided to `_GroupBlock`, `_default_continuous_indices(group_name, n)` treats all channels of `vasculature` and `microenv` groups as continuous. Existing configs do pass `channels`, so this is only a safety net.
- **Disk full.** Project disk full at handover from accumulated smoke checkpoints. Not deleted; clear under `output/` before next training run.

## Files / configs of interest

- Smoke configs (untracked): `configs/config_controlnet_exp_smoke_depth18_bs2_*_fixed.py`
- Probe entry point: `MultiGroupTMEModule.enable_debug_tme_probe()` (called once in train loop on first sync step when `debug_tme_probe=True`)
- Offender attribution: `_grad_health(..., top_k=8)` returns top-8 tensors by `max_abs` — useful when investigating future blow-ups.

## Why "safe clip" alone wasn't enough — for future readers

`_safe_clip_grad_norm_` returns `total_norm` **before** scaling. A reported norm of 1e31 means the optimizer sees `clip_coef = max_norm / 1e31 ≈ 1e-31`, effectively zeroing the TME update. Training looks stable (no NaNs, controlnet trains fine) while the TME branch silently freezes. Always check pre-clip `max_abs` per-tensor, not just the post-clip norm.
