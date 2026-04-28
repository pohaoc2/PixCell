# TME ∞ Grad Norm — Follow-up Debug Plan

Companion to `HANDOVER_DEBUG.md`. Handover left input-norm + AMP-unscale fixes staged but unverified on GPU. Plan ordered cheapest → deepest.

## Phase 0 — Verify current fix fired

Disk now ~55G free. Smoke runnable.

1. Use the depth-18 bs=2 smoke configs with `debug_tme_probe = True` and `save_final_checkpoint = False`.
2. Run depth=18 bs=2 max_train_samples=10 across 4 variants: concat / additive / grouped / per-channel.
3. Read first-step probe log + `grad_health_tme.top_tensors`.
4. Pass: per-tensor `max_abs < 1e3`, pre-clip `grad_norm_tme < 1e3`. Fail → Phase 1.

Runner:

```bash
python -m tools.debug.grad_explosion_tme_followup
```

Summary-only mode for already-run smoke logs:

```bash
python -m tools.debug.grad_explosion_tme_followup --summary-only
```

## Phase 1 — Isolate offender if still hot

Hypotheses ranked by likelihood given handover evidence (`stem.1.bias` GroupNorm post-Conv1 dominated).

### H1 — GroupNorm assumption violated on sparse channels

Vasculature near-zero except thin tubes → group mean/var dominated by zeros → tiny var → bias grad blows through `1/std`.

- **Probe**: log per-group `grad_health_tme.top_tensors`. If `vasculature.encoder.stem.1.bias` is offender → confirmed.
- **Fix**: swap `nn.GroupNorm(8, c1)` → `nn.InstanceNorm2d(c1, affine=True)` in `tme_encoder.py:77, 79, 91`. Per-sample-per-channel; no within-group stationarity assumption.

### H2 — `TMEInputNormalizer` numerically degenerate on near-constant channels

Oxygen/glucose near-uniform → tiny std → `centered/(std+eps)` ~1/eps in fp16 → overflow downstream.

- **Probe**: in `tme_encoder.py:65` log `(std+eps).min()` and `normalized.abs().max()` per channel. `> 1e3` → confirmed.
- **Fix**: floor std relatively, e.g. `std.clamp_min(scale * 1e-3)`. Or skip normalization when `scale < eps_abs` (return zeros).

### H3 — Cross-attention output projection still receives huge grad despite small init

`multi_group_tme.py:61` uses `normal_(std=0.02)`, not zero. Tiny weights × huge KV magnitudes → big upstream grad.

- **Probe**: existing `_proj_grad_norms` log. If `proj_grad` ≫ others → confirmed.
- **Fix**: revert to `zeros_(proj.weight)`. OR add `LayerNorm` at encoder output (before flatten), separate from existing post-flatten `norm_kv`.

### H4 — Double-unscale via Accelerate

`accelerator.clip_grad_norm_` internally unscales. Calling `_unscale_gradients_if_available` first then `accelerator.clip_grad_norm_` on else-branch → double-unscale → grads / scale² → fp16 underflow next step.

- **Probe**: log `accelerator.scaler.get_scale()` and grad samples pre/post unscale.
- **Fix**: call `unscale_gradients` exactly once. Verify `train_controlnet_exp.py:368-387` branch logic — `if gradients_unscaled` should gate against `accelerator.clip_grad_norm_`.

## Phase 2 — Defensive hardening (only after root cause known)

Priority order:

1. **Panic-skip** (handover §69): pre-clip `grad_norm_tme > 1e6` → `zero_grad` + log + skip step. Prevents silent dead-update mode.
2. **TME LR warmup** (handover §68): linear 0 → `tme_lr` over 500 steps. Reduces early shock as zero-init residuals activate.
3. **Per-group grad attribution in JSONL**: surface top-tensor group prefix so plots can split.

## Phase 3 — If H1–H4 clean but norm still hot

Suspect outside TME module.

- **`ctrl_latent`**: log `ctrl_latent.abs().max()` at `train_controlnet_exp.py:346`. Huge → fusion residual amplifying.
- **`zero_mask_latent` subtraction**: `fused = fused - vae_mask`. If TME diverges, big residual remains. Probe per-group δ via existing `_tme_residuals` log.

## Deliverable per phase

`checkpoints/<smoke_run>/train_log.jsonl` + `train_log.log` first-step probe lines. Compare 4 variants. Decision: norm < 1e3 → stop; else → next hypothesis.

## Don't do

- No new normalization layers before confirming H1/H2.
- No changes to `cfg_dropout_prob` / group dropout (orthogonal).
- Keep `_safe_clip_grad_norm_` as backstop even after fix.

## Next action

Free old smoke checkpoints if needed, run Phase 0 smoke, paste the runner summary plus first-step `[debug_tme_probe]` lines from `train_log.log`.
