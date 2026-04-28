# TME ∞ Grad Norm — Resolution Log

Companion to `grad_explosion_tme_followup.md`. Records root cause, fix, smoke results, and next steps for scale-up.

## Symptom

Phase 0 smoke (depth=18, bs=2, max_train_samples=10):

```
concat       PASS  grad_norm_tme=0
additive     FAIL  grad_norm_tme=2.97e31  top=groups.vasculature.encoder.stem.1.bias
grouped      FAIL  grad_norm_tme=3.43e31  top=groups.vasculature.encoder.stem.1.bias
per_channel  FAIL  grad_norm_tme=4.07e31  top=channels.cell_state_dead.encoder.stem.1.bias
```

Pattern: only sparse-input variants fail. Concat passes because shared encoder mixes sparse channels with dense `cell_masks` — input never near-zero.

## Root Cause

Three compounding issues. Concat avoided all three by construction.

1. **H3 — `cross_attn.proj.weight` not zero-init.**
   `multi_group_tme.py:61` and `per_channel_tme.py:28` used `normal_(std=0.02)`. Fusion was NOT identity at step 0, so real grad flowed on first step. Concat path (`tme_encoder.py:207`) used `zeros_` — clean.
2. **H2 — `TMEInputNormalizer` divide-by-eps on near-uniform continuous channels.**
   Vasculature near-zero except thin tubes → `std ≈ 0` → `centered / (std + eps)` blew up to 1/eps.
3. **H1 — GroupNorm `1/√(var+ε)` singularity on sparse conv outputs.**
   Sparse channel → conv output ~0 → group var ~0 → bias grad scaled by `1/√ε`. With default `ε=1e-5`, per-element gain ~316, amplified through chain.

H1+H3 explains both vasculature and `cell_state_dead`. H2 only fits vasculature (continuous-only).

## Fix Applied

| File | Change |
|------|--------|
| `diffusion/model/nets/multi_group_tme.py:61-62` | `normal_(std=0.02)` → `zeros_` on `cross_attn.proj.weight` |
| `diffusion/model/nets/per_channel_tme.py:28` | same as above |
| `diffusion/model/nets/tme_encoder.py:64` | floor std: `std = std.clamp_min(scale * 1e-3)` before divide |
| `diffusion/model/nets/tme_encoder.py:77,79,91,120` | `nn.GroupNorm(8, c)` → `nn.GroupNorm(8, c, eps=1e-3)` (4 sites: ResBlock×2, DownBlock, TMEEncoder.stem) |

Per-channel variant was missed in first pass; second patch fixed it.

## Verification (Phase 0 smoke, post-fix)

```
concat       PASS  grad_norm_tme=0.000e+00   max_abs=0.000e+00   elapsed=31s
additive     PASS  grad_norm_tme=8.146e-04   max_abs=1.054e-04   elapsed=33s
grouped      PASS  grad_norm_tme=1.187e-03   max_abs=1.764e-04   elapsed=33s
per_channel  PASS  grad_norm_tme=7.783e-04   max_abs=8.726e-05   elapsed=35s
```

All variants `< 1e-3`, well under `PASS_THRESHOLD = 1e3`. Tests in `tests/test_multi_group_tme.py` + `tests/test_grad_explosion_tme_followup_debug.py` pass in training env.

## Next Steps

### Step 1 — Commit fix

4 source files + this doc + `tests/test_grad_explosion_tme_followup_debug.py` + `tools/debug/`. Single commit, message references this doc and the followup runbook.

### Step 2 — Mid-length smoke at target depth

Goal: catch late-onset blowup as zero-init residuals activate during training. Step-1 health ≠ step-500 health.

- Config: depth=27, bs=2, **100–500 steps**, `debug_tme_probe=True`, `log_interval=10`, `save_final_checkpoint=False`.
- Run all 4 variants (concat / additive / grouped / per_channel).
- Pass criterion: `grad_norm_tme < 10` sustained through step 500. Spikes OK if isolated and `_safe_clip_grad_norm_` absorbs them.
- Watch for: monotonic grad_norm growth (residuals diverging), per-group grad attribution shifting toward sparse channels (residual H1).
- If fails → escalate to Phase 1 deeper hardening:
  - Swap stem GroupNorm → `InstanceNorm2d(affine=True)`.
  - TME LR warmup (linear 0 → `tme_lr` over 500 steps).
  - Panic-skip (`grad_norm_tme > 1e6` → zero_grad + skip step).

### Step 3 — Full training run

Only after Step 2 clean.

- depth=27, full dataset, target schedule.
- Keep `_safe_clip_grad_norm_` as backstop.
- Log `grad_norm_tme` + `grad_health_tme.top_tensors` at standard `log_interval`.
- First checkpoint: inspect attention maps + per-group residual magnitudes against expectation (sparse channels should NOT dominate).

### Step 4 — Validate output quality

- Stage3 inference on held-out tiles, all 4 variants.
- Compare against `tools/stage3/run_evaluation.py` figures.
- If concat baseline still wins: the per-group variants need architecture work, not grad-stability work — separate investigation.

## Don't Do

- Don't relax `_safe_clip_grad_norm_` — keep as backstop even with fix in place.
- Don't change `cfg_dropout_prob` / group dropout (orthogonal axis).
- Don't skip Step 2. Depth 18 → 27 is +50% ControlNet capacity; step-1 stability does not generalize.
