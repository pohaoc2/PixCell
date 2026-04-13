# TME Proj Gradient Starvation Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `proj` layer gradient starvation in `MultiGroupTMEModule` by splitting the TME optimizer into two param groups (proj at 3e-4, rest at 1e-5), resetting optimizer state on resume, and adding per-group residual + gradient-norm diagnostics.

**Architecture:** The `_GroupBlock.cross_attn.proj` layers are zero-initialized; they need a high LR to escape that regime while the healthy CNN/Q/K/V parameters must not be destabilized. A two-group AdamW optimizer targets proj layers specifically. The resume block drops old optimizer state (single-group checkpoint → two-group optimizer would hard-crash otherwise). Two diagnostic log lines give immediate visibility: residual magnitudes and proj gradient norms. Grad norms are captured in the first `sync_gradients` block (before `optimizer.step()`) and residuals are stored from every TME forward; both are emitted in the second `sync_gradients` block (after `global_step += 1`) so they always appear at the same step number.

**Tech Stack:** PyTorch `AdamW` with param groups, HuggingFace Accelerate, standard Python `logging`.

**Spec:** `docs/superpowers/specs/2026-03-25-tme-proj-gradient-fix-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `tests/test_train_controlnet_exp.py` | Modify | New tests: param-split filter, optimizer LRs, production path integration, resume-reset |
| `train_scripts/training_utils.py` | Modify | Split TME optimizer into two param groups |
| `train_scripts/train_controlnet_exp.py` | Modify | Resume block replacement; residual + grad-norm logging |
| `configs/config_controlnet_exp.py` | Modify | Revert `tme_lr=1e-5`; add `tme_proj_lr=3e-4`, `reset_tme_optimizer=True` |

---

## Task 1: Write failing tests — param-split filter, optimizer LRs, production path

**Files:**
- Modify: `tests/test_train_controlnet_exp.py`

- [ ] **Step 1: Append tests to `tests/test_train_controlnet_exp.py`**

```python
# ── Split-optimizer (TME proj LR fix) ────────────────────────────────────────

def _make_tme_module():
    from diffusion.model.nets.multi_group_tme import MultiGroupTMEModule
    channel_groups = [
        dict(name="cell_identity", n_channels=3),
        dict(name="cell_state",    n_channels=3),
        dict(name="vasculature",   n_channels=1),
        dict(name="microenv",      n_channels=2),
    ]
    return MultiGroupTMEModule(channel_groups=channel_groups)


def test_proj_param_filter_captures_all_proj_layers():
    """'cross_attn.proj' filter must capture exactly weight+bias for all 4 groups."""
    tme = _make_tme_module()
    proj_names  = [n for n, _ in tme.named_parameters() if "cross_attn.proj" in n]
    other_names = [n for n, _ in tme.named_parameters() if "cross_attn.proj" not in n]

    # 4 groups × (proj.weight + proj.bias) = 8
    assert len(proj_names) == 8, f"Expected 8 proj params, got {len(proj_names)}: {proj_names}"
    # filter is exhaustive — no param lost or double-counted
    total = sum(1 for _ in tme.named_parameters())
    assert len(proj_names) + len(other_names) == total
    for name in proj_names:
        assert "cross_attn.proj" in name


def test_split_tme_optimizer_has_correct_lrs():
    """Two-group AdamW must assign tme_proj_lr to proj params and tme_lr to the rest."""
    tme = _make_tme_module()
    proj_params  = [p for n, p in tme.named_parameters() if "cross_attn.proj" in n]
    other_params = [p for n, p in tme.named_parameters() if "cross_attn.proj" not in n]

    opt = torch.optim.AdamW(
        [{"params": proj_params,  "lr": 3e-4},
         {"params": other_params, "lr": 1e-5}],
        weight_decay=0.0,
    )

    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4), "proj group LR"
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-5), "other group LR"


def test_build_tme_creates_split_optimizer_when_proj_lr_set():
    """_build_tme_module_and_optimizers creates two param groups when tme_proj_lr is set."""
    from train_scripts.training_utils import _build_tme_module_and_optimizers
    from unittest.mock import MagicMock, patch
    import types

    config = types.SimpleNamespace(
        tme_model="MultiGroupTMEModule",
        tme_base_ch=32,
        tme_proj_lr=3e-4,
        tme_lr=1e-5,
        channel_groups=[
            dict(name="cell_identity", channels=["a", "b", "c"]),
            dict(name="microenv",      channels=["x", "y"]),
        ],
        optimizer={"type": "AdamW", "lr": 5e-6, "weight_decay": 0.0,
                   "betas": (0.9, 0.999), "eps": 1e-8},
        lr_schedule_args={"num_warmup_steps": 10},
        num_epochs=1,
    )
    controlnet = MagicMock()
    dataloader = MagicMock()
    dataloader.__len__ = lambda self: 10
    logger     = MagicMock()
    active_ch  = ["cell_masks", "a", "b", "c", "x", "y"]

    # Patch the controlnet optimizer and scheduler builders so only the TME path runs
    with patch("train_scripts.training_utils.build_optimizer") as mock_opt, \
         patch("train_scripts.training_utils.build_lr_scheduler") as mock_sched:
        mock_opt.return_value   = MagicMock()
        mock_sched.return_value = MagicMock()
        result = _build_tme_module_and_optimizers(
            config, controlnet, dataloader, active_ch, logger
        )

    opt = result["optimizer_tme"]
    assert len(opt.param_groups) == 2, f"Expected 2 param groups, got {len(opt.param_groups)}"
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4), "proj group should be 3e-4"
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-5), "other group should be 1e-5"
```

- [ ] **Step 2: Run tests — expect `test_build_tme_creates_split_optimizer_when_proj_lr_set` to FAIL, others to PASS**

```bash
cd /home/ec2-user/PixCell && python -m pytest \
  tests/test_train_controlnet_exp.py::test_proj_param_filter_captures_all_proj_layers \
  tests/test_train_controlnet_exp.py::test_split_tme_optimizer_has_correct_lrs \
  tests/test_train_controlnet_exp.py::test_build_tme_creates_split_optimizer_when_proj_lr_set \
  -v
```

Expected:
- `test_proj_param_filter_captures_all_proj_layers` — PASS (regression guard, no code change needed)
- `test_split_tme_optimizer_has_correct_lrs` — PASS (regression guard)
- `test_build_tme_creates_split_optimizer_when_proj_lr_set` — **FAIL** (`param_groups` has 1 group, not 2)

If the last test unexpectedly passes, stop and investigate before continuing.

---

## Task 2: Implement split-optimizer in `training_utils.py`

**Files:**
- Modify: `train_scripts/training_utils.py:69-72`

- [ ] **Step 1: Replace the TME optimizer construction block**

In `_build_tme_module_and_optimizers`, replace lines 69–72:

```python
    tme_optimizer_cfg       = deepcopy(config.optimizer)
    tme_optimizer_cfg['lr'] = getattr(config, "tme_lr", config.optimizer.get('lr', 1e-4))
    optimizer_tme    = build_optimizer(tme_module, tme_optimizer_cfg)
    lr_scheduler_tme = build_lr_scheduler(config, optimizer_tme, train_dataloader, lr_scale_ratio=1)
```

with:

```python
    tme_proj_lr = getattr(config, "tme_proj_lr", None)
    if tme_proj_lr is not None:
        proj_params  = [p for n, p in tme_module.named_parameters() if "cross_attn.proj" in n]
        other_params = [p for n, p in tme_module.named_parameters() if "cross_attn.proj" not in n]
        base_tme_lr  = getattr(config, "tme_lr", 1e-5)
        optimizer_tme = torch.optim.AdamW(
            [{"params": proj_params,  "lr": tme_proj_lr},
             {"params": other_params, "lr": base_tme_lr}],
            weight_decay=config.optimizer.get("weight_decay", 0.0),
            betas=tuple(config.optimizer.get("betas", (0.9, 0.999))),
            eps=config.optimizer.get("eps", 1e-8),
        )
    else:
        tme_optimizer_cfg       = deepcopy(config.optimizer)
        tme_optimizer_cfg['lr'] = getattr(config, "tme_lr", config.optimizer.get('lr', 1e-4))
        optimizer_tme = build_optimizer(tme_module, tme_optimizer_cfg)
    lr_scheduler_tme = build_lr_scheduler(config, optimizer_tme, train_dataloader, lr_scale_ratio=1)
```

- [ ] **Step 2: Run tests — all three new tests should now PASS**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_train_controlnet_exp.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add train_scripts/training_utils.py tests/test_train_controlnet_exp.py
git commit -m "feat: split TME optimizer — proj at 3e-4, others at 1e-5"
```

---

## Task 3: Failing test — resume-reset skips optimizer load

**Files:**
- Modify: `tests/test_train_controlnet_exp.py`

- [ ] **Step 1: Append test**

```python
def test_load_tme_checkpoint_reset_skips_optimizer(tmp_path):
    """With optimizer_tme=None, model weights load but optimizer state is not touched."""
    from diffusion.model.nets.multi_group_tme import MultiGroupTMEModule
    from train_scripts.training_utils import load_tme_checkpoint

    channel_groups = [dict(name="g1", n_channels=1)]
    src_tme = MultiGroupTMEModule(channel_groups=channel_groups)

    with torch.no_grad():
        for p in src_tme.parameters():
            p.fill_(0.42)

    ckpt = {
        "step": 4890,
        "epoch": 30,
        "model_state": src_tme.state_dict(),
        # Single-group state — loading into a two-group optimizer would crash without reset
        "optim_state": {"state": {}, "param_groups": [{"lr": 1e-5, "betas": (0.9, 0.999),
                                                        "eps": 1e-8, "weight_decay": 0.0,
                                                        "amsgrad": False, "params": []}]},
        "sched_state": {"last_epoch": 4890},
    }
    torch.save(ckpt, tmp_path / "tme_module.pth")

    dst_tme = MultiGroupTMEModule(channel_groups=channel_groups)
    step = load_tme_checkpoint(str(tmp_path), dst_tme, optimizer_tme=None, lr_scheduler_tme=None)

    assert step == 4890
    for p in dst_tme.parameters():
        assert torch.allclose(p, torch.tensor(0.42)), "model weights must transfer"
```

- [ ] **Step 2: Run test — expected PASS (`load_tme_checkpoint` already accepts `None`)**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_train_controlnet_exp.py::test_load_tme_checkpoint_reset_skips_optimizer -v
```

If it fails, there is a pre-existing bug in `load_tme_checkpoint` to fix before continuing.

---

## Task 4: Replace resume block in `train_controlnet_exp.py`

**Files:**
- Modify: `train_scripts/train_controlnet_exp.py:421-427`

- [ ] **Step 1: Replace (do not add a second block) the unconditional resume block at lines 421–427**

Find and replace this exact block:

```python
    tme_ckpt = getattr(config, "resume_tme_checkpoint", None)
    if tme_ckpt:
        step = load_tme_checkpoint(
            tme_ckpt, tme_module, optimizer_tme, lr_scheduler_tme,
            device=accelerator.device,
        )
        logger.info(f"Resumed TME module from step {step} ({tme_ckpt})")
```

with:

```python
    tme_ckpt = getattr(config, "resume_tme_checkpoint", None)
    if tme_ckpt:
        reset_opt = getattr(config, "reset_tme_optimizer", False)
        step = load_tme_checkpoint(
            tme_ckpt, tme_module,
            optimizer_tme=None if reset_opt else optimizer_tme,
            lr_scheduler_tme=None if reset_opt else lr_scheduler_tme,
            device=accelerator.device,
        )
        logger.info(
            f"Resumed TME module from step {step} ({tme_ckpt})"
            + (" [optimizer reset]" if reset_opt else "")
        )
```

- [ ] **Step 2: Run full test suite**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/ -v \
  --ignore=tests/test_paired_exp_dataset.py \
  --ignore=tests/test_multi_group_tme_integration.py -x
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add train_scripts/train_controlnet_exp.py tests/test_train_controlnet_exp.py
git commit -m "feat: reset TME optimizer on resume when reset_tme_optimizer=True"
```

---

## Task 5: Update config

**Files:**
- Modify: `configs/config_controlnet_exp.py`

- [ ] **Precondition: Locate the epoch-30 checkpoint directory**

```bash
ls /home/ec2-user/PixCell/checkpoints/pixcell_controlnet_exp/checkpoints/
```

Note the step directory that corresponds to epoch 30 (the one before the stall was identified). You will use it as `resume_tme_checkpoint` below. If the directory does not exist, training has not been run yet — set `resume_tme_checkpoint = None` and omit `reset_tme_optimizer` for now (add them when a checkpoint exists).

- [ ] **Step 1: Edit the TME LR block**

Find and replace in `configs/config_controlnet_exp.py`:

```python
tme_lr      = 3e-4   # was 1e-5; increased 30× to escape zero-init regime
```

with:

```python
tme_lr          = 1e-5   # encoder CNN + Q/K/V — already healthy, keep stable
tme_proj_lr     = 3e-4   # cross_attn.proj only — zero-init, needs the boost
# REQUIRED for the first resume after the optimizer-split is activated.
# Without this, loading the old single-group optimizer state into the new two-group
# optimizer raises: ValueError: loaded state dict has a different number of param groups.
reset_tme_optimizer = True
```

Also set `resume_tme_checkpoint` to the path found in the precondition step. Find:

```python
resume_from = None
```

and add after it (filling in the real path):

```python
resume_tme_checkpoint = f"{root}/checkpoints/pixcell_controlnet_exp/checkpoints/step_XXXXXXX"
```

- [ ] **Step 2: Verify the path exists and config parses**

```bash
cd /home/ec2-user/PixCell && python -c "
from mmcv import Config
import os
c = Config.fromfile('configs/config_controlnet_exp.py')
print('tme_lr:', c.tme_lr)
print('tme_proj_lr:', c.tme_proj_lr)
print('reset_tme_optimizer:', c.reset_tme_optimizer)
ckpt = getattr(c, 'resume_tme_checkpoint', None)
print('resume_tme_checkpoint:', ckpt)
if ckpt:
    assert os.path.exists(ckpt), f'Checkpoint not found: {ckpt}'
    print('Checkpoint path exists: OK')
"
```

Expected:
```
tme_lr: 1e-05
tme_proj_lr: 0.0003
reset_tme_optimizer: True
resume_tme_checkpoint: .../step_XXXXXXX
Checkpoint path exists: OK
```

- [ ] **Step 3: Commit**

```bash
git add configs/config_controlnet_exp.py
git commit -m "config: split tme_lr=1e-5 / tme_proj_lr=3e-4, reset optimizer on resume"
```

---

## Task 6: Add diagnostic logging

**Files:**
- Modify: `train_scripts/train_controlnet_exp.py`

No new tests — logging is a side-effect verified via run output. Manual check: both `delta_mean[*]` and `proj_grad[*]` lines must appear in training logs at every `log_interval` step.

The structure: residuals are stored from every TME forward (no extra forward pass); grad norms are captured in the **first** `sync_gradients` block (before `optimizer.step()`, while grads still exist); both are emitted in the **second** `sync_gradients` block after `global_step += 1`. This ensures both signals appear at the correct, consistent step number. When `gradient_accumulation_steps > 1`, `_tme_residuals` at each log step reflects the **last micro-step** within that accumulation window, not an average — this is fine for diagnostics but should be kept in mind when interpreting magnitudes.

- [ ] **Step 1: Add sentinel initializations before the epoch loop**

Immediately before the `for epoch in range(start_epoch + 1, ...)` line in `train_controlnet_exp`, add:

```python
    _proj_grad_norms: dict = {}
    _tme_residuals:   dict = {}
```

This prevents `NameError` if `gradient_accumulation_steps > 1` causes the first `sync_gradients` block to not fire on the very first micro-step.

- [ ] **Step 2: Always return residuals from the TME forward in the `use_multi_group` branch**

Find the final line of the `use_multi_group` block:

```python
                vae_mask = tme_module(vae_mask.to(dtype=tme_dtype), tme_channel_dict)
```

Replace with:

```python
                fused, _tme_residuals = tme_module(
                    vae_mask.to(dtype=tme_dtype), tme_channel_dict, return_residuals=True,
                )
                vae_mask = fused
```

- [ ] **Step 3: Capture proj grad norms before `optimizer.step()` in the first `sync_gradients` block**

Find the start of the first `if accelerator.sync_gradients:` block (inside `with accelerator.accumulate(...)`):

```python
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), config.gradient_clip)
                    optimizer.step()
```

Insert grad-norm capture at the top of that block (before `clip_grad_norm_`):

```python
                if accelerator.sync_gradients:
                    _proj_grad_norms = {}
                    if use_multi_group:
                        for _gname, _gblock in accelerator.unwrap_model(tme_module).groups.items():
                            _g = _gblock.cross_attn.proj.weight.grad
                            if _g is not None:
                                _proj_grad_norms[_gname] = (
                                    _g.norm().item(),
                                    _gblock.cross_attn.proj.weight.abs().max().item(),
                                )
                    accelerator.clip_grad_norm_(controlnet.parameters(), config.gradient_clip)
                    optimizer.step()
```

- [ ] **Step 4: Emit both signals in the second `sync_gradients` block after `global_step += 1`**

Find the second `if accelerator.sync_gradients:` block (outside `accumulate`, contains `global_step += 1`):

```python
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config.log_interval == 0:
                    time_cost       = time.time() - last_tic
```

Add the diagnostic lines immediately after `global_step += 1`:

```python
            if accelerator.sync_gradients:
                global_step += 1
                if (global_step % config.log_interval == 0
                        and accelerator.is_main_process
                        and use_multi_group):
                    for _gname, _delta in _tme_residuals.items():
                        logger.info(f"  delta_mean[{_gname}]={_delta.abs().mean():.3e}")
                    for _gname, (_gnorm, _wmax) in _proj_grad_norms.items():
                        logger.info(f"  proj_grad[{_gname}]={_gnorm:.3e}  proj_wmax={_wmax:.3e}")
                if global_step % config.log_interval == 0:
                    time_cost       = time.time() - last_tic
```

- [ ] **Step 5: Run test suite**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/ -v \
  --ignore=tests/test_paired_exp_dataset.py \
  --ignore=tests/test_multi_group_tme_integration.py -x
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add train_scripts/train_controlnet_exp.py
git commit -m "feat: add per-group delta_mean and proj_grad diagnostic logging"
```

---

## Verification After First Training Run

After the first `log_interval` steps (~100), check the training log for lines like:

```
  delta_mean[cell_identity]=1.234e-05
  delta_mean[cell_state]=9.876e-06
  delta_mean[vasculature]=1.111e-05
  delta_mean[microenv]=8.765e-06
  proj_grad[cell_identity]=2.345e-04  proj_wmax=3.456e-06
  proj_grad[cell_state]=1.987e-04  proj_wmax=2.876e-06
  ...
```

**Interpretation:**

| Signal | Healthy (LR fix working) | Problem (escalate) |
|--------|--------------------------|--------------------|
| `proj_grad[*]` | > 1e-4 | < 1e-5 → task-easiness; consider auxiliary TME loss |
| `proj_wmax` | growing each interval | stuck at 1e-6 despite LR boost |
| `delta_mean[*]` | ~1e-5 now, reaches ~1e-3 within 5k steps | flat or oscillating |
