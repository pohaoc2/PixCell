# Handover

Date: 2026-04-23

## Current status

The section 11 task-source layout under `src/` is in place and implemented to the level supported by this machine.

- CPU-complete and tested:
  - `a0_visibility_map`
  - `a0_tradeoff_scatter`
  - `a1_mask_targets`
  - `a1_probe_linear`
  - `a1_probe_mlp`
  - `a1_codex_targets`
- Planner-only and tested on this machine; actual generation/training still needs a GPU machine:
  - `a1_probe_encoders`
  - `a1_generated_probe`
  - `a2_decomposition`
  - `a3_combinatorial_sweep`

The most recent task-level documentation is in `README_TASKS.md`, which now lists the runnable CLI for every task package.

## What was completed

- Updated `docs/paper_action_plan_2026-04-22.md` to serve as the central spec for section 11 task layout and implementation status.
- Added isolated task packages under `src/<task_name>/`.
- Implemented shared task helpers under `src/_tasklib/`.
- Added a CLI for `src.a1_codex_targets.probe` so all task packages now have a documented command entrypoint.
- Added focused tests for the new task packages and the CODEX probe CLI.
- Added `README_TASKS.md` with copy-paste commands, expected outputs, and CPU vs GPU notes.

## Validated state

Focused task-suite validation has passed.

Most recent targeted command:

```bash
pytest -q tests/test_task_a1_codex_targets.py tests/test_task_a1_codex_probe_cli.py
```

Result:

```text
3 passed in 0.11s
```

Previously validated focused suite:

```bash
pytest -q \
  tests/test_task_a0_visibility_map.py \
  tests/test_task_a0_tradeoff_scatter.py \
  tests/test_task_a1_mask_targets.py \
  tests/test_task_a1_probe_linear.py \
  tests/test_task_a1_probe_mlp.py \
  tests/test_task_a1_codex_targets.py \
  tests/test_task_gpu_wrappers.py
```

That suite passed earlier in this session.

## Machine and data constraints

- This machine has no GPU.
- `torch` is present, but CUDA is unavailable.
- `h5py` is not available here, so lightweight task code avoids importing heavier dataset modules just to reuse helpers.
- Raw CRC33 CODEX data root:
  - `/home/pohaoc2/UW/bagherilab/he-feature-visualizer/data`
- Paired experimental PixCell root:
  - `data/orion-crc33`
- Inference outputs live under:
  - `inference_output/`

## Important outputs and entry points

- Task CLI summary:
  - `README_TASKS.md`
- Central planning/spec document:
  - `docs/paper_action_plan_2026-04-22.md`
- Shared task helpers:
  - `src/_tasklib/io.py`
  - `src/_tasklib/runtime.py`
  - `src/_tasklib/tile_ids.py`
- New CODEX probe CLI entrypoint:
  - `src/a1_codex_targets/probe.py`

## Open tasks

### High priority

- Run the CPU-safe tasks end-to-end on real data and materialize their outputs in the task `out/` folders.
- Move the planner-only GPU tasks onto a GPU machine and implement or connect the actual worker execution path behind the generated `plan.json` files.
- Decide whether to standardize task entrypoints to a single naming convention such as `python -m src.<task>.run`.

### GPU follow-up

- `a1_probe_encoders`: implement and run the actual Virchow/raw-CNN embedding and probe jobs on a GPU machine.
- `a1_generated_probe`: implement and run generated-H&E embedding plus downstream probe execution.
- `a2_decomposition`: implement and run the 2x2 UNI/TME generation workers and summary outputs.
- `a3_combinatorial_sweep`: implement and run the 27-condition generation workers plus downstream interaction analysis.

### Documentation and cleanup

- Keep `README_TASKS.md` and `docs/paper_action_plan_2026-04-22.md` synchronized as task CLIs evolve.
- Review section 11 details in `docs/paper_action_plan_2026-04-22.md` for stale implementation specifics now that the code is live.
- Fine-grained `cell_types` remains intentionally parked as TODO and should stay out of the active task list for now.

## Recommended next actions

1. Run the CPU-runnable tasks from `README_TASKS.md` against the real CRC33/PixCell data roots and capture their first real outputs.
2. Copy the needed inputs plus generated `plan.json` files to a GPU machine.
3. Implement the worker side for the planner-only GPU tasks, then add behavior tests around those execution paths.

## Git / branch state at handoff

- Current branch: `main`
- Remote: `origin git@github.com:pohaoc2/PixCell.git`
