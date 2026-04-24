# Codex Handover: Probe Speed Refactor

## Why it's slow

Running `src.a1_codex_targets.probe` with T2 + T3 targets has two bottlenecks:

### 1. Feature matrix loaded 4 separate times (main I/O bottleneck)

`run_probe_tasks` calls four runners sequentially:
`t2_linear` → `t2_mlp` → `t3_linear` → `t3_mlp`

Each runner independently calls `load_feature_matrix`, which opens **10,379 individual `.npy`
files** (one per tile). On EBS SSD this is ~5 s/load; on EFS or HDD it can be 50–200 s/load.
Total: 4 × that cost, for no reason — the feature matrix is identical every time.

### 2. MLP parallelism limited by GIL (secondary CPU bottleneck)

`run_cv_regression` in `src/a1_probe_linear/main.py` uses:

```python
with threadpool_limits(limits=1):
    results = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem")(...)
```

`require="sharedmem"` forces joblib to use threads. `sklearn.MLPRegressor`'s Adam optimizer
loop is partially Python-bound, so GIL contention limits effective parallelism to ~3–4× even
with `n_jobs=8`. Switching to the loky process backend gives true 8× parallelism — BLAS is
already limited to 1 thread via `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` env vars that the
subprocess inherits.

### Scale context

- 10,379 tiles, 1536-dim UNI features → 64 MB feature matrix
- T2: 19 CODEX marker targets; T3: 76 quantile targets
- 5-fold CV → 5 × (19 + 76) = **475 MLP fits total** across both MLP tasks
- Each fit: MLPRegressor(256, 64), max_iter=200 on ~8 k training samples → ~2–4 s serial

Estimated current runtime: **7–30 min** (lower end on NVMe, upper end on EFS).
Estimated after fix: **3–7 min**.

---

## Changes required

### File 1: `src/a1_probe_linear/main.py`

**Change A — `run_task`: add `preloaded_X` parameter**

Add `preloaded_X: np.ndarray | None = None` to the function signature (after all existing
params). Replace the feature-loading line:

```python
# before (around line 270)
X = load_feature_matrix(features_dir, tile_ids)

# after
X = preloaded_X if preloaded_X is not None else load_feature_matrix(features_dir, tile_ids)
```

**Change B — `run_cv_regression`: switch to process backend**

Replace lines ~170–182:

```python
# before
with threadpool_limits(limits=1):
    results = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem")(
        delayed(_fit_regression_target)(
            X, Y, train_idx, test_idx, target_idx,
            estimator_factory=estimator_factory,
        )
        for target_idx in range(n_targets)
    )

# after
results = Parallel(n_jobs=n_jobs)(
    delayed(_fit_regression_target)(
        X, Y, train_idx, test_idx, target_idx,
        estimator_factory=estimator_factory,
    )
    for target_idx in range(n_targets)
)
```

Drop `require="sharedmem"` and the `threadpool_limits` wrapper.
- Default joblib backend is loky (process-based, GIL-free).
- Subprocesses inherit `OMP_NUM_THREADS=1` etc., so BLAS stays single-threaded per worker.
- X (64 MB) is serialized per task dispatch; at 8 workers × ~10 target rounds this is
  acceptable (~0.5 s overhead per fold). If it becomes a bottleneck, switch to memmap — but
  start here.
- `_fit_regression_target` is a module-level function; `estimator_factory` is a lambda.
  loky uses cloudpickle, so lambdas serialize fine.
- The `threadpoolctl` import block can remain (it's used nowhere else after this change, but
  removing it is optional cleanup).

### File 2: `src/a1_probe_mlp/main.py`

**Add `preloaded_X` parameter to `run_task`** — identical pattern to File 1 Change A:

```python
# signature addition
preloaded_X: np.ndarray | None = None

# body replacement (around line 101)
X = preloaded_X if preloaded_X is not None else load_feature_matrix(features_dir, tile_ids)
```

No parallelism change needed here — `run_cv_regression` (imported from `a1_probe_linear`)
already has the process-backend fix from File 1.

### File 3: `src/a1_codex_targets/probe.py`

**`run_probe_tasks`: add `preloaded_X` param and load features once**

Add `preloaded_X: np.ndarray | None = None` to the signature. Before the first runner call,
add:

```python
if preloaded_X is None:
    from src.a1_probe_linear.main import load_feature_matrix, load_tile_ids
    _tile_ids = load_tile_ids(tile_ids_path)
    preloaded_X = load_feature_matrix(features_dir, _tile_ids)
```

Add `preloaded_X=preloaded_X` as a keyword argument to **all four** runner calls:

```python
results["t2_linear"] = linear_runner(
    features_dir, t2_targets_path, tile_ids_path, t2_linear_dir,
    target_names_path=marker_names_path,
    cv_splits_path=cv_splits_path,
    n_jobs=n_jobs,
    preloaded_X=preloaded_X,   # ADD THIS
)
results["t2_mlp"] = mlp_runner(
    features_dir, t2_targets_path, tile_ids_path, output_dir / "t2_mlp",
    target_names_path=marker_names_path,
    cv_splits_path=cv_splits_path,
    linear_results_json=results["t2_linear"]["json"],
    n_jobs=n_jobs,
    preloaded_X=preloaded_X,   # ADD THIS
)
# same for t3_linear and t3_mlp if t3_targets_path is not None
```

### File 4: `tests/test_task_a1_codex_targets.py`

**`test_run_probe_tasks_uses_supplied_runners`**: after the fix, `run_probe_tasks` will call
`load_feature_matrix` when `preloaded_X` is None, but the test has no real feature files.
Pass a dummy array to skip that load:

```python
outputs = run_probe_tasks(
    features_dir=tmp_path / "features",
    tile_ids_path=tmp_path / "tile_ids.txt",
    cv_splits_path=tmp_path / "cv_splits.json",
    t2_targets_path=tmp_path / "t2.npy",
    marker_names_path=tmp_path / "markers.json",
    out_dir=tmp_path / "out",
    linear_runner=fake_runner,
    mlp_runner=fake_runner,
    n_jobs=4,
    preloaded_X=np.zeros((1, 4), dtype=np.float32),   # ADD THIS
)
```

`fake_runner` accepts `**kwargs` so it silently receives and ignores `preloaded_X`.

---

## Files NOT needing changes

- `tests/test_task_a1_probe_mlp.py` — calls `run_task` directly and provides real feature
  files; `preloaded_X` defaults to None, behavior unchanged.
- `tests/test_task_a1_probe_encoders.py` — does not call `run_probe_tasks` or `run_task`.
- All other files in the repo.

---

## Verification

After implementing, run:

```bash
python -m pytest tests/test_task_a1_codex_targets.py \
                 tests/test_task_a1_probe_encoders.py \
                 tests/test_task_a1_probe_mlp.py -v
```

All existing tests must pass. The `test_run_cv_regression_parallel_matches_serial` test
(in `test_task_a1_probe_mlp.py`) explicitly checks serial vs parallel output equality —
this must still pass with the loky backend.
