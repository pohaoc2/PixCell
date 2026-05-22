from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_synthetic_dataset(n_tiles: int, n_patches: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_tiles, n_patches, n_features)).astype(np.float32)
    # Target = first feature + 0.1 * Gaussian noise -> easily learnable per-patch signal
    y = X[..., 0] + 0.1 * rng.normal(size=(n_tiles, n_patches)).astype(np.float32)
    return X, y[..., None]  # target tensor (N, P, 1)


def test_flatten_split_round_trips():
    from src.a1_probe_mlp_spatial.main import _flatten_split

    X, Y = _make_synthetic_dataset(4, 8, 3)
    flat_X, flat_y, tile_id_repeat = _flatten_split(X, Y, np.array([0, 2]))
    assert flat_X.shape == (16, 3)
    assert flat_y.shape == (16,)
    np.testing.assert_array_equal(tile_id_repeat[:8], 0)
    np.testing.assert_array_equal(tile_id_repeat[8:], 2)


def test_score_fold_distinguishes_good_and_trivial_predictions():
    from src.a1_probe_mlp_spatial.main import _score_fold

    rng = np.random.default_rng(1)
    n_tiles = 4
    n_patches = 16
    y_true = rng.normal(size=(n_tiles * n_patches,)).astype(np.float32)
    tile_repeat = np.repeat(np.arange(n_tiles), n_patches)

    # Perfect predictions -> R2 ~ 1, Pearson ~ 1
    perfect = _score_fold(y_true, y_true.copy(), tile_repeat)
    assert perfect["r2_global"] > 0.99
    assert perfect["r2_within"] > 0.99
    assert perfect["pearson_r"] > 0.99

    # Predict-the-tile-mean baseline -> R2_within ~ 0, R2_global > 0 if tiles differ
    tile_means = np.zeros_like(y_true)
    for tile_idx in range(n_tiles):
        mask = tile_repeat == tile_idx
        tile_means[mask] = y_true[mask].mean()
    baseline = _score_fold(y_true, tile_means, tile_repeat)
    assert baseline["r2_within"] < 1e-6


def test_run_task_returns_real_signal_and_handles_nans(tmp_path: Path):
    from src._tasklib.tile_ids import write_tile_ids
    from src.a1_probe_mlp_spatial.main import run_task

    n_tiles = 25
    n_patches = 8
    n_features = 6
    X, Y = _make_synthetic_dataset(n_tiles, n_patches, n_features, seed=2)
    # Inject NaN into a small fraction of patches.
    Y_flat = Y.reshape(-1)
    mask_nan = np.random.default_rng(7).random(Y_flat.size) < 0.05
    Y_flat[mask_nan] = np.nan
    Y = Y_flat.reshape(n_tiles, n_patches, 1)

    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    tile_ids = [f"{i * 512}_0" for i in range(n_tiles)]
    for tile_idx, tile_id in enumerate(tile_ids):
        np.save(features_dir / f"{tile_id}_uni_tokens.npy", X[tile_idx].astype(np.float16))

    targets_path = tmp_path / "targets.npy"
    np.save(targets_path, Y.astype(np.float32))
    tile_ids_path = write_tile_ids(tile_ids, tmp_path / "tile_ids.txt")

    out_dir = tmp_path / "out"
    paths = run_task(
        features_dir,
        targets_path,
        tile_ids_path,
        out_dir,
        n_splits=5,
        block_size_px=512,
        random_state=0,
    )
    import json

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    rows = payload["results"]
    assert len(rows) == 1
    row = rows[0]
    # Synthetic signal is highly predictable from the patch tokens.
    assert row["r2_global_mean"] > 0.5
    assert row["pearson_r_mean"] > 0.7
    assert row["n_valid_folds"] >= 4
