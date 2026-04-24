from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src._tasklib.tile_ids import tile_ids_sha1
from src.a1_probe_encoders.main import run_encoder_probe_to_csv


def _write_fake_cv_splits(path: Path, tile_ids: list[str]) -> None:
    n_tiles = len(tile_ids)
    midpoint = n_tiles // 2
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "tile_count": n_tiles,
                "tile_ids_sha1": tile_ids_sha1(tile_ids),
                "block_size_px": 2048,
                "n_splits": 2,
                "splits": [
                    {"train_idx": list(range(midpoint)), "test_idx": list(range(midpoint, n_tiles))},
                    {"train_idx": list(range(midpoint, n_tiles)), "test_idx": list(range(midpoint))},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_fake_manifest(cv_path: Path, target_names: list[str]) -> None:
    cv_path.with_name("manifest.json").write_text(
        json.dumps({"target_names": target_names}),
        encoding="utf-8",
    )


def test_run_encoder_probe_to_csv_creates_correct_schema(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_tiles = 20
    target_names = ["cell_density", "prolif_frac", "immune_frac"]
    tile_ids = [f"tile_{index:04d}" for index in range(n_tiles)]

    embeddings = rng.standard_normal((n_tiles, 8)).astype(np.float32)
    targets = rng.random((n_tiles, len(target_names))).astype(np.float32)

    embeddings_path = tmp_path / "embeddings.npy"
    targets_path = tmp_path / "targets.npy"
    tile_ids_path = tmp_path / "tile_ids.txt"
    cv_splits_path = tmp_path / "cv_splits.json"
    output_csv_path = tmp_path / "probe_results.csv"

    np.save(embeddings_path, embeddings)
    np.save(targets_path, targets)
    tile_ids_path.write_text("\n".join(tile_ids), encoding="utf-8")
    _write_fake_cv_splits(cv_splits_path, tile_ids)
    _write_fake_manifest(cv_splits_path, target_names)

    result = run_encoder_probe_to_csv(
        embeddings_path,
        targets_path=targets_path,
        tile_ids_path=tile_ids_path,
        cv_splits_path=cv_splits_path,
        output_csv_path=output_csv_path,
    )

    assert result == output_csv_path
    rows = list(csv.DictReader(output_csv_path.open(encoding="utf-8", newline="")))
    assert len(rows) == len(target_names)
    assert set(rows[0].keys()) == {"target", "r2_mean", "r2_sd", "n_valid_folds"}
    assert {row["target"] for row in rows} == set(target_names)
    for row in rows:
        assert row["n_valid_folds"] == "2"
        float(row["r2_mean"])
        float(row["r2_sd"])


def test_extract_ctranspath_embeddings_shape(monkeypatch, tmp_path: Path) -> None:
    from PIL import Image

    import src.a1_probe_encoders.main as probe_encoders

    class FakeExtractor:
        def extract_batch(self, images):
            batch_size = len(images)
            return np.zeros((batch_size, 768), dtype=np.float32)

    monkeypatch.setattr(probe_encoders, "_build_ctranspath_extractor", lambda model_name, *, device: FakeExtractor())

    he_dir = tmp_path / "he"
    he_dir.mkdir()
    tile_ids = [f"00{index:03d}_0000" for index in range(4)]
    for tile_id in tile_ids:
        image = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
        image.save(he_dir / f"{tile_id}.png")

    embeddings = probe_encoders.extract_ctranspath_embeddings(
        he_dir,
        tile_ids,
        device="cpu",
        batch_size=2,
        model_name="fake-model",
    )

    assert embeddings.shape == (4, 768)
    assert embeddings.dtype == np.float32
