from __future__ import annotations

import json
import sys
import types
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    class _DummyTensor:
        pass
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.Tensor = _DummyTensor
    sys.modules["torch"] = torch_stub

if "diffusers" not in sys.modules:
    diffusers_stub = types.ModuleType("diffusers")

    class _DummyScheduler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_timesteps(self, *args, **kwargs) -> None:
            pass

    diffusers_stub.DDPMScheduler = _DummyScheduler
    sys.modules["diffusers"] = diffusers_stub

from tools.compute_fid import (
    FOUR_GROUP_ORDER,
    ImageFeatureRecord,
    _generated_uni_feature_cache_path,
    all_features_cached,
    collect_condition_paths,
    compute_fid_from_stats,
    condition_metric_key,
    extract_uni_features,
)


def _write_rgb(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((12, 12, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_complete_ablation_cache(cache_dir: Path, tile_id: str, orion_root: Path) -> None:
    tile_dir = cache_dir / tile_id
    tile_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for index, groups in enumerate(
        combo
        for size in range(1, len(FOUR_GROUP_ORDER) + 1)
        for combo in combinations(FOUR_GROUP_ORDER, size)
    ):
        rel_path = Path(f"subset_{index}") / "generated_he.png"
        _write_rgb(tile_dir / rel_path, value=(index * 10) % 255)
        entries.append(
            {
                "active_groups": list(groups),
                "image_path": rel_path.as_posix(),
            }
        )

    _write_rgb(tile_dir / "all" / "generated_he.png", value=200)
    (tile_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tile_id": tile_id,
                "sections": [{"entries": entries}],
            }
        ),
        encoding="utf-8",
    )

    _write_rgb(orion_root / "he" / f"{tile_id}.png", value=25)
    features_dir = orion_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_dir / f"{tile_id}_uni.npy", np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_compute_fid_from_stats_zero_for_identical_distributions() -> None:
    mu = np.array([1.0, -2.0], dtype=np.float64)
    sigma = np.array([[2.0, 0.3], [0.3, 1.5]], dtype=np.float64)

    fid = compute_fid_from_stats(mu, sigma, mu.copy(), sigma.copy())

    assert fid == 0.0


def test_collect_condition_paths_uses_uni_feature_records(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    orion_root = tmp_path / "orion"
    tile_id = "tile_001"
    _write_complete_ablation_cache(cache_dir, tile_id, orion_root)

    tile_ids, real_records, condition_to_records = collect_condition_paths(
        cache_dir,
        orion_root,
        feature_backend="uni",
    )

    assert tile_ids == [tile_id]
    assert len(real_records) == 1
    assert real_records[0].feature_path == orion_root / "features" / f"{tile_id}_uni.npy"

    key = condition_metric_key(("cell_types",))
    record = condition_to_records[key][0]
    tile_dir = cache_dir / tile_id
    assert record.image_path == tile_dir / "subset_0" / "generated_he.png"
    assert record.feature_path == _generated_uni_feature_cache_path(
        tile_dir,
        Path("subset_0/generated_he.png"),
    )


def test_extract_uni_features_reuses_cached_features_and_saves_missing(tmp_path: Path) -> None:
    cached_feature_path = tmp_path / "cached.npy"
    missing_feature_path = tmp_path / "generated" / "sample_uni.npy"
    image_path = tmp_path / "generated" / "sample.png"

    np.save(cached_feature_path, np.array([1.0, 2.0], dtype=np.float32))
    _write_rgb(image_path, value=77)

    class FakeExtractor:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def extract_batch(self, images):
            self.batch_sizes.append(len(images))
            return np.array([[3.0, 4.0]], dtype=np.float32)

    extractor = FakeExtractor()
    records = [
        ImageFeatureRecord(
            image_path=tmp_path / "unused.png",
            feature_path=cached_feature_path,
        ),
        ImageFeatureRecord(
            image_path=image_path,
            feature_path=missing_feature_path,
        ),
    ]

    features = extract_uni_features(records, extractor=extractor, batch_size=8)

    assert extractor.batch_sizes == [1]
    np.testing.assert_allclose(features, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    assert missing_feature_path.is_file()
    np.testing.assert_allclose(np.load(missing_feature_path), np.array([3.0, 4.0], dtype=np.float32))


def test_extract_uni_features_uses_all_cached_without_extractor(tmp_path: Path) -> None:
    feature_a = tmp_path / "a.npy"
    feature_b = tmp_path / "b.npy"
    np.save(feature_a, np.array([1.0, 2.0], dtype=np.float32))
    np.save(feature_b, np.array([3.0, 4.0], dtype=np.float32))

    records = [
        ImageFeatureRecord(image_path=tmp_path / "unused_a.png", feature_path=feature_a),
        ImageFeatureRecord(image_path=tmp_path / "unused_b.png", feature_path=feature_b),
    ]

    assert all_features_cached(records) is True
    features = extract_uni_features(records, extractor=None, batch_size=8)

    np.testing.assert_allclose(features, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
