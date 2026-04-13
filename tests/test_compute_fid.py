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

if "scipy" not in sys.modules:
    scipy_stub = types.ModuleType("scipy")
    scipy_linalg_stub = types.ModuleType("scipy.linalg")

    def _sqrtm(matrix):
        values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
        values = np.clip(values, 0.0, None)
        root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
        return root.astype(np.complex128)

    scipy_linalg_stub.sqrtm = _sqrtm
    scipy_stub.linalg = scipy_linalg_stub
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.linalg"] = scipy_linalg_stub

from tools.compute_fid import (
    FOUR_GROUP_ORDER,
    ImageFeatureRecord,
    _generated_feature_cache_path,
    _generated_uni_feature_cache_path,
    all_features_cached,
    backfill_metrics,
    collect_condition_paths,
    compute_fid_from_stats,
    condition_metric_key,
    default_output_path,
    extract_uni_features,
    extract_virchow2_features,
    load_metric_scores,
    main,
    metric_key_for_backend,
    ordered_condition_keys,
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


def test_collect_condition_paths_uses_virchow2_feature_records(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    orion_root = tmp_path / "orion"
    tile_id = "tile_001"
    _write_complete_ablation_cache(cache_dir, tile_id, orion_root)
    np.save(
        orion_root / "features" / f"{tile_id}_virchow2.npy",
        np.array([0.4, 0.5, 0.6], dtype=np.float32),
    )

    tile_ids, real_records, condition_to_records = collect_condition_paths(
        cache_dir,
        orion_root,
        feature_backend="virchow2",
    )

    assert tile_ids == [tile_id]
    assert len(real_records) == 1
    assert real_records[0].feature_path == orion_root / "features" / f"{tile_id}_virchow2.npy"

    key = condition_metric_key(("cell_types",))
    record = condition_to_records[key][0]
    tile_dir = cache_dir / tile_id
    assert record.image_path == tile_dir / "subset_0" / "generated_he.png"
    assert record.feature_path == _generated_feature_cache_path(
        tile_dir,
        Path("subset_0/generated_he.png"),
        feature_backend="virchow2",
    )


def test_collect_condition_paths_uses_style_mapping_for_real_records(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    orion_root = tmp_path / "orion"
    tile_id = "tile_001"
    style_tile = "tile_999"
    _write_complete_ablation_cache(cache_dir, tile_id, orion_root)
    _write_rgb(orion_root / "he" / f"{style_tile}.png", value=77)
    np.save(orion_root / "features" / f"{style_tile}_uni.npy", np.array([0.9, 0.8, 0.7], dtype=np.float32))

    _, real_records, _ = collect_condition_paths(
        cache_dir,
        orion_root,
        feature_backend="uni",
        style_mapping={tile_id: style_tile},
    )

    assert real_records[0].image_path == orion_root / "he" / f"{style_tile}.png"
    assert real_records[0].feature_path == orion_root / "features" / f"{style_tile}_uni.npy"


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


def test_extract_virchow2_features_reuses_cached_features_and_saves_missing(tmp_path: Path) -> None:
    cached_feature_path = tmp_path / "cached.npy"
    missing_feature_path = tmp_path / "generated" / "sample_virchow2.npy"
    image_path = tmp_path / "generated" / "sample.png"

    np.save(cached_feature_path, np.array([5.0, 6.0], dtype=np.float32))
    _write_rgb(image_path, value=123)

    class FakeExtractor:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def extract_batch(self, images):
            self.batch_sizes.append(len(images))
            return np.array([[7.0, 8.0]], dtype=np.float32)

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

    features = extract_virchow2_features(records, extractor=extractor, batch_size=4)

    assert extractor.batch_sizes == [1]
    np.testing.assert_allclose(features, np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))
    assert missing_feature_path.is_file()
    np.testing.assert_allclose(np.load(missing_feature_path), np.array([7.0, 8.0], dtype=np.float32))


def test_metric_key_and_output_path_support_virchow2() -> None:
    cache_dir = Path("/tmp/example-cache")

    assert metric_key_for_backend("uni") == "fud"
    assert metric_key_for_backend("virchow2") == "fvd"
    assert metric_key_for_backend("inception") == "fid"
    assert default_output_path(cache_dir, "virchow2") == cache_dir / "fvd_scores.json"


def test_backfill_metrics_creates_missing_metrics_json(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    orion_root = tmp_path / "orion"
    tile_id = "tile_001"
    _write_complete_ablation_cache(cache_dir, tile_id, orion_root)

    metric_scores = {
        cond_key: float(index)
        for index, cond_key in enumerate(ordered_condition_keys(), start=1)
    }

    backfill_metrics(cache_dir, [tile_id], metric_scores, metric_key="fud")

    metrics_path = cache_dir / tile_id / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert payload["tile_id"] == tile_id
    per_condition = payload["per_condition"]
    assert sorted(per_condition) == sorted(ordered_condition_keys())
    for cond_key, score in metric_scores.items():
        assert per_condition[cond_key]["fud"] == score


def test_main_backfill_only_reuses_existing_scores(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    orion_root = tmp_path / "orion"
    tile_id = "tile_001"
    _write_complete_ablation_cache(cache_dir, tile_id, orion_root)

    output_path = cache_dir / "fud_scores.json"
    scores = {
        cond_key: float(index) / 10.0
        for index, cond_key in enumerate(ordered_condition_keys(), start=1)
    }
    output_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_fid.py",
            "--cache-dir",
            str(cache_dir),
            "--orion-root",
            str(orion_root),
            "--backfill-only",
        ],
    )

    main()

    payload = json.loads((cache_dir / tile_id / "metrics.json").read_text(encoding="utf-8"))
    for cond_key, score in load_metric_scores(output_path).items():
        assert payload["per_condition"][cond_key]["fud"] == score
