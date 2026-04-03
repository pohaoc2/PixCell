import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.pretrained_verify.cached_inference_features import load_or_compute_npy


class CachedInferenceFeaturesTests(unittest.TestCase):
    def test_load_or_compute_npy_loads_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "feature.npy"
            expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            np.save(cache_path, expected)

            calls = {"count": 0}

            def compute():
                calls["count"] += 1
                return np.array([9.0, 9.0, 9.0], dtype=np.float32)

            loaded = load_or_compute_npy(cache_path, compute)

            self.assertTrue(np.array_equal(loaded, expected))
            self.assertEqual(calls["count"], 0)

    def test_load_or_compute_npy_saves_new_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "feature.npy"
            expected = np.arange(6, dtype=np.float32).reshape(2, 3)

            calls = {"count": 0}

            def compute():
                calls["count"] += 1
                return expected

            loaded = load_or_compute_npy(cache_path, compute)

            self.assertEqual(calls["count"], 1)
            self.assertTrue(np.array_equal(loaded, expected))
            self.assertTrue(cache_path.exists())
            self.assertTrue(np.array_equal(np.load(cache_path), expected))

    def test_load_or_compute_npy_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "nested" / "cache" / "feature.npy"
            expected = np.array([4.0, 5.0], dtype=np.float32)

            loaded = load_or_compute_npy(cache_path, lambda: expected)

            self.assertTrue(np.array_equal(loaded, expected))
            self.assertTrue(cache_path.exists())
            self.assertTrue(cache_path.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
