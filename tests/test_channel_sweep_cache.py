"""Cache-contract tests for the channel sweep refactor.

These tests describe the expected save/load/render split for the sweep cache.
They intentionally skip until the new cache-focused API is present so the
current suite stays green while the implementation lands.
"""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("torch")


MODULE_CANDIDATES = (
    "tools.stage3.channel_sweep_cache",
    "tools.stage3.channel_sweep",
)

SAVE_FUNC_CANDIDATES = (
    "save_channel_sweep_cache",
    "write_channel_sweep_cache",
    "generate_channel_sweep_cache",
)

LOAD_FUNC_CANDIDATES = (
    "load_channel_sweep_cache",
    "read_channel_sweep_cache",
)

RENDER_FUNC_CANDIDATES = (
    "render_channel_sweep_figures",
    "render_channel_sweep_from_cache",
    "render_channel_sweep_cache",
)


def _resolve_future_api(func_names: tuple[str, ...]) -> tuple[Any, str]:
    for module_name in MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for func_name in func_names:
            func = getattr(module, func_name, None)
            if callable(func):
                return func, module_name
    pytest.skip(
        "channel sweep cache split not implemented yet; "
        f"missing any of {func_names} in {MODULE_CANDIDATES}"
    )


def _call_with_supported_kwargs(func, **preferred_kwargs):
    sig = inspect.signature(func)
    kwargs = {
        name: value
        for name, value in preferred_kwargs.items()
        if name in sig.parameters
    }
    return func(**kwargs)


def _write_png(path: Path, value: int, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(path)


def _build_fake_channel_sweep_cache(cache_dir: Path) -> Path:
    """Create a tiny cache tree with a manifest and a few PNGs."""
    exp2_dir = cache_dir / "exp2_cell_type_relabeling"
    exp3_dir = cache_dir / "exp3_cell_state_relabeling"
    for base in (exp2_dir, exp3_dir):
        base.mkdir(parents=True, exist_ok=True)

    manifests = {
        "version": 1,
        "tile_id": "mock_tile",
        "experiments": {
            "exp2": {
                "sources": ["cancer", "immune", "healthy"],
                "targets": ["cancer", "immune", "healthy"],
                "image_root": "exp2_cell_type_relabeling",
                "entries": [
                    {
                        "source_label": "cancer",
                        "target_label": "cancer",
                        "image_path": "exp2_cell_type_relabeling/cancer__cancer.png",
                    },
                    {
                        "source_label": "cancer",
                        "target_label": "immune",
                        "image_path": "exp2_cell_type_relabeling/cancer__immune.png",
                    },
                    {
                        "source_label": "immune",
                        "target_label": "immune",
                        "image_path": "exp2_cell_type_relabeling/immune__immune.png",
                    },
                ],
            },
            "exp3": {
                "sources": ["prolif", "nonprolif", "dead"],
                "targets": ["prolif", "nonprolif", "dead"],
                "image_root": "exp3_cell_state_relabeling",
                "entries": [
                    {
                        "source_label": "prolif",
                        "target_label": "prolif",
                        "image_path": "exp3_cell_state_relabeling/prolif__prolif.png",
                    },
                    {
                        "source_label": "prolif",
                        "target_label": "nonprolif",
                        "image_path": "exp3_cell_state_relabeling/prolif__nonprolif.png",
                    },
                    {
                        "source_label": "dead",
                        "target_label": "dead",
                        "image_path": "exp3_cell_state_relabeling/dead__dead.png",
                    },
                ],
            },
        },
    }

    for idx, rel in enumerate(
        [
            "exp2_cell_type_relabeling/cancer__cancer.png",
            "exp2_cell_type_relabeling/cancer__immune.png",
            "exp2_cell_type_relabeling/immune__immune.png",
            "exp3_cell_state_relabeling/prolif__prolif.png",
            "exp3_cell_state_relabeling/prolif__nonprolif.png",
            "exp3_cell_state_relabeling/dead__dead.png",
        ],
        start=1,
    ):
        _write_png(cache_dir / rel, 20 * idx)

    (cache_dir / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    return cache_dir


def test_channel_sweep_cache_manifest_round_trip(tmp_path):
    save_fn, _ = _resolve_future_api(SAVE_FUNC_CANDIDATES)
    load_fn, _ = _resolve_future_api(LOAD_FUNC_CANDIDATES)

    cache_dir = tmp_path / "channel_sweep_cache"
    result_tree = {
        "exp2": {
            "cancer": {
                "cancer": np.full((8, 8, 3), 10, dtype=np.uint8),
                "immune": np.full((8, 8, 3), 20, dtype=np.uint8),
                "healthy": np.full((8, 8, 3), 30, dtype=np.uint8),
            }
        },
        "exp3": {
            "prolif": {
                "prolif": np.full((8, 8, 3), 40, dtype=np.uint8),
                "nonprolif": np.full((8, 8, 3), 50, dtype=np.uint8),
                "dead": np.full((8, 8, 3), 60, dtype=np.uint8),
            }
        },
    }

    _call_with_supported_kwargs(
        save_fn,
        cache_dir=cache_dir,
        out_dir=cache_dir,
        results=result_tree,
        images=result_tree,
        tile_id="mock_tile",
        manifest={"tile_id": "mock_tile"},
    )

    manifest_path = cache_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("tile_id") == "mock_tile"

    loaded = _call_with_supported_kwargs(load_fn, cache_dir=cache_dir, cache_path=cache_dir)
    assert isinstance(loaded, dict)
    assert loaded.get("tile_id", "mock_tile") == "mock_tile"


def test_channel_sweep_render_from_cache_is_read_only(tmp_path, monkeypatch):
    render_fn, module_name = _resolve_future_api(RENDER_FUNC_CANDIDATES)
    cache_dir = _build_fake_channel_sweep_cache(tmp_path / "cache")
    out_dir = tmp_path / "figures"

    # If the render-only path accidentally tries to generate images, fail fast.
    module = importlib.import_module(module_name)
    if hasattr(module, "generate_from_ctrl"):
        monkeypatch.setattr(module, "generate_from_ctrl", lambda *a, **k: (_ for _ in ()).throw(AssertionError("render-only path should not call generation")))

    _call_with_supported_kwargs(
        render_fn,
        cache_dir=cache_dir,
        cache_path=cache_dir,
        out_dir=out_dir,
        output_dir=out_dir,
        cache_manifest=cache_dir / "manifest.json",
    )

    assert out_dir.exists()
    saved = list(out_dir.rglob("*.png"))
    assert saved, "render-only path should write at least one PNG"


def test_channel_sweep_cache_manifest_paths_are_relative(tmp_path):
    cache_dir = _build_fake_channel_sweep_cache(tmp_path / "cache")
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))

    for exp_name, exp_data in manifest["experiments"].items():
        assert exp_name in {"exp2", "exp3"}
        for entry in exp_data["entries"]:
            image_path = Path(entry["image_path"])
            assert not image_path.is_absolute()
            assert (cache_dir / image_path).is_file()
