from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

if "torch" not in sys.modules:
    torch_stub = ModuleType("torch")
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub

if "diffusers" not in sys.modules:
    diffusers_stub = ModuleType("diffusers")
    diffusers_stub.__path__ = []

    class _DummyScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def set_timesteps(self, *args, **kwargs):
            pass

    diffusers_utils_stub = ModuleType("diffusers.utils")
    diffusers_torch_utils_stub = ModuleType("diffusers.utils.torch_utils")

    def _randn_tensor(*args, **kwargs):
        raise RuntimeError("randn_tensor stub should not be called in this test module")

    diffusers_stub.DDPMScheduler = _DummyScheduler
    diffusers_torch_utils_stub.randn_tensor = _randn_tensor
    sys.modules["diffusers"] = diffusers_stub
    sys.modules["diffusers.utils"] = diffusers_utils_stub
    sys.modules["diffusers.utils.torch_utils"] = diffusers_torch_utils_stub

from tools.stage3.generate_ablation_subset_cache import (
    _build_parser,
    _build_worker_common_args,
    _is_cuda_device,
    _print_progress,
    _select_generation_tile_ids,
)
from tools.stage3.ablation_cache import is_complete_tile_cache_dir, list_complete_cached_tile_ids


def test_is_cuda_device_accepts_cuda_prefixes():
    assert _is_cuda_device("cuda")
    assert _is_cuda_device("cuda:0")
    assert not _is_cuda_device("cpu")


def test_build_worker_common_args_uses_feature_device_override(tmp_path):
    args = SimpleNamespace(
        config="configs/config_controlnet_exp.py",
        checkpoint_dir="checkpoints/example",
        device="cuda:0",
        guidance_scale=2.5,
        num_steps=20,
        seed=42,
        null_uni=False,
        cache_uni_features=True,
        force_uni_features=False,
        uni_model="pretrained_models/uni-2h",
        feature_device="cpu",
        style_mapping_json=None,
    )

    worker_args = _build_worker_common_args(args, data_root=tmp_path)

    assert worker_args["data_root"] == str(tmp_path)
    assert worker_args["device"] == "cuda:0"
    assert worker_args["feature_device"] == "cpu"


def test_parser_requires_one_mode_flag():
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_accepts_jobs_with_n_tiles():
    parser = _build_parser()

    args = parser.parse_args(["--n-tiles", "3", "--jobs", "4"])

    assert args.n_tiles == 3
    assert args.jobs == 4


def test_parser_accepts_target_total_tiles():
    parser = _build_parser()

    args = parser.parse_args(["--target-total-tiles", "5000"])

    assert args.target_total_tiles == 5000


def test_parser_accepts_skip_existing_with_n_tiles():
    parser = _build_parser()

    args = parser.parse_args(["--n-tiles", "3", "--skip-existing"])

    assert args.n_tiles == 3
    assert args.skip_existing is True


def test_parser_accepts_output_dir_alias():
    parser = _build_parser()

    args = parser.parse_args(["--tile-id", "tile_001", "--output-dir", "/tmp/full_ablation"])

    assert args.tile_id == "tile_001"
    assert args.output_dir == "/tmp/full_ablation"


def test_print_progress_finishes_with_newline(capsys):
    _print_progress(2, 2, prefix="Generation")

    captured = capsys.readouterr()
    assert "Generation [" in captured.err
    assert captured.err.endswith("\n")


def test_select_generation_tile_ids_skip_existing_only_samples_uncached():
    selected, message = _select_generation_tile_ids(
        all_ids=["tile_1", "tile_2", "tile_3", "tile_4"],
        existing_ids=["tile_1", "tile_3"],
        tile_sample_seed=7,
        requested_n=2,
        skip_existing=True,
    )

    assert sorted(selected) == ["tile_2", "tile_4"]
    assert "skipping 2 existing cache dirs" in message


def test_select_generation_tile_ids_target_total_grows_from_existing_count():
    selected, message = _select_generation_tile_ids(
        all_ids=["tile_1", "tile_2", "tile_3", "tile_4", "tile_5"],
        existing_ids=["tile_1", "tile_2"],
        tile_sample_seed=11,
        target_total_tiles=4,
    )

    assert len(selected) == 2
    assert set(selected).isdisjoint({"tile_1", "tile_2"})
    assert "grow cache from 2 to 4 total tiles" in message


def test_select_generation_tile_ids_target_total_noop_when_already_satisfied():
    selected, message = _select_generation_tile_ids(
        all_ids=["tile_1", "tile_2", "tile_3"],
        existing_ids=["tile_1", "tile_2", "tile_3"],
        tile_sample_seed=3,
        target_total_tiles=3,
    )

    assert selected == []
    assert "nothing to do" in message


def test_select_generation_tile_ids_target_total_errors_when_existing_exceeds_target():
    with pytest.raises(ValueError, match="exceeds target_total_tiles=2"):
        _select_generation_tile_ids(
            all_ids=["tile_1", "tile_2", "tile_3"],
            existing_ids=["tile_1", "tile_2", "tile_3"],
            tile_sample_seed=5,
            target_total_tiles=2,
        )


def _write_cache_manifest(cache_dir, *, include_all_entries: bool = True, include_all_file: bool = True):
    import json
    from pathlib import Path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "singles/01_cell_types.png",
        "singles/02_cell_state.png",
        "singles/03_vasculature.png",
        "singles/04_microenv.png",
        "pairs/01_cell_types__cell_state.png",
        "pairs/02_cell_types__vasculature.png",
        "pairs/03_cell_types__microenv.png",
        "pairs/04_cell_state__vasculature.png",
        "pairs/05_cell_state__microenv.png",
        "pairs/06_vasculature__microenv.png",
        "triples/01_cell_types__cell_state__vasculature.png",
        "triples/02_cell_types__cell_state__microenv.png",
        "triples/03_cell_types__vasculature__microenv.png",
        "triples/04_cell_state__vasculature__microenv.png",
    ]
    if include_all_entries:
        files.append("all/generated_he.png")
    for rel in files:
        path = cache_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    if not include_all_file:
        (cache_dir / "all" / "generated_he.png").unlink(missing_ok=True)
    (cache_dir / "cell_mask.png").write_bytes(b"mask")
    manifest = {
        "version": 1,
        "tile_id": cache_dir.name,
        "group_names": ["cell_types", "cell_state", "vasculature", "microenv"],
        "cell_mask_path": "cell_mask.png",
        "sections": [
            {
                "title": "1 active group",
                "subset_size": 1,
                "entries": [
                    {"active_groups": ["cell_types"], "condition_label": "a", "image_label": "a", "image_path": "singles/01_cell_types.png"},
                    {"active_groups": ["cell_state"], "condition_label": "b", "image_label": "b", "image_path": "singles/02_cell_state.png"},
                    {"active_groups": ["vasculature"], "condition_label": "c", "image_label": "c", "image_path": "singles/03_vasculature.png"},
                    {"active_groups": ["microenv"], "condition_label": "d", "image_label": "d", "image_path": "singles/04_microenv.png"},
                ],
            },
            {
                "title": "2 active groups",
                "subset_size": 2,
                "entries": [
                    {"active_groups": ["cell_types", "cell_state"], "condition_label": "1", "image_label": "1", "image_path": "pairs/01_cell_types__cell_state.png"},
                    {"active_groups": ["cell_types", "vasculature"], "condition_label": "2", "image_label": "2", "image_path": "pairs/02_cell_types__vasculature.png"},
                    {"active_groups": ["cell_types", "microenv"], "condition_label": "3", "image_label": "3", "image_path": "pairs/03_cell_types__microenv.png"},
                    {"active_groups": ["cell_state", "vasculature"], "condition_label": "4", "image_label": "4", "image_path": "pairs/04_cell_state__vasculature.png"},
                    {"active_groups": ["cell_state", "microenv"], "condition_label": "5", "image_label": "5", "image_path": "pairs/05_cell_state__microenv.png"},
                    {"active_groups": ["vasculature", "microenv"], "condition_label": "6", "image_label": "6", "image_path": "pairs/06_vasculature__microenv.png"},
                ],
            },
            {
                "title": "3 active groups",
                "subset_size": 3,
                "entries": [
                    {"active_groups": ["cell_types", "cell_state", "vasculature"], "condition_label": "1", "image_label": "1", "image_path": "triples/01_cell_types__cell_state__vasculature.png"},
                    {"active_groups": ["cell_types", "cell_state", "microenv"], "condition_label": "2", "image_label": "2", "image_path": "triples/02_cell_types__cell_state__microenv.png"},
                    {"active_groups": ["cell_types", "vasculature", "microenv"], "condition_label": "3", "image_label": "3", "image_path": "triples/03_cell_types__vasculature__microenv.png"},
                    {"active_groups": ["cell_state", "vasculature", "microenv"], "condition_label": "4", "image_label": "4", "image_path": "triples/04_cell_state__vasculature__microenv.png"},
                ],
            },
            {
                "title": "4 active groups",
                "subset_size": 4,
                "entries": [
                    {"active_groups": ["cell_types", "cell_state", "vasculature", "microenv"], "condition_label": "all", "image_label": "all", "image_path": "all/generated_he.png"},
                ],
            },
        ],
    }
    if not include_all_entries:
        manifest["sections"][-1]["entries"] = []
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_is_complete_tile_cache_dir_accepts_complete_cache(tmp_path):
    cache_dir = tmp_path / "tile_ok"
    _write_cache_manifest(cache_dir)

    assert is_complete_tile_cache_dir(cache_dir) is True


def test_is_complete_tile_cache_dir_rejects_missing_all_file(tmp_path):
    cache_dir = tmp_path / "tile_bad"
    _write_cache_manifest(cache_dir, include_all_file=False)

    assert is_complete_tile_cache_dir(cache_dir) is False


def test_list_complete_cached_tile_ids_filters_incomplete_dirs(tmp_path):
    _write_cache_manifest(tmp_path / "tile_complete")
    _write_cache_manifest(tmp_path / "tile_incomplete", include_all_file=False)

    assert list_complete_cached_tile_ids(tmp_path) == ["tile_complete"]
