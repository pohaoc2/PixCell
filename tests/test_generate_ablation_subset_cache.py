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
