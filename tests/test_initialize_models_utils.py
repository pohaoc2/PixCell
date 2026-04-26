import os

import pytest

from train_scripts.initialize_models_utils import find_checkpoint, parse_args, set_fsdp_env


def test_find_checkpoint_returns_file_path_unchanged(tmp_path):
    checkpoint_path = tmp_path / "epoch_1_step_7.pth"
    checkpoint_path.write_text("stub", encoding="utf-8")

    assert find_checkpoint(str(checkpoint_path)) == str(checkpoint_path)


def test_find_checkpoint_picks_highest_step_file(tmp_path):
    (tmp_path / "epoch_1_step_3.pth").write_text("a", encoding="utf-8")
    (tmp_path / "epoch_1_step_12.pth").write_text("b", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert find_checkpoint(str(tmp_path)).endswith("epoch_1_step_12.pth")


def test_find_checkpoint_raises_when_directory_has_no_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="No checkpoint found"):
        find_checkpoint(str(tmp_path))


def test_parse_args_applies_expected_defaults():
    args = parse_args(["configs/config_controlnet_exp.py"])

    assert args.config == "configs/config_controlnet_exp.py"
    assert args.work_dir is None
    assert args.resume_from is None
    assert args.batch_size is None
    assert args.report_to == "tensorboard"
    assert args.debug is False
    assert args.skip_step == 0
    assert args.seed is None


def test_parse_args_parses_optional_flags():
    args = parse_args(
        [
            "configs/config_controlnet_exp.py",
            "--work-dir",
            "runs/demo",
            "--resume-from",
            "runs/demo/checkpoints",
            "--batch-size",
            "8",
            "--report-to",
            "wandb",
            "--debug",
            "--skip-step",
            "11",
            "--seed",
            "7",
        ]
    )

    assert args.work_dir == "runs/demo"
    assert args.resume_from == "runs/demo/checkpoints"
    assert args.batch_size == 8
    assert args.report_to == "wandb"
    assert args.debug is True
    assert args.skip_step == 11
    assert args.seed == 7


def test_set_fsdp_env_sets_expected_environment_variables(monkeypatch):
    for key in (
        "ACCELERATE_USE_FSDP",
        "FSDP_AUTO_WRAP_POLICY",
        "FSDP_BACKWARD_PREFETCH",
        "FSDP_TRANSFORMER_CLS_TO_WRAP",
    ):
        monkeypatch.delenv(key, raising=False)

    set_fsdp_env()

    assert os.environ["ACCELERATE_USE_FSDP"] == "true"
    assert os.environ["FSDP_AUTO_WRAP_POLICY"] == "TRANSFORMER_BASED_WRAP"
    assert os.environ["FSDP_BACKWARD_PREFETCH"] == "BACKWARD_PRE"
    assert os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] == "PixArtBlock"
