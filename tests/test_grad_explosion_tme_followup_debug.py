from pathlib import Path

import pytest

from diffusion.utils.misc import read_config
from tools.debug.grad_explosion_tme_followup import (
    DEFAULT_VARIANTS,
    PASS_THRESHOLD,
    summarize_variant,
)


@pytest.mark.parametrize("name,config_path", DEFAULT_VARIANTS)
def test_phase0_smoke_configs_match_followup_runbook(name: str, config_path: str):
    config = read_config(config_path)

    assert config.controlnet_depth == 18
    assert config.train_batch_size == 2
    assert config.max_train_samples == 10
    assert config.data.max_train_samples == 10
    assert config.log_interval == 1
    assert config.debug_tme_probe is True
    assert config.save_final_checkpoint is False
    assert name in config.work_dir


def test_phase0_summary_passes_on_first_new_record(tmp_path: Path):
    config_path = tmp_path / "config.py"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    config_path.write_text(f"work_dir = {str(work_dir)!r}\n", encoding="utf-8")
    (work_dir / "train_log.jsonl").write_text(
        '{"step": 1, "grad_norm_tme": 10.0, '
        '"grad_health_tme": {"max_abs": 9.0, "top_tensors": '
        '[{"name": "groups.vasculature.encoder.stem.1.bias", '
        '"finite_norm": 2.0, "max_abs": 9.0}]}}\n',
        encoding="utf-8",
    )

    result = summarize_variant(
        name="grouped",
        config_path=config_path,
        returncode=0,
        elapsed_sec=1.5,
        pass_threshold=PASS_THRESHOLD,
    )

    assert result.passed is True
    assert result.grad_norm_tme == pytest.approx(10.0)
    assert result.max_abs == pytest.approx(9.0)
    assert result.top_tensors[0]["name"].endswith("stem.1.bias")


def test_phase0_summary_fails_on_hot_tme_grad(tmp_path: Path):
    config_path = tmp_path / "config.py"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    config_path.write_text(f"work_dir = {str(work_dir)!r}\n", encoding="utf-8")
    (work_dir / "train_log.jsonl").write_text(
        '{"step": 1, "grad_norm_tme": 1000000.0, '
        '"grad_health_tme": {"max_abs": 2000.0, "top_tensors": []}}\n',
        encoding="utf-8",
    )

    result = summarize_variant(
        name="grouped",
        config_path=config_path,
        returncode=0,
        elapsed_sec=1.5,
        pass_threshold=PASS_THRESHOLD,
    )

    assert result.passed is False
