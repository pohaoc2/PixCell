from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_codex_probe_cli_delegates_to_runner(tmp_path: Path, monkeypatch):
    from src.a1_codex_targets import probe as probe_module

    captured: dict[str, str] = {}

    def fake_run_probe_tasks(**kwargs):
        captured.update({key: str(value) if value is not None else "" for key, value in kwargs.items()})
        return {}

    monkeypatch.setattr(probe_module, "run_probe_tasks", fake_run_probe_tasks)

    exit_code = probe_module.main(
        [
            "--features-dir",
            str(tmp_path / "features"),
            "--tile-ids-path",
            str(tmp_path / "tile_ids.txt"),
            "--cv-splits-path",
            str(tmp_path / "cv_splits.json"),
            "--t2-targets-path",
            str(tmp_path / "t2.npy"),
            "--marker-names-path",
            str(tmp_path / "markers.json"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert captured["features_dir"] == str(tmp_path / "features")
    assert captured["tile_ids_path"] == str(tmp_path / "tile_ids.txt")
    assert captured["cv_splits_path"] == str(tmp_path / "cv_splits.json")
    assert captured["t2_targets_path"] == str(tmp_path / "t2.npy")
    assert captured["marker_names_path"] == str(tmp_path / "markers.json")
    assert captured["out_dir"] == str(tmp_path / "out")