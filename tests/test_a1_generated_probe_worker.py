from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image

from src._tasklib.runtime import RuntimeProbe


CPU_ONLY = RuntimeProbe(
    has_torch=True,
    has_cuda=False,
    has_diffusers=True,
    has_sklearn=True,
    has_matplotlib=True,
    warnings=("torch available but CUDA is unavailable",),
)


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def _make_args(base: dict[str, Path | str], *, worker: str | None = None) -> list[str]:
    args = [
        "--generated-root",
        str(base["generated_root"]),
        "--uni-model-path",
        str(base["uni_model_path"]),
        "--targets-path",
        str(base["targets_path"]),
        "--tile-ids-path",
        str(base["tile_ids_path"]),
        "--cv-splits-path",
        str(base["cv_splits_path"]),
        "--out-dir",
        str(base["out_dir"]),
    ]
    if worker is not None:
        args.extend(["--worker", worker])
    return args


def test_plan_task_includes_complete_worker_argv(tmp_path: Path):
    from src.a1_generated_probe.main import GeneratedProbeConfig, plan_task

    _write_png(tmp_path / "paired_ablation" / "ablation_results" / "0_0" / "all" / "generated_he.png", 32)
    (tmp_path / "uni").mkdir(parents=True)
    np.save(tmp_path / "targets.npy", np.zeros((2, 1), dtype=np.float32))
    (tmp_path / "tile_ids.txt").write_text("0_0\n2048_0\n", encoding="utf-8")
    (tmp_path / "cv_splits.json").write_text("{}", encoding="utf-8")

    config = GeneratedProbeConfig(
        generated_root=tmp_path / "paired_ablation",
        uni_model_path=tmp_path / "uni",
        targets_path=tmp_path / "targets.npy",
        tile_ids_path=tmp_path / "tile_ids.txt",
        cv_splits_path=tmp_path / "cv_splits.json",
        out_dir=tmp_path / "out",
    )

    plan = plan_task(config, runtime=CPU_ONLY)
    embed_argv = plan.jobs[0].command.argv
    assert embed_argv[0:4] == ("python", "-m", "src.a1_generated_probe.main", "--worker")
    assert embed_argv[4] == "embed"
    for flag, value in (
        ("--generated-root", str(config.generated_root)),
        ("--uni-model-path", str(config.uni_model_path)),
        ("--targets-path", str(config.targets_path)),
        ("--tile-ids-path", str(config.tile_ids_path)),
        ("--cv-splits-path", str(config.cv_splits_path)),
        ("--out-dir", str(config.out_dir)),
    ):
        assert flag in embed_argv
        assert value in embed_argv

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "generated_uni_embeddings.npy", np.zeros((1, 2), dtype=np.float32))
    (out_dir / "generated_tile_ids.txt").write_text("0_0\n", encoding="utf-8")

    probe_plan = plan_task(config, runtime=CPU_ONLY)
    probe_argv = probe_plan.jobs[1].command.argv
    assert probe_argv[4] == "probe"
    for flag, value in (
        ("--generated-root", str(config.generated_root)),
        ("--uni-model-path", str(config.uni_model_path)),
        ("--targets-path", str(config.targets_path)),
        ("--tile-ids-path", str(config.tile_ids_path)),
        ("--cv-splits-path", str(config.cv_splits_path)),
        ("--out-dir", str(config.out_dir)),
    ):
        assert flag in probe_argv
        assert value in probe_argv


def test_main_worker_embed_writes_subset_outputs(tmp_path: Path, monkeypatch):
    from src.a1_generated_probe import main as generated_probe_main

    _write_png(tmp_path / "paired_ablation" / "ablation_results" / "0_0" / "all" / "generated_he.png", 24)
    (tmp_path / "uni").mkdir(parents=True)
    np.save(tmp_path / "targets.npy", np.zeros((2, 1), dtype=np.float32))
    (tmp_path / "tile_ids.txt").write_text("0_0\n2048_0\n", encoding="utf-8")
    (tmp_path / "cv_splits.json").write_text("{}", encoding="utf-8")

    class DummyExtractor:
        def __init__(self, model_path, device="cuda"):
            self.model_path = Path(model_path)
            self.device = device

        def extract_batch(self, images):
            return np.stack(
                [
                    np.array([image.shape[0], image.shape[1], float(image.mean())], dtype=np.float32)
                    for image in images
                ],
                axis=0,
            )

    monkeypatch.setattr(generated_probe_main, "_load_uni_extractor_cls", lambda: DummyExtractor)

    base = {
        "generated_root": tmp_path / "paired_ablation",
        "uni_model_path": tmp_path / "uni",
        "targets_path": tmp_path / "targets.npy",
        "tile_ids_path": tmp_path / "tile_ids.txt",
        "cv_splits_path": tmp_path / "cv_splits.json",
        "out_dir": tmp_path / "out",
    }
    assert generated_probe_main.main(_make_args(base, worker="embed")) == 0

    embeddings = np.load(tmp_path / "out" / "generated_uni_embeddings.npy")
    assert embeddings.shape == (1, 3)
    assert np.allclose(embeddings[0], np.array([8.0, 8.0, 24.0], dtype=np.float32))
    assert (tmp_path / "out" / "generated_tile_ids.txt").read_text(encoding="utf-8") == "0_0\n"

    manifest = json.loads((tmp_path / "out" / "generated_probe_manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedded_tile_count"] == 1
    assert manifest["missing_generated_tile_ids"] == ["2048_0"]
    assert manifest["failed_tile_ids"] == []


def test_main_worker_probe_writes_generated_and_comparison_outputs(tmp_path: Path, monkeypatch):
    from src.a1_generated_probe import main as generated_probe_main

    tile_ids = ["0_0", "0_256", "0_512", "2048_0"]
    (tmp_path / "tile_ids.txt").write_text("\n".join(tile_ids) + "\n", encoding="utf-8")
    np.save(tmp_path / "targets.npy", np.arange(len(tile_ids), dtype=np.float32).reshape(-1, 1))
    (tmp_path / "cv_splits.json").write_text("{}", encoding="utf-8")
    (tmp_path / "uni").mkdir(parents=True)
    (tmp_path / "out").mkdir(parents=True)

    np.save(
        tmp_path / "out" / "generated_uni_embeddings.npy",
        np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
    )
    (tmp_path / "out" / "generated_tile_ids.txt").write_text("\n".join(tile_ids) + "\n", encoding="utf-8")
    (tmp_path / "out" / "generated_probe_manifest.json").write_text("{}", encoding="utf-8")

    real_features_dir = tmp_path / "real_features"
    real_features_dir.mkdir(parents=True)
    for index, tile_id in enumerate(tile_ids, start=1):
        np.save(real_features_dir / f"{tile_id}_uni.npy", np.array([float(index * 10)], dtype=np.float32))

    fake_probe_module = ModuleType("src.a1_probe_linear.main")

    def fake_load_cv_splits(loaded_tile_ids, cv_splits_path):
        assert loaded_tile_ids == tile_ids
        assert Path(cv_splits_path).name == "cv_splits.json"
        return [
            {"train_idx": [0, 1], "test_idx": [2, 3]},
            {"train_idx": [2, 3], "test_idx": [0, 1]},
        ]

    def fake_load_feature_matrix(features_dir, selected_tile_ids):
        return np.stack(
            [np.load(Path(features_dir) / f"{tile_id}_uni.npy").astype(np.float32) for tile_id in selected_tile_ids],
            axis=0,
        )

    def fake_run_cv_regression(X, Y, splits):
        fold_value = float(np.mean(X))
        return (
            np.full((len(splits), Y.shape[1]), fold_value, dtype=np.float32),
            np.zeros_like(Y, dtype=np.float32),
            np.zeros((Y.shape[1], X.shape[1]), dtype=np.float32),
        )

    def fake_summarize_probe_results(fold_scores, target_names):
        return [
            {
                "target": target_names[index],
                "r2_mean": float(np.mean(fold_scores[:, index])),
                "r2_sd": float(np.std(fold_scores[:, index])),
                "r2_folds": [float(value) for value in fold_scores[:, index]],
                "n_valid_folds": int(fold_scores.shape[0]),
            }
            for index in range(len(target_names))
        ]

    def fake_write_probe_results(rows, out_dir, *, prefix):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / f"{prefix}_results.json"
        csv_path = out_path / f"{prefix}_results.csv"
        json_path.write_text(json.dumps({"version": 1, "results": rows}), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["target", "r2_mean", "r2_sd", "n_valid_folds"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "target": row["target"],
                        "r2_mean": row["r2_mean"],
                        "r2_sd": row["r2_sd"],
                        "n_valid_folds": row["n_valid_folds"],
                    }
                )
        return {"json": json_path, "csv": csv_path}

    fake_probe_module.load_cv_splits = fake_load_cv_splits
    fake_probe_module.load_feature_matrix = fake_load_feature_matrix
    fake_probe_module.run_cv_regression = fake_run_cv_regression
    fake_probe_module.summarize_probe_results = fake_summarize_probe_results
    fake_probe_module.write_probe_results = fake_write_probe_results

    monkeypatch.setitem(sys.modules, "src.a1_probe_linear.main", fake_probe_module)
    monkeypatch.setattr(generated_probe_main, "_discover_real_features_dir", lambda: real_features_dir)

    base = {
        "generated_root": tmp_path / "paired_ablation",
        "uni_model_path": tmp_path / "uni",
        "targets_path": tmp_path / "targets.npy",
        "tile_ids_path": tmp_path / "tile_ids.txt",
        "cv_splits_path": tmp_path / "cv_splits.json",
        "out_dir": tmp_path / "out",
    }
    assert generated_probe_main.main(_make_args(base, worker="probe")) == 0

    generated_results = json.loads((tmp_path / "out" / "generated_probe_results.json").read_text(encoding="utf-8"))
    assert generated_results["results"][0]["target"] == "target_0"
    assert generated_results["results"][0]["r2_mean"] == 2.5

    with (tmp_path / "out" / "real_vs_generated_r2.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "target": "target_0",
            "real_r2": "25.0",
            "generated_r2": "2.5",
            "ratio": "0.1",
        }
    ]

    manifest = json.loads((tmp_path / "out" / "generated_probe_manifest.json").read_text(encoding="utf-8"))
    assert manifest["probe_tile_count"] == 4
    assert manifest["real_features_dir"] == str(real_features_dir.resolve())