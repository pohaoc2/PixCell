from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src._tasklib.io import write_json
from src._tasklib.runtime import JobState, RuntimeProbe
from src._tasklib.tile_ids import tile_ids_sha1


CPU_ONLY = RuntimeProbe(
    has_torch=True,
    has_cuda=False,
    has_diffusers=True,
    has_sklearn=True,
    has_matplotlib=True,
    warnings=("torch available but CUDA is unavailable",),
)


def _write_rgb(path: Path, value: int) -> None:
    image = np.full((20, 20, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def _write_linear_reference(linear_dir: Path, *, target_names: list[str], splits: list[dict[str, list[int]]], tile_ids: list[str]) -> Path:
    linear_dir.mkdir(parents=True, exist_ok=True)
    cv_splits_path = write_json(
        {
            "version": 1,
            "tile_count": len(tile_ids),
            "tile_ids_sha1": tile_ids_sha1(tile_ids),
            "block_size_px": 2048,
            "n_splits": len(splits),
            "splits": splits,
        },
        linear_dir / "cv_splits.json",
    )
    (linear_dir / "linear_probe_results.json").write_text(
        json.dumps(
            {
                "version": 1,
                "results": [
                    {"target": target_names[0], "r2_mean": 0.75},
                    {"target": target_names[1], "r2_mean": 0.25},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (linear_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "target_names": target_names}, indent=2) + "\n",
        encoding="utf-8",
    )
    return cv_splits_path


def _make_inputs(tmp_path: Path) -> dict[str, Path]:
    he_dir = tmp_path / "he"
    he_dir.mkdir(parents=True, exist_ok=True)
    tile_ids = [
        "0_0",
        "2048_0",
        "4096_0",
        "6144_0",
        "8192_0",
        "10240_0",
    ]
    brightness = [20, 50, 80, 120, 170, 220]
    for tile_id, value in zip(tile_ids, brightness):
        _write_rgb(he_dir / f"{tile_id}.png", value)

    targets = np.stack(
        [
            np.asarray(brightness, dtype=np.float32) / 255.0,
            1.0 - (np.asarray(brightness, dtype=np.float32) / 255.0),
        ],
        axis=1,
    )
    targets_path = tmp_path / "targets.npy"
    np.save(targets_path, targets)
    tile_ids_path = tmp_path / "tile_ids.txt"
    tile_ids_path.write_text("\n".join(tile_ids) + "\n", encoding="utf-8")

    splits = [
        {"train_idx": [0, 1, 2], "test_idx": [3, 4, 5]},
        {"train_idx": [3, 4, 5], "test_idx": [0, 1, 2]},
    ]
    target_names = ["tumor", "immune"]
    linear_dir = tmp_path / "linear_probe"
    cv_splits_path = _write_linear_reference(linear_dir, target_names=target_names, splits=splits, tile_ids=tile_ids)

    out_dir = tmp_path / "out"
    weights_path = tmp_path / "virchow.pt"
    weights_path.write_bytes(b"fake-weights")
    return {
        "he_dir": he_dir,
        "targets_path": targets_path,
        "tile_ids_path": tile_ids_path,
        "cv_splits_path": cv_splits_path,
        "out_dir": out_dir,
        "weights_path": weights_path,
    }


def _required_arg_map(argv: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    index = 3
    while index < len(argv):
        mapping[argv[index]] = argv[index + 1]
        index += 2
    return mapping


def _base_worker_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--he-dir",
        str(paths["he_dir"]),
        "--targets-path",
        str(paths["targets_path"]),
        "--tile-ids-path",
        str(paths["tile_ids_path"]),
        "--cv-splits-path",
        str(paths["cv_splits_path"]),
        "--out-dir",
        str(paths["out_dir"]),
        "--device",
        "cpu",
    ]


def test_plan_task_commands_include_required_worker_args(tmp_path: Path) -> None:
    from src.a1_probe_encoders.main import ProbeEncodersConfig, plan_task

    paths = _make_inputs(tmp_path)
    config = ProbeEncodersConfig(
        he_dir=paths["he_dir"],
        targets_path=paths["targets_path"],
        tile_ids_path=paths["tile_ids_path"],
        cv_splits_path=paths["cv_splits_path"],
        out_dir=paths["out_dir"],
        virchow_weights=paths["weights_path"],
    )

    initial_plan = plan_task(config, runtime=CPU_ONLY)
    assert initial_plan.jobs[0].state == JobState.DEFERRED
    assert initial_plan.jobs[1].state == JobState.DEFERRED

    virchow_args = _required_arg_map(initial_plan.jobs[0].command.argv)
    raw_cnn_args = _required_arg_map(initial_plan.jobs[1].command.argv)
    for arg_map, worker_name in ((virchow_args, "virchow"), (raw_cnn_args, "raw_cnn")):
        assert arg_map["--worker"] == worker_name
        assert arg_map["--he-dir"] == str(paths["he_dir"])
        assert arg_map["--targets-path"] == str(paths["targets_path"])
        assert arg_map["--tile-ids-path"] == str(paths["tile_ids_path"])
        assert arg_map["--cv-splits-path"] == str(paths["cv_splits_path"])
        assert arg_map["--out-dir"] == str(paths["out_dir"])
        assert arg_map["--virchow-weights"] == str(paths["weights_path"])

    np.save(paths["out_dir"] / "raw_cnn_embeddings.npy", np.zeros((6, 4), dtype=np.float32))
    compare_plan = plan_task(config, runtime=CPU_ONLY)
    assert compare_plan.jobs[2].state == JobState.READY
    compare_args = _required_arg_map(compare_plan.jobs[2].command.argv)
    assert compare_args["--worker"] == "compare"
    assert compare_args["--he-dir"] == str(paths["he_dir"])
    assert compare_args["--targets-path"] == str(paths["targets_path"])
    assert compare_args["--tile-ids-path"] == str(paths["tile_ids_path"])
    assert compare_args["--cv-splits-path"] == str(paths["cv_splits_path"])
    assert compare_args["--out-dir"] == str(paths["out_dir"])
    assert compare_args["--virchow-weights"] == str(paths["weights_path"])


def test_virchow_worker_writes_skip_marker_without_weights(tmp_path: Path) -> None:
    from src.a1_probe_encoders.main import main

    paths = _make_inputs(tmp_path)
    exit_code = main(["--worker", "virchow", *_base_worker_args(paths)])

    assert exit_code == 0
    skip_path = paths["out_dir"] / "virchow_SKIPPED.txt"
    assert skip_path.is_file()
    assert "virchow skipped" in skip_path.read_text(encoding="utf-8").lower()


def test_workers_write_embeddings_and_encoder_comparison(monkeypatch, tmp_path: Path) -> None:
    import src.a1_probe_encoders.main as probe_encoders

    paths = _make_inputs(tmp_path)

    monkeypatch.setattr(probe_encoders, "_RAW_CNN_EPOCHS", 2)
    monkeypatch.setattr(probe_encoders, "_RAW_CNN_BATCH_SIZE", 3)
    monkeypatch.setattr(probe_encoders, "_RAW_CNN_IMAGE_SIZE", 16)

    class FakeExtractor:
        def extract_batch(self, images):
            rows = []
            for image in images:
                pixels = np.asarray(image, dtype=np.float32) / 255.0
                mean_intensity = float(np.mean(pixels))
                rows.append(np.array([mean_intensity, mean_intensity * 2.0, 1.0], dtype=np.float32))
            return np.stack(rows, axis=0)

    monkeypatch.setattr(
        probe_encoders,
        "_build_virchow_extractor",
        lambda weights_path, *, device: FakeExtractor(),
    )

    worker_args = _base_worker_args(paths)
    assert probe_encoders.main(["--worker", "raw_cnn", *worker_args]) == 0
    cnn_embeddings = np.load(paths["out_dir"] / "raw_cnn_embeddings.npy")
    assert cnn_embeddings.shape == (6, 256)

    assert probe_encoders.main(
        [
            "--worker",
            "virchow",
            *worker_args,
            "--virchow-weights",
            str(paths["weights_path"]),
        ]
    ) == 0
    virchow_embeddings = np.load(paths["out_dir"] / "virchow_embeddings.npy")
    assert virchow_embeddings.shape == (6, 3)
    assert not (paths["out_dir"] / "virchow_SKIPPED.txt").exists()

    assert probe_encoders.main(
        [
            "--worker",
            "compare",
            *worker_args,
            "--virchow-weights",
            str(paths["weights_path"]),
        ]
    ) == 0

    csv_path = paths["out_dir"] / "encoder_comparison.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["target", "uni_r2", "virchow_r2", "cnn_r2"]
    assert [row["target"] for row in rows] == ["tumor", "immune"]
    assert [float(row["uni_r2"]) for row in rows] == [0.75, 0.25]
    assert all(row["virchow_r2"] for row in rows)
    assert all(row["cnn_r2"] for row in rows)