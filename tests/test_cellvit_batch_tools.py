from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cellvit.export_batch import export_cellvit_batch
from tools.cellvit.import_results import import_cellvit_results


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_cache(tmp_path: Path) -> Path:
    cache_root = tmp_path / "cache"
    tile_dir = cache_root / "17408_32768"
    _write_png(tile_dir / "singles" / "01_cell_types.png", 10)
    _write_png(tile_dir / "all" / "generated_he.png", 20)
    manifest = {
        "version": 1,
        "tile_id": "17408_32768",
        "group_names": ["cell_types", "cell_state", "vasculature", "microenv"],
        "sections": [
            {
                "title": "1 active group",
                "subset_size": 1,
                "entries": [
                    {
                        "active_groups": ["cell_types"],
                        "condition_label": "Only: cell types",
                        "image_label": "Only: cell types",
                        "image_path": "singles/01_cell_types.png",
                    }
                ],
            },
            {
                "title": "4 active groups",
                "subset_size": 4,
                "entries": [
                    {
                        "active_groups": ["cell_types", "cell_state", "vasculature", "microenv"],
                        "condition_label": "Only: all",
                        "image_label": "Only: all",
                        "image_path": "all/generated_he.png",
                    }
                ],
            },
        ],
    }
    (tile_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return cache_root


def test_export_and_import_cellvit_roundtrip(tmp_path: Path):
    cache_root = _make_cache(tmp_path)
    batch_dir = tmp_path / "cellvit_batch"

    manifest_path, _, count = export_cellvit_batch(cache_root, batch_dir)
    assert count == 2

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    flat_names = sorted(entry["flat_name"] for entry in payload["entries"])
    assert flat_names == [
        "17408_32768__all__generated_he.png",
        "17408_32768__singles__01_cell_types.png",
    ]

    results_dir = tmp_path / "cellvit_results"
    results_dir.mkdir()
    for flat_name in flat_names:
        stem = Path(flat_name).stem
        (results_dir / f"{stem}.json").write_text('{"patch":"x","cells":[]}', encoding="utf-8")

    imported, report_path = import_cellvit_results(
        manifest_path,
        results_dir,
    )
    assert len(imported) == 2
    assert report_path.is_file()
    assert (cache_root / "17408_32768" / "singles" / "01_cell_types_cellvit_instances.json").is_file()
    assert (cache_root / "17408_32768" / "all" / "generated_he_cellvit_instances.json").is_file()
