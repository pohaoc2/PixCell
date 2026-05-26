#!/usr/bin/env python3
"""One-tile leave-one-out visualization.

Row 1: H&E (full channels, then each group dropped).
Row 2: binary cell layout rasterized from CellViT instance contours.

Reads grouped-ablation cache produced by Fig-3 ablation runner:
  <ablation-root>/<tile_id>/all/generated_he.png
  <ablation-root>/<tile_id>/all/generated_he_cellvit_instances.json
  <ablation-root>/<tile_id>/manifest.json           (drives panel order)
  <ablation-root>/<tile_id>/triples/*.png           (1-group LOO renders)
  <ablation-root>/<tile_id>/triples/*_cellvit_instances.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOO_ROOT = ROOT / "inference_output" / "concat_ablation_1000" / "paired_ablation" / "ablation_results"
DEFAULT_TILE_ID = "11008_5888"
DEFAULT_OUT = ROOT / "inference_output" / "concat_ablation_1000" / "loo_tile_example.png"


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _rasterize_cells(json_path: Path, size: int) -> np.ndarray:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    for cell in data.get("cells", []):
        contour = cell.get("contour") or []
        if len(contour) < 3:
            continue
        draw.polygon([(int(x), int(y)) for x, y in contour], fill=255)
    return np.asarray(mask, dtype=np.uint8)


def _label(name: str) -> str:
    return name.replace("cell_types", "cell types").replace("cell_state", "cell state").replace("microenv", "nutrient")


def render(tile_id: str, loo_root: Path, out_path: Path) -> Path:
    tile_dir = loo_root / tile_id
    if not tile_dir.is_dir():
        raise SystemExit(f"missing tile dir: {tile_dir}")

    manifest = json.loads((tile_dir / "manifest.json").read_text(encoding="utf-8"))
    all_groups = set(manifest["group_names"])

    panels: list[tuple[str, Path, Path]] = [
        ("full", tile_dir / "all" / "generated_he.png", tile_dir / "all" / "generated_he_cellvit_instances.json"),
    ]
    for section in manifest["sections"]:
        if section["subset_size"] != len(all_groups) - 1:
            continue
        for entry in section["entries"]:
            dropped = (all_groups - set(entry["active_groups"])).pop()
            png = tile_dir / entry["image_path"]
            js = png.with_name(png.stem + "_cellvit_instances.json")
            panels.append((f"-{_label(dropped)}", png, js))

    missing = [str(p) for _, png, js in panels for p in (png, js) if not p.is_file()]
    if missing:
        raise SystemExit("missing assets:\n  " + "\n  ".join(missing))

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.0), constrained_layout=True)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (label, png_path, js_path) in enumerate(panels):
        he = _load_rgb(png_path)
        cells = _rasterize_cells(js_path, size=he.shape[0])

        ax_he = axes[0, col]
        ax_he.imshow(he)
        ax_he.set_title(label, fontsize=8)
        ax_he.set_xticks([])
        ax_he.set_yticks([])

        ax_mask = axes[1, col]
        ax_mask.imshow(cells, cmap="gray", vmin=0, vmax=255)
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])

    axes[0, 0].set_ylabel("H&E", fontsize=9)
    axes[1, 0].set_ylabel("CellViT mask", fontsize=9)

    fig.suptitle(f"Tile {tile_id} — leave-one-out sub-channel", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tile-id", default=DEFAULT_TILE_ID)
    p.add_argument("--loo-root", type=Path, default=DEFAULT_LOO_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out = render(args.tile_id, args.loo_root, args.out)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
