"""Normalize main concat figure PNGs to one manuscript column width.

This pads narrower figures on a white canvas. It never stretches or squashes the
rendered figure content, so text size and glyph aspect ratios stay unchanged.

Run:  python -m src.paper_figures.normalize_main_figure_widths
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.ablation_report.shared import ROOT

CONCAT_DIR = ROOT / "figures" / "pngs_updated" / "concat"

MAIN_WIDTH_NORMALIZE = [
    "fig1_approach_data.png",
    "fig2_architecture_performance.png",
    "fig3_uni_decomposition_v2.png",
    "fig4_per_channel_impact.png",
]


def _flatten_on_white(src: Image.Image) -> Image.Image:
    if src.mode == "RGBA":
        out = Image.new("RGBA", src.size, (255, 255, 255, 255))
        out.alpha_composite(src)
        return out.convert("RGB")
    return src.convert("RGB")


def normalize(paths: list[Path] | None = None) -> int:
    paths = paths or [CONCAT_DIR / name for name in MAIN_WIDTH_NORMALIZE]
    existing = [path for path in paths if path.is_file()]
    if not existing:
        raise FileNotFoundError("No main figure PNGs found to normalize.")

    images = []
    for path in existing:
        src = Image.open(path)
        images.append((path, _flatten_on_white(src), src.info.get("dpi")))

    target_w = max(im.width for _path, im, _dpi in images)
    for path, im, dpi in images:
        if im.width == target_w:
            print(f"kept {path} ({im.width} x {im.height})")
            continue
        pad = target_w - im.width
        left = pad // 2
        out = Image.new("RGB", (target_w, im.height), "white")
        out.paste(im, (left, 0))
        save_kwargs = {"dpi": dpi} if dpi else {}
        out.save(path, **save_kwargs)
        print(f"padded {path} ({im.width} -> {target_w}, left={left}, right={pad - left})")
    return target_w


def main() -> None:
    target_w = normalize()
    print(f"normalized main figures to {target_w}px wide")


if __name__ == "__main__":
    main()
