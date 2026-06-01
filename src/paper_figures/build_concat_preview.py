"""Print-size preview: stack every concat figure at its true physical size.

Each figure under ``figures/pngs_updated/concat/`` is placed on one tall canvas
at the same physical scale (resampled to a single master DPI from the DPI
embedded in each PNG), with no per-figure rescaling that would change its print
size. The caption for each figure is read from ``figures.md`` and rendered
beneath it, so the composite shows how the figures would look printed together.

Run:  python -m src.paper_figures.build_concat_preview
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CONCAT_DIR = ROOT / "figures" / "pngs_updated" / "concat"
FIGURES_MD = ROOT / "figures.md"
OUT_PATH = CONCAT_DIR / "_all_figures_preview.png"

MASTER_DPI = 150

# Order figures main-text first, then supplementary. sweep_grid_combined is
# omitted: it is panel C of uni_probe_overview, not a standalone figure.
FIGURE_ORDER = [
    "performance_paired_unpaired.png",
    "uni_probe_overview.png",
    "08_uni_tme_decomposition.png",
    "07d_t1_spatial_multi_encoder.png",
    "09b_channel_color_layout_impact.png",
    "ablation_grids_combined.png",
    "SI_A1_A2_unified.png",
]

# Figures intended for a single (half-page) column — left at their native width
# in the preview. Every other figure is a full-width figure and is resized up to
# the common full width, so figures that came out a bit too narrow (e.g. the SI
# composite, uni_probe) fill the column instead of floating narrow.
SINGLE_COLUMN = {
    "08_uni_tme_decomposition.png",
    "09b_channel_color_layout_impact.png",
}

# Point sizes (rendered at MASTER_DPI). Caption matches ~11 pt body text.
PT_CAPTION = 11
PT_HEADER = 13
PT_TITLE = 18

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def _pt_to_px(pt: float) -> int:
    return round(pt / 72.0 * MASTER_DPI)


def _font(bold: bool, pt: float) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(_FONT_DIR / name), _pt_to_px(pt))


def parse_captions(md_path: Path) -> dict[str, str]:
    """Map concat PNG basename -> raw caption markdown (the paragraph after
    '### Figure caption')."""
    text = md_path.read_text()
    captions: dict[str, str] = {}
    # Sections look like: ## name (`path`) ... ### Figure caption \n\n <para>
    section_re = re.compile(r"^##\s+.*?\(`([^`]+)`\)\s*$", re.MULTILINE)
    matches = list(section_re.finditer(text))
    for i, m in enumerate(matches):
        path = m.group(1)
        basename = Path(path).name
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        cap_m = re.search(r"###\s+Figure caption\s*\n+(.+?)(?:\n\s*\n|\Z)", block, re.DOTALL)
        if cap_m:
            captions[basename] = " ".join(cap_m.group(1).split())
    return captions


def md_to_runs(text: str) -> list[tuple[str, bool]]:
    """Split markdown into (text, is_bold) runs; drop italics markers."""
    text = text.replace(r"\*", "\x00")
    runs: list[tuple[str, bool]] = []
    bold = False
    for part in text.split("**"):
        part = part.replace("*", "").replace("\x00", "*")
        if part:
            runs.append((part, bold))
        bold = not bold
    return runs


def wrap_caption(draw, runs, max_w, pt):
    """Greedy word-wrap mixed bold/regular runs to lines of (word, font, width)."""
    f_reg, f_bold = _font(False, pt), _font(True, pt)
    space_w = draw.textlength(" ", font=f_reg)
    words: list[tuple[str, ImageFont.FreeTypeFont]] = []
    for txt, bold in runs:
        f = f_bold if bold else f_reg
        for w in txt.split():
            words.append((w, f))
    lines: list[list[tuple[str, ImageFont.FreeTypeFont, float]]] = []
    cur: list[tuple[str, ImageFont.FreeTypeFont, float]] = []
    cur_w = 0.0
    for w, f in words:
        ww = draw.textlength(w, font=f)
        sp = space_w if cur else 0
        if cur and cur_w + sp + ww > max_w:
            lines.append(cur)
            cur, cur_w = [(w, f, ww)], ww
        else:
            cur.append((w, f, ww))
            cur_w += sp + ww
    if cur:
        lines.append(cur)
    return lines, space_w


def build() -> Path:
    captions = parse_captions(FIGURES_MD)

    margin = _pt_to_px(28)            # ~0.39 in page margin
    gap_fig = _pt_to_px(34)           # between one caption and the next figure
    gap_cap = _pt_to_px(10)           # between image and its caption
    gap_hdr = _pt_to_px(6)            # between header label and image
    cap_leading = round(_pt_to_px(PT_CAPTION) * 1.42)
    hdr_h = round(_pt_to_px(PT_HEADER) * 1.5)

    # Load every figure and record its native physical width (inches).
    raw = []
    for idx, name in enumerate(FIGURE_ORDER, start=1):
        path = CONCAT_DIR / name
        if not path.is_file():
            print(f"skip (missing): {name}")
            continue
        im = Image.open(path).convert("RGB")
        dpi = float(im.info.get("dpi", (MASTER_DPI, MASTER_DPI))[0])
        raw.append((idx, name, im, im.width / dpi))

    # Full-width = widest of the full-width (non-single-column) figures. Resize
    # every full-width figure up/down to that width so any that came out too
    # narrow (SI composite, uni_probe) fill the column instead of floating narrow.
    # Single-column figures keep their native width (aspect preserved throughout).
    # The source PNGs are unchanged — this only affects the preview composite.
    full_w_in = max(w_in for _idx, name, _im, w_in in raw if name not in SINGLE_COLUMN)
    content_w = round(full_w_in * MASTER_DPI)
    items = []
    for idx, name, im, w_in in raw:
        target_w = round(w_in * MASTER_DPI) if name in SINGLE_COLUMN else content_w
        new_h = max(1, round(im.height / im.width * target_w))
        im = im.resize((target_w, new_h), Image.LANCZOS)
        items.append((idx, name, im, captions.get(name)))

    canvas_w = content_w + 2 * margin

    # Pass 1: measure heights (need a draw context for text metrics).
    dummy = Image.new("RGB", (10, 10), "white")
    ddraw = ImageDraw.Draw(dummy)

    title_lines, _ = wrap_caption(
        ddraw, [("PixCell concat figures - paper preview (full-width figures filled to column; single-column figures at native width, %d dpi)" % MASTER_DPI, True)],
        content_w, PT_TITLE,
    )
    title_h = len(title_lines) * round(_pt_to_px(PT_TITLE) * 1.4)

    layout = []
    for idx, name, im, cap in items:
        # Caption wraps to the figure's own width (paper convention) so it never
        # extends past the figure — single-column figures get taller captions.
        cap_w = im.width
        if cap:
            lines, space_w = wrap_caption(ddraw, md_to_runs(cap), cap_w, PT_CAPTION)
        else:
            lines, space_w = ([[("(no caption in figures.md)", _font(False, PT_CAPTION),
                                 ddraw.textlength("(no caption in figures.md)", font=_font(False, PT_CAPTION)))]],
                              ddraw.textlength(" ", font=_font(False, PT_CAPTION)))
        cap_h = len(lines) * cap_leading
        block_h = hdr_h + gap_hdr + im.height + gap_cap + cap_h
        layout.append((idx, name, im, lines, space_w, block_h))

    total_h = margin + title_h + gap_fig + sum(b[-1] + gap_fig for b in layout) + margin

    canvas = Image.new("RGB", (canvas_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    y = margin
    # Title
    tf = _font(True, PT_TITLE)
    for line in title_lines:
        x = margin
        for w, f, ww in line:
            draw.text((x, y), w, font=f, fill=(20, 20, 20))
            x += ww + draw.textlength(" ", font=tf)
        y += round(_pt_to_px(PT_TITLE) * 1.4)
    y += gap_fig

    hf = _font(True, PT_HEADER)
    for idx, name, im, lines, space_w, block_h in layout:
        # Header label
        draw.text((margin, y), f"Figure {idx}  -  {name}", font=hf, fill=(90, 90, 90))
        y += hdr_h + gap_hdr
        # Image (left-aligned at native physical size)
        canvas.paste(im, (margin, y))
        y += im.height + gap_cap
        # Caption
        for line in lines:
            x = margin
            for w, f, ww in line:
                draw.text((x, y), w, font=f, fill=(15, 15, 15))
                x += ww + space_w
            y += cap_leading
        y += gap_fig

    canvas.save(OUT_PATH, dpi=(MASTER_DPI, MASTER_DPI))
    print(f"wrote {OUT_PATH}  ({canvas_w} x {total_h} px @ {MASTER_DPI} dpi)")
    return OUT_PATH


if __name__ == "__main__":
    build()
