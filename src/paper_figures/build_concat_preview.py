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

from PIL import Image, ImageDraw, ImageFont
from PIL import JpegImagePlugin  # noqa: F401 - registers Pillow's PDF RGB writer

try:
    import matplotlib
except ModuleNotFoundError:  # preview/PDF export only needs the bundled fonts
    matplotlib = None

ROOT = Path(__file__).resolve().parents[2]
PNGS_ROOT = ROOT / "figures" / "pngs_updated"
CONCAT_DIR = PNGS_ROOT / "concat"
FIGURES_MD = ROOT / "figures.md"
OUT_PATH = CONCAT_DIR / "_all_figures_preview.png"
OUT_PATH_V2 = CONCAT_DIR / "_all_figures_preview_v2.png"
OUT_PATH_SI = CONCAT_DIR / "_all_SI_figures.png"
OUT_PATH_PDF = OUT_PATH.with_suffix(".pdf")
OUT_PATH_PDF_V2 = OUT_PATH_V2.with_suffix(".pdf")
OUT_PATH_SI_PDF = OUT_PATH_SI.with_suffix(".pdf")

MASTER_DPI = 150

# Entries containing "/" resolve relative to PNGS_ROOT (e.g. the graphical
# abstract under methods/); bare basenames resolve against CONCAT_DIR.
# Basenames listed here get an unnumbered "Graphical abstract" header instead of
# a "Figure N" header and do not advance the figure counter.
UNNUMBERED = {"overview_workflow.png"}

# Main-text figures: the graphical abstract on top, then the reorganized
# fig1-fig4 composites. Fig 3 uses the v2 side-by-side layout. Fig 4 uses the
# v1 bar-chart D/E variant; fig4_*_v2 is kept as an alternate render and is not
# listed in the main paper preview. The former standalone panels are embedded
# inside fig3/fig4 and are intentionally not listed separately.
MAIN_FIGURE_ORDER = [
    "methods/overview_workflow.png",
    "fig1_approach_data.png",
    "fig2_architecture_performance.png",
    "fig3_uni_decomposition_v2.png",
    "fig4_per_channel_impact.png",
]

# Supplementary figures: the panels that survive the reorganization as
# standalone SI items. performance Panel A/C and SI_A1_A2 Panel A/B were
# promoted into fig2, leaving only the ranking tables (Panel B) and the
# qualitative tile grid (Panel C) for the SI.
SI_FIGURE_ORDER = [
    "si_performance_ranking.png",
    "ablation_grids_combined.png",
    "si_a1a2_qualitative_tiles.png",
]

# Figures intended for a single (half-page) column — left at their native width
# in the preview. Every other figure is a full-width figure and is resized up to
# the common full width, so figures that came out a bit too narrow fill the
# column instead of floating narrow. The fig1-fig4 composites and the SI items
# are all full-width, so this is empty.
SINGLE_COLUMN: set[str] = set()

# Point sizes (rendered at MASTER_DPI). Caption matches ~11 pt body text.
PT_CAPTION = 11
PT_HEADER = 13
PT_TITLE = 18

_FONT_DIRS = [
    Path(matplotlib.get_data_path()) / "fonts" / "ttf" if matplotlib is not None else None,
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/dejavu-sans-fonts"),
]


def _pt_to_px(pt: float) -> int:
    return round(pt / 72.0 * MASTER_DPI)


def _font(bold: bool, pt: float) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for font_dir in _FONT_DIRS:
        if font_dir is None:
            continue
        font_path = font_dir / name
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), _pt_to_px(pt))
    return ImageFont.load_default(size=_pt_to_px(pt))


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


def build(
    figure_order: list[str],
    out_path: Path,
    *,
    title: str,
    label_prefix: str = "Figure",
) -> Path:
    captions = parse_captions(FIGURES_MD)

    margin = _pt_to_px(28)            # ~0.39 in page margin
    gap_fig = _pt_to_px(34)           # between one caption and the next figure
    gap_cap = _pt_to_px(10)           # between image and its caption
    gap_hdr = _pt_to_px(6)            # between header label and image
    cap_leading = round(_pt_to_px(PT_CAPTION) * 1.42)
    hdr_h = round(_pt_to_px(PT_HEADER) * 1.5)

    # Load every figure and record its native physical width (inches). Each entry
    # carries a precomputed header string (so the graphical abstract can be
    # unnumbered) and its basename (the caption / SINGLE_COLUMN key).
    raw = []
    fig_num = 0
    for name in figure_order:
        path = (PNGS_ROOT / name) if "/" in name else (CONCAT_DIR / name)
        if not path.is_file():
            print(f"skip (missing): {name}")
            continue
        basename = Path(name).name
        src = Image.open(path)
        dpi = float(src.info.get("dpi", (MASTER_DPI, MASTER_DPI))[0])
        if src.mode == "RGBA":
            # Flatten transparency onto white so a transparent abstract background
            # doesn't fall back to black under convert("RGB").
            im = Image.new("RGB", src.size, "white")
            im.paste(src, mask=src.split()[-1])
        else:
            im = src.convert("RGB")
        if basename in UNNUMBERED:
            header = "Graphical abstract"
        else:
            fig_num += 1
            header = f"{label_prefix}{fig_num}"
        raw.append((header, basename, im, im.width / dpi))

    # Full-width = widest of the full-width (non-single-column) figures. Resize
    # every full-width figure up/down to that width so any that came out too
    # narrow (SI composite, uni_probe) fill the column instead of floating narrow.
    # Single-column figures keep their native width (aspect preserved throughout).
    # The source PNGs are unchanged — this only affects the preview composite.
    full_w_in = max(w_in for _h, base, _im, w_in in raw if base not in SINGLE_COLUMN)
    content_w = round(full_w_in * MASTER_DPI)
    items = []
    for header, base, im, w_in in raw:
        target_w = round(w_in * MASTER_DPI) if base in SINGLE_COLUMN else content_w
        new_h = max(1, round(im.height / im.width * target_w))
        im = im.resize((target_w, new_h), Image.LANCZOS)
        items.append((header, base, im, captions.get(base)))

    canvas_w = content_w + 2 * margin

    # Pass 1: measure heights (need a draw context for text metrics).
    dummy = Image.new("RGB", (10, 10), "white")
    ddraw = ImageDraw.Draw(dummy)

    title_lines, _ = wrap_caption(
        ddraw, [(title, True)],
        content_w, PT_TITLE,
    )
    title_h = len(title_lines) * round(_pt_to_px(PT_TITLE) * 1.4)

    layout = []
    for header, base, im, cap in items:
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
        layout.append((header, base, im, lines, space_w, block_h))

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
    for header, base, im, lines, space_w, block_h in layout:
        # Header label
        draw.text((margin, y), f"({base})", font=hf, fill=(90, 90, 90))
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

    canvas.save(out_path, dpi=(MASTER_DPI, MASTER_DPI))
    print(f"wrote {out_path}  ({canvas_w} x {total_h} px @ {MASTER_DPI} dpi)")
    return out_path


def _load_pdf_items(figure_order: list[str]):
    captions = parse_captions(FIGURES_MD)
    raw = []
    for name in figure_order:
        path = (PNGS_ROOT / name) if "/" in name else (CONCAT_DIR / name)
        if not path.is_file():
            print(f"skip PDF page (missing): {name}")
            continue
        basename = Path(name).name
        src = Image.open(path)
        dpi = float(src.info.get("dpi", (MASTER_DPI, MASTER_DPI))[0])
        if src.mode == "RGBA":
            im = Image.new("RGB", src.size, "white")
            im.paste(src, mask=src.split()[-1])
        else:
            im = src.convert("RGB")
        raw.append((basename, im, im.width / dpi, captions.get(basename)))
    if not raw:
        return []

    full_w_in = max(w_in for base, _im, w_in, _cap in raw if base not in SINGLE_COLUMN)
    content_w = round(full_w_in * MASTER_DPI)
    items = []
    for base, im, w_in, cap in raw:
        target_w = round(w_in * MASTER_DPI) if base in SINGLE_COLUMN else content_w
        new_h = max(1, round(im.height / im.width * target_w))
        items.append((base, im.resize((target_w, new_h), Image.LANCZOS), cap))
    return items


def _draw_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    lines,
    *,
    x: int,
    y: int,
    space_w: float,
    fill: tuple[int, int, int],
    leading: int,
) -> int:
    for line in lines:
        xx = x
        for word, font, width in line:
            draw.text((xx, y), word, font=font, fill=fill)
            xx += width + space_w
        y += leading
    return y


def _render_pdf_page(
    items: list[tuple[str, Image.Image, str | None]],
    *,
    page_title: str | None = None,
) -> Image.Image:
    margin = _pt_to_px(28)
    gap_fig = _pt_to_px(28)
    gap_cap = _pt_to_px(10)
    gap_hdr = _pt_to_px(6)
    cap_leading = round(_pt_to_px(PT_CAPTION) * 1.42)
    hdr_h = round(_pt_to_px(PT_HEADER) * 1.5)
    title_leading = round(_pt_to_px(PT_TITLE) * 1.4)

    content_w = max(im.width for _base, im, _cap in items)
    dummy = Image.new("RGB", (10, 10), "white")
    ddraw = ImageDraw.Draw(dummy)

    title_lines = []
    title_h = 0
    if page_title:
        title_lines, _ = wrap_caption(ddraw, [(page_title, True)], content_w, PT_TITLE)
        title_h = len(title_lines) * title_leading + gap_fig

    blocks = []
    for base, im, cap in items:
        if cap:
            lines, space_w = wrap_caption(ddraw, md_to_runs(cap), im.width, PT_CAPTION)
        else:
            missing = "(no caption in figures.md)"
            lines = [[(missing, _font(False, PT_CAPTION), ddraw.textlength(missing, font=_font(False, PT_CAPTION)))]]
            space_w = ddraw.textlength(" ", font=_font(False, PT_CAPTION))
        cap_h = len(lines) * cap_leading
        block_h = hdr_h + gap_hdr + im.height + gap_cap + cap_h
        blocks.append((base, im, lines, space_w, block_h))

    page_w = content_w + 2 * margin
    page_h = margin + title_h + sum(block[-1] for block in blocks) + gap_fig * (len(blocks) - 1) + margin
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)

    y = margin
    if title_lines:
        y = _draw_wrapped_lines(
            draw,
            title_lines,
            x=margin,
            y=y,
            space_w=draw.textlength(" ", font=_font(True, PT_TITLE)),
            fill=(20, 20, 20),
            leading=title_leading,
        )
        y += gap_fig

    hf = _font(True, PT_HEADER)
    for idx, (base, im, lines, space_w, _block_h) in enumerate(blocks):
        draw.text((margin, y), f"({base})", font=hf, fill=(90, 90, 90))
        y += hdr_h + gap_hdr
        page.paste(im, (margin, y))
        y += im.height + gap_cap
        y = _draw_wrapped_lines(
            draw,
            lines,
            x=margin,
            y=y,
            space_w=space_w,
            fill=(15, 15, 15),
            leading=cap_leading,
        )
        if idx < len(blocks) - 1:
            y += gap_fig
    return page


def build_pdf(
    figure_order: list[str],
    out_path: Path,
    *,
    title: str,
    combine_first_two: bool = False,
) -> Path | None:
    items = _load_pdf_items(figure_order)
    if not items:
        print(f"skip PDF (no figures): {out_path}")
        return None

    page_groups: list[list[tuple[str, Image.Image, str | None]]] = []
    if combine_first_two and len(items) >= 2 and items[0][0] in UNNUMBERED:
        page_groups.append(items[:2])
        page_groups.extend([item] for item in items[2:])
    else:
        page_groups.extend([item] for item in items)

    pages = [
        _render_pdf_page(group, page_title=title if i == 0 else None)
        for i, group in enumerate(page_groups)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        out_path,
        save_all=True,
        append_images=pages[1:],
        resolution=MASTER_DPI,
    )
    print(f"wrote {out_path}  ({len(pages)} pages @ {MASTER_DPI} dpi)")
    return out_path


def main() -> None:
    main_title = "Main figures - paper preview (fig1-fig4, filled to common column width, %d dpi)" % MASTER_DPI
    si_title = "Supplementary figures - paper preview (filled to common column width, %d dpi)" % MASTER_DPI
    build(
        MAIN_FIGURE_ORDER,
        OUT_PATH_V2,
        title=main_title,
        label_prefix="Figure ",
    )
    build_pdf(
        MAIN_FIGURE_ORDER,
        OUT_PATH_PDF_V2,
        title=main_title,
        combine_first_two=True,
    )
    build(
        SI_FIGURE_ORDER,
        OUT_PATH_SI,
        title=si_title,
        label_prefix="Figure S",
    )
    build_pdf(
        SI_FIGURE_ORDER,
        OUT_PATH_SI_PDF,
        title=si_title,
    )


if __name__ == "__main__":
    main()
