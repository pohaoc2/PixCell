"""
color_constants.py — Shared color palette and channel display config for PixCell visualizations.

Import in any visualization script for consistent coloring across all figures:

    from tools.color_constants import (
        CELL_TYPE_COLORS, CELL_STATE_COLORS,
        CHANNEL_CMAP, CHANNEL_LABEL,
        GLUCOSE_PROXY_CMAP, OXYGEN_PROXY_CMAP,
        SECTION_BG, SECTION_TEXT,
    )
"""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap as _LSCmap

# ── Nutrient proxy RGBA-style gradients (continuous [0, 1]) ─────────────────────
# Used for matplotlib imshow + matching colorbars in Stage 3 overviews.

OXYGEN_PROXY_CMAP = _LSCmap.from_list(
    "oxygen_proxy",
    [
        (0.00, (0.35, 0.00, 0.08)),  # hypoxic — deep red
        (0.45, (0.75, 0.35, 0.20)),
        (1.00, (0.45, 0.92, 1.00)),  # oxygenated — light cyan
    ],
)

GLUCOSE_PROXY_CMAP = _LSCmap.from_list(
    "glucose_proxy",
    [
        (0.00, (0.12, 0.02, 0.22)),  # depleted — dark violet
        (0.50, (0.55, 0.40, 0.08)),
        (1.00, (0.98, 0.95, 0.35)),  # high — bright yellow
    ],
)


def _bk_cmap(r: int, g: int, b: int, name: str):
    """Colormap from black (value=0) to specified RGB (value=1), for binary channels."""
    return _LSCmap.from_list(name, [(0.0, 0.0, 0.0), (r / 255, g / 255, b / 255)])

# ── Cell biology RGBA overlay colors (0–255 per channel) ─────────────────────
# Used for cell-type/state overlays on H&E images.

CELL_TYPES: tuple[str, str, str] = ("cancer", "immune", "healthy")

CELL_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "cancer":  (220, 50,  50,  200),
    "immune":  (50,  100, 220, 200),
    "healthy": (50,  180, 50,  200),
    "other":   (150, 150, 150, 150),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferative": (240, 190, 0,   200),
    "quiescent":     (120, 120, 120, 200),
    "dead":          (110, 60,  20,  200),
    "other":         (80,  80,  80,  150),
}

# ── Per-channel matplotlib colormap names ─────────────────────────────────────
# Usage: plt.imshow(channel_img, cmap=CHANNEL_CMAP[channel_name])
#
# Design rationale:
#   cell types  — each uses the color family matching its CELL_TYPE_COLORS entry
#   cell states — warm/cool/earthy to distinguish biological significance
#   vasculature — Reds (blood vessels)
#   oxygen      — OXYGEN_PROXY_CMAP (hypoxic red → oxygenated cyan)
#   glucose     — GLUCOSE_PROXY_CMAP (depleted violet → high yellow)

CHANNEL_CMAP: dict = {
    # Binary channels: black background → cell color
    "cell_masks":           "gray",                                    # black → white
    "cell_type_cancer":     _bk_cmap(220, 50,  50,  "cmap_cancer"),  # black → red
    "cell_type_immune":     _bk_cmap(50,  100, 220, "cmap_immune"),  # black → blue
    "cell_type_healthy":    _bk_cmap(50,  180, 50,  "cmap_healthy"), # black → green
    "cell_state_prolif":    _bk_cmap(240, 190, 0,   "cmap_prolif"),  # black → yellow
    "cell_state_nonprolif": _bk_cmap(120, 120, 120, "cmap_nonprolif"), # black → grey
    "cell_state_dead":      _bk_cmap(110, 60,  20,  "cmap_dead"),    # black → brown
    # Continuous channels: keep standard colormaps
    "vasculature":          "Reds",
    "oxygen":               OXYGEN_PROXY_CMAP,
    "glucose":              GLUCOSE_PROXY_CMAP,
}

# ── Per-channel display labels ─────────────────────────────────────────────────

CHANNEL_LABEL: dict[str, str] = {
    "cell_masks":          "Cell Mask",
    "cell_type_cancer":    "Cancer",
    "cell_type_immune":    "Immune",
    "cell_type_healthy":   "Healthy",
    "cell_state_prolif":   "Prolif.",
    "cell_state_nonprolif":"Non-prolif.",
    "cell_state_dead":     "Dead",
    "vasculature":         "Vasculature",
    "oxygen":              "O\u2082",
    "glucose":             "Glucose",
}

# ── Section background/text colors ────────────────────────────────────────────
# Used to color-code figure panels by role.

SECTION_BG: dict[str, str] = {
    "input":     "#dce8f5",   # light blue   — TME input channels
    "output":    "#d5f0d5",   # light green  — model-generated H&E
    "reference": "#fde8cc",   # light orange — ground-truth / layout reference
    "style_ref": "#ece0f0",   # light purple — style source tile (unpaired mode)
    "analysis":  "#f0f0f0",   # light gray   — attention / residual analysis panels
}

SECTION_TEXT: dict[str, str] = {
    "input":     "#1a3a5c",
    "output":    "#1a6b1a",
    "reference": "#8b4500",
    "style_ref": "#4a1a6b",
    "analysis":  "#333333",
}

# ── Convenience: normalized RGBA (0–1) for matplotlib ─────────────────────────

def rgba_norm(r: int, g: int, b: int, a: int = 255) -> tuple[float, float, float, float]:
    """Convert 0-255 RGBA to 0-1 for matplotlib."""
    return (r / 255, g / 255, b / 255, a / 255)


CELL_TYPE_COLORS_NORM: dict[str, tuple[float, float, float, float]] = {
    k: rgba_norm(*v) for k, v in CELL_TYPE_COLORS.items()
}

CELL_STATE_COLORS_NORM: dict[str, tuple[float, float, float, float]] = {
    k: rgba_norm(*v) for k, v in CELL_STATE_COLORS.items()
}
