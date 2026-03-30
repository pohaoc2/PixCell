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
        (0.0, (0.0, 0.0, 0.0)),  # hypoxic — black
        (1.0, (0.0, 1.0, 1.0)),  # oxygenated — cyan
    ],
)

GLUCOSE_PROXY_CMAP = _LSCmap.from_list(
    "glucose_proxy",
    [
        (0.0, (0.0, 0.0, 0.0)),  # depleted — black
        (1.0, (1.0, 0.95, 0.12)),  # high — bright yellow
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
    "other":   (150, 150, 150, 120),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferative": (230, 50,  180, 200),  # magenta
    "nonprolif":     (240, 140, 30,  200),  # amber
    "dead":          (110, 40,  160, 200),  # purple
    "other":         (160, 160, 160, 120),
}

# ── Per-channel matplotlib colormap names ─────────────────────────────────────
# Usage: plt.imshow(channel_img, cmap=CHANNEL_CMAP[channel_name])
#
# Design rationale:
#   cell types  — each uses the color family matching its CELL_TYPE_COLORS entry
#   cell states — warm/cool/earthy to distinguish biological significance
#   vasculature — Reds (blood vessels)
#   oxygen      — OXYGEN_PROXY_CMAP (hypoxic black → oxygenated cyan)
#   glucose     — GLUCOSE_PROXY_CMAP (depleted black → high bright yellow)

CHANNEL_CMAP: dict = {
    # Binary channels: black background → cell color
    "cell_masks":           "gray",                                    # black → white
    "cell_type_cancer":     _bk_cmap(220, 50,  50,  "cmap_cancer"),  # black → red
    "cell_type_immune":     _bk_cmap(50,  100, 220, "cmap_immune"),  # black → blue
    "cell_type_healthy":    _bk_cmap(50,  180, 50,  "cmap_healthy"), # black → green
    "cell_state_prolif":    _bk_cmap(230, 50,  180, "cmap_prolif"),  # black → magenta
    "cell_state_nonprolif": _bk_cmap(240, 140, 30,  "cmap_nonprolif"), # black → amber
    "cell_state_dead":      _bk_cmap(110, 40,  160, "cmap_dead"),    # black → purple
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
