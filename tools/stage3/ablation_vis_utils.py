"""
Shared helpers for Stage 3 ablation publication figures and metrics.

Legacy manifests may still list ``cell_identity``; it is normalized to ``cell_types``.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

# Canonical four groups (display / config order; see docs/ablation_vis_spec.md)
FOUR_GROUP_ORDER: tuple[str, ...] = ("cell_types", "cell_state", "vasculature", "microenv")

_LEGACY_GROUP_ALIASES: dict[str, str] = {
    "cell_identity": "cell_types",
}


def normalize_group_name(name: str) -> str:
    """Map legacy manifest keys to current group names."""
    return _LEGACY_GROUP_ALIASES.get(name, name)


def normalize_active_groups(active_groups: Sequence[str]) -> tuple[str, ...]:
    """Normalize and sort for stable keys (same order as ``condition_metric_key``)."""
    pub = sorted(normalize_group_name(g) for g in active_groups)
    return tuple(pub)


def public_group_names(active_groups: tuple[str, ...]) -> tuple[str, ...]:
    """Sorted public names for one condition (legacy aliases applied)."""
    return normalize_active_groups(active_groups)


def condition_metric_key(active_groups: tuple[str, ...]) -> str:
    """Stable ``+``-joined key for JSON metrics (e.g. ``cell_types+cell_state``)."""
    return "+".join(public_group_names(active_groups))


def ordered_subset_condition_tuples() -> list[tuple[str, ...]]:
    """The 15 subset conditions: cardinality 1, 2, 3, then all four groups (same order as cache)."""
    result: list[tuple[str, ...]] = []
    for k in (1, 2, 3):
        result.extend(tuple(c) for c in combinations(FOUR_GROUP_ORDER, k))
    result.append(tuple(FOUR_GROUP_ORDER))
    return result


def default_orion_uni_npy_path(orion_root: Path, tile_id: str) -> Path:
    """``data/orion-crc33/features/<tile_id>_uni.npy`` style path."""
    return orion_root / "features" / f"{tile_id}_uni.npy"


def default_orion_he_png_path(orion_root: Path, tile_id: str) -> Path | None:
    """Paired real H&E tile under ``he/{tile_id}.png`` (or ``.jpg``)."""
    for ext in (".png", ".jpg", ".jpeg", ".tif"):
        p = orion_root / "he" / f"{tile_id}{ext}"
        if p.is_file():
            return p
    return None


def cosine_metric_title(metric_name: str | None) -> str:
    """Display title for a cosine metric identifier from ``uni_cosine_scores.json``."""
    return "RGB cosine" if metric_name == "rgb_pixel_cosine" else "UNI cosine"


def parse_uni_cosine_scores_json(
    cache_dir: str | Path,
    *,
    allow_missing: bool = True,
) -> tuple[dict[str, float], str]:
    """Parse ``uni_cosine_scores.json`` into ``(per_condition, title)``.

    Invalid entries (``None`` / NaN) are skipped.
    """
    path = Path(cache_dir) / "uni_cosine_scores.json"
    if not path.is_file():
        if allow_missing:
            return {}, "UNI cosine"
        raise FileNotFoundError(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    title = cosine_metric_title(raw.get("metric", "uni_cosine"))

    per = raw.get("per_condition")
    if not isinstance(per, dict):
        return {}, title

    out: dict[str, float] = {}
    for k, v in per.items():
        if v is None:
            continue
        fv = float(v)
        if np.isnan(fv):
            continue
        out[str(k)] = fv
    return out, title


def cache_manifest_uni_features(
    cache_dir: str | Path,
    *,
    uni_model: str | Path,
    device: str = "cuda",
    force: bool = False,
) -> int:
    """Cache UNI embeddings for every image listed in ``manifest.json``.

    Embeddings are stored under ``<cache_dir>/features/<section>/<stem>_uni.npy``.
    Returns the number of feature files written.
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending: list[tuple[Path, Path]] = []
    for section in manifest.get("sections", []):
        for entry in section.get("entries", []):
            rel = Path(entry["image_path"])
            img_path = cache_dir / rel
            feat_path = cache_dir / "features" / rel.parent / f"{rel.stem}_uni.npy"
            if not force and feat_path.is_file():
                continue
            pending.append((img_path, feat_path))

    if not pending:
        return 0

    from pipeline.extract_features import UNI2hExtractor

    extractor = UNI2hExtractor(model_path=str(uni_model), device=device)
    written = 0
    for img_path, feat_path in pending:
        if not img_path.is_file():
            raise FileNotFoundError(f"missing cached image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        emb = np.asarray(extractor.extract(img), dtype=np.float64).ravel()
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feat_path, emb)
        written += 1
    return written


def compute_rgb_pixel_cosine_scores(
    cache_dir: Path,
    orion_root: Path,
    *,
    resize_hw: tuple[int, int] = (256, 256),
) -> tuple[dict[str, float], dict]:
    """
    Cosine similarity of flattened RGB vectors (reference H&E vs each generated PNG).

    No torch/UNI — used when UNI-2h is unavailable. Returns ``(per_condition, reference_meta)``.
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    orion_root = Path(orion_root)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    tile_id = str(manifest.get("tile_id", "")).strip()
    if not tile_id:
        raise ValueError("manifest.json must contain tile_id")

    he_path = default_orion_he_png_path(orion_root, tile_id)
    if he_path is None:
        raise FileNotFoundError(
            f"No reference H&E for tile {tile_id!r} under {orion_root / 'he'}",
        )

    def _vec(path: Path) -> np.ndarray:
        im = Image.open(path).convert("RGB").resize(resize_hw, Image.BILINEAR)
        v = np.asarray(im, dtype=np.float64).ravel()
        v /= np.linalg.norm(v) + 1e-12
        return v

    ref_vec = _vec(he_path)
    per: dict[str, float] = {}
    for section in manifest["sections"]:
        for entry in section["entries"]:
            key = condition_metric_key(tuple(entry["active_groups"]))
            gen_path = cache_dir / entry["image_path"]
            if not gen_path.is_file():
                raise FileNotFoundError(gen_path)
            gen_vec = _vec(gen_path)
            per[key] = float(np.dot(ref_vec, gen_vec))

    meta = {
        "type": "rgb_pixels",
        "path": str(he_path.resolve()),
        "resize_hw": list(resize_hw),
    }
    return per, meta


def draw_image_border(
    ax,
    color: str,
    *,
    dashed: bool = False,
    linewidth: float = 2.5,
) -> None:
    """Draw a visible border around an image axes even when spines are hidden."""
    from matplotlib.patches import Rectangle

    patch = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle="--" if dashed else "-",
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(patch)


# --- Channel PNG discovery (paired patch folder; filenames vary) ---

_CHANNEL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cell_types", ("type", "celltype", "cell_type")),
    ("cell_state", ("state", "cellstate", "cell_state")),
    ("vasculature", ("vasc", "vessel", "vasculature")),
    ("microenv", ("nutrient", "oxygen", "glucose", "microenv")),
)


def discover_channel_pngs(patch_dir: Path) -> dict[str, Path]:
    """
    Map logical channel → first matching ``*.png`` path (case-insensitive substring on stem).

    Unmatched groups are omitted; caller may warn.
    """
    found: dict[str, Path] = {}
    for p in sorted(patch_dir.glob("*.png")):
        stem_l = p.stem.lower()
        for slot, keywords in _CHANNEL_RULES:
            if slot in found:
                continue
            if any(k in stem_l for k in keywords):
                found[slot] = p
                break
    return found


def load_uni_cosine_scores(cache_dir: str | Path) -> dict[str, float]:
    """
    Load ``uni_cosine_scores.json`` written by ``compute_ablation_uni_cosine``.

    Returns ``per_condition`` mapping only; raises if file missing or invalid.
    """
    scores, _ = parse_uni_cosine_scores_json(cache_dir, allow_missing=False)
    if not scores:
        path = Path(cache_dir) / "uni_cosine_scores.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        per = raw.get("per_condition")
        if not isinstance(per, dict):
            raise ValueError(f"invalid uni_cosine_scores.json: missing per_condition in {path}")
    return scores


def parse_uni_cosine_for_condition(raw: dict[str, Any], active_groups: tuple[str, ...]) -> float | None:
    """Look up cosine for a condition from a loaded JSON dict."""
    per = raw.get("per_condition")
    if not isinstance(per, dict):
        return None
    key = condition_metric_key(active_groups)
    val = per.get(key)
    if val is None:
        return None
    return float(val)


def load_exp_channel_plane(
    exp_channels_dir: Path,
    channel: str,
    tile_id: str,
    *,
    resolution: int = 256,
) -> np.ndarray:
    """Load one spatial channel as float32 ``[H, W]`` in ``[0, 1]``."""
    from diffusion.data.datasets.sim_controlnet_dataset import (
        _find_file,
        _load_spatial_file,
        get_channel_load_config,
        resolve_channel_dir,
    )

    ch_dir = resolve_channel_dir(exp_channels_dir, channel)
    cfg = get_channel_load_config(channel)
    path = _find_file(ch_dir, tile_id, exts=cfg["preferred_exts"])
    return _load_spatial_file(
        path,
        resolution=resolution,
        binary=bool(cfg["binary"]),
        normalization=str(cfg["normalization"]),
    )


def _apply_mpl_cmap(plane: np.ndarray, cmap) -> np.ndarray:
    """Apply a matplotlib colormap; return uint8 RGB ``[H, W, 3]``."""
    import matplotlib as mpl
    import matplotlib.cm as cm

    v = np.clip(plane.astype(np.float64), 0.0, 1.0)
    if isinstance(cmap, str):
        try:
            cmap = mpl.colormaps[cmap]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap(cmap)
    rgba = cmap(v)
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def rgb_cell_types_union(
    exp_channels_dir: Path,
    tile_id: str,
    *,
    resolution: int = 128,
) -> np.ndarray:
    """Union of ``cell_type_*`` planes with ``CELL_TYPE_COLORS`` (additive)."""
    from tools.color_constants import CELL_TYPE_COLORS

    pairs = [
        ("cell_type_cancer", "cancer"),
        ("cell_type_immune", "immune"),
        ("cell_type_healthy", "healthy"),
    ]
    rgb = np.zeros((resolution, resolution, 3), dtype=np.float32)
    for ch_name, sem in pairs:
        try:
            plane = load_exp_channel_plane(
                exp_channels_dir, ch_name, tile_id, resolution=resolution
            )
        except (FileNotFoundError, OSError):
            continue
        color = np.array(CELL_TYPE_COLORS[sem][:3], dtype=np.float32) / 255.0
        rgb += plane[..., None] * color
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def rgb_cell_states_union(
    exp_channels_dir: Path,
    tile_id: str,
    *,
    resolution: int = 128,
) -> np.ndarray:
    """Union of ``cell_state_*`` planes with ``CELL_STATE_COLORS`` (additive)."""
    from tools.color_constants import CELL_STATE_COLORS

    pairs = [
        ("cell_state_prolif", "proliferative"),
        ("cell_state_nonprolif", "nonprolif"),
        ("cell_state_dead", "dead"),
    ]
    rgb = np.zeros((resolution, resolution, 3), dtype=np.float32)
    for ch_name, sem in pairs:
        try:
            plane = load_exp_channel_plane(
                exp_channels_dir, ch_name, tile_id, resolution=resolution
            )
        except (FileNotFoundError, OSError):
            continue
        color = np.array(CELL_STATE_COLORS[sem][:3], dtype=np.float32) / 255.0
        rgb += plane[..., None] * color
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def rgb_vasculature_panel(
    exp_channels_dir: Path,
    tile_id: str,
    *,
    resolution: int = 128,
) -> np.ndarray:
    """Vasculature channel using ``CHANNEL_CMAP['vasculature']`` (``Reds``)."""
    from tools.color_constants import CHANNEL_CMAP

    plane = load_exp_channel_plane(
        exp_channels_dir, "vasculature", tile_id, resolution=resolution
    )
    return _apply_mpl_cmap(plane, CHANNEL_CMAP["vasculature"])


def rgb_microenv_union(
    exp_channels_dir: Path,
    tile_id: str,
    *,
    resolution: int = 128,
) -> np.ndarray:
    """Union of oxygen + glucose with ``OXYGEN_PROXY_CMAP`` and ``GLUCOSE_PROXY_CMAP`` (additive)."""
    from tools.color_constants import GLUCOSE_PROXY_CMAP, OXYGEN_PROXY_CMAP

    try:
        oxy = load_exp_channel_plane(exp_channels_dir, "oxygen", tile_id, resolution=resolution)
    except (FileNotFoundError, OSError):
        oxy = np.zeros((resolution, resolution), dtype=np.float32)
    try:
        glu = load_exp_channel_plane(exp_channels_dir, "glucose", tile_id, resolution=resolution)
    except (FileNotFoundError, OSError):
        glu = np.zeros((resolution, resolution), dtype=np.float32)

    ro = _apply_mpl_cmap(oxy, OXYGEN_PROXY_CMAP).astype(np.float32)
    rg = _apply_mpl_cmap(glu, GLUCOSE_PROXY_CMAP).astype(np.float32)
    return np.clip(ro + rg, 0.0, 255.0).astype(np.uint8)


def build_exp_channel_header_rgb(
    exp_channels_dir: Path,
    tile_id: str,
    *,
    resolution: int = 128,
) -> dict[str, np.ndarray]:
    """
    Build uint8 RGB thumbnails for the four TME header slots (``FOUR_GROUP_ORDER`` keys).

    On load failure, returns a dark gray placeholder for that slot.
    """
    gray = np.full((resolution, resolution, 3), 45, dtype=np.uint8)
    builders = {
        "cell_types": lambda: rgb_cell_types_union(
            exp_channels_dir, tile_id, resolution=resolution
        ),
        "cell_state": lambda: rgb_cell_states_union(
            exp_channels_dir, tile_id, resolution=resolution
        ),
        "vasculature": lambda: rgb_vasculature_panel(
            exp_channels_dir, tile_id, resolution=resolution
        ),
        "microenv": lambda: rgb_microenv_union(
            exp_channels_dir, tile_id, resolution=resolution
        ),
    }
    out: dict[str, np.ndarray] = {}
    for key in FOUR_GROUP_ORDER:
        try:
            out[key] = builders[key]()
        except (FileNotFoundError, OSError, ValueError, KeyError):
            out[key] = gray.copy()
    return out
