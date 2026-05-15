"""Per-tile label extraction for channel-derived and CellViT morphology attributes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.a1_mask_targets.main import (
    _find_file,
    _load_spatial_file,
    get_channel_load_config,
    resolve_channel_dir,
)
from src.a4_uni_probe.appearance_metrics import appearance_row_for_image


CHANNEL_ATTR_NAMES = (
    "cancer_fraction",
    "healthy_fraction",
    "immune_fraction",
    "prolif_fraction",
    "nonprolif_fraction",
    "dead_fraction",
    "vessel_area_pct",
    "mean_oxygen",
    "mean_glucose",
)
MORPHOLOGY_ATTR_NAMES = (
    "nuclear_area_mean",
    "eccentricity_mean",
    "nuclei_density",
    "intensity_mean_h",
    "intensity_mean_e",
)
APPEARANCE_ATTR_NAMES = (
    "h_mean",
    "e_mean",
    "texture_h_contrast",
    "texture_h_homogeneity",
    "texture_h_energy",
    "texture_e_contrast",
    "texture_e_homogeneity",
    "texture_e_energy",
)
ALL_ATTR_NAMES = CHANNEL_ATTR_NAMES + MORPHOLOGY_ATTR_NAMES + APPEARANCE_ATTR_NAMES

_FRACTION_CHANNELS = {
    "cancer_fraction": "cell_type_cancer",
    "healthy_fraction": "cell_type_healthy",
    "immune_fraction": "cell_type_immune",
    "prolif_fraction": "cell_state_prolif",
    "nonprolif_fraction": "cell_state_nonprolif",
    "dead_fraction": "cell_state_dead",
}
_OPTIONAL_CHANNELS = {
    "vessel_area_pct": "vasculature",
    "mean_oxygen": "oxygen",
    "mean_glucose": "glucose",
}


def _safe_nanmean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def _polygon_area(points: np.ndarray) -> float:
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return float("nan")
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return float(0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))))


def _contour_eccentricity(points: np.ndarray) -> float:
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return float("nan")
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2):
        return float("nan")
    eigvals = np.linalg.eigvalsh(cov)
    major = float(np.max(eigvals))
    minor = float(np.min(eigvals))
    if not np.isfinite(major) or not np.isfinite(minor) or major <= 0:
        return float("nan")
    ratio = max(0.0, min(1.0, minor / major))
    return float(np.sqrt(1.0 - ratio))


def _extract_nuclei_measurements(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    nuclei = payload.get("nuclei", [])
    if nuclei:
        areas = np.asarray([item.get("area", np.nan) for item in nuclei], dtype=np.float64)
        eccentricities = np.asarray([item.get("eccentricity", np.nan) for item in nuclei], dtype=np.float64)
        intensity_h = np.asarray([item.get("intensity_h", np.nan) for item in nuclei], dtype=np.float64)
        intensity_e = np.asarray([item.get("intensity_e", np.nan) for item in nuclei], dtype=np.float64)
        return areas, eccentricities, intensity_h, intensity_e, len(nuclei)

    cells = payload.get("cells", [])
    if not cells:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, 0

    areas: list[float] = []
    eccentricities: list[float] = []
    intensity_h: list[float] = []
    intensity_e: list[float] = []
    for cell in cells:
        contour = np.asarray(cell.get("contour", []), dtype=np.float64)
        areas.append(_polygon_area(contour))
        eccentricities.append(_contour_eccentricity(contour))
        intensity_h.append(float(cell.get("intensity_h", np.nan)))
        intensity_e.append(float(cell.get("intensity_e", np.nan)))

    return (
        np.asarray(areas, dtype=np.float64),
        np.asarray(eccentricities, dtype=np.float64),
        np.asarray(intensity_h, dtype=np.float64),
        np.asarray(intensity_e, dtype=np.float64),
        len(cells),
    )


def _load_channel_array(
    exp_channels_root: Path,
    tile_id: str,
    channel_name: str,
    *,
    resolution: int = 256,
    missing_ok: bool = False,
) -> np.ndarray | None:
    load_cfg = get_channel_load_config(channel_name)
    channel_dir = resolve_channel_dir(exp_channels_root, channel_name)
    try:
        path = _find_file(channel_dir, tile_id, exts=tuple(load_cfg["preferred_exts"]))
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    return _load_spatial_file(
        path,
        resolution=resolution,
        binary=bool(load_cfg["binary"]),
        normalization=str(load_cfg["normalization"]),
    )


def _masked_fraction(channel: np.ndarray | None, mask: np.ndarray) -> float:
    if channel is None:
        return float("nan")
    foreground = mask > 0.5
    if not np.any(foreground):
        return float("nan")
    return float(channel[foreground].mean())


def compute_channel_attributes(
    exp_channels_root: str | Path,
    tile_id: str,
    *,
    resolution: int = 256,
) -> dict[str, float]:
    exp_root = Path(exp_channels_root)
    mask = _load_channel_array(exp_root, tile_id, "cell_masks", resolution=resolution, missing_ok=False)
    if mask is None:
        raise FileNotFoundError(f"cell_masks missing for tile {tile_id} in {exp_root}")

    row: dict[str, float] = {}
    for attr_name, channel_name in _FRACTION_CHANNELS.items():
        row[attr_name] = _masked_fraction(
            _load_channel_array(exp_root, tile_id, channel_name, resolution=resolution, missing_ok=False),
            mask,
        )

    vasculature = _load_channel_array(exp_root, tile_id, "vasculature", resolution=resolution, missing_ok=True)
    row["vessel_area_pct"] = float("nan") if vasculature is None else float(np.mean(vasculature))

    oxygen = _load_channel_array(exp_root, tile_id, "oxygen", resolution=resolution, missing_ok=True)
    row["mean_oxygen"] = float("nan") if oxygen is None else float(np.mean(oxygen))

    glucose = _load_channel_array(exp_root, tile_id, "glucose", resolution=resolution, missing_ok=True)
    row["mean_glucose"] = float("nan") if glucose is None else float(np.mean(glucose))
    return row


def compute_morphology_attributes_from_cellvit(cellvit_json_path: str | Path) -> dict[str, float]:
    """Reduce one CellViT JSON sidecar into mean morphology statistics."""
    path = Path(cellvit_json_path)
    if not path.is_file():
        return {name: float("nan") for name in MORPHOLOGY_ATTR_NAMES}

    payload = json.loads(path.read_text(encoding="utf-8"))
    areas, eccentricities, intensity_h, intensity_e, nuclei_count = _extract_nuclei_measurements(payload)
    if nuclei_count == 0:
        return {name: float("nan") for name in MORPHOLOGY_ATTR_NAMES}

    tile_area = float(payload.get("tile_area", 256.0 * 256.0))

    return {
        "nuclear_area_mean": _safe_nanmean(areas),
        "eccentricity_mean": _safe_nanmean(eccentricities),
        "nuclei_density": float(nuclei_count / tile_area) if tile_area > 0 else float("nan"),
        "intensity_mean_h": _safe_nanmean(intensity_h),
        "intensity_mean_e": _safe_nanmean(intensity_e),
    }


def build_appearance_label_matrix(
    tile_ids: list[str],
    he_dir: str | Path,
) -> np.ndarray:
    he_root = Path(he_dir)
    rows: list[list[float]] = []
    for tile_id in tile_ids:
        he_path = he_root / f"{tile_id}.png"
        if he_path.is_file():
            row = appearance_row_for_image(he_path)
            rows.append([float(row.get(f"appearance.{name}", float("nan"))) for name in APPEARANCE_ATTR_NAMES])
        else:
            rows.append([float("nan")] * len(APPEARANCE_ATTR_NAMES))
    return np.asarray(rows, dtype=np.float32)


def build_label_matrix(
    tile_ids: list[str],
    exp_channels_root: str | Path,
    cellvit_real_dir: str | Path,
    *,
    he_dir: str | Path | None = None,
    resolution: int = 256,
) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    cellvit_root = Path(cellvit_real_dir)
    for tile_id in tile_ids:
        channel_row = compute_channel_attributes(exp_channels_root, tile_id, resolution=resolution)
        morph_row = compute_morphology_attributes_from_cellvit(cellvit_root / f"{tile_id}.json")
        row = {**channel_row, **morph_row}
        rows.append([float(row[attr_name]) for attr_name in CHANNEL_ATTR_NAMES + MORPHOLOGY_ATTR_NAMES])
    base_labels = np.asarray(rows, dtype=np.float32)

    if he_dir is not None:
        appearance_labels = build_appearance_label_matrix(tile_ids, he_dir)
    else:
        appearance_labels = np.full((len(tile_ids), len(APPEARANCE_ATTR_NAMES)), float("nan"), dtype=np.float32)

    return np.concatenate([base_labels, appearance_labels], axis=1), list(ALL_ATTR_NAMES)


def save_label_bundle(
    out_dir: str | Path,
    *,
    tile_ids: list[str],
    labels: np.ndarray,
    attr_names: list[str],
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bundle_path = out_path / "labels.npz"
    np.savez_compressed(
        bundle_path,
        tile_ids=np.asarray(tile_ids, dtype=str),
        attr_names=np.asarray(attr_names, dtype=str),
        labels=labels.astype(np.float32, copy=False),
    )
    return bundle_path
