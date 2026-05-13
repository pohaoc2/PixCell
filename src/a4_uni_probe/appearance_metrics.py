"""Appearance, stain, and texture metrics for generated H&E tiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hed

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:  # pragma: no cover
    from skimage.feature import greycomatrix as graycomatrix  # type: ignore
    from skimage.feature import greycoprops as graycoprops  # type: ignore


APPEARANCE_METRIC_NAMES = (
    "appearance.h_mean",
    "appearance.h_std",
    "appearance.e_mean",
    "appearance.e_std",
    "appearance.stain_vector_angle_deg",
    "appearance.texture_h_contrast",
    "appearance.texture_h_homogeneity",
    "appearance.texture_h_energy",
    "appearance.texture_e_contrast",
    "appearance.texture_e_homogeneity",
    "appearance.texture_e_energy",
)

_GLCM_DISTANCES = (1,)
_GLCM_ANGLES = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
_GLCM_LEVELS = 32


def _load_rgb_u8(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _hed_channels(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hed = rgb2hed(np.asarray(rgb_u8, dtype=np.float32) / 255.0)
    return np.asarray(hed[:, :, 0], dtype=np.float32), np.asarray(hed[:, :, 1], dtype=np.float32)


def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _quantize_channel(channel: np.ndarray, *, levels: int = _GLCM_LEVELS) -> np.ndarray:
    finite = np.asarray(channel, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    low = float(np.quantile(finite, 0.01))
    high = float(np.quantile(finite, 0.99))
    if high - low < 1e-6:
        return np.zeros_like(channel, dtype=np.uint8)
    clipped = np.clip(channel, low, high)
    scaled = np.rint((clipped - low) * (levels - 1) / (high - low)).astype(np.uint8)
    return scaled


def _haralick_row(channel: np.ndarray, prefix: str) -> dict[str, float]:
    quantized = _quantize_channel(channel)
    glcm = graycomatrix(
        quantized,
        distances=_GLCM_DISTANCES,
        angles=_GLCM_ANGLES,
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    return {
        f"appearance.texture_{prefix}_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        f"appearance.texture_{prefix}_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        f"appearance.texture_{prefix}_energy": float(np.mean(graycoprops(glcm, "energy"))),
    }


def _estimate_stain_vectors(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_u8, dtype=np.float32)
    tissue_mask = np.mean(rgb, axis=2) < 245.0
    optical_density = -np.log((rgb[tissue_mask] + 1.0) / 256.0)
    if optical_density.shape[0] < 16:
        return np.full((2, 3), np.nan, dtype=np.float32)
    optical_density = optical_density[np.sum(optical_density, axis=1) > 0.15]
    if optical_density.shape[0] < 16:
        return np.full((2, 3), np.nan, dtype=np.float32)

    _, _, vh = np.linalg.svd(optical_density, full_matrices=False)
    vectors = np.asarray(vh[:2], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        return np.full((2, 3), np.nan, dtype=np.float32)
    vectors = vectors / norms

    for index in range(vectors.shape[0]):
        axis = int(np.argmax(np.abs(vectors[index])))
        if vectors[index, axis] < 0.0:
            vectors[index] *= -1.0
    return vectors


def _vector_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return float("nan")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    cosine = float(np.clip(abs(cosine), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def stain_vector_angle_deg(rgb_u8: np.ndarray, reference_rgb_u8: np.ndarray) -> float:
    source_vectors = _estimate_stain_vectors(rgb_u8)
    reference_vectors = _estimate_stain_vectors(reference_rgb_u8)
    if not np.all(np.isfinite(source_vectors)) or not np.all(np.isfinite(reference_vectors)):
        return float("nan")

    aligned = 0.5 * (
        _vector_angle_deg(source_vectors[0], reference_vectors[0])
        + _vector_angle_deg(source_vectors[1], reference_vectors[1])
    )
    crossed = 0.5 * (
        _vector_angle_deg(source_vectors[0], reference_vectors[1])
        + _vector_angle_deg(source_vectors[1], reference_vectors[0])
    )
    return float(min(aligned, crossed))


def appearance_row_for_rgb(rgb_u8: np.ndarray, *, reference_rgb_u8: np.ndarray | None = None) -> dict[str, float]:
    hematoxylin, eosin = _hed_channels(rgb_u8)
    h_mean, h_std = _safe_mean_std(hematoxylin)
    e_mean, e_std = _safe_mean_std(eosin)
    row = {
        "appearance.h_mean": h_mean,
        "appearance.h_std": h_std,
        "appearance.e_mean": e_mean,
        "appearance.e_std": e_std,
        "appearance.stain_vector_angle_deg": (
            stain_vector_angle_deg(rgb_u8, reference_rgb_u8)
            if reference_rgb_u8 is not None
            else float("nan")
        ),
    }
    row.update(_haralick_row(hematoxylin, "h"))
    row.update(_haralick_row(eosin, "e"))
    return row


def appearance_row_for_image(image_path: str | Path, *, reference_image_path: str | Path | None = None) -> dict[str, float]:
    rgb = _load_rgb_u8(image_path)
    reference_rgb = _load_rgb_u8(reference_image_path) if reference_image_path is not None and Path(reference_image_path).is_file() else None
    return appearance_row_for_rgb(rgb, reference_rgb_u8=reference_rgb)


def _find_reference_he_path(data_root: Path, tile_id: str) -> Path | None:
    he_root = data_root / "he"
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = he_root / f"{tile_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _safe_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom == 0.0:
        return 0.0
    y_centered = y - float(np.mean(y))
    return float(np.dot(x_centered, y_centered) / denom)


def _append_appearance_metrics(rows: list[dict[str, object]], *, data_root: Path) -> list[dict[str, object]]:
    cache: dict[tuple[str, str], dict[str, float]] = {}
    updated_rows: list[dict[str, object]] = []
    for row in rows:
        tile_id = str(row["tile_id"])
        image_path = str(row["image_path"])
        reference_path = _find_reference_he_path(data_root, tile_id)
        cache_key = (image_path, str(reference_path) if reference_path is not None else "")
        appearance = cache.get(cache_key)
        if appearance is None:
            appearance = appearance_row_for_image(image_path, reference_image_path=reference_path)
            cache[cache_key] = appearance
        merged = dict(row)
        merged.update({name: float(appearance[name]) for name in APPEARANCE_METRIC_NAMES})
        updated_rows.append(merged)
    return updated_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_sweep_rows(rows: list[dict[str, object]], *, attr: str) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for metric_name in APPEARANCE_METRIC_NAMES:
        metric_summary: dict[str, object] = {"attr": attr, "metric": metric_name}
        for direction_name in ("targeted", "random"):
            direction_rows = [row for row in rows if row.get("direction") == direction_name]
            alphas = np.asarray([_safe_float(row, "alpha") for row in direction_rows], dtype=np.float32)
            values = np.asarray([_safe_float(row, metric_name) for row in direction_rows], dtype=np.float32)
            valid = np.isfinite(alphas) & np.isfinite(values)
            if int(valid.sum()) < 2:
                metric_summary[f"{direction_name}_slope_mean"] = float("nan")
                metric_summary[f"{direction_name}_slope_ci95_low"] = float("nan")
                metric_summary[f"{direction_name}_slope_ci95_high"] = float("nan")
                metric_summary[f"{direction_name}_n"] = int(valid.sum())
                continue
            alphas = alphas[valid]
            values = values[valid]
            rng = np.random.default_rng(0)
            slopes = []
            for _ in range(400):
                choice = rng.integers(0, len(alphas), size=len(alphas))
                slopes.append(_linear_slope(alphas[choice], values[choice]))
            slope_arr = np.asarray(slopes, dtype=np.float32)
            metric_summary[f"{direction_name}_slope_mean"] = float(np.mean(slope_arr))
            metric_summary[f"{direction_name}_slope_ci95_low"] = float(np.quantile(slope_arr, 0.025))
            metric_summary[f"{direction_name}_slope_ci95_high"] = float(np.quantile(slope_arr, 0.975))
            metric_summary[f"{direction_name}_n"] = int(len(alphas))
        summary_rows.append(metric_summary)
    return summary_rows


def _summarize_null_rows(rows: list[dict[str, object]], *, attr: str) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for metric_name in APPEARANCE_METRIC_NAMES:
        metric_summary: dict[str, object] = {"attr": attr, "metric": metric_name}
        for condition_name in ("targeted", "random", "full_uni_null"):
            values = np.asarray(
                [_safe_float(row, metric_name) for row in rows if row.get("condition") == condition_name],
                dtype=np.float32,
            )
            finite = values[np.isfinite(values)]
            metric_summary[f"{condition_name}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            metric_summary[f"{condition_name}_n"] = int(finite.size)
        summary_rows.append(metric_summary)
    return summary_rows


def run_appearance(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = Path(args.out_dir)
    data_root = Path(args.data_root)
    outputs: dict[str, Path] = {}

    sweep_summary_rows: list[dict[str, object]] = []
    for metrics_path in sorted(out_dir.glob("sweep/*/metrics.csv")):
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        updated_rows = _append_appearance_metrics(rows, data_root=data_root)
        _write_csv(metrics_path, updated_rows)
        attr = metrics_path.parent.name
        summary_rows = _summarize_sweep_rows(updated_rows, attr=attr)
        summary_path = metrics_path.parent / "appearance_summary.csv"
        _write_csv(summary_path, summary_rows)
        sweep_summary_rows.extend(summary_rows)

    null_summary_rows: list[dict[str, object]] = []
    for metrics_path in sorted(out_dir.glob("null/*/metrics.csv")):
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        updated_rows = _append_appearance_metrics(rows, data_root=data_root)
        _write_csv(metrics_path, updated_rows)
        attr = metrics_path.parent.name
        summary_rows = _summarize_null_rows(updated_rows, attr=attr)
        summary_path = metrics_path.parent / "appearance_summary.csv"
        _write_csv(summary_path, summary_rows)
        null_summary_rows.extend(summary_rows)

    if sweep_summary_rows:
        sweep_path = out_dir / "appearance_sweep_summary.csv"
        _write_csv(sweep_path, sweep_summary_rows)
        outputs["sweep"] = sweep_path
    if null_summary_rows:
        null_path = out_dir / "appearance_null_summary.csv"
        _write_csv(null_path, null_summary_rows)
        outputs["null"] = null_path
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_appearance(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())