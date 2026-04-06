"""Stage 4 figure generation helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

_ARCHETYPE_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
]


def _color(k: int) -> str:
    return _ARCHETYPE_COLORS[k % len(_ARCHETYPE_COLORS)]


def _load_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def _add_thumbnail(ax, image: np.ndarray, xy: tuple[float, float], zoom: float, color: str) -> None:
    """Place a small image thumbnail anchored on the scatter point."""
    imagebox = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        xy,
        frameon=True,
        pad=0.18,
        bboxprops=dict(edgecolor=color, linewidth=1.4, facecolor="white"),
        zorder=6,
    )
    ax.add_artist(ab)


def _maybe_pca(param_vectors: np.ndarray) -> np.ndarray:
    """Project parameters to 2D with PCA, padding a zero axis if needed."""
    param_vectors = np.asarray(param_vectors, dtype=np.float32)
    if param_vectors.ndim != 2:
        raise ValueError(f"Expected 2D parameter matrix, got shape {param_vectors.shape}")
    if param_vectors.shape[1] == 0:
        raise ValueError("Parameter matrix must have at least one column")
    if param_vectors.shape[1] == 1:
        return np.c_[param_vectors[:, 0], np.zeros(len(param_vectors), dtype=np.float32)]
    centered = param_vectors - param_vectors.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.zeros((centered.shape[0], 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.c_[coords[:, 0], np.zeros(len(coords), dtype=np.float32)]
    return coords.astype(np.float32, copy=False)


def save_fig1_archetype_umap(
    umap_coords: np.ndarray,
    labels: np.ndarray,
    medoid_indices: list[int],
    tile_ids: list[str],
    he_dir: Path,
    output_path: Path,
    k: int,
    thumbnail_size: int = 42,
) -> None:
    """Fig 1: patient UNI UMAP colored by archetype, with medoid thumbnails."""
    umap_coords = np.asarray(umap_coords, dtype=np.float32)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx in range(k):
        mask = labels == idx
        if not np.any(mask):
            continue
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            s=6,
            color=_color(idx),
            alpha=0.45,
            label=f"Archetype {idx}",
            linewidths=0,
        )

    zoom = thumbnail_size / 256.0
    for idx, tile_index in enumerate(medoid_indices):
        ax.scatter(
            umap_coords[tile_index, 0],
            umap_coords[tile_index, 1],
            s=140,
            marker="*",
            color=_color(idx),
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )
        if 0 <= tile_index < len(tile_ids):
            thumb = _load_rgb(he_dir / f"{tile_ids[tile_index]}.png")
            if thumb is not None:
                _add_thumbnail(ax, thumb, tuple(umap_coords[tile_index]), zoom=zoom, color=_color(idx))

    ax.set_title("CRC patient UNI UMAP by archetype", fontsize=14, pad=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_fig2_umap_overlay(
    crc_umap: np.ndarray,
    gn_umap: np.ndarray,
    gn_assignments: np.ndarray,
    output_path: Path,
    k: int,
) -> None:
    """Fig 2: CRC background with simulation embeddings colored by archetype."""
    crc_umap = np.asarray(crc_umap, dtype=np.float32)
    gn_umap = np.asarray(gn_umap, dtype=np.float32)
    gn_assignments = np.asarray(gn_assignments)
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        crc_umap[:, 0],
        crc_umap[:, 1],
        s=4,
        color="#d9d9d9",
        alpha=0.22,
        linewidths=0,
        label="CRC tiles",
    )
    for idx in range(k):
        mask = gn_assignments == idx
        if not np.any(mask):
            continue
        ax.scatter(
            gn_umap[mask, 0],
            gn_umap[mask, 1],
            s=20,
            color=_color(idx),
            alpha=0.85,
            label=f"G_N -> A{idx}",
            linewidths=0,
        )
    ax.set_title("Simulation outputs in CRC UNI space", fontsize=14, pad=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_fig3_side_by_side(
    style_results_dir: Path,
    output_path: Path,
    k: int,
) -> None:
    """Fig 3: medoid, style tile, and generated image per archetype."""
    fig, axes = plt.subplots(k, 3, figsize=(11, max(3.2 * k, 3.2)))
    axes = np.atleast_2d(axes)
    col_titles = ["CRC medoid", "Style reference", "Generated G_N"]
    file_names = ["medoid_he.png", "style_tile.png", "generated_he.png"]

    for row in range(k):
        row_dir = style_results_dir / f"archetype_{row}"
        for col, (title, fname) in enumerate(zip(col_titles, file_names, strict=True)):
            ax = axes[row, col]
            path = row_dir / fname
            img = _load_rgb(path)
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=11)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=11, pad=8)
            if col == 0:
                ax.set_ylabel(f"A{row}", rotation=0, labelpad=18, fontsize=11, color=_color(row))
                ax.yaxis.set_label_coords(-0.1, 0.5)
    fig.suptitle("Real CRC medoid vs style-conditioned generation", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_fig4_param_space(
    param_vectors: np.ndarray,
    assignments: np.ndarray,
    param_names: list[str],
    output_path: Path,
    k: int,
) -> None:
    """Fig 4: parameter space projected to 2D and colored by archetype."""
    coords = _maybe_pca(param_vectors)
    assignments = np.asarray(assignments)

    fig, ax = plt.subplots(figsize=(9, 7))
    for idx in range(k):
        mask = assignments == idx
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=20,
            color=_color(idx),
            alpha=0.82,
            label=f"Archetype {idx}",
            linewidths=0,
        )

    ax.set_title("Simulation parameter space by archetype assignment", fontsize=14, pad=12)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if param_vectors.shape[1] > 1:
        ax.legend(frameon=False, fontsize=9, loc="best")
    if param_names:
        ax.text(
            0.01,
            0.01,
            "Parameters: " + ", ".join(param_names),
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            ha="left",
            va="bottom",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
