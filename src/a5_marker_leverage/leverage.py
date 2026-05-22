"""Marker-leverage pipeline.

Predicts per-marker generative impact ΔE without rerunning diffusion:
  ΔE_m(tile) = L2 over g of ( Δfrac_g(tile, m) · ΔE_g(tile) / baseline_frac_g(tile) )

where Δfrac_g comes from a counterfactual reassignment under drop-the-dimension
kmeans (cell-type markers) or threshold removal (Ki67 → cell_state). ΔE_g is the
existing per-tile LOO ΔE from `inference_output/subchannel_loo_n300`.

Outputs: out/predicted_delta_e_per_tile.csv (long-form, one row per (tile, marker, cluster))
         out/predicted_delta_e_summary.csv (per-marker mean ± SEM).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# Repo roots
ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = Path.home() / "he-feature-visualizer" / "data" / "features_crc33.csv"
LOO_DIR = ROOT / "inference_output" / "subchannel_loo_n300"
TILE_LIST_SHARDS = [
    LOO_DIR / "_tile_lists" / "shard_a.txt",
    LOO_DIR / "_tile_lists" / "shard_b.txt",
]
OUT_DIR = ROOT / "src" / "a5_marker_leverage" / "out"

# 13 cell-type kmeans markers (assign_cells.CODEX_FINE_TYPE_MARKERS, deduped).
# Names match columns in features_crc33.csv.
TYPE_MARKERS: list[str] = [
    "Pan-CK", "E-cadherin",
    "CD45", "CD3e", "CD4", "CD45RO", "CD8a", "FOXP3", "CD20", "CD68", "CD163",
    "CD31", "SMA",
]

# Ki67 drives the state pathway (threshold, not kmeans).
STATE_MARKER = "Ki67"

# Fine-type signature markers per cluster label (matches assign_cells.CODEX_FINE_TYPE_MARKERS).
FINE_TYPE_MARKERS: dict[str, list[str]] = {
    "epithelial": ["Pan-CK", "E-cadherin"],
    "cd4_t": ["CD45", "CD3e", "CD4", "CD45RO"],
    "cd8_t": ["CD45", "CD3e", "CD8a", "CD45RO"],
    "treg": ["CD45", "CD3e", "CD4", "FOXP3"],
    "b_cell": ["CD45", "CD20"],
    "macrophage": ["CD45", "CD68", "CD163"],
    "endothelial": ["CD31"],
    "sma_stromal": ["SMA"],
}

# Map fine cluster → coarse cell_type (matches CODEX_FINE_TO_FINAL_WEIGHTS).
FINE_TO_COARSE: dict[str, str] = {
    "epithelial": "cancer",
    "cd4_t": "immune", "cd8_t": "immune", "treg": "immune",
    "b_cell": "immune", "macrophage": "immune",
    "endothelial": "healthy", "sma_stromal": "healthy",
}

CELL_TYPE_CHANNELS = {"cancer": "cell_type_cancer", "immune": "cell_type_immune", "healthy": "cell_type_healthy"}
CELL_STATE_CHANNELS = {"prolif": "cell_state_prolif", "nonprolif": "cell_state_nonprolif", "dead": "cell_state_dead"}

TILE_SIZE = 256
KMEANS_SEED = 0
CLUSTER_PENALTY = 0.35
ZSCORE_EPS = 1e-6


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------

def load_loo_tile_ids() -> list[str]:
    tile_ids: list[str] = []
    for shard in TILE_LIST_SHARDS:
        tile_ids.extend([line.strip() for line in shard.read_text().splitlines() if line.strip()])
    return sorted(set(tile_ids))


def centroid_to_tile_id(x: float, y: float) -> str:
    row_px = int(y // TILE_SIZE) * TILE_SIZE
    col_px = int(x // TILE_SIZE) * TILE_SIZE
    return f"{row_px}_{col_px}"


def load_cells_in_tiles(features_csv: Path, tile_ids: list[str]) -> pd.DataFrame:
    """Load per-cell rows whose centroids fall inside any of `tile_ids`."""
    tile_set = set(tile_ids)
    df = pd.read_csv(features_csv)
    df["tile_id"] = [centroid_to_tile_id(x, y) for x, y in zip(df["X_centroid"], df["Y_centroid"])]
    keep = df[df["tile_id"].isin(tile_set)].reset_index(drop=True)
    return keep


# ----------------------------------------------------------------------------
# Preprocessing (mirrors assign_cells._preprocess_codex_matrix)
# ----------------------------------------------------------------------------

def preprocess_z(matrix: np.ndarray, winsor_pct: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Background-subtract, winsorize, z-score. Returns (Z, mean, std)."""
    arr = np.clip(matrix.astype(np.float64), 0.0, None)
    bg = np.percentile(arr, 5.0, axis=0)
    arr = np.clip(arr - bg, 0.0, None)
    lo = np.percentile(arr, winsor_pct, axis=0)
    hi = np.percentile(arr, 100.0 - winsor_pct, axis=0)
    arr = np.clip(arr, lo, hi)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std <= ZSCORE_EPS] = 1.0
    return (arr - mean) / std, mean, std


# ----------------------------------------------------------------------------
# Kmeans + fine-type assignment
# ----------------------------------------------------------------------------

def fit_kmeans(Z: np.ndarray, n_clusters: int = 8) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=KMEANS_SEED, n_init=20)
    km.fit(Z)
    return km


def score_centers_to_fine_types(centers: np.ndarray, marker_names: list[str]) -> dict[int, str]:
    """For each cluster, return the fine_type whose signature scores highest."""
    df = pd.DataFrame(centers, columns=marker_names)
    pos = df.clip(lower=0.0)
    scores = pd.DataFrame(index=df.index, columns=list(FINE_TYPE_MARKERS), dtype=float)
    for ftype, sig_markers in FINE_TYPE_MARKERS.items():
        present = [m for m in sig_markers if m in df.columns]
        signal = df[present].mean(axis=1) if present else pd.Series(0.0, index=df.index)
        other = [m for m in df.columns if m not in present]
        penalty = pos[other].mean(axis=1) if other else pd.Series(0.0, index=df.index)
        scores[ftype] = signal - CLUSTER_PENALTY * penalty
    return {int(cid): scores.loc[cid].idxmax() for cid in scores.index}


def assign_drop_dim(Z: np.ndarray, centers: np.ndarray, drop_idx: int) -> np.ndarray:
    """Assign cells to nearest centroid after dropping dimension `drop_idx` from both sides."""
    cols = [j for j in range(Z.shape[1]) if j != drop_idx]
    Zr = Z[:, cols]
    Cr = centers[:, cols]
    d2 = ((Zr[:, None, :] - Cr[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def assign_refit(Z: np.ndarray, drop_idx: int, n_clusters: int = 8) -> np.ndarray:
    """SI (a.2): re-fit kmeans on D-1 markers."""
    cols = [j for j in range(Z.shape[1]) if j != drop_idx]
    km = KMeans(n_clusters=n_clusters, random_state=KMEANS_SEED, n_init=20)
    return km.fit_predict(Z[:, cols]), km.cluster_centers_


# ----------------------------------------------------------------------------
# Δfrac per tile per cluster
# ----------------------------------------------------------------------------

def cluster_to_coarse_map(cluster_fine: dict[int, str]) -> dict[int, str]:
    return {cid: FINE_TO_COARSE[ftype] for cid, ftype in cluster_fine.items()}


def per_tile_fracs(tile_ids: list[str], cell_tile: np.ndarray, coarse_labels: np.ndarray) -> pd.DataFrame:
    """Return DataFrame (tile_id, coarse_label) → frac of cells in that tile."""
    counts = pd.crosstab(pd.Series(cell_tile, name="tile_id"), pd.Series(coarse_labels, name="coarse"))
    counts = counts.reindex(index=tile_ids, columns=["cancer", "immune", "healthy"], fill_value=0)
    fracs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return fracs


# ----------------------------------------------------------------------------
# Ki67 threshold pathway (mirrors assign_cells._derive_cell_state)
# ----------------------------------------------------------------------------

def ki67_threshold(ki67_values: np.ndarray) -> float:
    """Pipeline uses 75th percentile of non-zero Ki67 as the prolif threshold (heuristic
    consistent with assign_cells)."""
    nz = ki67_values[ki67_values > 0]
    if nz.size == 0:
        return float("inf")
    return float(np.percentile(nz, 75.0))


def assign_state(ki67_values: np.ndarray, threshold: float) -> np.ndarray:
    """Return array of {prolif, nonprolif} per cell. (dead handled by CellViT prior elsewhere — we
    treat dead as a held-out class not flippable by Ki67 zeroing.)"""
    state = np.where(ki67_values >= threshold, "prolif", "nonprolif")
    return state


def per_tile_state_fracs(tile_ids: list[str], cell_tile: np.ndarray, state_labels: np.ndarray) -> pd.DataFrame:
    counts = pd.crosstab(pd.Series(cell_tile, name="tile_id"), pd.Series(state_labels, name="state"))
    counts = counts.reindex(index=tile_ids, columns=["prolif", "nonprolif"], fill_value=0)
    fracs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return fracs


# ----------------------------------------------------------------------------
# LOO ΔE per tile per cluster
# ----------------------------------------------------------------------------

def load_per_tile_delta_e(tile_ids: list[str]) -> pd.DataFrame:
    """Read subchannel_loo_diff_stats.json per tile. Returns (tile_id × sub_channel) ΔE_mean."""
    cols = (
        list(CELL_TYPE_CHANNELS.values()) + list(CELL_STATE_CHANNELS.values())
    )
    out = pd.DataFrame(index=tile_ids, columns=cols, dtype=float)
    for tid in tile_ids:
        p = LOO_DIR / tid / "subchannel_loo_diff_stats.json"
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        for ch in cols:
            if ch in data:
                out.at[tid, ch] = data[ch].get("delta_e_mean", np.nan)
    return out


# ----------------------------------------------------------------------------
# Δfrac → predicted ΔE
# ----------------------------------------------------------------------------

def predicted_delta_e(
    baseline_fracs: pd.DataFrame,           # (tile × coarse) baseline frac
    new_fracs: pd.DataFrame,                # (tile × coarse) after dropping marker
    delta_e: pd.DataFrame,                  # (tile × sub_channel) ΔE_mean from LOO
    channels: dict[str, str],               # coarse → sub_channel name
    min_baseline: float = 0.02,
) -> pd.Series:
    """Per-tile predicted ΔE for one marker.

    For each cluster g: fraction of g's mass perturbed = |baseline_frac - new_frac| / baseline_frac
    (clipped if baseline ≈ 0). Multiplied by ΔE_g (full-channel LOO) gives the predicted
    contribution from cluster g. Summed across clusters (L1) since simultaneous mask perturbations
    add roughly linearly in CIELAB at small magnitudes.
    """
    total = pd.Series(0.0, index=baseline_fracs.index)
    for coarse, ch in channels.items():
        b = baseline_fracs[coarse]
        n = new_fracs[coarse]
        de = delta_e[ch].reindex(baseline_fracs.index)
        ratio = (b - n).abs() / b.where(b > min_baseline, np.nan)
        total = total.add((ratio * de).fillna(0.0), fill_value=0.0)
    return total


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tile_ids = load_loo_tile_ids()
    print(f"[a5] {len(tile_ids)} LOO tiles loaded")

    cells = load_cells_in_tiles(FEATURES_CSV, tile_ids)
    print(f"[a5] {len(cells)} cells in those tiles")

    raw = cells[TYPE_MARKERS].to_numpy(dtype=np.float64)
    Z, _, _ = preprocess_z(raw)
    km = fit_kmeans(Z, n_clusters=8)
    cluster_ids = km.predict(Z)
    centers = km.cluster_centers_
    fine_map = score_centers_to_fine_types(centers, TYPE_MARKERS)
    coarse_map = cluster_to_coarse_map(fine_map)
    print(f"[a5] fine map: {fine_map}")

    coarse_labels = np.array([coarse_map[int(c)] for c in cluster_ids])
    baseline_type_fracs = per_tile_fracs(tile_ids, cells["tile_id"].to_numpy(), coarse_labels)

    # State pathway: Ki67 threshold from this cell population.
    ki67_raw = cells[STATE_MARKER].to_numpy(dtype=np.float64)
    thr = ki67_threshold(ki67_raw)
    print(f"[a5] Ki67 prolif threshold = {thr:.3f}")
    baseline_state = assign_state(ki67_raw, thr)
    baseline_state_fracs = per_tile_state_fracs(tile_ids, cells["tile_id"].to_numpy(), baseline_state)

    delta_e = load_per_tile_delta_e(tile_ids)

    # ---------- (a.1) drop-the-dim per cell-type marker ----------
    long_rows: list[dict] = []
    for j, m in enumerate(TYPE_MARKERS):
        new_clusters = assign_drop_dim(Z, centers, drop_idx=j)
        new_coarse = np.array([coarse_map[int(c)] for c in new_clusters])
        new_fracs = per_tile_fracs(tile_ids, cells["tile_id"].to_numpy(), new_coarse)
        pred = predicted_delta_e(baseline_type_fracs, new_fracs, delta_e, CELL_TYPE_CHANNELS)
        for tid, v in pred.items():
            long_rows.append({"marker": m, "tile_id": tid, "predicted_delta_e": float(v), "pathway": "type_kmeans"})
        print(f"[a5] type marker {m}: predicted ΔE mean = {pred.mean():.3f} (sem {pred.sem():.3f})")

    # ---------- Ki67 threshold pathway ----------
    # Zero Ki67 → all cells flip to nonprolif. Frac flipped per tile = baseline prolif frac.
    ki67_zero = np.zeros_like(ki67_raw)
    new_state = assign_state(ki67_zero, thr)
    new_state_fracs = per_tile_state_fracs(tile_ids, cells["tile_id"].to_numpy(), new_state)
    # Use only prolif and nonprolif channels (dead unaffected by Ki67 zeroing).
    ki67_channels = {"prolif": "cell_state_prolif", "nonprolif": "cell_state_nonprolif"}
    pred_ki67 = predicted_delta_e(baseline_state_fracs, new_state_fracs, delta_e, ki67_channels)
    for tid, v in pred_ki67.items():
        long_rows.append({"marker": STATE_MARKER, "tile_id": tid, "predicted_delta_e": float(v), "pathway": "ki67_threshold"})
    print(f"[a5] Ki67: predicted ΔE mean = {pred_ki67.mean():.3f}")

    long_df = pd.DataFrame(long_rows)
    long_path = OUT_DIR / "predicted_delta_e_per_tile.csv"
    long_df.to_csv(long_path, index=False)

    summary = (
        long_df.groupby("marker")["predicted_delta_e"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    summary_path = OUT_DIR / "predicted_delta_e_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[a5] wrote {long_path}")
    print(f"[a5] wrote {summary_path}")
    return summary_path


if __name__ == "__main__":
    run()
