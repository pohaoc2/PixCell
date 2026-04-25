"""Figures 05/06: Representative ablation grid for paired and unpaired datasets."""
from __future__ import annotations

from pathlib import Path

from tools.ablation_report.data import load_leave_one_out_summary_stats, resolve_loo_root
from tools.ablation_report.shared import ROOT
from tools.stage3.ablation_cache import load_manifest, resolve_all_image_path
from tools.stage3.ablation_grid_figure import (
    FOUR_GROUP_ORDER,
    METRIC_BAR_PRESETS,
    render_ablation_grid_figure,
)
from tools.stage3.style_mapping import load_style_mapping


def build_representative_ablation_grid(
    *,
    metrics_root: Path,
    dataset_root: Path,
    orion_root: Path,
    out_png: Path,
    dpi: int = 300,
    style_mapping_json: Path | None = None,
    tile_id: str | None = None,
) -> Path | None:
    """Find the representative tile and render its ablation grid.

    Returns the output path, or None if no representative tile is found.
    """
    metrics_root = Path(metrics_root)
    dataset_root = Path(dataset_root)
    orion_root = Path(orion_root)

    loo_root = resolve_loo_root(metrics_root, dataset_root)
    if tile_id is not None:
        representative_tile = tile_id
    else:
        _, _, representative_tile = load_leave_one_out_summary_stats(loo_root)
    if representative_tile is None:
        print(f"No representative tile found under {loo_root}")
        return None

    # Cache dir: metrics_root/<tile_id> or dataset_root/ablation_results/<tile_id>
    candidates = [
        metrics_root / representative_tile,
        dataset_root / "ablation_results" / representative_tile,
    ]
    cache_dir: Path | None = None
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            cache_dir = candidate
            break
    if cache_dir is None:
        print(f"manifest.json not found for tile {representative_tile!r} in {candidates}")
        return None

    manifest = load_manifest(cache_dir)
    all4ch_image = resolve_all_image_path(
        cache_dir, manifest,
        n_groups=len(manifest.get("group_names") or FOUR_GROUP_ORDER),
    )
    if all4ch_image is None:
        print(f"All-4-ch image not found for tile {representative_tile!r}")
        return None

    style_mapping = load_style_mapping(style_mapping_json)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    return render_ablation_grid_figure(
        cache_dir,
        all4ch_image=all4ch_image,
        orion_root=orion_root,
        tile_id=representative_tile,
        out_png=out_png,
        dpi=dpi,
        auto_cosine=False,
        metric_bars=METRIC_BAR_PRESETS["paired"],
        style_mapping=style_mapping,
    )
