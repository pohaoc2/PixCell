"""Metric export/import/summary utilities for the SI A1/A2 cache."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ablation_a1_a2.cache_io import load_cache, merge_metrics, save_cache
from tools.compute_ablation_metrics import (
    _compute_binary_segmentation_metrics,
    _compute_pq,
    _load_gt_instance_mask,
    compute_style_hed_for_pair,
    run_cellvit,
)
from tools.compute_fid import (
    ImageFeatureRecord,
    compute_fid_from_stats,
    compute_statistics,
    extract_uni_features,
    load_uni_extractor,
)


METRIC_VARIANTS = ("production", "a1_concat", "a1_per_channel", "a2_bypass", "a2_bypass_full_tme", "a2_off_shelf")
METRIC_KEYS = ("fud", "dice", "pq", "lpips", "style_hed")


def _metric_tile_ids(cache: dict, tile_dir: Path, variants: tuple[str, ...]) -> list[str]:
    cached = [str(tile_id) for tile_id in cache.get("metric_tile_ids", [])]
    if cached:
        return cached
    common: set[str] | None = None
    for variant in variants:
        variant_ids = {
            path.stem
            for path in (tile_dir / variant).glob("*.png")
            if not path.name.endswith("_cellvit_instances.png")
        }
        common = variant_ids if common is None else common & variant_ids
    return sorted(common or set())


def _image_path(tile_dir: Path, variant: str, tile_id: str) -> Path:
    return tile_dir / variant / f"{tile_id}.png"


def _flat_name(variant: str, tile_id: str) -> str:
    return f"{variant}__{tile_id}.png"


def select_metric_tiles_from_paired_ablation(
    *,
    cache_dir: Path,
    paired_ablation_root: Path,
    n_tiles: int,
    overwrite: bool = True,
) -> Path:
    """Select metric tiles from paired ablation and reuse production images/CellViT sidecars."""
    cache_dir = Path(cache_dir)
    paired_root = Path(paired_ablation_root)
    source_root = paired_root / "ablation_results"
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing paired ablation results: {source_root}")

    eligible: list[str] = []
    for tile_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if (
            (tile_dir / "all" / "generated_he.png").is_file()
            and (tile_dir / "all" / "generated_he_cellvit_instances.json").is_file()
            and (tile_dir / "metrics.json").is_file()
        ):
            eligible.append(tile_dir.name)
    if len(eligible) < n_tiles:
        raise ValueError(f"requested {n_tiles} tiles but only found {len(eligible)} eligible paired-ablation tiles")

    selected = eligible[:n_tiles]
    cache = load_cache(cache_dir / "cache.json")
    cache["metric_tile_ids"] = selected
    production_dir = cache_dir / "tiles" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = cache_dir / "features" / "production"
    feature_dir.mkdir(parents=True, exist_ok=True)

    for tile_id in selected:
        src_dir = source_root / tile_id
        dst_png = production_dir / f"{tile_id}.png"
        dst_cellvit = production_dir / f"{tile_id}_cellvit_instances.json"
        dst_uni = feature_dir / f"{tile_id}_uni.npy"
        if overwrite or not dst_png.exists():
            shutil.copy2(src_dir / "all" / "generated_he.png", dst_png)
        if overwrite or not dst_cellvit.exists():
            shutil.copy2(src_dir / "all" / "generated_he_cellvit_instances.json", dst_cellvit)
        src_uni = src_dir / "features" / "all" / "generated_he_uni.npy"
        if src_uni.is_file() and (overwrite or not dst_uni.exists()):
            shutil.copy2(src_uni, dst_uni)

    save_cache(cache, cache_dir / "cache.json")
    out = cache_dir / "metric_tile_ids.txt"
    out.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return out


def export_cellvit(
    *,
    cache_dir: Path,
    out_dir: Path,
    variants: tuple[str, ...] = ("a1_concat", "a1_per_channel", "a2_bypass", "a2_bypass_full_tme", "a2_off_shelf"),
    overwrite: bool = True,
    mode: str = "copy",
) -> Path:
    """Export generated H&E PNGs into a flat CellViT batch folder."""
    cache_dir = Path(cache_dir)
    tile_dir = cache_dir / "tiles"
    cache = load_cache(cache_dir / "cache.json")
    tile_ids = _metric_tile_ids(cache, tile_dir, variants)
    if not tile_ids:
        raise FileNotFoundError("no metric_tile_ids and no common generated PNGs found")

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    for variant in variants:
        for tile_id in tile_ids:
            source = _image_path(tile_dir, variant, tile_id)
            if not source.is_file():
                raise FileNotFoundError(f"missing generated PNG: {source}")
            flat_name = _flat_name(variant, tile_id)
            dest = images_dir / flat_name
            if mode == "hardlink":
                if dest.exists():
                    dest.unlink()
                try:
                    dest.hardlink_to(source.resolve())
                except OSError:
                    shutil.copy2(source, dest)
            elif mode == "copy":
                shutil.copy2(source, dest)
            else:
                raise ValueError(f"unsupported mode: {mode!r}")
            entries.append(
                {
                    "variant": variant,
                    "tile_id": tile_id,
                    "source_path": str(source.resolve()),
                    "flat_name": flat_name,
                    "flat_path": str(dest.resolve()),
                }
            )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "si_a1_a2_cellvit_export",
                "cache_dir": str(cache_dir.resolve()),
                "images_dir": str(images_dir.resolve()),
                "entries": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def import_cellvit(*, manifest_path: Path, results_dir: Path, result_pattern: str = "{stem}.json") -> Path:
    """Copy CellViT JSON outputs back beside their source generated PNGs."""
    manifest_path = Path(manifest_path)
    results_dir = Path(results_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    imported: list[dict[str, str]] = []
    for entry in entries:
        flat_name = str(entry["flat_name"])
        stem = Path(flat_name).stem
        matches = sorted(results_dir.glob(result_pattern.format(name=flat_name, stem=stem)))
        if not matches:
            raise FileNotFoundError(f"no CellViT result for {flat_name!r} under {results_dir}")
        matched = matches[0]
        source = Path(entry["source_path"])
        dest = source.with_name(f"{source.stem}_cellvit_instances{matched.suffix.lower()}")
        shutil.copy2(matched, dest)
        imported.append(
            {
                "variant": str(entry["variant"]),
                "tile_id": str(entry["tile_id"]),
                "matched_result": str(matched.resolve()),
                "imported_path": str(dest.resolve()),
            }
        )
    report_path = manifest_path.parent / "import_report.json"
    report_path.write_text(
        json.dumps(
            {
                "version": 1,
                "manifest_path": str(manifest_path.resolve()),
                "results_dir": str(results_dir.resolve()),
                "entries": imported,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _std(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.std(finite)) if finite else None


def _load_lpips(device: str):
    import torch
    import lpips

    resolved = device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        resolved = "cpu"
    model = lpips.LPIPS(net="alex").to(resolved)
    model.eval()
    return model, resolved


def _lpips_pair(loss_fn, device: str, ref_path: Path, gen_path: Path) -> float:
    import torch

    def tensor(path: Path):
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        return float(loss_fn(tensor(ref_path), tensor(gen_path)).reshape(-1)[0].item())


def _compute_fud_by_variant(
    *,
    tile_dir: Path,
    tile_ids: list[str],
    variants: tuple[str, ...],
    orion_root: Path,
    uni_model: Path,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    real_records = [
        ImageFeatureRecord(
            image_path=orion_root / "he" / f"{tile_id}.png",
            feature_path=orion_root / "features" / f"{tile_id}_uni.npy",
        )
        for tile_id in tile_ids
    ]
    all_records = list(real_records)
    for variant in variants:
        for tile_id in tile_ids:
            all_records.append(
                ImageFeatureRecord(
                    image_path=_image_path(tile_dir, variant, tile_id),
                    feature_path=tile_dir.parent / "features" / variant / f"{tile_id}_uni.npy",
                )
            )
    extractor = None
    if not all(record.feature_path is not None and record.feature_path.is_file() for record in all_records):
        extractor = load_uni_extractor(uni_model=uni_model, device=device)
    real_features = extract_uni_features(real_records, extractor=extractor, batch_size=batch_size)
    real_mu, real_sigma = compute_statistics(real_features)

    fud: dict[str, float] = {}
    for variant in variants:
        gen_records = [
            ImageFeatureRecord(
                image_path=_image_path(tile_dir, variant, tile_id),
                feature_path=tile_dir.parent / "features" / variant / f"{tile_id}_uni.npy",
            )
            for tile_id in tile_ids
        ]
        gen_features = extract_uni_features(gen_records, extractor=extractor, batch_size=batch_size)
        gen_mu, gen_sigma = compute_statistics(gen_features)
        fud[variant] = compute_fid_from_stats(real_mu, real_sigma, gen_mu, gen_sigma)
    return fud


def compute_and_update_metrics(
    *,
    cache_dir: Path,
    orion_root: Path = ROOT / "data/orion-crc33",
    variants: tuple[str, ...] = METRIC_VARIANTS,
    device: str = "cuda",
    uni_model: Path = ROOT / "pretrained_models/uni-2h",
    batch_size: int = 16,
) -> dict[str, dict[str, float | int | None]]:
    """Compute FUD, LPIPS, HED, DICE, and PQ, then merge summaries into cache.json."""
    cache_dir = Path(cache_dir)
    tile_dir = cache_dir / "tiles"
    cache_path = cache_dir / "cache.json"
    cache = load_cache(cache_path)
    tile_ids = _metric_tile_ids(cache, tile_dir, variants)
    if not tile_ids:
        raise FileNotFoundError("no metric tile IDs available")

    lpips_model, lpips_device = _load_lpips(device)
    fud = _compute_fud_by_variant(
        tile_dir=tile_dir,
        tile_ids=tile_ids,
        variants=variants,
        orion_root=Path(orion_root),
        uni_model=Path(uni_model),
        device=device,
        batch_size=batch_size,
    )

    summary: dict[str, dict[str, float | int | None]] = {}
    for variant in variants:
        dice_values: list[float] = []
        pq_values: list[float] = []
        lpips_values: list[float] = []
        hed_values: list[float] = []
        for tile_id in tile_ids:
            gen_path = _image_path(tile_dir, variant, tile_id)
            ref_path = Path(orion_root) / "he" / f"{tile_id}.png"
            lpips_values.append(_lpips_pair(lpips_model, lpips_device, ref_path, gen_path))

            ref_img = Image.open(ref_path).convert("RGB")
            gen_img = Image.open(gen_path).convert("RGB")
            # Reuse the per-tile HED scorer through a tiny manifest-free local equivalent.
            from tools.stage3.hed_utils import masked_mean_std, rgb_to_hed, tissue_mask_from_rgb

            ref_hed = rgb_to_hed(ref_img)
            gen_hed = rgb_to_hed(gen_img)
            tissue_mask = tissue_mask_from_rgb(ref_img) | tissue_mask_from_rgb(gen_img)
            hed_score = 0.0
            for channel in (0, 1):
                ref_mean, ref_std = masked_mean_std(ref_hed[..., channel], tissue_mask)
                gen_mean, gen_std = masked_mean_std(gen_hed[..., channel], tissue_mask)
                hed_score += abs(gen_mean - ref_mean) + abs(gen_std - ref_std)
            hed_values.append(float(hed_score))

            pred_inst = run_cellvit(gen_path)
            gt_inst = _load_gt_instance_mask(Path(orion_root), tile_id, shape=pred_inst.shape)
            _, _, pq = _compute_pq(gt_inst, pred_inst)
            dice, _iou, _accuracy = _compute_binary_segmentation_metrics(gt_inst, pred_inst)
            pq_values.append(float(pq))
            dice_values.append(float(dice))

        summary[variant] = {
            "n_tiles": len(tile_ids),
            "fud": fud.get(variant),
            "dice": _mean(dice_values),
            "dice_std": _std(dice_values),
            "pq": _mean(pq_values),
            "pq_std": _std(pq_values),
            "lpips": _mean(lpips_values),
            "lpips_std": _std(lpips_values),
            "style_hed": _mean(hed_values),
            "style_hed_std": _std(hed_values),
        }
        merge_metrics(cache, variant, summary[variant])

    save_cache(cache, cache_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_select = sub.add_parser("select-tiles", help="Select metric tiles and reuse paired-ablation production outputs")
    p_select.add_argument("--cache-dir", type=Path, default=Path("inference_output/si_a1_a2"))
    p_select.add_argument("--paired-ablation-root", type=Path, default=Path("inference_output/paired_ablation"))
    p_select.add_argument("--n-tiles", type=int, default=100)

    p_export = sub.add_parser("export-cellvit", help="Export generated SI PNGs for CellViT")
    p_export.add_argument("--cache-dir", type=Path, default=Path("inference_output/si_a1_a2"))
    p_export.add_argument("--out-dir", type=Path, default=Path("inference_output/si_a1_a2/export_vit"))
    p_export.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    p_export.add_argument("--variants", nargs="+", choices=METRIC_VARIANTS, default=None)

    p_import = sub.add_parser("import-cellvit", help="Import CellViT JSON results")
    p_import.add_argument("--manifest", type=Path, default=Path("inference_output/si_a1_a2/export_vit/manifest.json"))
    p_import.add_argument("--results-dir", type=Path, required=True)
    p_import.add_argument("--result-pattern", default="{stem}.json")

    p_compute = sub.add_parser("compute", help="Compute SI metrics and update cache.json")
    p_compute.add_argument("--cache-dir", type=Path, default=Path("inference_output/si_a1_a2"))
    p_compute.add_argument("--orion-root", type=Path, default=ROOT / "data/orion-crc33")
    p_compute.add_argument("--device", default="cuda")
    p_compute.add_argument("--uni-model", type=Path, default=ROOT / "pretrained_models/uni-2h")
    p_compute.add_argument("--batch-size", type=int, default=16)
    p_compute.add_argument("--variants", nargs="+", choices=METRIC_VARIANTS, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "select-tiles":
        out = select_metric_tiles_from_paired_ablation(
            cache_dir=args.cache_dir,
            paired_ablation_root=args.paired_ablation_root,
            n_tiles=args.n_tiles,
        )
        print(f"Selected {args.n_tiles} metric tiles -> {out}")
        print("Production images and CellViT sidecars were copied from paired_ablation.")
        return 0
    if args.cmd == "export-cellvit":
        export_variants = tuple(args.variants) if args.variants else ("a1_concat", "a1_per_channel", "a2_bypass", "a2_bypass_full_tme", "a2_off_shelf")
        manifest = export_cellvit(cache_dir=args.cache_dir, out_dir=args.out_dir, variants=export_variants, mode=args.mode)
        count = len(json.loads(manifest.read_text(encoding="utf-8")).get("entries", []))
        print(f"Exported {count} PNGs -> {args.out_dir / 'images'}")
        print(f"Manifest -> {manifest}")
        return 0
    if args.cmd == "import-cellvit":
        report = import_cellvit(manifest_path=args.manifest, results_dir=args.results_dir, result_pattern=args.result_pattern)
        print(f"Import report -> {report}")
        return 0
    if args.cmd == "compute":
        summary = compute_and_update_metrics(
            cache_dir=args.cache_dir,
            orion_root=args.orion_root,
            variants=tuple(args.variants) if args.variants else METRIC_VARIANTS,
            device=args.device,
            uni_model=args.uni_model,
            batch_size=args.batch_size,
        )
        print(json.dumps(summary, indent=2))
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
