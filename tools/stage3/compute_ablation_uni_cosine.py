#!/usr/bin/env python3
"""
Compute UNI-2h cosine similarity between reference embeddings and each ablation PNG.

Default reference: ``{orion_root}/features/{tile_id}_uni.npy`` where ``tile_id`` comes
from ``manifest.json`` (paired dataset layout, e.g. ``data/orion-crc33``).

Override with ``--reference-uni``. If that is unset, **``{orion_root}/features/<tile_id>_uni.npy``**
is used automatically (``tile_id`` from ``manifest.json``). Encode from ``--reference-he`` only when
that file is missing (or pass ``--reference-uni`` explicitly).

Writes ``uni_cosine_scores.json`` next to ``--cache-dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import (
    condition_metric_key,
    default_orion_uni_npy_path,
)
from tools.stage3.style_mapping import load_style_mapping
from tools.stage3.uni_cosine_similarity import cosine_similarity_uni, flatten_uni_npy


def resolve_reference_embedding(
    *,
    cache_dir: Path,
    orion_root: Path,
    style_mapping: dict[str, str] | None,
    reference_uni: Path | None,
    reference_he: Path | None,
    extractor_for_he: UNI2hExtractor | None,
) -> tuple[np.ndarray, dict]:
    """
    Returns (ref_emb, reference_meta) where reference_meta is stored in the output JSON.
    """
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    tile_id = str(manifest.get("tile_id", "")).strip()
    if not tile_id:
        raise ValueError("manifest.json must contain a non-empty tile_id")

    if reference_uni is not None:
        path = reference_uni.resolve()
        return flatten_uni_npy(np.load(path)), {"type": "uni_npy", "path": str(path)}

    auto_npy = default_orion_uni_npy_path(orion_root, tile_id, style_mapping=style_mapping)
    if auto_npy.is_file():
        return flatten_uni_npy(np.load(auto_npy)), {
            "type": "orion_features_uni_npy",
            "path": str(auto_npy.resolve()),
            "tile_id": tile_id,
        }

    if reference_he is not None:
        if extractor_for_he is None:
            raise ValueError("UNI extractor required when using --reference-he")
        img = Image.open(reference_he).convert("RGB")
        emb = np.asarray(extractor_for_he.extract(img), dtype=np.float64).ravel()
        return emb, {"type": "he_png", "path": str(reference_he.resolve())}

    raise FileNotFoundError(
        f"Reference UNI not found at {auto_npy} (expected for tile_id={tile_id!r}). "
        "Pass --reference-uni PATH, place {tile_id}_uni.npy under orion features/, "
        "or pass --reference-he to encode from an H&E image."
    )


def compute_scores_for_manifest(
    cache_dir: Path,
    ref_emb: np.ndarray,
    extractor: UNI2hExtractor,
) -> dict[str, float]:
    """Compute per-condition cosine scores.

    UNI embeddings are cached under ``<cache_dir>/features/<subdir>/<stem>_uni.npy``
    (mirroring the image sub-path) so subsequent runs skip model inference.
    """
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    per_condition: dict[str, float] = {}

    for section in manifest["sections"]:
        for entry in section["entries"]:
            raw_groups = tuple(entry["active_groups"])
            key = condition_metric_key(raw_groups)
            gen_path = cache_dir / entry["image_path"]
            if not gen_path.is_file():
                raise FileNotFoundError(f"missing generated image: {gen_path}")

            img_rel = Path(entry["image_path"])
            feat_path = cache_dir / "features" / img_rel.parent / (img_rel.stem + "_uni.npy")

            if feat_path.is_file():
                gen_emb = np.load(feat_path).astype(np.float64).ravel()
            else:
                gen_img = Image.open(gen_path).convert("RGB")
                gen_emb = np.asarray(extractor.extract(gen_img), dtype=np.float64).ravel()
                feat_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(feat_path, gen_emb)

            per_condition[key] = cosine_similarity_uni(ref_emb, gen_emb)

    return per_condition


def load_uni_extractor(
    *,
    uni_model: Path | None = None,
    device: str = "cuda",
):
    """Load one UNI extractor with the usual cuda->cpu fallback."""
    from pipeline.extract_features import UNI2hExtractor

    uni_model = Path(uni_model) if uni_model is not None else ROOT / "pretrained_models/uni-2h"

    devices_try: list[str]
    if str(device).lower() == "cuda":
        devices_try = ["cuda", "cpu"]
    else:
        devices_try = [str(device)]

    last_err: Exception | None = None
    extractor = None
    for dev in devices_try:
        try:
            extractor = UNI2hExtractor(model_path=str(uni_model), device=dev)
            break
        except Exception as exc:
            last_err = exc
            continue
    if extractor is None:
        assert last_err is not None
        raise last_err
    return extractor


def compute_and_write_uni_cosine_scores(
    cache_dir: Path,
    *,
    orion_root: Path,
    style_mapping: dict[str, str] | None = None,
    reference_uni: Path | None = None,
    reference_he: Path | None = None,
    uni_model: Path | None = None,
    device: str = "cuda",
    output_path: Path | None = None,
    extractor: Any | None = None,
) -> tuple[Path, dict, dict[str, float]]:
    """
    Run UNI embedding + cosine for every manifest entry; write ``uni_cosine_scores.json``.

    Returns ``(out_path, reference_meta, per_condition)``.
    """
    cache_dir = Path(cache_dir).resolve()
    orion_root = Path(orion_root).resolve()
    uni_model = Path(uni_model) if uni_model is not None else ROOT / "pretrained_models/uni-2h"
    out_path = output_path if output_path is not None else cache_dir / "uni_cosine_scores.json"

    if extractor is None:
        extractor = load_uni_extractor(uni_model=uni_model, device=device)

    ref_emb, ref_meta = resolve_reference_embedding(
        cache_dir=cache_dir,
        orion_root=orion_root,
        style_mapping=style_mapping,
        reference_uni=reference_uni,
        reference_he=reference_he,
        extractor_for_he=extractor,
    )
    per_condition = compute_scores_for_manifest(cache_dir, ref_emb, extractor)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    tile_id = manifest.get("tile_id", "")
    payload = {
        "version": 1,
        "metric": "uni_cosine",
        "tile_id": tile_id,
        "orion_root": str(orion_root),
        "style_mapping_tile_count": 0 if style_mapping is None else len(style_mapping),
        "reference": ref_meta,
        "uni_model": str(uni_model.resolve()),
        "per_condition": per_condition,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path, ref_meta, per_condition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UNI cosine similarity vs reference for each ablation cache image.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing manifest.json and subset PNGs",
    )
    parser.add_argument(
        "--orion-root",
        type=Path,
        default=ROOT / "data/orion-crc33",
        help="Paired dataset root containing features/<tile_id>_uni.npy (default: data/orion-crc33)",
    )
    parser.add_argument(
        "--style-mapping-json",
        type=Path,
        default=None,
        help="Optional layout->style mapping JSON for unpaired reference lookup.",
    )
    parser.add_argument(
        "--reference-uni",
        type=Path,
        default=None,
        help="Explicit UNI .npy for reference (overrides auto path under orion-root/features/)",
    )
    parser.add_argument(
        "--reference-he",
        type=Path,
        default=None,
        help=(
            "Reference H&E RGB; UNI embedding is computed from this PNG only if "
            "{orion-root}/features/<tile_id>_uni.npy does not exist (auto path is preferred)"
        ),
    )
    parser.add_argument(
        "--uni-model",
        type=Path,
        default=ROOT / "pretrained_models/uni-2h",
        help="UNI-2h model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: {cache-dir}/uni_cosine_scores.json)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    orion_root = args.orion_root.resolve()
    if not (cache_dir / "manifest.json").is_file():
        raise SystemExit(f"manifest not found: {cache_dir / 'manifest.json'}")

    out_path = args.output if args.output is not None else cache_dir / "uni_cosine_scores.json"

    out_path, ref_meta, per_condition = compute_and_write_uni_cosine_scores(
        cache_dir,
        orion_root=orion_root,
        style_mapping=load_style_mapping(args.style_mapping_json),
        reference_uni=args.reference_uni,
        reference_he=args.reference_he,
        uni_model=args.uni_model,
        device=args.device,
        output_path=out_path,
    )
    if (
        args.reference_he is not None
        and args.reference_uni is None
        and ref_meta.get("type") != "he_png"
    ):
        print(
            "Note: --reference-he not used; reference embedding was loaded from "
            f"{ref_meta.get('path', ref_meta)}",
            file=sys.stderr,
        )

    print(f"Reference embedding source: {ref_meta}")
    print(f"Wrote {len(per_condition)} scores → {out_path}")


if __name__ == "__main__":
    main()
