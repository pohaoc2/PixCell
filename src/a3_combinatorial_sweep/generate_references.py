"""Generate original-TME reference H&E images for A3 anchors.

Writes to ``inference_output/paired_ablation/ablation_results/<anchor>/all/generated_he.png``,
matching the existing paired-ablation convention. Idempotent: skips anchors that
already have a reference. Failures on individual anchors are logged but do not
abort the batch.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def target_path(output_root: Path, anchor_id: str) -> Path:
    """Return the paired-ablation reference image path for one anchor."""
    return Path(output_root) / anchor_id / "all" / "generated_he.png"


def read_anchor_list(anchors_path: Path) -> list[str]:
    """Read non-empty anchor IDs from a newline-delimited text file."""
    return [
        line.strip()
        for line in Path(anchors_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def plan_missing_anchors(*, anchors_path: Path, output_root: Path) -> list[str]:
    """Return anchors whose reference image is not already present."""
    anchors = read_anchor_list(anchors_path)
    return [anchor_id for anchor_id in anchors if not target_path(output_root, anchor_id).is_file()]


def render_and_save_reference(
    anchor_id: str,
    output_path: Path,
    *,
    config_path: Path,
    checkpoint_dir: Path,
    data_root: Path,
    device: str = "cuda",
    guidance_scale: float = 2.5,
    num_steps: int = 20,
    seed: int = 42,
) -> Path:
    """Render the anchor's H&E from its original, unmodified TME channels."""
    from src.a3_combinatorial_sweep.main import (
        _load_anchor_ctrl,
        _load_anchor_uni,
        _load_generation_runtime,
        _make_generation_noise,
        _render_generated_image,
        _save_image,
    )

    models, runtime_config, scheduler, exp_channels_dir, feat_dir = _load_generation_runtime(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
        device=device,
        num_steps=num_steps,
    )
    active_channels = list(runtime_config.data.active_channels)
    base_ctrl = _load_anchor_ctrl(
        anchor_id,
        active_channels=active_channels,
        image_size=runtime_config.image_size,
        exp_channels_dir=exp_channels_dir,
    )
    uni_embeds = _load_anchor_uni(anchor_id, feat_dir=feat_dir)
    fixed_noise = _make_generation_noise(
        config=runtime_config,
        scheduler=scheduler,
        device=device,
        seed=seed,
    )
    generated = _render_generated_image(
        base_ctrl,
        models=models,
        config=runtime_config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        fixed_noise=fixed_noise,
        seed=seed,
    )
    return _save_image(generated, output_path)


ORIGINAL_RENDER_AND_SAVE_REFERENCE = render_and_save_reference


def _load_reference_runtime(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    data_root: Path,
    device: str,
    num_steps: int,
    seed: int,
) -> dict[str, Any]:
    from src.a3_combinatorial_sweep.main import _load_generation_runtime, _make_generation_noise

    models, runtime_config, scheduler, exp_channels_dir, feat_dir = _load_generation_runtime(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
        device=device,
        num_steps=num_steps,
    )
    fixed_noise = _make_generation_noise(
        config=runtime_config,
        scheduler=scheduler,
        device=device,
        seed=seed,
    )
    return {
        "models": models,
        "runtime_config": runtime_config,
        "scheduler": scheduler,
        "exp_channels_dir": exp_channels_dir,
        "feat_dir": feat_dir,
        "fixed_noise": fixed_noise,
    }


def _render_and_save_reference_with_runtime(
    anchor_id: str,
    output_path: Path,
    *,
    runtime: dict[str, Any],
    device: str,
    guidance_scale: float,
    seed: int,
) -> Path:
    from src.a3_combinatorial_sweep.main import (
        _load_anchor_ctrl,
        _load_anchor_uni,
        _render_generated_image,
        _save_image,
    )

    runtime_config = runtime["runtime_config"]
    active_channels = list(runtime_config.data.active_channels)
    base_ctrl = _load_anchor_ctrl(
        anchor_id,
        active_channels=active_channels,
        image_size=runtime_config.image_size,
        exp_channels_dir=runtime["exp_channels_dir"],
    )
    uni_embeds = _load_anchor_uni(anchor_id, feat_dir=runtime["feat_dir"])
    generated = _render_generated_image(
        base_ctrl,
        models=runtime["models"],
        config=runtime_config,
        scheduler=runtime["scheduler"],
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        fixed_noise=runtime["fixed_noise"],
        seed=seed,
    )
    return _save_image(generated, output_path)


def run(
    *,
    anchors_path: Path,
    output_root: Path,
    config_path: Path,
    checkpoint_dir: Path,
    data_root: Path,
    device: str = "cuda",
    guidance_scale: float = 2.5,
    num_steps: int = 20,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Process every anchor in ``anchors_path`` and return outcome lists."""
    anchors = read_anchor_list(anchors_path)
    skipped: list[str] = []
    generated: list[str] = []
    failed: list[str] = []
    runtime: dict[str, Any] | None = None

    for anchor_id in anchors:
        output_path = target_path(output_root, anchor_id)
        if output_path.is_file():
            skipped.append(anchor_id)
            LOGGER.info("skip %s (reference already exists)", anchor_id)
            continue

        try:
            if render_and_save_reference is ORIGINAL_RENDER_AND_SAVE_REFERENCE:
                if runtime is None:
                    runtime = _load_reference_runtime(
                        config_path=config_path,
                        checkpoint_dir=checkpoint_dir,
                        data_root=data_root,
                        device=device,
                        num_steps=num_steps,
                        seed=seed,
                    )
                _render_and_save_reference_with_runtime(
                    anchor_id,
                    output_path,
                    runtime=runtime,
                    device=device,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
            else:
                render_and_save_reference(
                    anchor_id,
                    output_path,
                    config_path=config_path,
                    checkpoint_dir=checkpoint_dir,
                    data_root=data_root,
                    device=device,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    seed=seed,
                )
        except Exception:
            LOGGER.exception("reference generation failed for %s", anchor_id)
            failed.append(anchor_id)
            continue

        generated.append(anchor_id)
        LOGGER.info("generated reference for %s -> %s", anchor_id, output_path)

    return {"skipped": skipped, "generated": generated, "failed": failed}


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_cli(argv)
    summary = run(
        anchors_path=args.anchors,
        output_root=args.output_root,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
