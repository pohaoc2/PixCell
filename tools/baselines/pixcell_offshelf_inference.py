"""Off-the-shelf PixCell ControlNet inference for the A2 baseline.

This wrapper runs the published PixCell ControlNet checkpoint without
fine-tuning, using the paired-test cell-mask channel as mask-only spatial
conditioning and cached UNI embeddings for style conditioning.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class OffShelfPixCellInference:
    """Lazy-loading off-the-shelf PixCell ControlNet runner."""

    controlnet_path: str
    base_model_path: str
    vae_path: str
    uni_path: str
    device: str = "cuda"
    config_path: str = "configs/config_controlnet_exp.py"

    def __post_init__(self) -> None:
        self._loaded = False
        self._vae = None
        self._base_model = None
        self._controlnet = None

    def _ensure_loaded(self) -> None:  # pragma: no cover - heavyweight model path
        if self._loaded:
            return
        from train_scripts.inference_controlnet import (
            load_controlnet_model_from_checkpoint,
            load_pixcell_controlnet_model_from_checkpoint,
            load_vae,
        )

        self._vae = load_vae(self.vae_path, self.device)
        self._base_model = load_pixcell_controlnet_model_from_checkpoint(
            self.config_path,
            _resolve_weight_file(self.base_model_path),
        )
        self._base_model.to(self.device).eval()
        self._controlnet = load_controlnet_model_from_checkpoint(
            self.config_path,
            _resolve_weight_file(self.controlnet_path),
            self.device,
        )
        self._loaded = True

    def encode_mask_to_latent(self, cell_mask: np.ndarray) -> torch.Tensor:
        """VAE-encode a cell mask into the ControlNet latent convention."""
        from train_scripts.inference_controlnet import encode_ctrl_mask_latent

        self._ensure_loaded()
        mask = np.asarray(cell_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        if mask.size and float(np.nanmax(mask)) > 1.0:
            mask = mask / 255.0
        ctrl_full = torch.from_numpy(mask[None, ...])
        return encode_ctrl_mask_latent(
            ctrl_full,
            self._vae,
            vae_shift=getattr(self._vae.config, "shift_factor", 0.0609),
            vae_scale=getattr(self._vae.config, "scaling_factor", 1.5305),
            device=self.device,
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        )

    def run_on_tile(
        self,
        *,
        tile_id: str,
        cell_mask: np.ndarray,
        uni_embedding: np.ndarray,
        out_dir: Path,
        num_steps: int = 30,
        guidance_scale: float = 1.5,
    ) -> Path:
        """Generate one PNG and return its output path."""
        from tools.stage3.common import make_inference_scheduler
        from train_scripts.inference_controlnet import denoise
        from tools.stage3.tile_pipeline import _decode_latents_to_image

        self._ensure_loaded()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_latent = self.encode_mask_to_latent(cell_mask)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        scheduler = make_inference_scheduler(num_steps=num_steps, device=self.device)
        latents = torch.randn_like(mask_latent, device=self.device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma
        y = torch.from_numpy(np.asarray(uni_embedding)).view(1, 1, 1, 1536)

        denoised = denoise(
            latents=latents,
            uni_embeds=y,
            controlnet_input_latent=mask_latent,
            scheduler=scheduler,
            controlnet_model=self._controlnet,
            pixcell_controlnet_model=self._base_model,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            device=self.device,
        )
        image = _decode_latents_to_image(
            denoised,
            vae=self._vae,
            vae_scale=getattr(self._vae.config, "scaling_factor", 1.5305),
            vae_shift=getattr(self._vae.config, "shift_factor", 0.0609),
            dtype=dtype,
        )
        out_path = out_dir / f"{tile_id}.png"
        Image.fromarray(image).save(out_path)
        return out_path


def _resolve_weight_file(path: str | Path) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    candidates = sorted(p.glob("*.safetensors")) + sorted(p.glob("*.pth"))
    if not candidates:
        candidates = sorted(p.glob("**/*.safetensors")) + sorted(p.glob("**/*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No model weights found under {p}")
    return str(candidates[0])


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def _load_uni(path: Path) -> np.ndarray:
    return np.load(path).reshape(-1)[:1536]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlnet", "--controlnet-path", dest="controlnet_path", required=True)
    parser.add_argument("--base", "--base-model-path", dest="base_model_path", required=True)
    parser.add_argument("--vae", "--vae-path", dest="vae_path", required=True)
    parser.add_argument("--uni", "--uni-path", dest="uni_path", required=True)
    parser.add_argument("--tile-ids", "--tile_ids", dest="tile_ids", required=True)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    parser.add_argument("--config", default="configs/config_controlnet_exp.py")
    parser.add_argument("--num-steps", "--num_steps", dest="num_steps", type=int, default=30)
    parser.add_argument("--guidance-scale", "--guidance_scale", dest="guidance_scale", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    runner = OffShelfPixCellInference(
        controlnet_path=args.controlnet_path,
        base_model_path=args.base_model_path,
        vae_path=args.vae_path,
        uni_path=args.uni_path,
        device=args.device,
        config_path=args.config,
    )
    tile_ids = [line.strip() for line in Path(args.tile_ids).read_text().splitlines() if line.strip()]
    mask_dir = Path(args.mask_dir) if args.mask_dir else Path("data/orion-crc33/exp_channels/cell_masks")
    uni_dir = Path(args.uni_path)
    for tile_id in tile_ids:
        uni_path = uni_dir / f"{tile_id}_uni.npy"
        out = runner.run_on_tile(
            tile_id=tile_id,
            cell_mask=_load_mask(mask_dir / f"{tile_id}.png"),
            uni_embedding=_load_uni(uni_path),
            out_dir=Path(args.out_dir),
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
