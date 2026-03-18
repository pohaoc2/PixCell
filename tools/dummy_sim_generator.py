"""
dummy_sim_generator.py

Generates a minimal fake dataset on disk to test SimControlNetData
without requiring real simulation outputs or TCGA features.

Creates:
    sim_data_root/
    ├── metadata/
    │   ├── sim_index.hdf5
    │   └── real_index.hdf5
    ├── sim_channels/
    │   ├── cell_mask/   {sim_id}.png   — binary hex-patch pattern
    │   ├── oxygen/      {sim_id}.png   — smooth gradient noise
    │   ├── glucose/     {sim_id}.png   — smooth gradient noise
    │   └── tgf/         {sim_id}.png   — smooth gradient noise
    ├── features/
    │   └── {tile_id}_uni.npy           — random [1152] float32
    └── vae_features/
        ├── {tile_id}_sd3_vae.npy       — random [32, 32, 32] float32
        └── {tile_id}_mask_sd3_vae.npy  — random [32, 32, 32] float32

Usage:
    # Generate dummy data and run a quick dataloader test
    python dummy_sim_generator.py --out_dir ./dummy_sim_data --n_sim 20 --n_real 10

    # Then test the dataset
    python dummy_sim_generator.py --out_dir ./dummy_sim_data --test_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────

RESOLUTION  = 256
VAE_CH      = 16     # VAE latent channels (mean only)
VAE_FULL_CH = 32     # mean + std concatenated
VAE_LT_SZ   = 32     # spatial latent size (256 // 8)
SSL_DIM     = 1536   # UNI-2h embedding dimension


# ── Helpers ───────────────────────────────────────────────────────────────────

def _smooth_noise(resolution: int, rng: np.random.Generator, scale: int = 32) -> np.ndarray:
    """
    Generate smooth noise by upsampling a small random patch.
    Mimics the low-frequency structure of TME concentration maps.
    Returns float32 [H, W] in [0, 1].
    """
    small = rng.random((scale, scale)).astype(np.float32)
    big   = cv2.resize(small, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    vmin, vmax = big.min(), big.max()
    return ((big - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)


def _hex_cell_mask(resolution: int, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a fake binary cell mask by placing filled circles at random positions.
    Returns uint8 [H, W] with values in {0, 255}.
    """
    canvas = np.zeros((resolution, resolution), dtype=np.uint8)
    radius = max(4, resolution // 40)
    xs = rng.integers(radius, resolution - radius, size=n_cells)
    ys = rng.integers(radius, resolution - radius, size=n_cells)
    for x, y in zip(xs, ys):
        cv2.circle(canvas, (int(x), int(y)), radius, 255, thickness=-1)
    return canvas


def _save_png_gray(arr: np.ndarray, path: Path) -> None:
    """Save a [H, W] float32 [0,1] or uint8 [0,255] array as grayscale PNG."""
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _write_h5_index(path: Path, key: str, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        dt   = h5py.special_dtype(vlen=str)
        dset = h5.create_dataset(key, shape=(len(ids),), dtype=dt)
        for i, sid in enumerate(ids):
            dset[i] = sid


# ── Generators ────────────────────────────────────────────────────────────────

def generate_sim_snapshot(
    sim_id: str,
    sim_channels_dir: Path,
    resolution: int,
    rng: np.random.Generator,
    channels: list[str],
    n_cells: int = 60,
) -> None:
    """
    Write all requested channel PNG files for one simulation snapshot.

    Args:
        sim_id:            Identifier string used in filenames.
        sim_channels_dir:  Root of the sim_channels/ directory.
        resolution:        Image spatial resolution.
        rng:               NumPy random generator.
        channels:          List of channel names to generate.
        n_cells:           Approximate number of cells in the mask.
    """
    for ch in channels:
        out_dir = sim_channels_dir / ch
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sim_id}.png"

        if ch == "cell_mask":
            arr = _hex_cell_mask(resolution, n_cells, rng)
            _save_png_gray(arr, out_path)

        else:
            # TME fields: smooth noise with a realistic spatial structure
            arr = _smooth_noise(resolution, rng, scale=rng.integers(8, 48))
            _save_png_gray(arr, out_path)


def generate_real_tile(
    tile_id: str,
    features_dir: Path,
    vae_features_dir: Path,
    rng: np.random.Generator,
    ssl_prefix: str = "uni",
    vae_prefix: str = "sd3_vae",
    include_vae_mask: bool = True,
) -> None:
    """
    Write dummy UNI embedding and VAE latent files for one "real" H&E tile.

    File format mirrors what PanCancerControlNetData expects:
        features/{tile_id}_{ssl_prefix}.npy          → [1152]      float32
        vae_features/{tile_id}_{vae_prefix}.npy      → [32, 32, 32] float32 (mean+std)
        vae_features/{tile_id}_mask_{vae_prefix}.npy → [32, 32, 32] float32 (optional)

    Args:
        tile_id:            Identifier string.
        features_dir:       Path to features/ directory.
        vae_features_dir:   Path to vae_features/ directory.
        rng:                NumPy random generator.
        ssl_prefix:         UNI file suffix. Default: "uni".
        vae_prefix:         VAE file suffix. Default: "sd3_vae".
        include_vae_mask:   Whether to also write the VAE-encoded mask file.
    """
    features_dir.mkdir(parents=True, exist_ok=True)
    vae_features_dir.mkdir(parents=True, exist_ok=True)

    # UNI-2h embedding: random unit-normalized vector [1152]
    ssl = rng.standard_normal(SSL_DIM).astype(np.float32)
    ssl /= np.linalg.norm(ssl) + 1e-8
    np.save(features_dir / f"{tile_id}_{ssl_prefix}.npy", ssl)

    # VAE latent: [32, lt_sz, lt_sz] (mean + std concatenated along channel dim)
    # Mean sampled from N(0,1), std from Softplus(N(0,0.1)) to ensure positivity
    mean = rng.standard_normal((VAE_CH, VAE_LT_SZ, VAE_LT_SZ)).astype(np.float32)
    std  = np.log1p(np.exp(rng.standard_normal((VAE_CH, VAE_LT_SZ, VAE_LT_SZ)).astype(np.float32) * 0.1))
    vae  = np.concatenate([mean, std], axis=0)   # [32, 32, 32]
    np.save(vae_features_dir / f"{tile_id}_{vae_prefix}.npy", vae)

    # VAE-encoded mask (optional)
    if include_vae_mask:
        mask_mean = rng.standard_normal((VAE_CH, VAE_LT_SZ, VAE_LT_SZ)).astype(np.float32) * 0.3
        mask_std  = np.abs(rng.standard_normal((VAE_CH, VAE_LT_SZ, VAE_LT_SZ)).astype(np.float32) * 0.1)
        mask_vae  = np.concatenate([mask_mean, mask_std], axis=0)
        np.save(vae_features_dir / f"{tile_id}_mask_{vae_prefix}.npy", mask_vae)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_dummy_dataset(
    out_dir: str | Path,
    n_sim: int = 20,
    n_real: int = 10,
    resolution: int = RESOLUTION,
    channels: list[str] | None = None,
    seed: int = 0,
) -> Path:
    """
    Generate a complete dummy dataset for testing SimControlNetData.

    Args:
        out_dir:     Root directory to create (will be created if absent).
        n_sim:       Number of fake simulation snapshots.
        n_real:      Number of fake real H&E tiles.
        resolution:  Spatial resolution (default: 256).
        channels:    Which sim channels to generate. Default: all 4.
        seed:        Random seed for reproducibility.

    Returns:
        Path to the created dataset root.
    """
    out_dir  = Path(out_dir)
    channels = channels or ["cell_mask", "oxygen", "glucose", "tgf"]
    rng      = np.random.default_rng(seed)

    print(f"\nGenerating dummy dataset in: {out_dir}")
    print(f"  {n_sim} sim snapshots × channels={channels}")
    print(f"  {n_real} real tiles\n")

    sim_channels_dir = out_dir / "sim_channels"
    features_dir     = out_dir / "features"
    vae_features_dir = out_dir / "vae_features"

    # ── Sim snapshots ──────────────────────────────────────────────────────────
    sim_ids: list[str] = []
    for i in range(n_sim):
        sim_id   = f"sim_{i:04d}_{rng.integers(1000, 99999):06d}"
        n_cells  = int(rng.integers(20, 80))
        generate_sim_snapshot(
            sim_id=sim_id,
            sim_channels_dir=sim_channels_dir,
            resolution=resolution,
            rng=rng,
            channels=channels,
            n_cells=n_cells,
        )
        sim_ids.append(sim_id)
    print(f"  ✓ Generated {n_sim} sim snapshots")

    # ── Real tiles ─────────────────────────────────────────────────────────────
    real_ids: list[str] = []
    for i in range(n_real):
        tile_id = f"TCGA_dummy_{i:04d}"
        generate_real_tile(
            tile_id=tile_id,
            features_dir=features_dir,
            vae_features_dir=vae_features_dir,
            rng=rng,
        )
        real_ids.append(tile_id)
    print(f"  ✓ Generated {n_real} real tiles")

    # ── HDF5 indices ───────────────────────────────────────────────────────────
    _write_h5_index(out_dir / "metadata/sim_index.hdf5",  f"sim_{resolution}",  sim_ids)
    _write_h5_index(out_dir / "metadata/real_index.hdf5", f"real_{resolution}", real_ids)
    print(f"  ✓ Wrote HDF5 indices\n")

    return out_dir


# ── Dataset smoke test ────────────────────────────────────────────────────────

def test_dataset(
    data_root: str | Path,
    active_channels: list[str] | None = None,
    resolution: int = RESOLUTION,
    n_batches: int = 3,
    batch_size: int = 2,
) -> None:
    """
    Instantiate SimControlNetData and run a few batches through a DataLoader.
    Prints shapes and value ranges for each output tensor.
    """
    # Import here so this script can be run standalone without the full package
    import torch
    from torch.utils.data import DataLoader

    # Local import of the dataset class
    import importlib, sys
    spec = importlib.util.spec_from_file_location(
        "sim_controlnet_dataset",
        Path(__file__).parent / "diffusion/data/datasets/sim_controlnet_dataset.py",
    )
    mod    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SimControlNetData = mod.SimControlNetData

    active_channels = active_channels or ["cell_mask", "oxygen", "glucose", "tgf"]

    print(f"\n{'─'*60}")
    print(f"Testing SimControlNetData")
    print(f"  root={data_root}")
    print(f"  active_channels={active_channels}")
    print(f"  resolution={resolution}")
    print(f"{'─'*60}")

    dataset = SimControlNetData(
        root=str(data_root),
        resolution=resolution,
        active_channels=active_channels,
    )
    print(f"\n  Dataset length: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break

        vae_feat, ssl_feat, ctrl_tensor, vae_mask, data_info = batch

        print(f"\n  Batch {batch_idx + 1}/{n_batches}")
        print(f"    vae_feat    : {tuple(vae_feat.shape)}  "
              f"range=[{vae_feat.min():.3f}, {vae_feat.max():.3f}]")
        print(f"    ssl_feat    : {tuple(ssl_feat.shape)}  "
              f"range=[{ssl_feat.min():.3f}, {ssl_feat.max():.3f}]")
        print(f"    ctrl_tensor : {tuple(ctrl_tensor.shape)}  "
              f"range=[{ctrl_tensor.min():.3f}, {ctrl_tensor.max():.3f}]")
        print(f"    vae_mask    : {tuple(vae_mask.shape)}  "
              f"range=[{vae_mask.min():.3f}, {vae_mask.max():.3f}]")
        #print(f"    sim_ids     : {data_info['sim_id']}")
        #print(f"    tile_ids    : {data_info['tile_id']}")

        # Check expected shapes
        B  = batch_size
        lt = resolution // 8
        C  = len(active_channels)
        assert vae_feat.shape    == (B, 16, lt, lt),                    "vae_feat shape wrong"
        assert ssl_feat.shape    == (B, 1, 1, SSL_DIM),                       "ssl_feat shape wrong"
        assert ctrl_tensor.shape == (B, C, resolution, resolution),     "ctrl_tensor shape wrong"
        assert vae_mask.shape    == (B, 16, lt, lt),                    "vae_mask shape wrong"

        # Cell mask should be binary
        mask_ch    = active_channels.index("cell_mask")
        mask_vals  = ctrl_tensor[:, mask_ch, :, :].unique()
        assert set(mask_vals.tolist()).issubset({0.0, 1.0}), \
            f"cell_mask is not binary! Unique values: {mask_vals}"

        print(f"    ✓ All shape assertions passed")

    print(f"\n{'─'*60}")
    print("All tests passed!")
    print(f"{'─'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate dummy simulation dataset and/or test SimControlNetData"
    )
    p.add_argument(
        "--out_dir", default="./dummy_sim_data",
        help="Root directory for the dummy dataset (default: ./dummy_sim_data)",
    )
    p.add_argument("--n_sim",  type=int, default=20, help="Number of sim snapshots")
    p.add_argument("--n_real", type=int, default=10, help="Number of real tiles")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument(
        "--channels", nargs="+",
        default=["cell_mask", "oxygen", "glucose", "tgf"],
        help="Channels to generate and test",
    )
    p.add_argument(
        "--test_only", action="store_true",
        help="Skip generation, only run the dataset test (data must already exist)",
    )
    p.add_argument("--n_batches", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.test_only:
        generate_dummy_dataset(
            out_dir=args.out_dir,
            n_sim=args.n_sim,
            n_real=args.n_real,
            resolution=args.resolution,
            channels=args.channels,
            seed=args.seed,
        )

    test_dataset(
        data_root=args.out_dir,
        active_channels=args.channels,
        resolution=args.resolution,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
    )