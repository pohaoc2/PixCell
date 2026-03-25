"""
verify_stage2_model_loading.py

End-to-end verification that stage2_train.py loads the correct base model and
ControlNet, and that a 0-epoch checkpoint produces the same H&E output as
verify_pretrained_inference.py.

Steps:
  1. Create a minimal dummy paired-exp dataset (1 tile, fake features).
  2. Run stage2_train.py with num_epochs=0 → saves initial checkpoint.
  3. Run stage3_inference.py from that checkpoint using the sample test mask.
  4. Create a side-by-side comparison figure vs vis_pretrained_verification_test_mask.png.

The key property being verified:
  - At epoch 0, MultiGroupTMEModule has zero-init residuals → fused = mask_latent.
  - Therefore, output == verify_pretrained_inference.py output (same mask + UNI).

Usage:
    python tools/verify_stage2_model_loading.py [--seed 42] [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent


# ── 1. Dummy dataset creation ─────────────────────────────────────────────────

REQUIRED_CHANNELS = [
    "cell_masks",
    "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
    "cell_state_prolif",  "cell_state_nonprolif", "cell_state_dead",
]
OPTIONAL_CHANNELS = ["vasculature", "oxygen", "glucose"]
ALL_CHANNELS = REQUIRED_CHANNELS + OPTIONAL_CHANNELS

VAE_FULL_CH = 32   # mean (16) + std (16) concatenated on ch dim
VAE_LT_SZ   = 32   # 256 // 8
SSL_DIM     = 1536  # UNI-2h


def _make_dummy_exp_dataset(root: Path, tile_id: str = "dummy_tile_0000") -> None:
    """
    Create the minimal directory/file structure needed by PairedExpControlNetData.

    The channel PNGs are 256×256 zeros (binary channels → black).
    The VAE / UNI numpy arrays are small random tensors.
    """
    exp_ch_dir   = root / "exp_channels"
    feat_dir     = root / "features"
    vae_dir      = root / "vae_features"
    meta_dir     = root / "metadata"

    for d in [exp_ch_dir, feat_dir, vae_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    blank = Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode="L")

    for ch in ALL_CHANNELS:
        ch_dir = exp_ch_dir / ch
        ch_dir.mkdir(exist_ok=True)
        blank.save(ch_dir / f"{tile_id}.png")

    rng = np.random.default_rng(0)

    # VAE latent: [32, 32, 32] — first 16 mean, last 16 std
    vae_feat = rng.standard_normal((VAE_FULL_CH, VAE_LT_SZ, VAE_LT_SZ)).astype(np.float32)
    np.save(vae_dir / f"{tile_id}_sd3_vae.npy", vae_feat)

    # Mask VAE latent (optional fallback exists, but provide it anyway)
    mask_vae = rng.standard_normal((VAE_FULL_CH, VAE_LT_SZ, VAE_LT_SZ)).astype(np.float32)
    np.save(vae_dir / f"{tile_id}_mask_sd3_vae.npy", mask_vae)

    # UNI embedding: [1536]
    uni_feat = rng.standard_normal(SSL_DIM).astype(np.float32)
    np.save(feat_dir / f"{tile_id}_uni.npy", uni_feat)

    # HDF5 index
    h5_path = meta_dir / "exp_index.hdf5"
    with h5py.File(h5_path, "w") as h5:
        ids_bytes = np.array([tile_id.encode("utf-8")], dtype=object)
        h5.create_dataset("exp_256", data=ids_bytes)

    print(f"[dummy dataset] Created 1-tile dummy dataset at {root}")


# ── 2. Prepare inference_data/sample for stage3_inference ─────────────────────

def _ensure_test_mask_in_sample(sample_dir: Path) -> None:
    """
    Copy inference_data/sample/test_mask.png into sample/cell_mask/ as
    test_mask.png, and place a blank 256×256 PNG named test_mask.png in all
    other channel dirs.  This allows stage3_inference.py to run with
    --sim-id test_mask against the same mask used in verify_pretrained_inference.py.

    Skips files that already exist.
    """
    test_mask_src = sample_dir / "test_mask.png"
    if not test_mask_src.exists():
        raise FileNotFoundError(f"test_mask.png not found at {test_mask_src}")

    blank = Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode="L")
    channel_dirs = [
        "cell_mask",
        "cell_type_healthy", "cell_type_cancer", "cell_type_immune",
        "cell_state_prolif",  "cell_state_nonprolif", "cell_state_dead",
        "vasculature", "oxygen", "glucose",
    ]
    for ch in channel_dirs:
        ch_dir = sample_dir / ch
        ch_dir.mkdir(exist_ok=True)
        dest = ch_dir / "test_mask.png"
        if dest.exists():
            continue
        if ch == "cell_mask":
            shutil.copy(test_mask_src, dest)
            print(f"  Copied test_mask.png → {dest}")
        else:
            blank.save(dest)
            print(f"  Created blank test_mask.png → {dest}")


# ── 3. Comparison figure ───────────────────────────────────────────────────────

def _save_comparison(
    stage3_output: Path,
    reference_vis: Path,
    mask_path: Path,
    save_path: Path,
) -> None:
    """Side-by-side: [Reference H&E mask | verify_pretrained output | stage2 output]."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    mask_img = np.array(Image.open(mask_path).convert("RGB").resize((256, 256)))
    ref_img  = np.array(Image.open(reference_vis).convert("RGB"))
    gen_img  = np.array(Image.open(stage3_output).convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(mask_img);  axes[0].set_title("Input mask", fontsize=11)
    axes[1].imshow(ref_img);   axes[1].set_title("verify_pretrained_inference.py", fontsize=11)
    axes[2].imshow(gen_img);   axes[2].set_title("stage2 (0-epoch) → stage3", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        "Stage-2 model loading verification\n"
        "Expected: right ≈ middle (zero-init TME → fused = mask_latent)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n[comparison] Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 model loading end-to-end verification")
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip stage2_train (use existing checkpoint if present)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    sample_dir     = ROOT / "inference_data" / "sample"
    dummy_data_dir = ROOT / "inference_data" / "dummy_exp_train_data"
    checkpoint_dir = (
        ROOT / "checkpoints" / "pixcell_controlnet_0epoch_verify"
        / "checkpoints" / "step_0000000"
    )
    output_path    = ROOT / "inference_data" / "results" / "stage2_verify_output.png"
    comparison_out = ROOT / "inference_data" / "results" / "stage2_vs_pretrained_verify.png"
    reference_vis  = sample_dir / "vis_pretrained_verification_test_mask.png"
    reference_uni  = sample_dir / "test_control_image_uni.npy"
    mask_path      = sample_dir / "test_mask.png"
    config_path    = ROOT / "configs" / "config_0epoch_verify.py"

    # ── Step 1: Dummy dataset ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 1 — Creating dummy paired-exp training dataset")
    print("=" * 70)
    _make_dummy_exp_dataset(dummy_data_dir)

    # ── Step 2: Prepare sample/cell_mask/test_mask.png ────────────────────────
    print("\n" + "=" * 70)
    print("Step 2 — Preparing inference_data/sample channel dirs (sim_id=test_mask)")
    print("=" * 70)
    _ensure_test_mask_in_sample(sample_dir)

    # ── Step 3: Zero-epoch training (load models + save checkpoint) ───────────
    if not args.skip_train or not checkpoint_dir.exists():
        print("\n" + "=" * 70)
        print("Step 3 — Running stage2_train.py (num_epochs=0)")
        print("=" * 70)
        cmd = [sys.executable, str(ROOT / "stage2_train.py"), str(config_path)]
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            sys.exit(f"stage2_train.py failed with exit code {result.returncode}")
    else:
        print(f"\nStep 3 — Skipped (checkpoint exists at {checkpoint_dir})")

    if not checkpoint_dir.exists():
        sys.exit(f"Checkpoint not found at {checkpoint_dir}. Training may have failed.")

    # ── Step 4: Stage-3 inference from the 0-epoch checkpoint ────────────────
    print("\n" + "=" * 70)
    print("Step 4 — Running stage3_inference.py from 0-epoch checkpoint")
    print("=" * 70)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "stage3_inference.py"),
        "--config",           str(ROOT / "configs" / "config_controlnet_exp.py"),
        "--checkpoint-dir",   str(checkpoint_dir),
        "--sim-channels-dir", str(sample_dir),
        "--sim-id",           "test_mask",
        "--reference-uni",    str(reference_uni),
        "--output",           str(output_path),
        "--device",           args.device,
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        sys.exit(f"stage3_inference.py failed with exit code {result.returncode}")

    # ── Step 5: Comparison figure ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Step 5 — Creating comparison figure")
    print("=" * 70)

    if not reference_vis.exists():
        print(f"  WARNING: Reference vis not found at {reference_vis}.")
        print(f"  Run verify_pretrained_inference.py first to generate it.")
    else:
        _save_comparison(output_path, reference_vis, mask_path, comparison_out)

    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)
    print(f"  Stage-3 output:  {output_path}")
    if reference_vis.exists():
        print(f"  Comparison:      {comparison_out}")
    print()
    print("Expected result: the stage-3 output should look visually similar to")
    print("vis_pretrained_verification_test_mask.png — realistic H&E tissue with")
    print("cell boundaries matching the input mask.")
    print()
    print("If models were loaded correctly, pixel-level similarity is expected")
    print("because at epoch 0 the TME module contributes zero residuals,")
    print("making the conditioning identical to verify_pretrained_inference.py.")


if __name__ == "__main__":
    main()
