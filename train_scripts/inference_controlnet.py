# %%
import torch
import numpy as np
import os
import copy
from PIL import Image
from train_scripts.mapping_weights_helper import (
    load_controlnet_weights_flexible,
    load_model_weights_flexible,
)


def null_uni_embed(device="cuda", dtype=torch.float16):
    """
    Zero UNI embedding for TME-only (no style reference) inference.

    Pass as `uni_embeds` to `denoise()` to run purely TME-conditioned generation.
    CFG guidance_scale still applies — higher values increase TME adherence.

    Returns:
        Tensor shape [1, 1, 1, 1536], all zeros.
    """
    return torch.zeros(1, 1, 1, 1536, device=device, dtype=dtype)


def encode_ctrl_mask_latent(
    ctrl_full: torch.Tensor,
    vae,
    *,
    vae_shift,
    vae_scale,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """VAE-encode the cell-mask channel (index 0) of ctrl_full to scaled latent space.

    Matches training/inference convention: repeat mask to RGB, map [0,1] → [-1,1], encode mean,
    then apply (latent - vae_shift) * vae_scale.

    Args:
        ctrl_full: [C, H, W] control stack; channel 0 is the binary mask in [0, 1].
        vae: Frozen VAE (caller sets ``.eval()`` and device).
        vae_shift: Config ``shift_factor``.
        vae_scale: Config ``scale_factor``.
        device: Encode device.
        dtype: Encode dtype (e.g. float16 on CUDA).

    Returns:
        Scaled mask latent [1, 16, H/8, W/8].
    """
    cell_mask_img = ctrl_full[0:1].unsqueeze(0).repeat(1, 3, 1, 1)
    cell_mask_img = 2 * (cell_mask_img - 0.5)
    with torch.no_grad():
        lat = vae.encode(cell_mask_img.to(device=device, dtype=dtype)).latent_dist.mean
    return (lat - vae_shift) * vae_scale


def load_controlnet_model(module_name, file_path, checkpoints_folder, device="cuda"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    controlnet_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = controlnet_mod
    spec.loader.exec_module(controlnet_mod)
    PixCellControlNet = controlnet_mod.PixCellControlNet
    model = PixCellControlNet.from_pretrained(checkpoints_folder)
    model.to(device)
    model.eval()
    return model


def load_pixcell_controlnet_model_from_checkpoint(config_file_path, state_file_path):
    from diffusion.utils.misc import read_config
    from diffusion.model.builder import build_model

    config = read_config(config_file_path)
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    max_length = config.model_max_length
    latent_size = int(config.image_size) // 8
    pixcell_controlnet_model_kwargs = {
        "pe_interpolation": config.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.qk_norm,
        "kv_compress_config": kv_compress_config,
        "micro_condition": config.micro_condition,
        "add_pos_embed_to_cond": getattr(config, "add_pos_embed_to_cond", False),
        **config.get("base_model_kwargs", {}),
    }
    pixcell_controlnet = build_model(
        config.base_model,  # e.g., 'PixCell_Transformer_XL_2_UNI'
        False,  # No grad checkpointing for frozen model
        config.get("fp32_attention", False),
        input_size=latent_size,
        learn_sigma=True,
        pred_sigma=True,
        **pixcell_controlnet_model_kwargs,
    )
    load_model_weights_flexible(
        pixcell_controlnet,
        state_file_path,
        remap_pixcell_safetensors=True,
        verbose=True,
    )
    pixcell_controlnet.eval()
    return pixcell_controlnet


def load_base_model_checkpoint(base_model, checkpoint_path):
    finetuned = torch.load(checkpoint_path, map_location="cpu")
    finetuned_sd = finetuned["state_dict"] if "state_dict" in finetuned else finetuned

    # NO remapping needed — both checkpoint and model use 'blocks.' naming
    missing, unexpected = base_model.load_state_dict(finetuned_sd, strict=False)

    print(f"Total missing: {len(missing)}, Total unexpected: {len(unexpected)}")
    if missing:
        print(f"Missing examples: {missing[:3]}")
    if unexpected:
        print(f"Unexpected examples: {unexpected[:3]}")
    return base_model


def load_controlnet_model_from_checkpoint(config_file_path, state_file_path, device="cuda"):
    from diffusion.utils.misc import read_config
    from diffusion.model.builder import build_model

    config = read_config(config_file_path)
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    max_length = config.model_max_length
    latent_size = int(config.image_size) // 8
    controlnet_model_kwargs = {
        "pe_interpolation": config.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.qk_norm,
        "kv_compress_config": kv_compress_config,
        "conditioning_channels": config.controlnet_conditioning_channels,  # e.g., 16 for cell masks
        "n_controlnet_blocks": getattr(config, "n_controlnet_blocks", None),
        **config.get("controlnet_model_kwargs", {}),
    }

    # Build ControlNet (all parameters trainable)
    controlnet = build_model(
        config.controlnet_model,  # e.g., 'PixCell_ControlNet_XL_2_UNI'
        config.grad_checkpointing,
        config.get("fp32_attention", False),
        input_size=latent_size,
        learn_sigma=False,  # ControlNet doesn't predict sigma
        pred_sigma=False,
        **controlnet_model_kwargs,
    )
    stats = load_controlnet_weights_flexible(controlnet, state_file_path, verbose=True)
    print(f"Loaded ControlNet tensors: {stats['loaded']}")
    controlnet.to(device)
    controlnet.eval()
    return controlnet


def save_keys_comparison_controlnet(controlnet, state_file_path, device="cuda"):
    from safetensors.torch import load_file
    import pandas as pd

    sd = load_file(state_file_path)
    controlnet_sd = controlnet.state_dict()
    sd_keys = list(sd.keys())
    controlnet_keys = list(controlnet_sd.keys())

    # Check counts first
    print(f"Pretrained keys: {len(sd_keys)}")
    print(f"ControlNet keys: {len(controlnet_keys)}")

    # Build comparison df by position
    max_len = max(len(sd_keys), len(controlnet_keys))
    df = pd.DataFrame(
        {
            "sd_key": sd_keys + [None] * (max_len - len(sd_keys)),
            "controlnet_key": controlnet_keys + [None] * (max_len - len(controlnet_keys)),
            "sd_shape": [
                str(sd[k].shape) if k else None for k in sd_keys + [None] * (max_len - len(sd_keys))
            ],
            "controlnet_shape": [
                str(controlnet_sd[k].shape) if k else None
                for k in controlnet_keys + [None] * (max_len - len(controlnet_keys))
            ],
            "shape_match": [
                (
                    str(sd[sd_keys[i]].shape) == str(controlnet_sd[controlnet_keys[i]].shape)
                    if i < len(sd_keys) and i < len(controlnet_keys)
                    else False
                )
                for i in range(max_len)
            ],
        }
    )

    df.to_csv("keys.csv", index=False)

    # Quick summary
    print(f"Shape mismatches: {(~df['shape_match']).sum()}")
    print(f"Missing in sd: {df['sd_key'].isna().sum()}")
    print(f"Missing in controlnet: {df['controlnet_key'].isna().sum()}")


def test_load_controlnet(controlnet, state_file_path, device="cuda"):
    mapped = torch.load(state_file_path)
    controlnet_sd = controlnet.state_dict()
    not_loaded = [k for k in controlnet_sd.keys() if k not in mapped]
    print(f"Not loaded ({len(not_loaded)} keys):")
    for k in not_loaded:
        print(f"  {k:60s} {tuple(controlnet_sd[k].shape)}")
    missing, unexpected = controlnet.load_state_dict(mapped, strict=False)

    print(f"MISSING (random init): {len(missing)}")
    print(f"UNEXPECTED (dropped): {len(unexpected)}")
    controlnet.to(device)
    controlnet.eval()
    return controlnet


def load_pixcell_controlnet_model(module_name, file_path, checkpoints_folder, device="cuda"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    pixcell_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = pixcell_mod
    spec.loader.exec_module(pixcell_mod)
    PixCellTransformer2DModelControlNet = pixcell_mod.PixCellTransformer2DModelControlNet
    model = PixCellTransformer2DModelControlNet.from_pretrained(
        checkpoints_folder,
        # subfolder="transformer"
    )
    model.to(device)
    model.eval()
    return model


def load_vae(vae_folder, device="cuda"):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        vae_folder,
        local_files_only=True,
        use_safetensors=True,  # Explicitly tell it to look for the safetensors file you have
        trust_remote_code=True,  # Sometimes required if the VAE uses custom scaling
    )
    vae.to(device)
    vae.eval()
    return vae


def initialize_pixcell_controlnet_model(module_name, file_path, checkpoints_folder, device="cuda"):
    import importlib.util
    import sys

    # Standard dynamic import logic
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    pixcell_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = pixcell_mod
    spec.loader.exec_module(pixcell_mod)

    PixCellTransformer2DModelControlNet = pixcell_mod.PixCellTransformer2DModelControlNet

    # 1. Load only the configuration dictionary
    config = PixCellTransformer2DModelControlNet.load_config(checkpoints_folder)

    # 2. Initialize the model with random weights based on that config
    model = PixCellTransformer2DModelControlNet.from_config(config)

    model.to(device)
    model.eval()
    return model


def initialize_controlnet_model(module_name, file_path, checkpoints_folder, device="cuda"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    controlnet_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = controlnet_mod
    spec.loader.exec_module(controlnet_mod)
    PixCellControlNet = controlnet_mod.PixCellControlNet
    config = PixCellControlNet.load_config(checkpoints_folder)
    model = PixCellControlNet.from_config(config)
    model.to(device)
    model.eval()
    return model


def denoise(
    latents,
    uni_embeds,
    controlnet_input_latent,
    scheduler,
    controlnet_model,
    pixcell_controlnet_model,
    guidance_scale=1.5,  # Standard for PixCell/PixArt
    num_inference_steps=20,
    conditioning_scale=1.0,
    device="cuda",
):

    # 1. Prepare Tensors & Dtype (Crucial for VRAM)
    dtype = torch.float16 if device == "cuda" else torch.float32
    latents = latents.to(device, dtype=dtype)
    uni_embeds = uni_embeds.to(device, dtype=dtype)
    controlnet_input_latent = controlnet_input_latent.to(device, dtype=dtype)
    controlnet_model = controlnet_model.to(device, dtype=dtype)
    # 2. Create Unconditional Embeddings
    # Using zeros or random is standard, but must match dtype/device
    uncond_uni_embeds = torch.randn(1, 1, 1, 1536).to(device, dtype=dtype)
    uncond_uni_embeds /= 1536**0.5

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    latent_channels = getattr(pixcell_controlnet_model, "in_channels", 16)

    # 3. Inference Loop with Autocast
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            for t in timesteps:
                # Expand for Classifier-Free Guidance (CFG) batching
                # This runs cond and uncond in ONE pass
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                latent_single = scheduler.scale_model_input(latents, t)
                current_timestep = t.expand(latent_model_input.shape[0])
                # print(f"t={t.item()}: latents mean={latents.mean():.3f} std={latents.std():.3f}")

                # --- ControlNet Pass ---
                # We only need ControlNet for the conditional part (the second half of the batch)
                # but many pipelines prefer passing a batch or zero-filled residuals
                controlnet_outputs = controlnet_model(
                    hidden_states=latent_single,  # Single pass for control signals
                    conditioning=controlnet_input_latent,
                    encoder_hidden_states=uni_embeds,
                    timestep=t.expand(latents.shape[0]),
                    return_dict=False,
                    conditioning_scale=conditioning_scale,
                )[0]
                if 0:  # t == timesteps[0]:
                    print(f"len(controlnet_outputs): {len(controlnet_outputs)}")
                # for i in range(len(controlnet_outputs)):
                #    print(f"controlnet_outputs[{i}] mean: {controlnet_outputs[i].mean():.6f}, std: {controlnet_outputs[i].std():.3f}")
                # asd()
                # controlnet_outputs = controlnet_outputs[:14]
                # --- Transformer Pass (The Memory Hog) ---
                # Concatenate embeds: [uncond, cond]
                # controlnet_outputs = [torch.zeros_like(res) for res in controlnet_outputs]
                batch_embeds = torch.cat([uncond_uni_embeds, uni_embeds])

                # Prepare ControlNet residuals for the batch
                # Uncond gets None/Zeros, Cond gets the outputs
                uncond_residuals = [torch.zeros_like(res) for res in controlnet_outputs]
                batched_residuals = [
                    torch.cat([u, c]) for u, c in zip(uncond_residuals, controlnet_outputs)
                ]
                noise_pred_batch = pixcell_controlnet_model(
                    x=latent_model_input,
                    y=batch_embeds,
                    controlnet_outputs=batched_residuals,
                    timestep=current_timestep,
                    # controlnet_outputs=None,
                    return_dict=False,
                )
                # --- CFG Logic ---
                noise_pred_uncond, noise_pred_cond = noise_pred_batch.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # --- Learned Sigma (Variance) Handling ---
                if pixcell_controlnet_model.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                # --- Step ---
                # ... forward pass ...
                # print(f"  noise_pred mean={noise_pred.mean():.3f} std={noise_pred.std():.3f}")

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # print(f"  after step: mean={latents.mean():.3f} std={latents.std():.3f}")
                # if t == timesteps[2]: break  # just first 3 steps
    return latents


def prepare_controlnet_input(idx):

    latent_shape = (1, 16, 32, 32)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32).to(device)
    latents = latents * scheduler.init_noise_sigma

    uni_embeds = torch.from_numpy(np.load(f"../features_consep/sample_{idx}_uni.npy"))
    # uni_embeds = torch.from_numpy(np.load(f"uni_emb_control.npy"))
    uni_embeds = torch.from_numpy(np.load(f"../data/features_tcga_3660/0_{idx}_uni.npy"))
    uni_embeds = uni_embeds.view(1, 1, 1, 1536).to(device)
    mask_path = "../test_mask.png"
    mask_path = f"../consep_masks/sample_{idx}_mask.png"
    # mask_path = f"../data/tcga_3660_masks/0_{idx}_mask.png"
    controlnet_input = np.asarray(Image.open(mask_path).convert("RGB").resize((256, 256)))
    # controlnet_input = Image.open(mask_path).convert('L')
    # controlnet_input = np.array(controlnet_input)
    # controlnet_input = torch.from_numpy(controlnet_input > 0).float()
    # resize to 256x256
    # import torchvision.transforms as T
    # controlnet_input = T.Resize((256, 256))(controlnet_input)
    # controlnet_input = np.array(Image.open(f"../masks/sample_{idx}_mask.png"))
    # controlnet_input = np.repeat(controlnet_input[..., None], 3, axis=-1)

    controlnet_input_torch = torch.from_numpy(controlnet_input.copy() / 255.0).float().to(device)
    controlnet_input_torch = controlnet_input_torch.permute(2, 0, 1).unsqueeze(0)
    controlnet_input_torch = 2 * (controlnet_input_torch - 0.5)
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    controlnet_input_latent = vae.encode(controlnet_input_torch).latent_dist.mean
    controlnet_input_latent = (controlnet_input_latent - vae_shift) * vae_scale
    # controlnet_input_latent, _ = torch.from_numpy(np.load(f"../features_consep_masks/sample_{idx}_mask_sd3_vae.npy")).chunk(2)
    # controlnet_input_latent = (controlnet_input_latent-vae_shift)*vae_scale

    return latents, uni_embeds, controlnet_input_latent, controlnet_input


def decode_latents(vae, latents, hist_image, mask_image, save_path):
    import cv2
    import matplotlib.pyplot as plt

    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    latents_for_decode = latents.float()

    with torch.no_grad():
        generated_image = vae.decode(
            (latents_for_decode / vae_scale) + vae_shift, return_dict=False
        )[0]
    generated_image = (generated_image / 2 + 0.5).clamp(0, 1)
    generated_image = generated_image.cpu().permute(0, 2, 3, 1).numpy()
    generated_image = (generated_image * 255).round().astype(np.uint8)

    # Apply mask to generated image
    gen_img = generated_image[0].copy()  # (H, W, 3), uint8

    # --- Get contours of each cell in the mask ---
    # mask_image expected shape: (H, W, 3) or (H, W), values 0 or 1
    if mask_image.ndim == 3:
        mask_gray = mask_image[..., 0].astype(np.uint8)  # (H, W), 0 or 1
    else:
        mask_gray = mask_image.astype(np.uint8)

    # Label connected components so each cell gets its own contour
    num_labels, labeled = cv2.connectedComponents(mask_gray)

    contour_overlay = gen_img.copy()
    for label_id in range(1, num_labels):  # skip background (0)
        cell_mask = (labeled == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_overlay, contours, -1, color=(255, 0, 0), thickness=1)
    # --- Plot ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(hist_image)
    ax[0].set_title("Original Image (TCGA)")
    ax[1].imshow(mask_image)
    ax[1].set_title("Mask Image")
    ax[2].imshow(contour_overlay)
    ax[2].set_title("Generated Image")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return gen_img


# %%
if __name__ == "__main__":
    from diffusers import DPMSolverMultistepScheduler

    # %%
    device = "cuda"
    only_init_models = False
    from_checkpoint = True
    pixcell_controlnet_module_name = "pixcell_controlnet_transformer"
    pixcell_controlnet_file_path = (
        "../pretrained_models/pixcell-256-controlnet/transformer/pixcell_controlnet_transformer.py"
    )
    pixcell_controlnet_checkpoints_folder = (
        "../pretrained_models/pixcell-256-controlnet/transformer/"
    )
    controlnet_module_name = "pixcell_controlnet"
    controlnet_file_path = (
        "../pretrained_models/pixcell-256-controlnet/controlnet/pixcell_controlnet.py"
    )
    controlnet_checkpoints_folder = "../pretrained_models/pixcell-256-controlnet/controlnet/"

    config_file_path = "../configs/pan_cancer/config_controlnet_gan.py"
    state_name = "controlnet_epoch_50_step_1050.pth"
    state_file_path = (
        f"../pretrained_models/pixcell-256/transformer/diffusion_pytorch_model.safetensors"
    )
    state_name = "base_model_unfrozen_epoch_700_step_9100.pth"
    state_file_path = f"../checkpoints/pixcell_controlnet_full/checkpoints/{state_name}"
    # pixcell_controlnet_model = load_pixcell_controlnet_model_from_checkpoint(config_file_path, state_file_path, device)
    # base_model = load_base_model_checkpoint(pixcell_controlnet_model, state_file_path)
    # asd()
    # %%
    state_file_path = (
        f"../pretrained_models/pixcell-256/transformer/diffusion_pytorch_model.safetensors"
    )
    pixcell_controlnet_model_base = load_pixcell_controlnet_model_from_checkpoint(
        config_file_path, state_file_path
    )
    state_name = "base_model_unfrozen_epoch_700_step_9100.pth"
    state_file_path = f"../checkpoints/pixcell_controlnet_full/checkpoints/{state_name}"
    pixcell_controlnet_model = copy.deepcopy(pixcell_controlnet_model_base)
    pixcell_controlnet_model_base.to(device)
    # pixcell_controlnet_model = load_base_model_checkpoint(pixcell_controlnet_model, state_file_path)
    pixcell_controlnet_model.to(device)
    # %%
    n_blocks = len(pixcell_controlnet_model.blocks)  # should be 28
    print(f"n_blocks: {n_blocks}")
    for i in range(4):
        block_idx = n_blocks - 4 + i  # 24, 25, 26, 27
        cn_weight = pixcell_controlnet_model.blocks[block_idx].scale_shift_table
        base_weight = pixcell_controlnet_model_base.blocks[block_idx].scale_shift_table
        diff = (cn_weight - base_weight).abs().max().item()
        print(
            f"blocks[{block_idx}]: controlnet_max={cn_weight.abs().max():.6f}, base_max={base_weight.abs().max():.6f}, diff={diff:.6f}"
        )

    # %%
    if only_init_models:
        # pixcell_controlnet_model = initialize_pixcell_controlnet_model(pixcell_controlnet_module_name, pixcell_controlnet_file_path, pixcell_controlnet_checkpoints_folder, device)
        controlnet_model = initialize_controlnet_model(
            controlnet_module_name, controlnet_file_path, controlnet_checkpoints_folder, device
        )
    else:
        # pixcell_controlnet_model = load_pixcell_controlnet_model(pixcell_controlnet_module_name, pixcell_controlnet_file_path, pixcell_controlnet_checkpoints_folder, device)
        if from_checkpoint:
            print("Loading ControlNet from checkpoint")
            config_file_path = "../configs/pan_cancer/config_controlnet_gan.py"
            state_name = "controlnet_epoch_20_step_17940_tcga.pth"
            state_file_path = f"../checkpoints/pixcell_controlnet_full/checkpoints/{state_name}"
            controlnet_model = load_controlnet_model_from_checkpoint(
                config_file_path, state_file_path, device
            )
            print(f"Loaded {state_name}!")
        else:
            print("Loading ControlNet from pretrained model")
            controlnet_model = load_controlnet_model(
                controlnet_module_name, controlnet_file_path, controlnet_checkpoints_folder, device
            )
    # %%
    state_file_path = f"./controlnet_mapped_weights.pt"  # from pretrained
    controlnet_model = test_load_controlnet(controlnet_model, state_file_path, device)

    # %%
    print(
        f"controlnet_blocks[0].weight max: {controlnet_model.controlnet_blocks[0].weight.abs().max():.6f}"
    )
    # If this is 0.0, the model was re-initialized after loading
    with torch.no_grad():
        weight_0 = controlnet_model.controlnet_blocks[0].weight
        has_changed = (weight_0 != 0).any().item()
        max_val = weight_0.abs().max().item()

        print(f"Did weights move from zero? {has_changed}")
        print(f"Absolute max weight value: {max_val:.20f}")
    # asd()
    # %%
    device = "cuda"
    vae = load_vae("../pretrained_models/sd-3.5-vae/vae", device)
    scheduler_folder = "../pretrained_models/pixcell-256/scheduler/"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        scheduler_folder,
    )
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    scheduler.set_timesteps(20, device=device)
    print(type(scheduler))
    print(scheduler.config)
    # %%

    mask_path = "../test_mask.png"
    mask_path = "../consep_masks/sample_0_mask.png"
    controlnet_input = np.asarray(Image.open(mask_path).convert("RGB").resize((256, 256)))
    # controlnet_input = np.array(Image.open(f"../masks/sample_{idx}_mask.png"))
    # controlnet_input = np.repeat(controlnet_input[..., None], 3, axis=-1)
    controlnet_input_torch = torch.from_numpy(controlnet_input.copy() / 255.0).float().to(device)
    controlnet_input_torch = controlnet_input_torch.permute(2, 0, 1).unsqueeze(0)
    controlnet_input_torch = 2 * (controlnet_input_torch - 0.5)
    print(f"controlnet_input_torch.shape: {controlnet_input_torch.shape}")
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    controlnet_input_latent = vae.encode(controlnet_input_torch).latent_dist.mean
    controlnet_input_latent = (controlnet_input_latent - vae_shift) * vae_scale
    print(f"controlnet_input_latent.shape: {controlnet_input_latent.shape}")
    # asd()
    decoded_image = vae.decode(controlnet_input_latent).sample
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).detach().numpy()
    decoded_image = (decoded_image * 255).round().astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(controlnet_input)
    ax[0].set_title("Controlnet Input")
    ax[1].imshow(decoded_image[0])
    ax[1].set_title("Decoded Image")
    plt.show()

    # %%
    idx = 0
    hist_image = Image.open(f"../consep/sample_{idx}.png")
    hist_image = Image.open(f"../consep_masks/sample_{idx}_mask.png")
    latent = np.load(f"../features_consep_masks/sample_{idx}_mask_sd3_vae.npy")
    latent = torch.from_numpy(latent).to(device)[0]
    latent = latent.unsqueeze(0)
    # latent = (latent - vae.config.shift_factor)*vae.config.scaling_factor
    vae.to(latent.device, dtype=latent.dtype)
    # Now decode
    with torch.no_grad():
        decoded_image = vae.decode(latent).sample
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).detach().numpy()
    decoded_image = (decoded_image * 255).round().astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(hist_image)
    ax[0].set_title("Controlnet Input")
    ax[1].imshow(decoded_image[0])
    ax[1].set_title("Decoded Image")
    plt.show()
    # %%
    idx = 300
    latents, uni_embeds, controlnet_input_latent, controlnet_input = prepare_controlnet_input(idx)
    if 0:
        print("UNI shape: ", uni_embeds.shape)
        print(f"UNI mean: {uni_embeds.mean():.6f}")
        print(f"UNI std: {uni_embeds.std():.6f}")
        print(f"UNI min: {uni_embeds.min():.6f}")
        print(f"UNI max: {uni_embeds.max():.6f}")
        print(f"UNI L2 Norm: {torch.norm(uni_embeds, p=2).item()}")
        print("=" * 50)
        print(f"controlnet_input.shape: {controlnet_input.shape}")
        print(f"controlnet_input mean: {controlnet_input.mean():.6f}")
        print(f"controlnet_input std: {controlnet_input.std():.6f}")
        print(f"controlnet_input min: {controlnet_input.min():.6f}")
        print(f"controlnet_input max: {controlnet_input.max():.6f}")
        print(
            f"controlnet_input L2 Norm: {torch.norm(torch.from_numpy(controlnet_input).float(), p=2).item()}"
        )
        print("=" * 50)
        print(f"controlnet_input_latent.shape: {controlnet_input_latent.shape}")
        print(f"controlnet_input_latent mean: {controlnet_input_latent.mean():.6f}")
        print(f"controlnet_input_latent std: {controlnet_input_latent.std():.6f}")
        print(f"controlnet_input_latent min: {controlnet_input_latent.min():.6f}")
        print(f"controlnet_input_latent max: {controlnet_input_latent.max():.6f}")
        print(f"controlnet_input_latent L2 Norm: {torch.norm(controlnet_input_latent, p=2).item()}")
        print("=" * 50)
    # %%
    denoised_latents = denoise(
        latents,
        uni_embeds,
        controlnet_input_latent,
        scheduler,
        controlnet_model,
        pixcell_controlnet_model=pixcell_controlnet_model,
        guidance_scale=2.5,
        num_inference_steps=50,
        conditioning_scale=1.0,
        device="cuda",
    )
    # %%
    hist_image = Image.open(f"../consep/sample_{idx}.png")
    # hist_image = Image.open(f"../test_control_image.png")
    hist_image = cv2.imread(f"../data/tcga_3660/0_{idx}.png")
    hist_image = cv2.cvtColor(hist_image, cv2.COLOR_BGR2RGB)
    hist_image = Image.fromarray(hist_image)
    # mask_image = controlnet_input.cpu().numpy()
    # mask_image = mask_image[0, 0, :, :]
    mask_image = controlnet_input
    generated_image = decode_latents(
        vae, denoised_latents, hist_image, mask_image, "generated_image.png"
    )
    # if generated_image = hist_image
    if np.array_equal(generated_image, hist_image):
        print("Generated image is the same as the original image")
    else:
        print("Generated image is different from the original image")
    # %%
