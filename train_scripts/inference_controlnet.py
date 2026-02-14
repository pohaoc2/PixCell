# %%
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from diffusion.model.builder import MODELS
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.utils.misc import read_config
from diffusion.model.builder import build_model


def load_controlnet_model(module_name, file_path, checkpoints_folder, device='cuda'):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    controlnet_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = controlnet_mod
    spec.loader.exec_module(controlnet_mod)
    PixCellControlNet = controlnet_mod.PixCellControlNet
    model = PixCellControlNet.from_pretrained(checkpoints_folder)
    model.to(device)
    model.eval();
    return model

def load_pixcell_controlnet_model_from_checkpoint(config_file_path, state_file_path, device='cuda'):
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
        "conditioning_channels": config.controlnet_conditioning_channels,  # e.g., 16 for cell masks
        "n_controlnet_blocks": getattr(config, 'n_controlnet_blocks', None),
        **config.get('pixcell_controlnet_model_kwargs', {})
    }
    pixcell_controlnet = build_model(
        config.base_model,  # e.g., 'PixCell_Transformer_XL_2_UNI'
        config.grad_checkpointing,
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=False,  # ControlNet doesn't predict sigma
        pred_sigma=False,
        **pixcell_controlnet_model_kwargs
    )
    _ = load_checkpoint(state_file_path, model=pixcell_controlnet)
    pixcell_controlnet.to(device)
    pixcell_controlnet.eval();
    return pixcell_controlnet

def load_controlnet_model_from_checkpoint(config_file_path, state_file_path, device='cuda'):
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
        "n_controlnet_blocks": getattr(config, 'n_controlnet_blocks', None),
        **config.get('controlnet_model_kwargs', {})
    }
    
    # Build ControlNet (all parameters trainable)
    controlnet = build_model(
        config.controlnet_model,  # e.g., 'PixCell_ControlNet_XL_2_UNI'
        config.grad_checkpointing,
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=False,  # ControlNet doesn't predict sigma
        pred_sigma=False,
        **controlnet_model_kwargs
    )
    _ = load_checkpoint(state_file_path, controlnet=controlnet)
    controlnet.to(device)
    controlnet.eval();
    return controlnet

def load_pixcell_controlnet_model(module_name, file_path, checkpoints_folder, device='cuda'):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    pixcell_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = pixcell_mod
    spec.loader.exec_module(pixcell_mod)
    PixCellTransformer2DModelControlNet = pixcell_mod.PixCellTransformer2DModelControlNet
    model = PixCellTransformer2DModelControlNet.from_pretrained(
        checkpoints_folder,
        #subfolder="transformer"
    )
    model.to(device)
    model.eval();
    return model
def load_vae(vae_folder, device='cuda'):
    vae = AutoencoderKL.from_pretrained(
        vae_folder,
        local_files_only=True,
        use_safetensors=True,  # Explicitly tell it to look for the safetensors file you have
        trust_remote_code=True # Sometimes required if the VAE uses custom scaling
    )
    vae.to(device)
    vae.eval()
    return vae

def initialize_pixcell_controlnet_model(module_name, file_path, checkpoints_folder, device='cuda'):
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

def initialize_controlnet_model(module_name, file_path, checkpoints_folder, device='cuda'):
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

def denoise(latents,
            uni_embeds,
            controlnet_input_latent,
            scheduler,
            controlnet_model,
            pixcell_controlnet_model,
            guidance_scale=1.5, # Standard for PixCell/PixArt
            num_inference_steps=20,
            device='cuda'):
    
    # 1. Prepare Tensors & Dtype (Crucial for VRAM)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    latents = latents.to(device, dtype=dtype)
    uni_embeds = uni_embeds.to(device, dtype=dtype)
    controlnet_input_latent = controlnet_input_latent.to(device, dtype=dtype)
    
    # 2. Create Unconditional Embeddings
    # Using zeros or random is standard, but must match dtype/device
    uncond_uni_embeds = torch.randn(1, 1, 1, 1536).to(device, dtype=dtype)
    uncond_uni_embeds /= 1536 ** 0.5
    
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    latent_channels = pixcell_controlnet_model.config.in_channels

    # 3. Inference Loop with Autocast
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=(device=='cuda')):
            for t in timesteps:
                # Expand for Classifier-Free Guidance (CFG) batching
                # This runs cond and uncond in ONE pass
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                latent_single = scheduler.scale_model_input(latents, t)
                current_timestep = t.expand(latent_model_input.shape[0])

                # --- ControlNet Pass ---
                # We only need ControlNet for the conditional part (the second half of the batch)
                # but many pipelines prefer passing a batch or zero-filled residuals
                controlnet_outputs = controlnet_model(
                    hidden_states=latent_single, # Single pass for control signals
                    conditioning=controlnet_input_latent,
                    encoder_hidden_states=uni_embeds,
                    timestep=t.expand(latents.shape[0]),
                    return_dict=False,
                    conditioning_scale=1.0,
                )[0]
                # --- Transformer Pass (The Memory Hog) ---
                # Concatenate embeds: [uncond, cond]
                #controlnet_outputs = [torch.zeros_like(res) for res in controlnet_outputs]
                batch_embeds = torch.cat([uncond_uni_embeds, uni_embeds])
                
                # Prepare ControlNet residuals for the batch
                # Uncond gets None/Zeros, Cond gets the outputs
                uncond_residuals = [torch.zeros_like(res) for res in controlnet_outputs]
                batched_residuals = [torch.cat([u, c]) for u, c in zip(uncond_residuals, controlnet_outputs)]
                noise_pred_batch = pixcell_controlnet_model(
                    latent_model_input,
                    encoder_hidden_states=batch_embeds,
                    controlnet_outputs=batched_residuals,
                    timestep=current_timestep,
                    return_dict=False,
                )[0]

                # --- CFG Logic ---
                noise_pred_uncond, noise_pred_cond = noise_pred_batch.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # --- Learned Sigma (Variance) Handling ---
                if pixcell_controlnet_model.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # --- Step ---
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


def prepare_controlnet_input(idx):
    
    latent_shape = (1, 16, 32, 32)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32).to(device)
    latents = latents * scheduler.init_noise_sigma
    
    uni_embeds = torch.from_numpy(np.load(f"uni_emb_control.npy"))
    uni_embeds = torch.from_numpy(np.load(f"../features_consep/sample_{idx}_uni.npy"))
    uni_embeds = uni_embeds.view(1, 1, 1, 1536).to(device)
    mask_path = "../test_mask.png"
    controlnet_input = np.asarray(Image.open(mask_path).convert("RGB"))
    #controlnet_input = np.array(Image.open(f"../masks/sample_{idx}_mask.png"))
    #controlnet_input = np.repeat(controlnet_input[..., None], 3, axis=-1)
    controlnet_input_torch = torch.from_numpy(controlnet_input.copy()/255.).float().to(device)
    controlnet_input_torch = controlnet_input_torch.permute(2, 0, 1).unsqueeze(0)
    controlnet_input_torch = 2 * (controlnet_input_torch - 0.5)
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    controlnet_input_latent = vae.encode(controlnet_input_torch).latent_dist.mean
    #controlnet_input_latent = (controlnet_input_latent-vae_shift)*vae_scale
    return latents, uni_embeds, controlnet_input_latent

def decode_latents(latents, vae, hist_image, mask_image, save_path):
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    latents_for_decode = latents.float()

    with torch.no_grad():
        generated_image = vae.decode(
            (latents_for_decode / vae_scale) + vae_shift,
            return_dict=False
        )[0]
    generated_image = (generated_image / 2 + 0.5).clamp(0, 1)
    generated_image = generated_image.cpu().permute(0, 2, 3, 1).numpy()
    generated_image = (generated_image * 255).round().astype(np.uint8)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(hist_image)
    ax[0].set_title("Original Image")
    ax[1].imshow(mask_image)
    ax[1].set_title("Mask Image")
    ax[2].imshow(generated_image[0])
    ax[2].set_title("Generated Image")

    plt.savefig(save_path)
    plt.show()
    return generated_image[0]
# %%
if __name__ == "__main__":
    # %%
    device = 'cuda'
    only_init_models = False
    from_checkpoint = True
    pixcell_controlnet_module_name = "pixcell_controlnet_transformer"
    pixcell_controlnet_file_path = "../pretrained_models/pixcell-256-controlnet/transformer/pixcell_controlnet_transformer.py"
    pixcell_controlnet_checkpoints_folder = "../pretrained_models/pixcell-256-controlnet/transformer/"
    controlnet_module_name = "pixcell_controlnet"
    controlnet_file_path = "../pretrained_models/pixcell-256-controlnet/controlnet/pixcell_controlnet.py"
    controlnet_checkpoints_folder = "../pretrained_models/pixcell-256-controlnet/controlnet/"
    if only_init_models:
        pixcell_controlnet_model = initialize_pixcell_controlnet_model(pixcell_controlnet_module_name, pixcell_controlnet_file_path, pixcell_controlnet_checkpoints_folder, device)
        controlnet_model = initialize_controlnet_model(controlnet_module_name, controlnet_file_path, controlnet_checkpoints_folder, device)
    else:
        pixcell_controlnet_model = load_pixcell_controlnet_model(pixcell_controlnet_module_name, pixcell_controlnet_file_path, pixcell_controlnet_checkpoints_folder, device)
        if from_checkpoint:
            print("Loading ControlNet from checkpoint")
            config_file_path = '../configs/pan_cancer/config_controlnet_gan.py'
            state_name = 'controlnet_epoch_1_step_32.pth'
            state_file_path = f'../checkpoints/pixcell_controlnet_full/checkpoints/{state_name}'
            controlnet_model = load_controlnet_model_from_checkpoint(config_file_path, state_file_path, device)
            print(f"Loaded {state_name}!")
        else:
            print("Loading ControlNet from pretrained model")
            controlnet_model = load_controlnet_model(controlnet_module_name, controlnet_file_path, controlnet_checkpoints_folder, device)
    # %%
    with torch.no_grad():
        weight_0 = controlnet_model.controlnet_blocks[0].weight
        has_changed = (weight_0 != 0).any().item()
        max_val = weight_0.abs().max().item()
        
        print(f"Did weights move from zero? {has_changed}")
        print(f"Absolute max weight value: {max_val:.20f}")
    #asd()
    # %%
    device = 'cuda'
    vae = load_vae("../pretrained_models/sd-3.5-vae/vae", device)
    scheduler_folder = "../pretrained_models/pixcell-256/scheduler/"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        scheduler_folder,
    )
    # %%
    
    mask_path = "../test_mask.png"
    mask_path = "../consep_masks/sample_0_mask.png"
    controlnet_input = np.asarray(Image.open(mask_path).convert("RGB"))
    #controlnet_input = np.array(Image.open(f"../masks/sample_{idx}_mask.png"))
    #controlnet_input = np.repeat(controlnet_input[..., None], 3, axis=-1)
    controlnet_input_torch = torch.from_numpy(controlnet_input.copy()/255.).float().to(device)
    controlnet_input_torch = controlnet_input_torch.permute(2, 0, 1).unsqueeze(0)
    controlnet_input_torch = 2 * (controlnet_input_torch - 0.5)
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0)
    controlnet_input_latent = vae.encode(controlnet_input_torch).latent_dist.mean
    controlnet_input_latent = (controlnet_input_latent-vae_shift)*vae_scale

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
    #latent = (latent - vae.config.shift_factor)*vae.config.scaling_factor
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
    idx = 10
    latents, uni_embeds, controlnet_input_latent = prepare_controlnet_input(idx)
    print(f"UNI L2 Norm: {torch.norm(uni_embeds, p=2).item()}")
    print(f"Controlnet Input L2 Norm: {torch.norm(controlnet_input_latent, p=2).item()}")
    #uni_embeds /= uni_embeds.shape[-1] ** 0.5
    #controlnet_input_latent /= controlnet_input_latent.shape[-1] ** 0.5
    #print(f"Normalized UNI L2 Norm: {torch.norm(uni_embeds, p=2).item()}")
    #print(f"Normalized Controlnet Input L2 Norm: {torch.norm(controlnet_input_latent, p=2).item()}")
    print("UNI shape: ", uni_embeds.shape)
    # %%
    denoised_latents = denoise(latents,
            uni_embeds,
            controlnet_input_latent,
            scheduler,
            controlnet_model,
            pixcell_controlnet_model=pixcell_controlnet_model,
            guidance_scale=1.5,
            num_inference_steps=50,
            device='cuda')
    # %%
    
    hist_image = Image.open(f"../test_control_image.png")
    hist_image = Image.open(f"../consep/sample_{idx}.png")
    mask_image = Image.open(f"../masks/sample_{idx}_mask.png")
    mask_path = "../test_mask.png"
    mask_image = Image.open(mask_path).convert("RGB")
    generated_image = decode_latents(denoised_latents, vae, hist_image, mask_image, "generated_image.png")
    # %%
