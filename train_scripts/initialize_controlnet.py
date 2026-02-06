# %%
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
import torch
import numpy as np
from train_scripts.initialize_models import extract_uni_emb
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download

def _load_pixcell_model(module_name, file_path, checkpoints_folder, device='cuda'):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    pixcell_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = pixcell_mod
    spec.loader.exec_module(pixcell_mod)
    PixCellTransformer2DModel = pixcell_mod.PixCellTransformer2DModel
    model = PixCellTransformer2DModel.from_pretrained(
        checkpoints_folder,
        #subfolder="transformer"
    )
    model.to(device)
    model.eval();
    return model

def _load_controlnet_model(module_name, file_path, checkpoints_folder, device='cuda'):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    controlnet_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = controlnet_mod
    spec.loader.exec_module(controlnet_mod)
    PixCellControlNet = controlnet_mod.PixCellControlNet
    model = PixCellControlNet.from_pretrained(
        checkpoints_folder,
        #subfolder="transformer"
    )
    model.to(device)
    model.eval();
    return model

def _load_pixcell_controlnet_model(module_name, file_path, checkpoints_folder, device='cuda'):
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
def _load_vae(vae_folder, device='cuda'):
    vae = AutoencoderKL.from_pretrained(
        vae_folder,
        local_files_only=True,
        use_safetensors=True,  # Explicitly tell it to look for the safetensors file you have
        trust_remote_code=True # Sometimes required if the VAE uses custom scaling
    )
    vae.to(device)
    vae.eval()
    return vae
import torch

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
                
                current_timestep = t.expand(latent_model_input.shape[0])

                # --- ControlNet Pass ---
                # We only need ControlNet for the conditional part (the second half of the batch)
                # but many pipelines prefer passing a batch or zero-filled residuals
                controlnet_outputs = controlnet_model(
                    hidden_states=latents, # Single pass for control signals
                    conditioning=controlnet_input_latent,
                    encoder_hidden_states=uni_embeds,
                    timestep=t.expand(latents.shape[0]),
                    return_dict=False,
                )[0]

                # --- Transformer Pass (The Memory Hog) ---
                # Concatenate embeds: [uncond, cond]
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


def _prepare_controlnet_input(idx):
    
    latent_shape = (1, 16, 32, 32)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32).to(device)
    latents = latents * scheduler.init_noise_sigma
    uni_embeds = torch.from_numpy(np.load(f"../features/sample_{idx}_uni.npy"))
    uni_embeds = uni_embeds.view(1, 1, 1, 1536).to(device)
    mask_path = hf_hub_download(repo_id="StonyBrook-CVLab/PixCell-256-Cell-ControlNet", filename="test_mask.png")
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
device = 'cuda'
module_name = "pixcell_controlnet_transformer"
file_path = "../pretrained_models/pixcell-256-controlnet/transformer/pixcell_controlnet_transformer.py"
checkpoints_folder = "../pretrained_models/pixcell-256-controlnet/transformer/"
pixcell_controlnet_model = _load_pixcell_controlnet_model(module_name, file_path, checkpoints_folder, device)
module_name = "pixcell_controlnet"
file_path = "../pretrained_models/pixcell-256-controlnet/controlnet/pixcell_controlnet.py"
checkpoints_folder = "../pretrained_models/pixcell-256-controlnet/controlnet/"
controlnet_model = _load_controlnet_model(module_name, file_path, checkpoints_folder, device)

vae = _load_vae("../pretrained_models/sd-3.5-vae/vae", device)
scheduler_folder = "../pretrained_models/pixcell-256/scheduler/"
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    scheduler_folder,
)
# %%
idx = 3
latents, uni_embeds, controlnet_input_latent = _prepare_controlnet_input(idx)
print(f"UNI L2 Norm: {torch.norm(uni_embeds, p=2).item()}")
print(f"Controlnet Input L2 Norm: {torch.norm(controlnet_input_latent, p=2).item()}")
uni_embeds /= uni_embeds.shape[-1] ** 0.5
controlnet_input_latent /= controlnet_input_latent.shape[-1] ** 0.5
print(f"UNI L2 Norm: {torch.norm(uni_embeds, p=2).item()}")
print(f"Controlnet Input L2 Norm: {torch.norm(controlnet_input_latent, p=2).item()}")
# %%
denoised_latents = denoise(latents,
        uni_embeds,
        controlnet_input_latent,
        scheduler,
        controlnet_model,
        pixcell_controlnet_model=pixcell_controlnet_model,
        guidance_scale=2.0,
        num_inference_steps=50,
        device='cuda')
# %%
hist_image = Image.open(f"../tcga_subset_0.1k/sample_{idx}.png")
mask_image = Image.open(f"../masks/sample_{idx}_mask.png")
mask_path = hf_hub_download(repo_id="StonyBrook-CVLab/PixCell-256-Cell-ControlNet", filename="test_mask.png")
mask_image = Image.open(mask_path).convert("RGB")
generated_image = decode_latents(denoised_latents, vae, hist_image, mask_image, "generated_image.png")
# %%
