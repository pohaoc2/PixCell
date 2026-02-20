import os
import re
import torch
from safetensors.torch import load_file as load_safetensors
from diffusion.utils.logger import get_root_logger


def save_checkpoint(work_dir,
                    epoch,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    keep_last=False,
                    step=None,
                    model_type='controlnet',
                    ):
    os.makedirs(work_dir, exist_ok=True)
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict['state_dict_ema'] = model_ema.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['scheduler'] = lr_scheduler.state_dict()
    if epoch is not None:
        state_dict['epoch'] = epoch
        file_path = os.path.join(work_dir, f"{model_type}_epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"
    logger = get_root_logger()
    torch.save(state_dict, file_path)
    logger.info(f'Saved checkpoint of epoch {epoch} to {file_path.format(epoch)}.')
    if keep_last:
        for i in range(epoch):
            previous_ckgt = file_path.format(i)
            if os.path.exists(previous_ckgt):
                os.remove(previous_ckgt)

def save_checkpoint_controlnet(
    accelerator, 
    controlnet, 
    model_ema, 
    optimizer, 
    lr_scheduler, 
    global_step, 
    epoch, 
    config, 
    logger
):
    """
    Save ControlNet checkpoint.
    
    Note: Only saves ControlNet weights, not the frozen base model.
    """
    if not accelerator.is_main_process:
        return
    
    import os
    
    save_dir = os.path.join(config.work_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(save_dir, f'controlnet_epoch_{epoch}_step_{global_step}.pth')
    
    # Prepare checkpoint
    checkpoint = {
        'controlnet_state_dict': accelerator.unwrap_model(controlnet).state_dict(),
        'ema_state_dict': accelerator.unwrap_model(model_ema).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'config': config,
    }
    controlnet.save_config(save_directory=save_dir)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✓ Saved ControlNet checkpoint to {checkpoint_path}")
    logger.info(f"✓ Saved ControlNet config to {os.path.join(save_dir, 'config.json')}")


def remap_pixcell_to_pixart_alpha(state_dict):
    """
    Remap PixCell (Diffusers PixArt-Sigma format) to PixArt-Alpha format
    """
    import torch
    
    new_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # 1. Timestep embedder
        new_key = new_key.replace('adaln_single.emb.timestep_embedder.linear_1', 't_embedder.mlp.0')
        new_key = new_key.replace('adaln_single.emb.timestep_embedder.linear_2', 't_embedder.mlp.2')
        
        # 2. AdaLN linear
        new_key = new_key.replace('adaln_single.linear', 't_block.1')
        
        # 3. Caption projection
        new_key = new_key.replace('caption_projection.linear_1', 'y_embedder.y_proj.fc1')
        new_key = new_key.replace('caption_projection.linear_2', 'y_embedder.y_proj.fc2')
        new_key = new_key.replace('caption_projection.uncond_embedding', 'y_embedder.y_embedding')
        
        # 4. Patch embed
        new_key = new_key.replace('pos_embed.proj', 'x_embedder.proj')
        
        # 5. Transformer blocks
        new_key = new_key.replace('transformer_blocks.', 'blocks.')
        
        # 6. MLP layers (feedforward)
        # ff.net.0.proj -> mlp.fc1
        # ff.net.2 -> mlp.fc2
        new_key = new_key.replace('.ff.net.0.proj', '.mlp.fc1')
        new_key = new_key.replace('.ff.net.2', '.mlp.fc2')
        
        # 7. Scale-shift table in blocks
        # blocks.N.final_layer.scale_shift_table -> blocks.N.scale_shift_table
        import re
        new_key = re.sub(r'(blocks\.\d+)\.final_layer\.scale_shift_table', r'\1.scale_shift_table', new_key)
        
        # 8. Skip attention keys for now - we'll handle them separately
        if '.attn1.' in new_key or '.attn2.' in new_key:
            new_dict[key] = value  # Keep original key temporarily
            continue
        
        # 9. Final layer (at the end of the model, not within blocks)
        new_key = new_key.replace('proj_out', 'final_layer.linear')
        # Don't replace scale_shift_table at model level if already replaced at block level
        if 'blocks.' not in new_key:
            new_key = new_key.replace('scale_shift_table', 'final_layer.scale_shift_table')
        
        new_dict[new_key] = value
    
    # Process attention layers
    final_dict = {}
    processed_keys = set()
    
    for key, value in new_dict.items():
        if key in processed_keys:
            continue
            
        # Handle attn1 (self-attention) - merge to qkv
        if 'attn1.to_q.weight' in key:
            block_prefix = key.replace('transformer_blocks.', 'blocks.').split('.attn1')[0]
            
            q_w = new_dict[key.replace('to_q', 'to_q')]
            k_w = new_dict[key.replace('to_q', 'to_k')]
            v_w = new_dict[key.replace('to_q', 'to_v')]
            
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
            final_dict[f'{block_prefix}.attn.qkv.weight'] = qkv_w
            
            processed_keys.add(key)
            processed_keys.add(key.replace('to_q', 'to_k'))
            processed_keys.add(key.replace('to_q', 'to_v'))
            
        elif 'attn1.to_q.bias' in key:
            block_prefix = key.replace('transformer_blocks.', 'blocks.').split('.attn1')[0]
            
            q_b = new_dict[key.replace('to_q', 'to_q')]
            k_b = new_dict[key.replace('to_q', 'to_k')]
            v_b = new_dict[key.replace('to_q', 'to_v')]
            
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
            final_dict[f'{block_prefix}.attn.qkv.bias'] = qkv_b
            
            processed_keys.add(key)
            processed_keys.add(key.replace('to_q', 'to_k'))
            processed_keys.add(key.replace('to_q', 'to_v'))
            
        elif 'attn1.to_out.0' in key:
            block_prefix = key.replace('transformer_blocks.', 'blocks.').split('.attn1')[0]
            new_key = key.replace('transformer_blocks.', 'blocks.').replace('attn1.to_out.0', 'attn.proj')
            final_dict[new_key] = value
            processed_keys.add(key)
            
        # Handle attn2 (cross-attention) - keep separate as q_linear and kv_linear
        elif 'attn2.to_q' in key:
            new_key = key.replace('transformer_blocks.', 'blocks.').replace('attn2.to_q', 'cross_attn.q_linear')
            final_dict[new_key] = value
            processed_keys.add(key)
            
        elif 'attn2.to_k.weight' in key:
            block_prefix = key.replace('transformer_blocks.', 'blocks.').split('.attn2')[0]
            
            k_w = new_dict[key]
            v_w = new_dict[key.replace('to_k', 'to_v')]
            
            kv_w = torch.cat([k_w, v_w], dim=0)
            final_dict[f'{block_prefix}.cross_attn.kv_linear.weight'] = kv_w
            
            processed_keys.add(key)
            processed_keys.add(key.replace('to_k', 'to_v'))
            
        elif 'attn2.to_k.bias' in key:
            block_prefix = key.replace('transformer_blocks.', 'blocks.').split('.attn2')[0]
            
            k_b = new_dict[key]
            v_b = new_dict[key.replace('to_k', 'to_v')]
            
            kv_b = torch.cat([k_b, v_b], dim=0)
            final_dict[f'{block_prefix}.cross_attn.kv_linear.bias'] = kv_b
            
            processed_keys.add(key)
            processed_keys.add(key.replace('to_k', 'to_v'))
            
        elif 'attn2.to_out.0' in key:
            new_key = key.replace('transformer_blocks.', 'blocks.').replace('attn2.to_out.0', 'cross_attn.proj')
            final_dict[new_key] = value
            processed_keys.add(key)
            
        # Copy non-attention keys
        elif 'attn1' not in key and 'attn2' not in key:
            final_dict[key] = value
            processed_keys.add(key)
    
    return final_dict


def load_checkpoint(checkpoint,
                    model=None,
                    controlnet=None,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    load_ema=False,
                    resume_optimizer=True,
                    resume_lr_scheduler=True,
                    max_length=120,
                    ignore_keys=[]
                    ):
    assert isinstance(checkpoint, str)
    ckpt_file = checkpoint
    is_safetensors = ckpt_file.endswith('.safetensors')

    if is_safetensors:
        print(f"Detected Safetensors format. Loading {ckpt_file}...")
        state_dict = load_safetensors(ckpt_file, device="cpu")
        checkpoint_data = {'state_dict': state_dict}
    else:
        checkpoint_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    # 2. Extract state_dict based on format
    if is_safetensors:
        state_dict = checkpoint_data['state_dict']
        # Remap PixCell (Diffusers) keys to PixArt-Alpha format
        state_dict = remap_pixcell_to_pixart_alpha(state_dict)  # ← ADD THIS
    else:
        if load_ema:
            state_dict = checkpoint_data.get('state_dict_ema', checkpoint_data.get('state_dict', checkpoint_data))
        else:
            state_dict = checkpoint_data.get('state_dict', checkpoint_data)

    # 3. Handle Positional Embedding key deletions
    state_dict_keys_to_delete = ['pos_embed', 'base_model.pos_embed', 'model.pos_embed']
    for key in state_dict_keys_to_delete:
        if key in state_dict:
            del state_dict[key]
            break

    # 4. Handle Ignore Keys
    keys = list(state_dict.keys())
    for key in keys:
        for ignore_key in ignore_keys:
            if key.startswith(ignore_key):
                print(f"Ignore key: {key}")
                if key in state_dict:
                    del state_dict[key]

    # 5. Load into Model
    if model is not None:
        missing, unexpect = model.load_state_dict(state_dict, strict=False)
    else:
        missing, unexpect = [], []

    # 6. Optional: Load EMA/Optimizer/Scheduler (Usually only in .pth)
    if not is_safetensors:
        print("is_safetensors is False")
        if controlnet is not None and 'controlnet_state_dict' in checkpoint_data:
            print("controlnet is not None and 'controlnet_state_dict' in checkpoint_data")
            controlnet.load_state_dict(checkpoint_data['controlnet_state_dict'], strict=False)
        if model_ema is not None and 'state_dict_ema' in checkpoint_data:
            model_ema.load_state_dict(checkpoint_data['state_dict_ema'], strict=False)
        if optimizer is not None and resume_optimizer and 'optimizer' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer'])
        if lr_scheduler is not None and resume_lr_scheduler and 'scheduler' in checkpoint_data:
            lr_scheduler.load_state_dict(checkpoint_data['scheduler'])

    # 7. Logging and Return
    logger = get_root_logger()
    if optimizer is not None and not is_safetensors:
        epoch_match = re.match(r'.*epoch_(\d*).*.pth', ckpt_file)
        epoch = checkpoint_data.get('epoch', epoch_match.group(1) if epoch_match else 0)
        logger.info(f'Resume checkpoint of epoch {epoch} from {ckpt_file}. Load ema: {load_ema}.')
        return epoch, missing, unexpect

    logger.info(f'Load checkpoint from {ckpt_file}. Load ema: {load_ema}.')
    return missing, unexpect
