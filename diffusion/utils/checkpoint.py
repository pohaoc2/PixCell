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
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
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


def load_checkpoint(checkpoint,
                    model,
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
        # Treat the loaded dict as the state_dict directly
        checkpoint_data = {'state_dict': state_dict} 
    else:
        # Standard .pth loading
        checkpoint_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    # 2. Extract state_dict based on format
    # Safetensors are usually just the state_dict itself.
    # .pth files are often nested dictionaries.
    if is_safetensors:
        state_dict = checkpoint_data
    else:
        # Pytorch nested logic
        if load_ema:
            state_dict = checkpoint_data.get('state_dict_ema', checkpoint_data.get('state_dict', checkpoint_data))
        else:
            state_dict = checkpoint_data.get('state_dict', checkpoint_data)

    # 3. Handle Positional Embedding key deletions
    state_dict_keys_to_delete = ['pos_embed', 'base_model.pos_embed', 'model.pos_embed']
    for key in state_dict_keys_to_delete:
        if key in state_dict:
            del state_dict[key]
            # No need to loop through other variations if found
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
    missing, unexpect = model.load_state_dict(state_dict, strict=False)

    # 6. Optional: Load EMA/Optimizer/Scheduler (Usually only in .pth)
    if not is_safetensors:
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
