# Copyright (c) 2024 PixCell Team

import torch
import torch.nn as nn
from typing import Optional, Tuple
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, PatchEmbed
from diffusers.models.controlnets.controlnet import zero_module
from diffusers.models.activations import deprecate, FP32SiLU
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
# PixArt imports for registration and core blocks
from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import (
    t2i_modulate, 
    CaptionEmbedder, 
    AttentionKVCompress, 
    MultiHeadCrossAttention, 
    TimestepEmbedder,
)


def pixcell_get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
    device: Optional[torch.device] = None,
    phase=0,
    output_type: str = "pt",  # Default to 'pt' for torch
):
    """
    Creates 2D sinusoidal positional embeddings.
    
    This is from your existing PixCell codebase - works with torch tensors directly.
    """
    if output_type == "np":
        deprecation_message = (
            "`get_2d_sincos_pos_embed` uses `torch` and supports `device`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='pt' to use the new version now."
        )
        deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
        raise ValueError("Not supported")
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = (
        torch.arange(grid_size[0], device=device, dtype=torch.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size[1], device=device, dtype=torch.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = pixcell_get_2d_sincos_pos_embed_from_grid(embed_dim, grid, phase=phase, output_type=output_type)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.concat([torch.zeros([extra_tokens, embed_dim], device=device), pos_embed], dim=0)
    return pos_embed


def pixcell_get_2d_sincos_pos_embed_from_grid(embed_dim, grid, phase=0, output_type="pt"):
    """
    This function generates 2D sinusoidal positional embeddings from a grid.
    
    From your existing PixCell codebase.
    """
    if output_type == "np":
        deprecation_message = (
            "`get_2d_sincos_pos_embed_from_grid` uses `torch` and supports `device`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='pt' to use the new version now."
        )
        deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
        raise ValueError("Not supported")
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    emb_h = pixcell_get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], phase=phase, output_type=output_type)
    emb_w = pixcell_get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], phase=phase, output_type=output_type)

    emb = torch.concat([emb_h, emb_w], dim=1)
    return emb


def pixcell_get_1d_sincos_pos_embed_from_grid(embed_dim, pos, phase=0, output_type="pt"):
    """
    This function generates 1D positional embeddings from a grid.
    
    From your existing PixCell codebase.
    """
    if output_type == "np":
        deprecation_message = (
            "`get_1d_sincos_pos_embed_from_grid` uses `torch` and supports `device`."
            " `from_numpy` is no longer required."
            "  Pass `output_type='pt' to use the new version now."
        )
        deprecate("output_type=='np'", "0.34.0", deprecation_message, standard_warn=False)
        raise ValueError("Not supported")
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1) + phase
    out = torch.outer(pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.concat([emb_sin, emb_cos], dim=1)
    return emb


class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    Used in ControlNet - these blocks are TRAINABLE copies of the base model blocks.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0, 
                 input_size=None, sampling=None, sr_ratio=1, qk_norm=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionKVCompress(
            hidden_size, num_heads=num_heads, qkv_bias=True, 
            sampling=sampling, sr_ratio=sr_ratio, qk_norm=qk_norm, **block_kwargs
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, 
            hidden_features=int(hidden_size * mlp_ratio), 
            act_layer=approx_gelu, 
            drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        self.sampling = sampling
        self.sr_ratio = sr_ratio

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        
        x = x + self.drop_path(
            gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C)
        )
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(
            gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        
        return x


@MODELS.register_module()
class PixCellControlNet(ModelMixin, ConfigMixin):
    """
    ControlNet for PixCell/PixArt architecture.
    
    Maximally reuses existing functions from:
    - diffusers: PatchEmbed, zero_module
    - PixCell: positional embedding functions
    - PixArt: CaptionEmbedder, TimestepEmbedder, transformer blocks
    
    Architecture:
    1. Copies transformer blocks from base model (TRAINABLE)
    2. Adds zero-initialized output projection per block (TRAINABLE)
    3. Processes conditioning signal (cell masks) along with latents
    
    All parameters in this model are trainable during ControlNet training.
    The base transformer model should be frozen and loaded separately.
    """
    
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1152,
        controlnet_depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.0,
        caption_channels=1536,  # UNI embedding dimension
        pe_interpolation=1.0,
        model_max_length=1,
        qk_norm=False,
        kv_compress_config=None,
        conditioning_channels=16,  # Cell mask conditioning channels
        n_controlnet_blocks=None,  # Optional: use subset of blocks
        config=None,
        **kwargs,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.controlnet_depth = controlnet_depth
        self.hidden_size = hidden_size
        
        # 1. Input embeddings
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # Reusing existing TimestepEmbedder from PixArt
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        
        # Positional embeddings (fixed, not trained)
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        
        # 2. Timestep block
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # 3. Caption (UNI) embedder - reusing existing CaptionEmbedder
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=0.1,  # For classifier-free guidance
            act_layer=approx_gelu,
            token_num=model_max_length
        )
        
        # 4. Conditioning embedder (cell masks) - using diffusers PatchEmbed + zero_module
        self.cond_embedder = PatchEmbed(input_size, patch_size, conditioning_channels, hidden_size, bias=True)
        # Zero-initialize using diffusers' zero_module
        self.cond_embedder = zero_module(self.cond_embedder)
        
        # 5. TRAINABLE Transformer blocks (copied from base model)
        drop_path_list = [x.item() for x in torch.linspace(0, drop_path, controlnet_depth)]
        
        self.kv_compress_config = kv_compress_config or {
            'sampling': None,
            'scale_factor': 1,
            'kv_compress_layer': [],
        }
        
        self.blocks = nn.ModuleList([
            PixArtBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                drop_path=drop_path_list[i],
                input_size=(input_size // patch_size, input_size // patch_size),
                sampling=self.kv_compress_config['sampling'],
                sr_ratio=int(self.kv_compress_config['scale_factor']) 
                    if i in self.kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
            )
            for i in range(controlnet_depth)
        ])
        
        # Optional: use only subset of blocks
        self.n_controlnet_blocks = n_controlnet_blocks or controlnet_depth
        if self.n_controlnet_blocks < controlnet_depth:
            self.blocks = self.blocks[:self.n_controlnet_blocks]
        
        # 6. TRAINABLE ControlNet output blocks (ZERO INITIALIZED using diffusers' zero_module)
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.blocks)):
            controlnet_block = nn.Linear(hidden_size, hidden_size)
            controlnet_block = zero_module(controlnet_block)  # Reusing zero_module
            self.controlnet_blocks.append(controlnet_block)
        
        self.initialize_weights()
        
        if config:
            from diffusion.utils.logger import get_root_logger
            import os
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"PixCellControlNet initialized with {len(self.blocks)} blocks")
            logger.warning(f"position embed interpolation: {self.pe_interpolation}, base size: {self.base_size}")
        else:
            print(f"PixCellControlNet: {len(self.blocks)} trainable blocks")
            print(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def initialize_weights(self):
        """Initialize transformer layers"""
        def _basic_init(module):
                if isinstance(module, nn.Linear):
                    # Check if this linear layer is inside our controlnet_blocks
                    # We don't want to overwrite the zeros!
                    is_controlnet_output = any(module is b for b in self.controlnet_blocks)
                    
                    if not is_controlnet_output:
                        torch.nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Initialize positional embeddings using PixCell's existing function
        pos_embed = pixcell_get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
            interpolation_scale=self.pe_interpolation,
            base_size=self.base_size,
            device=self.pos_embed.device,
            output_type="pt"  # Use torch tensors
        )
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        
        # Initialize caption embedding MLP
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        
        # Zero-out cross-attention projection in transformer blocks
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        
        # ControlNet blocks already zero-initialized using zero_module
        for block in self.controlnet_blocks:
                nn.init.constant_(block.weight, 0)
                nn.init.constant_(block.bias, 0)
    def forward(
        self, 
        hidden_states, 
        conditioning,
        encoder_hidden_states,
        timestep, 
        conditioning_scale=1.0,
        mask=None, 
        data_info=None,
        **kwargs
    ):
        """
        Forward pass of PixCellControlNet.
        
        Args:
            hidden_states: (N, C, H, W) tensor of latent images
            conditioning: (N, C_cond, H, W) tensor of cell mask conditioning
            encoder_hidden_states: (N, 1, 1, D) tensor of UNI embeddings
            timestep: (N,) tensor of diffusion timesteps
            conditioning_scale: float, scale factor for controlnet outputs
            mask: Optional attention mask
            data_info: Optional metadata dict
        
        Returns:
            Tuple of (controlnet_outputs,) where controlnet_outputs is a list of 
            tensors to be added to the base model's transformer blocks.
        """
        # Convert to proper dtype
        hidden_states = hidden_states.to(self.dtype)
        conditioning = conditioning.to(self.dtype)
        timestep = timestep.to(self.dtype)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype)
        
        # Ensure encoder_hidden_states has correct shape
        if len(encoder_hidden_states.shape) == 2:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1).unsqueeze(1)
        if len(encoder_hidden_states.shape) == 3:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        
        pos_embed = self.pos_embed.to(self.dtype)
        
        # 1. Embed inputs with positional embeddings
        x = self.x_embedder(hidden_states) + pos_embed
        
        # Add conditioning (cell masks)
        x = x + self.cond_embedder(conditioning)
        
        # 2. Embed timestep
        t = self.t_embedder(timestep.to(x.dtype))
        t0 = self.t_block(t)
        
        # 3. Embed caption (UNI features)
        y = self.y_embedder(encoder_hidden_states, self.training)
        
        # Handle masking (from PixArt)
        if mask is not None and data_info is not None and set(data_info["mask_type"]) != {"null"}:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        
        # 4. Forward through transformer blocks (TRAINABLE)
        block_outputs = []
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)
            block_outputs.append(x)
        
        # 5. Apply ControlNet blocks (TRAINABLE zero-initialized projections)
        controlnet_outputs = []
        for block_output, controlnet_block in zip(block_outputs, self.controlnet_blocks):
            controlnet_out = controlnet_block(block_output)
            controlnet_outputs.append(controlnet_out * conditioning_scale)
        return (controlnet_outputs,)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @classmethod
    def from_pretrained_base_model(cls, base_model_path, **controlnet_kwargs):
        """
        Initialize ControlNet by copying weights from a pretrained base model.
        
        This is the recommended way to initialize the ControlNet:
        1. Loads base model checkpoint
        2. Creates ControlNet with specified config
        3. Copies transformer block weights from base to ControlNet
        4. Leaves controlnet_blocks zero-initialized
        
        Usage:
            controlnet = PixCellControlNet.from_pretrained_base_model(
                "path/to/base_model.pth",
                conditioning_channels=16,
                n_controlnet_blocks=28
            )
        
        Args:
            base_model_path: Path to pretrained base model checkpoint
            **controlnet_kwargs: Additional arguments for ControlNet config
        
        Returns:
            Initialized ControlNet with copied base model weights
        """
        import os
        
        print(f"Loading base model from: {base_model_path}")
        
        # Load base model checkpoint
        base_checkpoint = torch.load(base_model_path, map_location='cpu')
        
        if 'state_dict' in base_checkpoint:
            base_state_dict = base_checkpoint['state_dict']
        elif 'model' in base_checkpoint:
            base_state_dict = base_checkpoint['model']
        else:
            base_state_dict = base_checkpoint
        
        # Extract config from base model if available
        if 'config' in base_checkpoint:
            base_config = base_checkpoint['config']
            config = {
                'input_size': base_config.get('input_size', 32),
                'patch_size': base_config.get('patch_size', 2),
                'in_channels': base_config.get('in_channels', 16),
                'hidden_size': base_config.get('hidden_size', 1152),
                'depth': base_config.get('depth', 28),
                'num_heads': base_config.get('num_heads', 16),
                'caption_channels': base_config.get('caption_channels', 1536),
            }
        else:
            # Default PixCell config
            config = {
                'input_size': 32,
                'patch_size': 2,
                'in_channels': 16,
                'hidden_size': 1152,
                'depth': 28,
                'num_heads': 16,
                'caption_channels': 1536,
            }
        
        # Update with user-provided kwargs
        config.update(controlnet_kwargs)
        
        # Create ControlNet
        print("Creating ControlNet...")
        controlnet = cls(**config)
        
        # Copy weights from base model
        print("Copying transformer block weights from base model...")
        
        copied_keys = []
        not_found_keys = []
        
        for name, param in controlnet.named_parameters():
            # Try to find corresponding key in base model
            possible_keys = [
                name,
                name.replace('blocks.', 'transformer_blocks.'),
            ]
            
            found = False
            for key in possible_keys:
                if key in base_state_dict:
                    if param.shape == base_state_dict[key].shape:
                        param.data.copy_(base_state_dict[key])
                        copied_keys.append(key)
                        found = True
                        break
            
            if not found and 'controlnet_blocks' not in name and 'cond_embedder' not in name:
                not_found_keys.append(name)
        
        print(f"✓ Copied {len(copied_keys)} parameter tensors from base model")
        
        if not_found_keys:
            print(f"Note: {len(not_found_keys)} parameters not found in base model (expected for ControlNet-specific layers)")
        
        print(f"✓ ControlNet initialized successfully!")
        print(f"  Total parameters: {sum(p.numel() for p in controlnet.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in controlnet.parameters() if p.requires_grad):,}")
        
        return controlnet


# Convenience constructors
@MODELS.register_module()
def PixCell_ControlNet_XL_2_UNI(**kwargs):
    """PixCell ControlNet XL with UNI conditioning"""
    depth = kwargs.get('controlnet_depth', 28)
    return PixCellControlNet(
        controlnet_depth=14, 
        hidden_size=1152, 
        patch_size=2, 
        num_heads=16,
        in_channels=16,
        caption_channels=1536,
        **kwargs
    )