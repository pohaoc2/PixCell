import math
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import t2i_modulate, CaptionEmbedder, AttentionKVCompress, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, CONCHEmbedder
from diffusion.utils.logger import get_root_logger

from diffusion.model.nets.PixArt import PixArtBlock
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


#################################################################################
#                           PixArt ControlNet Implementation                    #
#################################################################################

class PixArtControlNet(nn.Module):
    """Lightweight ControlNet with reduced depth."""
    
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=1,
            hidden_size=1152,
            depth=28,
            controlnet_depth=None,  # NEW: Allow custom depth
            num_heads=16,
            mlp_ratio=4.0,
            drop_path: float = 0.,
            qk_norm=False,
            kv_compress_config=None,
            use_checkpoint=False,  # ← Add this with default
            **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # Use reduced depth for ControlNet if specified
        self.controlnet_depth = controlnet_depth if controlnet_depth is not None else depth
        
        # ControlNet input embedding
        self.control_x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        
        num_patches = self.control_x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        
        # ControlNet blocks - REDUCED DEPTH
        drop_path_list = [x.item() for x in torch.linspace(0, drop_path, self.controlnet_depth)]
        self.kv_compress_config = kv_compress_config or {
            'sampling': None,
            'scale_factor': 1,
            'kv_compress_layer': [],
        }
        
        self.control_blocks = nn.ModuleList([
            PixArtBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_list[i],
                input_size=(input_size // patch_size, input_size // patch_size),
                sampling=self.kv_compress_config['sampling'],
                sr_ratio=int(
                    self.kv_compress_config['scale_factor']
                ) if i in self.kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
            )
            for i in range(self.controlnet_depth)
        ])
        
        # Zero-initialized linear layers - one per FULL depth
        # We'll repeat features for skipped layers
        self.zero_convs = nn.ModuleList([
            nn.Conv2d(hidden_size, hidden_size, 1)  # ✓ Preserves spatial structure
            for _ in range(depth)
        ])
        
        # Mapping: which controlnet block maps to which base block
        self.block_mapping = self._create_block_mapping(depth, self.controlnet_depth)
        
        self.initialize_weights()
    
    def _create_block_mapping(self, full_depth, control_depth):
        """Create mapping from base model blocks to controlnet blocks."""
        if control_depth == full_depth:
            return list(range(full_depth))
        
        # Evenly distribute controlnet blocks across base model depth
        mapping = []
        step = full_depth / control_depth
        for i in range(full_depth):
            control_idx = min(int(i / step), control_depth - 1)
            mapping.append(control_idx)
        return mapping
    
    def forward(self, control_input, y, t0, y_lens):
        """Forward pass with reduced depth."""
        B = control_input.shape[0]
        h = w = int(self.control_x_embedder.num_patches ** 0.5)
        
        x = self.control_x_embedder(control_input) + self.pos_embed
        
        # Process through reduced control blocks
        block_outputs = []
        for block in self.control_blocks:
            if self.use_checkpoint:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens)
            else:
                x = block(x, y, t0, y_lens)
            block_outputs.append(x)
        
        # Create outputs for all base model blocks
        control_features = []
        for i, control_idx in enumerate(self.block_mapping):
            mapped_output = block_outputs[control_idx]  # (B, H*W, C)
            
            # Reshape to spatial: (B, C, H, W)
            spatial = mapped_output.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            
            # Apply zero conv
            gated = self.zero_convs[i](spatial)  # (B, C, H, W)
            
            # Reshape back to sequence: (B, H*W, C)
            gated = gated.permute(0, 2, 3, 1).reshape(B, -1, gated.shape[1])
            
            control_features.append(gated)
        
        return control_features
    
    def initialize_weights(self):
        """Initialize weights following ControlNet paper."""
        # Basic initialization for all modules
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.control_x_embedder.num_patches ** 0.5),
            pe_interpolation=1.0, 
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.control_x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        for zero_conv in self.zero_convs:
            nn.init.constant_(zero_conv.weight, 0)
            nn.init.constant_(zero_conv.bias, 0)
        
        for block in self.control_blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype


@MODELS.register_module()
class PixArt_UNI_ControlNet(nn.Module):
    """
    Combined PixArt model with ControlNet for cell mask conditioning.
    Base model is frozen, ControlNet is trainable.
    """
    
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=16,  # Base model input channels (UNI features)
            control_channels=4,  # ControlNet input channels (cell masks)
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            controlnet_depth=None,
            pred_sigma=True,
            drop_path: float = 0.,
            caption_channels=1536,  # UNI caption channels
            pe_interpolation=1.0,
            config=None,
            model_max_length=120,
            qk_norm=False,
            kv_compress_config=None,
            freeze_base=True,
            **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.freeze_base = freeze_base
        
        # Base Model Components (to be frozen)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Caption embedder
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length
        )
        
        if config and config.get("use_cond_pos_embed", False):
            self.register_buffer("y_pos_embed", torch.zeros(1, model_max_length, caption_channels))
        
        # Base model blocks
        drop_path_list = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.kv_compress_config = kv_compress_config or {
            'sampling': None,
            'scale_factor': 1,
            'kv_compress_layer': [],
        }
        
        self.blocks = nn.ModuleList([
            PixArtBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path_list[i],
                input_size=(input_size // patch_size, input_size // patch_size),
                sampling=self.kv_compress_config['sampling'],
                sr_ratio=int(
                    self.kv_compress_config['scale_factor']
                ) if i in self.kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ])
        
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)
        # Use reduced depth if specified, otherwise use full depth
        self.controlnet_depth = controlnet_depth if controlnet_depth is not None else depth
        
        # ControlNet Branch (trainable)
        self.controlnet = PixArtControlNet(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=control_channels,
            hidden_size=hidden_size,
            depth=depth,
            controlnet_depth=self.controlnet_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            qk_norm=qk_norm,
            kv_compress_config=kv_compress_config,
        )
        
        self.initialize_weights()
        
        # Freeze base model if specified
        if self.freeze_base:
            self._freeze_base_model()
        
        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"PixArt + ControlNet initialized")
            logger.warning(f"Base model frozen: {self.freeze_base}")
            logger.warning(f"ControlNet depth: {self.controlnet_depth}/{depth}")  # Log this
            logger.warning(f"Position embed interpolation: {self.pe_interpolation}, base size: {self.base_size}")
            logger.warning(f"KV compress config: {self.kv_compress_config}")
    
    def _freeze_base_model(self):
        """Freeze all base model parameters."""
        for param in self.x_embedder.parameters():
            param.requires_grad = False
        for param in self.t_embedder.parameters():
            param.requires_grad = False
        for param in self.t_block.parameters():
            param.requires_grad = False
        for param in self.y_embedder.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.final_layer.parameters():
            param.requires_grad = False
        if hasattr(self, 'y_pos_embed'):
            self.y_pos_embed.requires_grad = False
        self.pos_embed.requires_grad = False
    
    def forward(self, x, timestep, y, control_input, mask=None, data_info=None, **kwargs):
        """Forward pass combining base model and ControlNet."""
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        control_input = control_input.to(self.dtype)
        
        # Process conditioning
        if len(y.shape) == 2:
            y = y.unsqueeze(1).unsqueeze(1)
        if len(y.shape) == 3:
            y = y.unsqueeze(1)
        
        original_batch_size = y.shape[0]
        original_token_length = y.shape[2]
        if hasattr(self, 'y_pos_embed'):
            y_pos_embed = self.y_pos_embed.to(self.dtype)
            y = y + y_pos_embed.unsqueeze(0)
        
        # Base model forward
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x) + pos_embed
        t = self.t_embedder(timestep.to(x.dtype))
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)
        
        # Process mask
        if mask is not None and data_info is not None and set(data_info["mask_type"]) != {"null"}:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [original_token_length] * original_batch_size
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        
        # Get ControlNet features
        control_features = self.controlnet(control_input, y, t0, y_lens)
        
        # Forward through base blocks with ControlNet additions
        for i, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)
            # Add ControlNet features
            x = x + control_features[i]
        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x
    
    def forward_with_dpmsolver(self, x, timestep, y, control_input, mask=None, **kwargs):
        """DPM solver forward (no variance prediction)."""
        model_out = self.forward(x, timestep, y, control_input, mask)
        return model_out.chunk(2, dim=1)[0]
    
    def forward_with_cfg(self, x, timestep, y, control_input, cfg_scale, mask=None, **kwargs):
        """Classifier-free guidance forward."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        control_combined = torch.cat([control_input[:len(control_input)//2], 
                                      control_input[:len(control_input)//2]], dim=0)
        model_out = self.forward(combined, timestep, y, control_combined, mask, kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    def unpatchify(self, x):
        """Convert patches back to image."""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def initialize_weights(self):
        """Initialize all weights."""
        # Base model initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
            pe_interpolation=self.pe_interpolation,
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Patch embedding
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        
        # Caption embedding
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        
        # Zero-out cross-attention in base blocks
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        
        # Zero-out final layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        # Y positional embeddings if present
        if hasattr(self, 'y_pos_embed'):
            y_pos_embed = get_2d_sincos_pos_embed(
                self.y_pos_embed.shape[-1],
                int(self.y_pos_embed.shape[1] ** 0.5),
                pe_interpolation=self.pe_interpolation,
                base_size=self.base_size,
                phase=self.base_size // self.y_pos_embed.shape[1]
            )
            self.y_pos_embed.data.copy_(torch.from_numpy(y_pos_embed).float().unsqueeze(0))
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    
    @torch.no_grad()
    def forward_without_controlnet(self, x, timestep, y, mask=None, data_info=None):
        """
        Forward pass WITHOUT ControlNet for testing base model loading.
        Uses zero control features.
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        
        # Process conditioning
        if len(y.shape) == 2:
            y = y.unsqueeze(1).unsqueeze(1)
        if len(y.shape) == 3:
            y = y.unsqueeze(1)
        
        if hasattr(self, 'y_pos_embed'):
            y_pos_embed = self.y_pos_embed.to(self.dtype)
            y = y + y_pos_embed.unsqueeze(0)
        
        # Base model forward
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x) + pos_embed
        t = self.t_embedder(timestep.to(x.dtype))
        t0 = self.t_block(t)
        
        batch_size = y.shape[0]
        seq_length = y.shape[2]
        
        y = self.y_embedder(y, False)
        y_lens = [seq_length] * batch_size
        y = y.squeeze(1).view(1, -1, x.shape[-1])
        
        for i, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)
        
        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

@MODELS.register_module()
def PixArt_XL_2_UNI_ControlNet(**kwargs):
    """Memory-efficient PixArt XL with ControlNet for T4 GPU."""
    kwargs.setdefault('control_channels', 1)
    kwargs.setdefault('controlnet_depth', 14)  # Half depth
    
    return PixArt_UNI_ControlNet(
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16,
        in_channels=16,
        caption_channels=1536,
        freeze_base=True,
        **kwargs
    )