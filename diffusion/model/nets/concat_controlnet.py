"""A1 concat ControlNet wrappers."""

from diffusion.model.builder import MODELS
from diffusion.model.nets.PixArtControlNet import PixCellControlNet


@MODELS.register_module()
def PixCell_ControlNet_XL_2_UNI_Concat(**kwargs):
    """ControlNet variant that embeds raw 10-channel 256x256 conditioning directly."""
    kwargs.setdefault("controlnet_depth", 27)
    kwargs.setdefault("hidden_size", 1152)
    kwargs.setdefault("patch_size", 2)
    kwargs.setdefault("num_heads", 16)
    kwargs.setdefault("in_channels", 16)
    kwargs.setdefault("caption_channels", 1536)
    kwargs.setdefault("conditioning_channels", 10)
    kwargs.setdefault("conditioning_input_size", 256)
    kwargs.setdefault("conditioning_patch_size", 16)
    return PixCellControlNet(**kwargs)