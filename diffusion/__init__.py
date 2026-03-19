# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

"""
Keep package import lightweight.

Historically this module eagerly imported core classes, which pulled in optional
training dependencies (for example mmcv) even when callers only needed dataset
helpers. We expose the same public symbols lazily to avoid those side effects.
"""

__all__ = ["IDDPM", "DPMS", "SASolverSampler"]


def __getattr__(name):
    if name == "IDDPM":
        from .iddpm import IDDPM

        return IDDPM
    if name == "DPMS":
        from .dpm_solver import DPMS

        return DPMS
    if name == "SASolverSampler":
        from .sa_sampler import SASolverSampler

        return SASolverSampler
    raise AttributeError(f"module 'diffusion' has no attribute '{name}'")
