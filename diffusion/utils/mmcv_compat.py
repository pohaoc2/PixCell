"""Thin wrappers — re-exports from mmcv 1.7.0 under a stable import path."""
from mmcv import Registry, Config, build_from_cfg
from mmcv.runner import (
    get_dist_info,
    LogBuffer,
    build_optimizer,
    OPTIMIZER_BUILDERS,
    DefaultOptimizerConstructor,
    OPTIMIZERS,
)
from mmcv.utils import _BatchNorm, _InstanceNorm
from mmcv.utils.logging import logger_initialized
