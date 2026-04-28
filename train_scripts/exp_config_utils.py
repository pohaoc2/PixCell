"""Helpers for resolving paired experimental dataset config."""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_ACTIVE_CHANNELS: list[str] = [
    "cell_masks",
    "cell_type_healthy",
    "cell_type_cancer",
    "cell_type_immune",
    "cell_state_prolif",
    "cell_state_nonprolif",
    "cell_state_dead",
    "vasculature",
    "oxygen",
    "glucose",
]

DEFAULT_DATASET_KWARGS: dict[str, str] = {
    "exp_channels_dir": "exp_channels",
    "features_dir": "features",
    "vae_features_dir": "vae_features",
    "exp_index_h5": "metadata/exp_index.hdf5",
    "vae_prefix": "sd3_vae",
    "ssl_prefix": "uni",
}

_CHANNEL_ALIASES: dict[str, str] = {
    "cell_mask": "cell_masks",
}


def _cfg_get(container, key: str, default=None):
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def canonicalize_exp_channels(channels: list[str]) -> list[str]:
    canonical: list[str] = []
    seen: set[str] = set()
    for channel in channels:
        name = _CHANNEL_ALIASES.get(channel, channel)
        if name not in seen:
            canonical.append(name)
            seen.add(name)
    return canonical


def resolve_exp_active_channels(config) -> list[str]:
    data_cfg = _cfg_get(config, "data", None)
    active_channels = _cfg_get(config, "active_channels", None)
    if active_channels is None:
        active_channels = _cfg_get(data_cfg, "active_channels", DEFAULT_ACTIVE_CHANNELS)
    return canonicalize_exp_channels(list(active_channels))


def resolve_exp_dataset_kwargs(config) -> dict[str, object]:
    data_cfg = _cfg_get(config, "data", None)
    dataset_kwargs: dict[str, object] = {
        "active_channels": resolve_exp_active_channels(config),
    }
    for key, default in DEFAULT_DATASET_KWARGS.items():
        value = _cfg_get(config, key, None)
        if value is None:
            value = _cfg_get(data_cfg, key, default)
        dataset_kwargs[key] = value
    max_train_samples = _cfg_get(config, "max_train_samples", None)
    if max_train_samples is None:
        max_train_samples = _cfg_get(data_cfg, "max_train_samples", None)
    if max_train_samples is not None:
        dataset_kwargs["max_train_samples"] = max_train_samples
    return dataset_kwargs
