"""Utilities for splitting TME channels into named groups."""
from __future__ import annotations
import torch


def channel_index_map(active_channels: list[str]) -> dict[str, int]:
    """Map channel name to its runtime index in the current config order."""
    return {name: i for i, name in enumerate(active_channels)}


def split_channels_to_groups(
    control_input: torch.Tensor,
    active_channels: list[str],
    channel_groups: list[dict],
) -> dict[str, torch.Tensor]:
    ch_to_idx = channel_index_map(active_channels)
    result = {}
    for group in channel_groups:
        indices = [ch_to_idx[ch] for ch in group["channels"]]
        result[group["name"]] = control_input[:, indices]
    return result
