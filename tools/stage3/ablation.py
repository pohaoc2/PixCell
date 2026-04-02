"""
Pure helpers for Stage 3 ablation condition planning and labeling.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any


_GROUP_DISPLAY_NAMES = {
    "cell_types": "cell types",
    "cell_state": "cell state",
    "microenv": "nutrient",
}


@dataclass(frozen=True)
class AblationCondition:
    """One ablation condition to render through the shared image generator."""

    label: str
    active_groups: tuple[str, ...]


@dataclass(frozen=True)
class AblationVisSection:
    """One grouped set of ablation conditions and their rendered images."""

    title: str
    conditions: tuple[AblationCondition, ...]
    images: tuple[tuple[str, Any], ...]


def group_display_name(group_name: str) -> str:
    """Human-readable label for a TME group."""
    return _GROUP_DISPLAY_NAMES.get(group_name, group_name.replace("_", " "))


def group_names_from_channel_groups(channel_groups: Sequence[dict]) -> tuple[str, ...]:
    """Extract group names from the config's channel_groups structure."""
    return tuple(group["name"] for group in channel_groups)


def reorder_channel_groups(channel_groups: Sequence[dict], group_order: Sequence[str]) -> list[dict]:
    """Return channel_groups reordered to match the provided group name order."""
    by_name = {group["name"]: group for group in channel_groups}
    return [by_name[name] for name in group_order]


def order_slug(group_order: Sequence[str]) -> str:
    """Filesystem-safe stem for one progressive group order."""
    return "__".join(group_order)


def baseline_condition(zero_mask_latent: bool) -> AblationCondition:
    """Baseline condition used by progressive group ablations."""
    label = "No conditioning\n(zero TME)" if zero_mask_latent else "Mask only\n(no TME groups)"
    return AblationCondition(label=label, active_groups=())


def default_condition_label(active_groups: Iterable[str]) -> str:
    """Fallback label for an arbitrary set of active groups."""
    group_names = tuple(active_groups)
    if not group_names:
        return "No active groups"
    return "Only:\n" + "\n".join(group_display_name(name) for name in group_names)


def build_subset_conditions(
    group_names: Sequence[str],
    subset_size: int,
) -> list[AblationCondition]:
    """All unique size-k subsets of the configured group names."""
    if subset_size < 1:
        raise ValueError(f"subset_size must be >= 1, got {subset_size}")
    if subset_size > len(group_names):
        return []

    return [
        AblationCondition(
            label=default_condition_label(group_combo),
            active_groups=tuple(group_combo),
        )
        for group_combo in combinations(group_names, subset_size)
    ]


def build_subset_ablation_sections(
    group_names: Sequence[str],
    *,
    single_images: Sequence[tuple[str, Any]],
    pair_images: Sequence[tuple[str, Any]],
    triple_images: Sequence[tuple[str, Any]],
    all_four_images: Sequence[tuple[str, Any]] | None = None,
) -> list[AblationVisSection]:
    """Build standard 1/2/3-group sections; optional 4-group (all channels) section."""
    sections: list[AblationVisSection] = [
        AblationVisSection(
            title="1 active group",
            conditions=tuple(build_subset_conditions(group_names, subset_size=1)),
            images=tuple(single_images),
        ),
        AblationVisSection(
            title="2 active groups",
            conditions=tuple(build_subset_conditions(group_names, subset_size=2)),
            images=tuple(pair_images),
        ),
        AblationVisSection(
            title="3 active groups",
            conditions=tuple(build_subset_conditions(group_names, subset_size=3)),
            images=tuple(triple_images),
        ),
    ]
    if all_four_images is not None:
        conds = build_subset_conditions(group_names, subset_size=4)
        if len(conds) != len(all_four_images):
            raise ValueError(
                f"all_four: expected {len(conds)} conditions, got {len(all_four_images)} images"
            )
        sections.append(
            AblationVisSection(
                title="4 active groups",
                conditions=tuple(conds),
                images=tuple(all_four_images),
            )
        )
    return sections


def build_progressive_conditions(
    group_order: Sequence[str],
    *,
    zero_mask_latent: bool,
    include_baseline: bool = True,
) -> list[AblationCondition]:
    """Progressive cumulative additions for one specific group order."""
    conditions: list[AblationCondition] = []
    if include_baseline:
        conditions.append(baseline_condition(zero_mask_latent))

    for n_groups in range(1, len(group_order) + 1):
        active = tuple(group_order[:n_groups])
        conditions.append(
            AblationCondition(
                label="Groups:\n" + "\n".join(group_display_name(name) for name in active),
                active_groups=active,
            )
        )
    return conditions


def build_progressive_order_conditions(
    group_names: Sequence[str],
    *,
    zero_mask_latent: bool,
    include_baseline: bool = True,
) -> list[tuple[tuple[str, ...], list[AblationCondition]]]:
    """All progressive cumulative addition sweeps across every group order."""
    return [
        (
            tuple(group_order),
            build_progressive_conditions(
                group_order,
                zero_mask_latent=zero_mask_latent,
                include_baseline=include_baseline,
            ),
        )
        for group_order in permutations(group_names)
    ]
