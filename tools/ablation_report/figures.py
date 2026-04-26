from __future__ import annotations

import base64
import io
import textwrap
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .data import load_rgb_pil_local
from .shared import (
    CHANNEL_EFFECT_CMAP,
    FOUR_GROUP_ORDER,
    GROUP_LABELS,
    GROUP_SHORT_LABELS,
    INK,
    METRIC_LABELS,
    METRIC_SPEC_BY_KEY,
    OKABE_GRAY,
    SOFT_GRID,
    TRADEOFF_HIGHER_IS_BETTER,
    TRADEOFF_REFERENCE_BANDS,
    TRADEOFF_METRIC_ORDER,
    DatasetSummary,
    _format_mean_sd,
    _metric_caption,
    _ranked_best_worst_selection,
    _ranked_labels,
    _reference_band_limits,
    _tight_range,
    comparison_metric_keys,
    condition_indicator_text,
    humanize_token,
    metric_tradeoff_keys,
    plain_takeaway_text,
    plt,
    np,
)


def figure_to_data_uri(fig: plt.Figure, *, dpi: int = 180) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_figure_png(fig: plt.Figure, output_path: Path, *, dpi: int = 220) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _comparison_metric_order(summaries: list[DatasetSummary]) -> list[str]:
    metrics = [
        metric
        for metric in TRADEOFF_METRIC_ORDER
        if any(metric in comparison_metric_keys(summary) for summary in summaries)
    ]
    if metrics:
        return metrics
    for summary in summaries:
        summary_metrics = comparison_metric_keys(summary)
        if summary_metrics:
            return summary_metrics
    return []


def _render_condition_glyph_axes(ax: plt.Axes, condition: object, *, center_x: float, center_y: float, dx: float) -> None:
    active_groups = set()
    for token in str(condition).split("+"):
        normalized = token.strip()
        if normalized in FOUR_GROUP_ORDER:
            active_groups.add(normalized)
    xs = [center_x + dx * offset for offset in (-1.5, -0.5, 0.5, 1.5)]
    ys = [center_y] * len(FOUR_GROUP_ORDER)
    facecolors = [INK if group in active_groups else "white" for group in FOUR_GROUP_ORDER]
    edgecolors = [INK if group in active_groups else "#A0A0A0" for group in FOUR_GROUP_ORDER]
    linewidths = [0.9 if group in active_groups else 1.0 for group in FOUR_GROUP_ORDER]
    sizes = [22 if group in active_groups else 20 for group in FOUR_GROUP_ORDER]
    ax.scatter(
        xs,
        ys,
        s=sizes,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        transform=ax.transAxes,
        clip_on=False,
        zorder=3,
    )


def _draw_comparison_metric_panel(ax: plt.Axes, summary: DatasetSummary, metric_key: str) -> None:
    best_entries, worst_entries, total = _ranked_best_worst_selection(summary.best_worst.get(metric_key))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    caption = _metric_caption(metric_key)
    ax.text(
        0.5,
        0.975,
        caption,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="normal",
        color=INK,
        family="DejaVu Serif",
    )

    if not best_entries and not worst_entries:
        ax.text(0.5, 0.55, "No ranked entries", ha="center", va="center", fontsize=8.0, color=OKABE_GRAY)
        return

    top_labels = _ranked_labels(total, len(best_entries))
    bottom_labels = _ranked_labels(total, len(worst_entries), tail=True)
    rows: list[tuple[str, object]] = []
    rows.extend((rank_label, entry) for rank_label, entry in zip(top_labels, best_entries))
    if best_entries and worst_entries:
        rows.append(("···", None))
    rows.extend((rank_label, entry) for rank_label, entry in zip(bottom_labels, worst_entries))

    x_rank = 0.09
    x_cond = 0.42
    x_value = 0.985
    top_line_y = 0.88
    header_y = 0.835
    header_bottom_y = 0.80
    bottom_line_y = 0.08

    ax.plot([0.0, 1.0], [top_line_y, top_line_y], color=INK, linewidth=1.1, transform=ax.transAxes, clip_on=False)
    ax.plot([0.0, 1.0], [header_bottom_y, header_bottom_y], color=INK, linewidth=0.8, transform=ax.transAxes, clip_on=False)
    ax.text(x_rank, header_y, "Rank", ha="center", va="center", fontsize=8, color=INK, family="DejaVu Serif")
    ax.text(
        x_cond,
        header_y,
        "CT/CS/VAS/NUC",
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
        family="DejaVu Serif",
    )
    ax.text(
        x_value,
        header_y,
        "Mean" if metric_key == "fud" else "Mean ± SD",
        ha="right",
        va="center",
        fontsize=8,
        color=INK,
        family="DejaVu Serif",
    )

    available_h = header_bottom_y - bottom_line_y
    row_h = available_h / max(len(rows), 1)
    for index, (rank_label, entry) in enumerate(rows):
        y = header_bottom_y - (index + 0.5) * row_h
        if entry is None:
            sep_y = header_bottom_y - index * row_h
            ax.plot([0.0, 1.0], [sep_y, sep_y], color=INK, linewidth=0.7, linestyle=(0, (2, 2)), transform=ax.transAxes)
            ax.plot([0.0, 1.0], [sep_y - row_h, sep_y - row_h], color=INK, linewidth=0.7, linestyle=(0, (2, 2)), transform=ax.transAxes)
            ax.text(x_rank, y, rank_label, ha="center", va="center", fontsize=8.5, color=INK, family="DejaVu Serif")
            continue

        condition, mean_value, sd_value = entry
        ax.text(x_rank, y, rank_label, ha="center", va="center", fontsize=8.5, color="#555555", family="DejaVu Serif")
        _render_condition_glyph_axes(ax, condition, center_x=x_cond, center_y=y, dx=0.055)
        value_text = f"{float(mean_value):.3f}" if metric_key == "fud" else _format_mean_sd(mean_value, sd_value)
        ax.text(x_value, y, value_text, ha="right", va="center", fontsize=8, color=INK, family="DejaVu Serif")

    ax.plot([0.0, 1.0], [bottom_line_y, bottom_line_y], color=INK, linewidth=1.1, transform=ax.transAxes, clip_on=False)


def build_metric_trends_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    metrics = metric_tradeoff_keys(summaries)
    n_cols = 5
    n_rows = 1
    fig = plt.figure(figsize=(20.0, 5.5))
    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.22, hspace=0.28)

    from tools.stage3.ablation_vis_utils import condition_metric_key, ordered_subset_condition_tuples

    condition_tuples = ordered_subset_condition_tuples()
    condition_keys = [condition_metric_key(cond) for cond in condition_tuples]
    x_positions = list(range(len(condition_keys)))
    spans: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(condition_tuples)):
        if len(condition_tuples[index]) != len(condition_tuples[index - 1]):
            spans.append((len(condition_tuples[index - 1]), start, index - 1))
            start = index
    spans.append((len(condition_tuples[-1]), start, len(condition_tuples) - 1))
    dataset_styles = {
        "paired": {"marker": "^", "markerfacecolor": "white", "markeredgecolor": INK, "error_linestyle": "solid"},
        "unpaired": {"marker": "s", "markerfacecolor": "white", "markeredgecolor": INK, "error_linestyle": (0, (4, 2))},
    }
    x_offsets = {"paired": -0.12, "unpaired": 0.12}

    for index, metric_key in enumerate(metrics):
        row, col = divmod(index, n_cols)
        subgrid = outer[row, col].subgridspec(2, 1, height_ratios=[7.0, 2.0], hspace=0.02)
        ax = fig.add_subplot(subgrid[0, 0])
        dot_ax = fig.add_subplot(subgrid[1, 0], sharex=ax)
        all_values: list[float] = []
        for summary in summaries:
            valid: list[tuple[int, float, float]] = []
            for x_value, cond_key in zip(x_positions, condition_keys, strict=True):
                stats = summary.condition_stats.get(cond_key, {}).get(metric_key)
                if stats is None:
                    continue
                valid.append((x_value, float(stats[0]), float(stats[1])))
            if not valid:
                continue
            xs, ys, stds = zip(*valid)
            for value, std_value in zip(ys, stds, strict=True):
                all_values.extend([value - std_value, value + std_value])
            style = dataset_styles[summary.slug]
            shifted_xs = [x + x_offsets[summary.slug] for x in xs]
            container = ax.errorbar(
                shifted_xs,
                ys,
                yerr=stds,
                color=INK,
                linestyle="none",
                marker=style["marker"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=1.1,
                linewidth=1.6,
                markersize=8.0,
                capsize=2.0,
                elinewidth=0.9,
                zorder=4,
            )
            _, _, barlinecols = container.lines
            for barlinecol in barlinecols:
                barlinecol.set_linestyle(style["error_linestyle"])

        reference = TRADEOFF_REFERENCE_BANDS.get(metric_key)
        if reference is not None:
            mean = float(reference["mean"])
            std = reference.get("std")
            all_values.append(mean)
            if std is not None:
                all_values.extend([mean - float(std), mean + float(std)])

        if not all_values:
            ax.axis("off")
            dot_ax.axis("off")
            continue

        lo, hi = _tight_range(all_values)
        spec = METRIC_SPEC_BY_KEY.get(metric_key)
        higher_is_better = spec.higher_is_better if spec is not None else TRADEOFF_HIGHER_IS_BETTER.get(metric_key, True)
        arrow = "↑" if higher_is_better else "↓"
        ax.set_title(
            f"{METRIC_LABELS.get(metric_key, metric_key)} ({arrow})",
            color=INK,
            fontsize=18.5,
            fontweight="normal",
            pad=4,
        )
        for _, _, end_idx in spans[:-1]:
            ax.axvline(end_idx + 0.5, color="#BEBEBE", linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=1)
            dot_ax.axvline(end_idx + 0.5, color="#D0D0D0", linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=1)
        ax.set_xlim(-0.55, len(condition_keys) - 0.45)
        ax.set_xticks([])
        ax.set_ylim(lo, hi)
        band_limits = _reference_band_limits(metric_key)
        if band_limits is not None and reference is not None:
            band_lo, band_hi = band_limits
            band = ax.axhspan(band_lo, band_hi, color="#A8A8A8", alpha=0.22, zorder=0)
            band.set_hatch("////")
            band.set_edgecolor("#8A8A8A")
            band.set_linewidth(0.0)
            y_span = hi - lo
            label_y = min(hi - 0.03 * y_span, band_hi + 0.04 * y_span)
            ax.text(
                0.02,
                label_y,
                str(reference["label"]),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=16.5,
                color="#666666",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
                zorder=5,
            )
        elif reference is not None:
            mean = float(reference["mean"])
            ax.axhline(mean, color="#8A8A8A", linewidth=1.1, linestyle=(0, (5, 2.5)), zorder=1)
            y_span = hi - lo
            label_y = min(hi - 0.03 * y_span, mean + 0.035 * y_span)
            ax.text(
                0.02,
                label_y,
                str(reference["label"]),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=16.5,
                color="#666666",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
                zorder=5,
            )
        ax.grid(True, axis="y", color=SOFT_GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(INK)
        ax.spines["left"].set_color(INK)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", colors=INK, labelsize=17.5)
        ax.set_axisbelow(True)
        dot_ax.set_facecolor("white")
        for x_value, cond in zip(x_positions, condition_tuples, strict=True):
            active_groups = set(cond)
            for row_index, group in enumerate(FOUR_GROUP_ORDER):
                y_value = 3 - row_index
                is_active = group in active_groups
                dot_ax.scatter(
                    x_value,
                    y_value,
                    s=24 if is_active else 21,
                    facecolors=INK if is_active else "white",
                    edgecolors=INK if is_active else "#A0A0A0",
                    linewidths=0.9 if is_active else 1.0,
                    zorder=3,
                )
        dot_ax.set_xlim(-0.55, len(condition_keys) - 0.45)
        dot_ax.set_ylim(-0.6, 3.6)
        dot_ax.axis("off")
        if col == 0:
            ymin, ymax = dot_ax.get_ylim()
            for row_index, group in enumerate(FOUR_GROUP_ORDER):
                y_value = 3 - row_index
                y_axes = (y_value - ymin) / (ymax - ymin)
                dot_ax.text(
                    -0.05,
                    y_axes,
                    GROUP_SHORT_LABELS[group],
                    transform=dot_ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=17.0,
                    color=INK,
                )

    for index in range(len(metrics), n_rows * n_cols):
        row, col = divmod(index, n_cols)
        subgrid = outer[row, col].subgridspec(2, 1, height_ratios=[7.0, 2.0], hspace=0.02)
        fig.add_subplot(subgrid[0, 0]).axis("off")
        fig.add_subplot(subgrid[1, 0]).axis("off")

    handles = [
        Line2D([0], [0], color=INK, linestyle="None", marker="^", markerfacecolor="white", markeredgecolor=INK, markersize=8.0, linewidth=1.6, label="Paired"),
        Line2D([0], [0], color=INK, linestyle="None", marker="s", markerfacecolor="white", markeredgecolor=INK, markersize=8.0, linewidth=1.6, label="Unpaired"),
        Patch(facecolor="#A8A8A8", edgecolor="#8A8A8A", hatch="////", alpha=0.22, label="Benchmark band (mean ± SD)"),
        Line2D([0], [0], color="#8A8A8A", linestyle=(0, (5, 2.5)), linewidth=1.1, label="Benchmark line"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=4, frameon=False, bbox_to_anchor=(0.99, 0.04), fontsize=13.5)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.14, top=0.95)
    return fig


def heatmap_metric_keys(summary: DatasetSummary) -> list[str]:
    present = set(summary.metric_keys)
    return [metric for metric in TRADEOFF_METRIC_ORDER if metric in present]


def _shared_heatmap_metric_keys(summaries: list[DatasetSummary]) -> list[str]:
    if not summaries:
        return []
    shared = set(heatmap_metric_keys(summaries[0]))
    for summary in summaries[1:]:
        shared &= set(heatmap_metric_keys(summary))
    return [metric for metric in TRADEOFF_METRIC_ORDER if metric in shared]


def _heatmap_matrix(
    summary: DatasetSummary,
    source: dict[str, dict[str, tuple[float, float]]],
    metric_keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(FOUR_GROUP_ORDER), len(metric_keys)), np.nan, dtype=float)
    stds = np.full((len(FOUR_GROUP_ORDER), len(metric_keys)), np.nan, dtype=float)
    for row, group in enumerate(FOUR_GROUP_ORDER):
        for col, metric_key in enumerate(metric_keys):
            value = source.get(group, {}).get(metric_key)
            if value is not None:
                matrix[row, col] = float(value[0])
                stds[row, col] = float(value[1])
    return matrix, stds


def _heatmap_metric_scales(summaries: list[DatasetSummary]) -> dict[str, float]:
    per_metric_values: dict[str, list[float]] = {}
    for summary in summaries:
        metric_keys = heatmap_metric_keys(summary)
        matrix, _ = _heatmap_matrix(summary, summary.added_effect_stats, metric_keys)
        for col, metric_key in enumerate(metric_keys):
            values = matrix[:, col]
            finite_values = [float(value) for value in values[np.isfinite(values)]]
            if finite_values:
                per_metric_values.setdefault(metric_key, []).extend(finite_values)

    scales: dict[str, float] = {}
    for metric_key, values in per_metric_values.items():
        scales[metric_key] = max(abs(min(values)), abs(max(values)))
    return scales


def _normalize_heatmap_matrix(
    matrix: np.ndarray,
    metric_keys: list[str],
    metric_scales: dict[str, float],
) -> np.ndarray:
    normalized = np.full(matrix.shape, np.nan, dtype=float)
    for col, metric_key in enumerate(metric_keys):
        scale = metric_scales.get(metric_key, 0.0)
        column = matrix[:, col]
        finite_mask = np.isfinite(column)
        if not np.any(finite_mask):
            continue
        if scale > 0:
            normalized[finite_mask, col] = np.clip(column[finite_mask] / scale, -1.0, 1.0)
        else:
            normalized[finite_mask, col] = 0.0
    return normalized


def _metric_label_with_arrow(metric_key: str) -> str:
    label = METRIC_LABELS.get(metric_key, metric_key)
    spec = METRIC_SPEC_BY_KEY.get(metric_key)
    if spec is not None:
        arrow = "↑" if spec.higher_is_better else "↓"
        return f"{label} ({arrow})"
    return label


def build_channel_effect_heatmaps_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    shared_metric_keys = _shared_heatmap_metric_keys(summaries)
    fig = plt.figure(figsize=(7.9 * len(summaries), 3.35))
    grid = fig.add_gridspec(1, len(summaries) + 1, width_ratios=[1] * len(summaries) + [0.04], wspace=0.05)
    metric_scales = _heatmap_metric_scales(summaries)
    im = None

    for col, summary in enumerate(summaries):
        ax = fig.add_subplot(grid[0, col])
        raw_matrix, stds = _heatmap_matrix(summary, summary.added_effect_stats, shared_metric_keys)
        normalized_matrix = _normalize_heatmap_matrix(raw_matrix, shared_metric_keys, metric_scales)
        masked = np.ma.masked_invalid(normalized_matrix)
        im = ax.imshow(masked, cmap=CHANNEL_EFFECT_CMAP, vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_yticks(range(len(FOUR_GROUP_ORDER)))
        ax.set_yticklabels([GROUP_LABELS[group] for group in FOUR_GROUP_ORDER], fontsize=11.5, color=INK)
        ax.set_xticks(range(len(shared_metric_keys)))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xticklabels(
            [_metric_label_with_arrow(metric_key) for metric_key in shared_metric_keys],
            rotation=0,
            ha="center",
            fontsize=10.9,
            color=INK,
        )
        if col > 0:
            ax.set_yticklabels([])
        for r in range(raw_matrix.shape[0]):
            for c in range(raw_matrix.shape[1]):
                value = raw_matrix[r, c]
                if not np.isfinite(value):
                    continue
                norm_value = normalized_matrix[r, c]
                raw_std = stds[r, c]
                metric_key = shared_metric_keys[c]
                spec = METRIC_SPEC_BY_KEY.get(metric_key)
                display_value = value if (spec is None or spec.higher_is_better) else -value
                ax.text(
                    c,
                    r,
                    f"{display_value:+.2f}\n±{raw_std:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="white" if abs(norm_value) >= 0.55 else "#111111",
                )
        ax.set_xticks(np.arange(-0.5, len(shared_metric_keys), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FOUR_GROUP_ORDER), 1), minor=True)
        ax.grid(which="minor", color="#D4D4D4", linewidth=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(INK)
            spine.set_linewidth(1.0)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=0)

    cax = fig.add_subplot(grid[0, len(summaries)])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["most\nneg.", "0\n(no effect)", "most\npos."], fontsize=9.5)
    cbar.ax.yaxis.set_tick_params(color=INK, labelcolor=INK)
    cbar.outline.set_edgecolor(INK)
    cbar.outline.set_linewidth(1.0)
    fig.subplots_adjust(left=0.13, right=0.93, top=0.96, bottom=0.10)
    return fig


def build_leave_one_out_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    fig, bar_ax = plt.subplots(1, 1, figsize=(5.8, 3.7))
    x = np.arange(len(FOUR_GROUP_ORDER))
    width = 0.28
    styles = {
        "paired": {"offset": -width / 2, "facecolor": "white", "alpha": 1.0, "hatch": None},
        "unpaired": {"offset": width / 2, "facecolor": "white", "alpha": 1.0, "hatch": "//"},
    }
    x_tick_labels = ["Cell\ntypes", "Cell\nstate", "Vasc.", "Nutrient"]

    values_for_range: list[float] = []
    for group_index, group in enumerate(FOUR_GROUP_ORDER):
        for summary in summaries:
            style = styles[summary.slug]
            value = float(summary.loo_summary.get(group, {}).get("mean_diff", 0.0)) / 255.0
            std_value = float(summary.loo_stats.get(group, {}).get("mean_diff", (0.0, 0.0))[1]) / 255.0
            values_for_range.extend([value - std_value, value + std_value])
            bar_ax.bar(
                group_index + style["offset"],
                value,
                width=width,
                color=style["facecolor"],
                alpha=style["alpha"],
                edgecolor=INK,
                hatch=style["hatch"],
                linewidth=1.0,
                yerr=std_value,
                capsize=3,
                error_kw={"ecolor": INK, "elinewidth": 1.0, "capthick": 1.0},
            )
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(x_tick_labels)
    for tick in bar_ax.get_xticklabels():
        tick.set_color(INK)
        tick.set_fontsize(9.5)
    lo, hi = _tight_range(values_for_range)
    bar_ax.set_ylim(max(0.0, lo), hi)
    bar_ax.grid(True, axis="y", color=SOFT_GRID, linewidth=0.8)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.tick_params(axis="y", labelsize=9.5)
    bar_ax.set_axisbelow(True)
    bar_ax.set_ylabel("Mean |\u0394 pixel| (normalized [0, 1])", fontsize=10.0, color=INK)

    handles = [
        Patch(facecolor="white", edgecolor=INK, label="Paired"),
        Patch(facecolor="white", edgecolor=INK, hatch="//", label="Unpaired"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=10.0)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.77)
    return fig


def build_comparison_table_figure(summaries: list[DatasetSummary]) -> plt.Figure:
    metric_keys = _comparison_metric_order(summaries)
    rendered_summaries = [summary for summary in summaries if any(metric in comparison_metric_keys(summary) for metric in metric_keys)]
    if not rendered_summaries:
        fig, ax = plt.subplots(figsize=(12.0, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No paired vs unpaired comparison data available.", ha="center", va="center", fontsize=12.0, color=OKABE_GRAY)
        return fig

    fig = plt.figure(figsize=(14.2, max(3.0, 1.95 * len(rendered_summaries) + 0.35)))
    outer = fig.add_gridspec(len(rendered_summaries), 1, hspace=0.12)

    for row_index, summary in enumerate(rendered_summaries):
        row_grid = outer[row_index].subgridspec(2, 1, height_ratios=[0.11, 1.0], hspace=0.03)
        label_ax = fig.add_subplot(row_grid[0, 0])
        label_ax.axis("off")
        label_ax.text(
            0.0,
            0.62,
            summary.title.upper(),
            ha="left",
            va="center",
            fontsize=11,
            fontweight="normal",
            color=INK,
            family="DejaVu Serif",
            transform=label_ax.transAxes,
            clip_on=False,
        )
        label_ax.plot([0.0, 1.0], [0.08, 0.08], color=INK, linewidth=2.0, transform=label_ax.transAxes, clip_on=False)

        metric_grid = row_grid[1, 0].subgridspec(1, len(metric_keys), wspace=0.08)
        for col_index, metric_key in enumerate(metric_keys):
            ax = fig.add_subplot(metric_grid[0, col_index])
            _draw_comparison_metric_panel(ax, summary, metric_key)

    fig.subplots_adjust(left=0.028, right=0.992, bottom=0.05, top=0.97)
    return fig


def draw_evidence_panel(ax: plt.Axes, label: str, path: Path | None) -> None:
    ax.set_title(label, fontsize=11.5, fontweight="bold", pad=10, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D9D5CA")
        spine.set_linewidth(1.0)
    if path is None or not path.is_file():
        ax.set_facecolor("#F6F4EF")
        ax.text(0.5, 0.5, "Figure not found", ha="center", va="center", color=OKABE_GRAY, fontsize=11.0)
        return
    ax.imshow(np.asarray(load_rgb_pil_local(path)))


def build_dataset_summary_figure(summary: DatasetSummary) -> plt.Figure:
    fig = plt.figure(figsize=(15.0, 7.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.96, 1.0, 1.0], wspace=0.08)
    text_ax = fig.add_subplot(grid[0, 0])
    text_ax.axis("off")

    representative = summary.representative_tile or "not found"
    metrics_text = ", ".join(
        f"{humanize_token(metric_key)} {'↑' if METRIC_SPEC_BY_KEY[metric_key].higher_is_better else '↓'}"
        for metric_key in summary.metric_keys
    )
    metrics_text = textwrap.fill(metrics_text, width=34)
    takeaway_blocks = [
        textwrap.fill(
            plain_takeaway_text(note.lstrip("- ").strip()),
            width=40,
            initial_indent="- ",
            subsequent_indent="  ",
        )
        for note in summary.key_takeaways[:6]
    ]
    takeaways_text = "\n\n".join(takeaway_blocks) if takeaway_blocks else "Representative findings unavailable."

    text_ax.text(0.0, 0.99, summary.title, fontsize=19, fontweight="bold", color=INK, va="top")
    text_ax.text(
        0.0,
        0.91,
        f"{summary.tile_count} tiles\nRepresentative tile: {representative}",
        fontsize=11.0,
        color=OKABE_GRAY,
        va="top",
        linespacing=1.5,
    )
    text_ax.text(0.0, 0.77, "Metrics", fontsize=12.5, fontweight="bold", color=INK, va="top")
    text_ax.text(0.0, 0.72, metrics_text, fontsize=10.8, color=INK, va="top", linespacing=1.4)
    text_ax.text(0.0, 0.58, "Key takeaways", fontsize=12.5, fontweight="bold", color=INK, va="top")
    text_ax.text(0.0, 0.53, takeaways_text, fontsize=10.7, color=INK, va="top", linespacing=1.45)

    draw_evidence_panel(fig.add_subplot(grid[0, 1]), "Representative ablation grid", summary.ablation_grid_path)
    draw_evidence_panel(fig.add_subplot(grid[0, 2]), "Representative leave-one-out diff", summary.loo_diff_path)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.06)
    return fig


def export_report_png_pages(summaries: list[DatasetSummary], png_dir: Path) -> list[Path]:
    page_builders: list[tuple[str, plt.Figure]] = [
        ("01_metric_tradeoffs.png", build_metric_trends_figure(summaries)),
        ("02_paired_vs_unpaired.png", build_comparison_table_figure(summaries)),
        ("03_channel_effect_sizes.png", build_channel_effect_heatmaps_figure(summaries)),
        ("04_leave_one_out_impact.png", build_leave_one_out_figure(summaries)),
    ]
    for index, summary in enumerate(summaries, start=5):
        page_builders.append((f"{index:02d}_{summary.slug}_summary.png", build_dataset_summary_figure(summary)))

    output_paths: list[Path] = []
    for filename, fig in page_builders:
        output_path = png_dir / filename
        save_figure_png(fig, output_path)
        output_paths.append(output_path)
    return output_paths
