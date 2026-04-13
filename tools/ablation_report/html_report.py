from __future__ import annotations

import base64
import html
import os
import statistics
from pathlib import Path

from .figures import (
    build_channel_effect_heatmaps_figure,
    build_leave_one_out_figure,
    build_metric_trends_figure,
    figure_to_data_uri,
)
from .shared import (
    FOUR_GROUP_ORDER,
    GROUP_LABELS,
    GROUP_SHORT_LABELS,
    METRIC_SPEC_BY_KEY,
    OKABE_BLUE,
    OKABE_GREEN,
    OKABE_ORANGE,
    OKABE_PURPLE,
    TRADEOFF_METRIC_ORDER,
    DatasetSummary,
    _format_mean_sd,
    _metric_caption,
    _ranked_best_worst_selection,
    _ranked_labels,
    comparison_metric_keys,
    compact_condition_order_label,
    condition_groups,
    condition_order_label,
    format_condition,
    humanize_token,
)


def render_condition_glyph(condition: object) -> str:
    active_groups = condition_groups(condition)
    label = format_condition(condition)
    dots = "".join(
        (
            f"<span class='condition-dot{' is-active' if group in active_groups else ''}' "
            f"title='{html.escape(GROUP_SHORT_LABELS[group], quote=True)}'></span>"
        )
        for group in FOUR_GROUP_ORDER
    )
    return (
        f"<span class='condition-glyph' aria-label='{html.escape(label, quote=True)}' "
        f"title='{html.escape(label, quote=True)}'>{dots}</span>"
    )


def _render_ranked_metric_block(record: object, metric_key: str) -> str:
    best_entries, worst_entries, total = _ranked_best_worst_selection(record)
    if not best_entries and not worst_entries:
        return ""

    rows: list[str] = []
    for rank_label, entry in zip(_ranked_labels(total, len(best_entries)), best_entries):
        rows.append(
            "<tr>"
            f"<td class='rank-cell'>{html.escape(rank_label)}</td>"
            f"<td class='condition-cell'>{render_condition_glyph(entry[0])}</td>"
            f"<td class='value-cell'>{f'{float(entry[1]):.3f}' if metric_key == 'fud' else _format_mean_sd(entry[1], entry[2])}</td>"
            "</tr>"
        )
    if best_entries and worst_entries:
        rows.append("<tr class='rank-sep'><td colspan='3'>···</td></tr>")
    if worst_entries:
        rows.extend(
            [
                "<tr>"
                f"<td class='rank-cell'>{html.escape(rank_label)}</td>"
                f"<td class='condition-cell'>{render_condition_glyph(entry[0])}</td>"
                f"<td class='value-cell'>{f'{float(entry[1]):.3f}' if metric_key == 'fud' else _format_mean_sd(entry[1], entry[2])}</td>"
                "</tr>"
                for rank_label, entry in zip(_ranked_labels(total, len(worst_entries), tail=True), worst_entries)
            ]
        )
    return "".join(rows)


def _render_ranked_metric_table(record: object, metric_key: str) -> str:
    rows = _render_ranked_metric_block(record, metric_key)
    if not rows:
        return ""
    return (
        "<table class='ranked-table'>"
        "<thead><tr>"
        "<th class='rank-cell'>Rank</th>"
        f"<th class='condition-cell'>{html.escape(compact_condition_order_label())}</th>"
        f"<th class='value-cell'>{'Mean' if metric_key == 'fud' else 'Mean ± SD'}</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _render_ranked_metric_group(summary: DatasetSummary, metric_keys: list[str]) -> str:
    metric_tables: list[str] = []
    for metric_key in metric_keys:
        table_html = _render_ranked_metric_table(summary.best_worst.get(metric_key), metric_key)
        if not table_html:
            continue
        metric_tables.append(
            f"<div class='metric-wrap metric-wrap--{html.escape(metric_key)}'>"
            f"<div class='metric-caption'>{html.escape(_metric_caption(metric_key))}</div>"
            f"{table_html}"
            "</div>"
        )
    if not metric_tables:
        return ""
    return f"<div class='metric-group metric-group--five-up'>{''.join(metric_tables)}</div>"


def render_comparison_table(summaries: list[DatasetSummary]) -> str:
    metric_keys = [
        metric
        for metric in TRADEOFF_METRIC_ORDER
        if any(metric in comparison_metric_keys(summary) for summary in summaries)
    ]

    sections: list[str] = []
    rendered_groups = 0
    for summary in summaries:
        if not metric_keys:
            metric_keys = comparison_metric_keys(summary)
        group_html = _render_ranked_metric_group(summary, metric_keys)
        if not group_html:
            continue
        section_label = html.escape(summary.title.upper())
        if rendered_groups == 0:
            sections.append(f"<div class='dataset-label'>{section_label}</div>")
        else:
            sections.append(
                "<div class='comparison-divider'>"
                f"<span class='comparison-divider-label'>{section_label}</span>"
                "<span class='comparison-divider-line'></span>"
                "</div>"
            )
        sections.append(group_html)
        rendered_groups += 1
    if not sections:
        return ""
    return (
        "<div class='comparison-section'>"
        "<section class='dataset-section dataset-section--comparison'>"
        f"{''.join(sections)}"
        "</section>"
        "</div>"
    )


def render_notes_list(notes: list[str]) -> str:
    items = "".join(f"<li>{render_takeaway_text(note.lstrip('- ').strip())}</li>" for note in notes)
    return f"<ul class='takeaways'>{items}</ul>"


def path_to_html_src(path: Path, output_dir: Path) -> str:
    rel_path = os.path.relpath(path.resolve(), start=output_dir.resolve())
    return html.escape(rel_path.replace(os.sep, "/"), quote=True)


def image_file_to_data_uri(path: Path) -> str:
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_evidence_block(
    summary: DatasetSummary,
    output_dir: Path,
    *,
    self_contained: bool = False,
) -> str:
    cards: list[str] = []
    for label, path in (
        ("Representative ablation grid", summary.ablation_grid_path),
        ("Representative leave-one-out diff", summary.loo_diff_path),
    ):
        if path is None:
            continue
        # Evidence images are embedded by default so the report remains portable
        # and renders reliably in local file viewers and IDE previews.
        src = image_file_to_data_uri(path)
        cards.append(
            "<figure class='evidence-card'>"
            f"<img src='{src}' alt='{html.escape(label)}' loading='lazy' />"
            f"<figcaption>{html.escape(label)}"
            f"<span>{html.escape(str(path))}</span></figcaption>"
            "</figure>"
        )
    if not cards:
        return "<p class='muted'>Representative figures were not found for this dataset.</p>"
    return "<div class='evidence-grid'>" + "".join(cards) + "</div>"


def metric_direction_badge(metric_key: str) -> str:
    spec = METRIC_SPEC_BY_KEY[metric_key]
    arrow = "↑" if spec.higher_is_better else "↓"
    return f"<span class='metric-chip'>{html.escape(humanize_token(metric_key))} {arrow}</span>"


def render_takeaway_text(text: str) -> str:
    parts = text.split("`")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            rendered.append(f"<strong>{html.escape(humanize_token(part))}</strong>")
        else:
            rendered.append(html.escape(part))
    return "".join(rendered)


def render_dataset_section(
    summary: DatasetSummary,
    output_dir: Path,
    *,
    self_contained: bool = False,
) -> str:
    chips = "".join(metric_direction_badge(metric_key) for metric_key in summary.metric_keys)
    representative = summary.representative_tile or "not found"
    return (
        "<section class='dataset-section'>"
        f"<div class='section-header'><h2>{html.escape(summary.title)}</h2>"
        f"<p>{summary.tile_count} tiles · representative tile {html.escape(representative)}</p></div>"
        f"<div class='chips'>{chips}</div>"
        "<div class='card'>"
        "<h3>Key takeaways</h3>"
        f"{render_notes_list(summary.key_takeaways[:6])}"
        "</div>"
        "<div class='card'>"
        "<h3>Representative evidence</h3>"
        f"{render_evidence_block(summary, output_dir, self_contained=self_contained)}"
        "</div>"
        "</section>"
    )


def render_report_html(
    title: str,
    summaries: list[DatasetSummary],
    output_path: Path,
    *,
    self_contained: bool = False,
) -> str:
    trend_uri = figure_to_data_uri(build_metric_trends_figure(summaries))
    heatmap_uri = figure_to_data_uri(build_channel_effect_heatmaps_figure(summaries))
    loo_uri = figure_to_data_uri(build_leave_one_out_figure(summaries))
    comparison_table = render_comparison_table(summaries)

    best_structure = []
    for summary in summaries:
        group_scores = {
            group: statistics.mean(
                [
                    summary.added_effects.get(group, {}).get(metric_key, 0.0)
                    for metric_key in ("pq", "dice")
                    if metric_key in summary.metric_keys
                ]
            )
            for group in FOUR_GROUP_ORDER
            if any(metric_key in summary.metric_keys for metric_key in ("pq", "dice"))
        }
        if group_scores:
            winner = max(group_scores, key=group_scores.get)
            best_structure.append(f"{summary.title}: {GROUP_LABELS[winner]}")

    overview_line = " | ".join(best_structure)
    dataset_sections = "".join(
        render_dataset_section(summary, output_path.parent, self_contained=self_contained)
        for summary in summaries
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f7f5;
      --paper: #ffffff;
      --ink: #1b1b1b;
      --muted: #686760;
      --line: #dddcd6;
      --blue: {OKABE_BLUE};
      --orange: {OKABE_ORANGE};
      --green: {OKABE_GREEN};
      --purple: {OKABE_PURPLE};
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --sans: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      line-height: 1.45;
    }}
    .page {{
      max-width: 1420px;
      margin: 0 auto;
      padding: 28px 22px 44px;
    }}
    .hero {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 24px 26px 20px;
      box-shadow: 0 8px 24px rgba(28, 28, 28, 0.04);
      margin-bottom: 22px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 2rem;
      line-height: 1.1;
      font-family: var(--serif);
      letter-spacing: -0.02em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 1000px;
    }}
    .overview-bar {{
      margin-top: 14px;
      padding: 10px 12px;
      border-radius: 8px;
      background: #f3f3ef;
      color: #3f3c35;
      font-size: 0.95rem;
    }}
    .report-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    .figure-card, .card, .dataset-section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: 0 6px 20px rgba(28, 28, 28, 0.03);
    }}
    .figure-card {{
      padding: 18px 18px 14px;
    }}
    .figure-card h2, .dataset-section h2, .card h3 {{
      margin: 0 0 10px;
      font-family: var(--serif);
      letter-spacing: -0.01em;
    }}
    .figure-card p {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .figure-card img {{
      width: 100%;
      border-radius: 6px;
      display: block;
      border: 1px solid #e6e4dd;
      background: white;
    }}
    .dataset-section {{
      padding: 18px;
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .section-header p {{
      margin: 0;
      color: var(--muted);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .metric-chip {{
      display: inline-block;
      padding: 0;
      background: transparent;
      color: var(--ink);
      border: none;
      box-shadow: none;
      font-size: 0.9rem;
      font-weight: 600;
      letter-spacing: 0.01em;
    }}
    .card {{
      padding: 16px;
      margin-top: 14px;
    }}
    .comparison-section {{
      margin-bottom: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .comparison-section .dataset-section {{
      margin-bottom: 0;
      overflow-x: auto;
    }}
    .dataset-label {{
      font-family: "Times New Roman", Times, serif;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 2px solid #000;
      padding-bottom: 2px;
      margin-bottom: 5px;
    }}
    .comparison-divider {{
      margin: 10px 0 8px;
    }}
    .comparison-divider-label {{
      font-family: "Times New Roman", Times, serif;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.05em;
      color: #000;
      display: block;
      margin-bottom: 4px;
    }}
    .comparison-divider-line {{
      display: block;
      border-top: 3px solid #000;
    }}
    .metric-group {{
      display: flex;
      gap: 10px;
      flex-wrap: nowrap;
      align-items: flex-start;
    }}
    .metric-group--five-up {{
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 7px;
      min-width: 740px;
      align-items: start;
    }}
    .metric-wrap {{
      flex: 1 1 0;
      min-width: 0;
    }}
    .metric-wrap--fud {{
      flex: 1.18 1 0;
    }}
    .metric-caption {{
      font-family: "Times New Roman", Times, serif;
      font-size: 9px;
      font-weight: 700;
      margin-bottom: 1px;
      text-align: center;
    }}
    .ranked-table {{
      border-collapse: collapse;
      width: 100%;
      font-family: "Times New Roman", Times, serif;
      font-size: 8.6px;
      table-layout: fixed;
    }}
    .ranked-table thead tr {{
      border-top: 1.5px solid #000;
      border-bottom: 1px solid #000;
    }}
    .ranked-table tbody tr:last-child td {{
      border-bottom: 1.5px solid #000;
    }}
    .ranked-table th, .ranked-table td {{
      padding: 1px 2px;
      line-height: 1.12;
    }}
    .ranked-table th.rank-cell {{
      text-align: center;
      width: 22px;
    }}
    .ranked-table th.condition-cell {{
      text-align: center;
      white-space: nowrap;
      font-size: 7.7px;
      letter-spacing: 0.02em;
    }}
    .ranked-table th.value-cell {{
      text-align: right;
      white-space: nowrap;
    }}
    .ranked-table td.rank-cell {{
      text-align: center;
      color: #555;
      font-size: 9px;
      white-space: nowrap;
    }}
    .ranked-table td.condition-cell {{
      white-space: nowrap;
      text-align: center;
    }}
    .ranked-table td.value-cell {{
      text-align: right;
      white-space: nowrap;
    }}
    .ranked-table tr.rank-sep td {{
      text-align: center;
      color: #000;
      font-size: 9px;
      padding: 1px 4px;
      border-top: 1px dashed #000;
      border-bottom: 1px dashed #000;
    }}
    .ranked-table tr.dataset-divider td {{
      padding: 0;
      height: 7px;
      border-top: 1.2px solid #000;
      border-bottom: none;
    }}
    .takeaways {{
      margin: 0;
      padding-left: 18px;
    }}
    .takeaways li {{
      margin-bottom: 8px;
    }}
    .takeaways strong {{
      font-weight: 700;
    }}
    .condition-glyph {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      white-space: nowrap;
    }}
    .condition-dot {{
      width: 9px;
      height: 9px;
      border-radius: 999px;
      border: 1.2px solid #000;
      background: #fff;
      display: inline-block;
      flex: 0 0 auto;
    }}
    .condition-dot.is-active {{
      background: #000;
    }}
    .evidence-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .evidence-card {{
      margin: 0;
      border: 1px solid #e6e4dd;
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    .evidence-card img {{
      width: 100%;
      display: block;
      background: #fff;
    }}
    .evidence-card figcaption {{
      padding: 10px 12px 12px;
      font-size: 0.92rem;
    }}
    .evidence-card span {{
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      margin-top: 4px;
      word-break: break-all;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 960px) {{
      .evidence-grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 18px 14px 28px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>
        This report reframes the ablation summaries in a journal-style layout:
        metric trajectories, oriented channel-effect matrices with uncertainty, representative leave-one-out evidence,
        and a direct paired versus unpaired comparison table.
      </p>
      <div class="overview-bar">{html.escape(overview_line)}</div>
    </section>
    <div class="report-grid">
      <section class="figure-card">
        <h2>Metric Tradeoffs</h2>
        <img src="{trend_uri}" alt="Metric trends by active group count" />
      </section>
      <section class="figure-card">
        <h2>Paired vs Unpaired Ranked Conditions</h2>
        <p>Each paired and unpaired section uses matching ranked mini-tables per metric, with the top three and bottom three conditions isolated so the value scales are not visually comparable across metrics. Filled circles indicate included groups in {html.escape(condition_order_label())} order.</p>
        {comparison_table}
      </section>
      <section class="figure-card">
        <h2>Channel Effect Sizes</h2>
        <p>Each cell shows the raw metric change from adding that channel group &#177; SD; sign follows the metric’s natural direction (e.g. negative FID = improvement). Background color encodes goodness: green = improvement, red = worsening, scaled across all channels and datasets.</p>
        <img src="{heatmap_uri}" alt="Channel effect heatmaps" />
      </section>
      <section class="figure-card">
        <h2>Leave-One-Out Impact</h2>
        <p>Grouped bars show mean normalized leave-one-out pixel change ± SD across tiles, with paired bars on white fill and unpaired bars hatched for contrast.</p>
        <img src="{loo_uri}" alt="Leave one out summary bars" />
      </section>
      {dataset_sections}
    </div>
  </div>
</body>
</html>
"""
