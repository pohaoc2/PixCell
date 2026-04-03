#!/usr/bin/env python3
"""Generate a self-contained HTML proposal for dataset-level ablation metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import FOUR_GROUP_ORDER

METRICS: tuple[str, ...] = ("fid", "cosine", "lpips", "aji", "pq")
METRIC_LABELS: dict[str, str] = {
    "fid": "FID",
    "cosine": "Cosine",
    "lpips": "LPIPS",
    "aji": "AJI",
    "pq": "PQ",
}
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "fid": False,
    "cosine": True,
    "lpips": False,
    "aji": True,
    "pq": True,
}
GROUP_SHORT: dict[str, str] = {
    "cell_types": "CT",
    "cell_state": "CS",
    "vasculature": "Vas",
    "microenv": "Env",
}
CARD_COLORS: dict[int, str] = {
    1: "#087f5b",
    2: "#1864ab",
    3: "#b04a00",
    4: "#7b4cc2",
}


def _condition_label(cond_key: str) -> str:
    groups = set(cond_key.split("+")) if cond_key else set()
    ordered = [GROUP_SHORT[g] for g in FOUR_GROUP_ORDER if g in groups]
    return "+".join(ordered)


def _condition_cardinality(cond_key: str) -> int:
    return 0 if not cond_key else len(cond_key.split("+"))


def _aggregate_metrics(cache_root: Path) -> list[dict]:
    grouped: dict[str, dict[str, list[float]]] = {}
    tile_count = 0
    for metrics_path in sorted(cache_root.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_condition = payload.get("per_condition", {})
        if not isinstance(per_condition, dict):
            continue
        tile_count += 1
        for cond_key, record in per_condition.items():
            if not isinstance(record, dict):
                continue
            bucket = grouped.setdefault(cond_key, {})
            for metric in ("cosine", "lpips", "aji", "pq"):
                value = record.get(metric)
                if value is None:
                    continue
                bucket.setdefault(metric, []).append(float(value))

    rows: list[dict] = []
    for cond_key in sorted(grouped):
        metrics = {}
        for metric in METRICS:
            values = grouped[cond_key].get(metric, [])
            if values:
                metrics[metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            else:
                metrics[metric] = None
        rows.append(
            {
                "condition_key": cond_key,
                "condition_label": _condition_label(cond_key),
                "cardinality": _condition_cardinality(cond_key),
                "metrics": metrics,
            }
        )
    return rows


def _build_payload(cache_root: Path) -> dict:
    rows = _aggregate_metrics(cache_root)
    return {
        "cache_root": str(cache_root),
        "tile_count": len(list(cache_root.glob("*/metrics.json"))),
        "metrics": list(METRICS),
        "metric_labels": METRIC_LABELS,
        "higher_is_better": METRIC_HIGHER_IS_BETTER,
        "card_colors": CARD_COLORS,
        "rows": rows,
    }


def _html(payload: dict) -> str:
    payload_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Dataset-Level Ablation Metrics Proposal</title>
<style>
  :root {{
    --bg: #f7f4eb;
    --panel: rgba(255,255,255,0.88);
    --ink: #171717;
    --muted: #5d5d5d;
    --line: rgba(23,23,23,0.12);
    --accent: #111111;
    --shadow: 0 16px 50px rgba(36, 27, 14, 0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: var(--ink);
    background:
      radial-gradient(circle at top left, rgba(197, 170, 126, 0.18), transparent 28%),
      radial-gradient(circle at bottom right, rgba(120, 90, 40, 0.10), transparent 24%),
      linear-gradient(180deg, #fbf8f1 0%, var(--bg) 100%);
  }}
  .shell {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 32px 28px 56px;
  }}
  .hero {{
    display: grid;
    grid-template-columns: 1.3fr 0.7fr;
    gap: 24px;
    align-items: end;
    margin-bottom: 28px;
  }}
  .hero-card, .meta-card, .option {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 22px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(12px);
  }}
  .hero-card {{
    padding: 28px;
  }}
  .hero h1 {{
    margin: 0 0 10px;
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 0.95;
    letter-spacing: -0.04em;
  }}
  .hero p {{
    margin: 0;
    max-width: 70ch;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.5;
  }}
  .meta-card {{
    padding: 22px;
    display: grid;
    gap: 12px;
  }}
  .meta-kicker {{
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    color: var(--muted);
  }}
  .meta-big {{
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
  }}
  .meta-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }}
  .meta-pill {{
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 12px;
    background: rgba(255,255,255,0.65);
  }}
  .section-label {{
    margin: 34px 0 12px;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
  }}
  .options {{
    display: grid;
    gap: 20px;
  }}
  .option {{
    padding: 22px;
  }}
  .option-head {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: baseline;
    margin-bottom: 16px;
  }}
  .option h2 {{
    margin: 0;
    font-size: 1.2rem;
    letter-spacing: -0.03em;
  }}
  .option-note {{
    font-size: 0.85rem;
    color: var(--muted);
  }}
  .caption {{
    margin: 6px 0 18px;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.45;
  }}
  .small-multiples {{
    display: grid;
    grid-template-columns: repeat(3, minmax(240px, 1fr));
    gap: 14px;
  }}
  .chart-card {{
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 12px 14px 10px;
    background: rgba(255,255,255,0.72);
  }}
  .chart-title {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-weight: 700;
    font-size: 0.92rem;
  }}
  .placeholder {{
    min-height: 280px;
    display: grid;
    place-items: center;
    text-align: center;
    color: var(--muted);
    border: 1px dashed rgba(23,23,23,0.22);
    border-radius: 14px;
    background: repeating-linear-gradient(-45deg, rgba(0,0,0,0.03), rgba(0,0,0,0.03) 8px, transparent 8px, transparent 16px);
  }}
  .heatmap-wrap {{
    overflow-x: auto;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255,255,255,0.72);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
  }}
  thead th {{
    position: sticky;
    top: 0;
    background: rgba(255,255,255,0.96);
    z-index: 1;
  }}
  th, td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    white-space: nowrap;
  }}
  td.metric-cell {{
    min-width: 122px;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }}
  .chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }}
  .dot {{
    width: 10px;
    height: 10px;
    border-radius: 999px;
    display: inline-block;
  }}
  .metric-main {{
    font-weight: 700;
    display: block;
  }}
  .metric-sub {{
    font-size: 0.74rem;
    color: var(--muted);
    display: block;
  }}
  .control-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 14px;
    align-items: center;
  }}
  .control-row select {{
    appearance: none;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 10px 12px;
    background: white;
    font: inherit;
    color: var(--ink);
  }}
  .ranking {{
    display: grid;
    gap: 10px;
  }}
  .rank-row {{
    display: grid;
    grid-template-columns: 28px 110px 1fr 80px;
    gap: 10px;
    align-items: center;
    font-size: 0.88rem;
  }}
  .rank-bar {{
    height: 18px;
    border-radius: 999px;
    background: #ece8df;
    overflow: hidden;
    position: relative;
  }}
  .rank-fill {{
    height: 100%;
    background: var(--accent);
    border-radius: 999px;
  }}
  .rank-tick {{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background: rgba(255,255,255,0.85);
  }}
  .footer {{
    margin-top: 24px;
    color: var(--muted);
    font-size: 0.82rem;
  }}
  @media (max-width: 980px) {{
    .hero {{ grid-template-columns: 1fr; }}
    .small-multiples {{ grid-template-columns: 1fr; }}
    .rank-row {{ grid-template-columns: 24px 90px 1fr 70px; }}
  }}
</style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="hero-card">
        <h1>Dataset-Level Ablation Metrics</h1>
        <p>
          Three web-based options for summarizing the 15 channel combinations across the full cache.
          The current proposal uses aggregated means and standard deviations from the existing
          tile-level <code>metrics.json</code> files. FID is shown as a dataset-level placeholder
          because it has not been computed yet.
        </p>
      </div>
      <div class="meta-card">
        <div class="meta-kicker">Proposal Input</div>
        <div class="meta-big" id="tileCount"></div>
        <div class="meta-grid">
          <div class="meta-pill"><strong>15</strong><br />conditions</div>
          <div class="meta-pill"><strong>5</strong><br />metrics</div>
          <div class="meta-pill"><strong>4+6+4+1</strong><br />combination layout</div>
          <div class="meta-pill"><strong>mean ± std</strong><br />default summary</div>
        </div>
      </div>
    </div>

    <div class="section-label">Option Set</div>
    <div class="options">
      <section class="option">
        <div class="option-head">
          <h2>Option A: Small-Multiple Bar Charts</h2>
          <div class="option-note">Closest to your suggestion: nominal x-axis, value on y-axis, std whiskers</div>
        </div>
        <p class="caption">
          Best when we want a familiar publication-style view. Each metric gets its own compact panel, which
          keeps the scales interpretable and lets viewers compare means and variability without interacting first.
        </p>
        <div class="small-multiples" id="smallMultiples"></div>
      </section>

      <section class="option">
        <div class="option-head">
          <h2>Option B: Heatmap Matrix + Mean/Std Cell Labels</h2>
          <div class="option-note">Best overview density; fastest way to scan all combinations at once</div>
        </div>
        <p class="caption">
          This trades detailed bar geometry for a dense comparison matrix. Color encodes the mean, and each
          cell still prints mean ± std so the exact summary is not hidden.
        </p>
        <div class="heatmap-wrap" id="heatmapWrap"></div>
      </section>

      <section class="option">
        <div class="option-head">
          <h2>Option C: Ranked Explorer</h2>
          <div class="option-note">Interactive review mode for choosing the “best” combination by a selected metric</div>
        </div>
        <p class="caption">
          This view emphasizes ranking and uncertainty. It is the most decision-oriented option and works well
          alongside the static tile-level grid because it answers “which combination wins on average?”
        </p>
        <div class="control-row">
          <label for="metricSelect"><strong>Rank by</strong></label>
          <select id="metricSelect"></select>
        </div>
        <div class="ranking" id="ranking"></div>
      </section>
    </div>

    <div class="footer">
      Proposed next step: keep Option A as the default dataset summary, add Option B as the compact appendix view,
      and include Option C in the exploratory web dashboard once FID is available.
    </div>
  </div>

<script>
const DATA = {payload_json};
const METRICS = DATA.metrics;
const METRIC_LABELS = DATA.metric_labels;
const HIGHER_IS_BETTER = DATA.higher_is_better;
const CARD_COLORS = DATA.card_colors;
const rows = DATA.rows;

document.getElementById("tileCount").textContent = `${{DATA.tile_count}} tiles aggregated`;

function metricRange(metric) {{
  if (metric === "fid" || metric === "lpips" || metric === "aji" || metric === "pq") {{
    return [0, 1];
  }}
  return [-1, 1];
}}

function colorForMetricValue(metric, value) {{
  if (value == null) return "transparent";
  const [min, max] = metricRange(metric);
  const frac = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const shade = HIGHER_IS_BETTER[metric] ? 92 - frac * 54 : 92 - (1 - frac) * 54;
  return `hsl(42 60% ${{shade}}%)`;
}}

function svgSmallMultiple(metric) {{
  const width = 420;
  const height = 280;
  const margin = {{top: 20, right: 18, bottom: 92, left: 42}};
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const data = rows.map(r => {{
    const stat = r.metrics[metric];
    return {{
      label: r.condition_label,
      mean: stat ? stat.mean : null,
      std: stat ? stat.std : null,
      card: r.cardinality,
    }};
  }});

  if (metric === "fid") {{
    return `<div class="chart-card"><div class="chart-title"><span>${{METRIC_LABELS[metric]}}</span><span>dataset-only</span></div><div class="placeholder">FID panel reserved<br/>once dataset-level FID is computed per combination</div></div>`;
  }}

  const [yMin, yMax] = metricRange(metric);
  const xStep = innerW / data.length;
  const barW = Math.max(10, xStep * 0.62);
  const y = (v) => margin.top + innerH - ((v - yMin) / (yMax - yMin)) * innerH;
  const axisTicks = [yMin, (yMin + yMax) / 2, yMax];

  let svg = `<div class="chart-card"><div class="chart-title"><span>${{METRIC_LABELS[metric]}}</span><span>${{HIGHER_IS_BETTER[metric] ? "higher is better" : "lower is better"}}</span></div>`;
  svg += `<svg viewBox="0 0 ${{width}} ${{height}}" width="100%" height="auto" aria-label="${{METRIC_LABELS[metric]}}">`;
  axisTicks.forEach(t => {{
    const yy = y(t);
    svg += `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{yy}}" y2="${{yy}}" stroke="rgba(0,0,0,0.08)" />`;
    svg += `<text x="${{margin.left - 8}}" y="${{yy + 4}}" font-size="10" text-anchor="end" fill="#666">${{t.toFixed(1)}}</text>`;
  }});
  svg += `<line x1="${{margin.left}}" x2="${{margin.left}}" y1="${{margin.top}}" y2="${{margin.top + innerH}}" stroke="#222" />`;
  svg += `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{margin.top + innerH}}" y2="${{margin.top + innerH}}" stroke="#222" />`;

  data.forEach((d, i) => {{
    const cx = margin.left + xStep * i + xStep / 2;
    const x = cx - barW / 2;
    const baseY = y(0);
    const topY = y(d.mean);
    const barH = Math.max(0, baseY - topY);
    const errTop = y(Math.min(yMax, d.mean + d.std));
    const errBot = y(Math.max(yMin, d.mean - d.std));
    svg += `<rect x="${{x}}" y="${{topY}}" width="${{barW}}" height="${{barH}}" rx="4" fill="#111" />`;
    svg += `<line x1="${{cx}}" x2="${{cx}}" y1="${{errTop}}" y2="${{errBot}}" stroke="#111" stroke-width="1.4" />`;
    svg += `<line x1="${{cx - 5}}" x2="${{cx + 5}}" y1="${{errTop}}" y2="${{errTop}}" stroke="#111" stroke-width="1.4" />`;
    svg += `<line x1="${{cx - 5}}" x2="${{cx + 5}}" y1="${{errBot}}" y2="${{errBot}}" stroke="#111" stroke-width="1.4" />`;
    svg += `<circle cx="${{x + 4}}" cy="${{topY + 7}}" r="3" fill="${{CARD_COLORS[String(d.card)]}}" />`;
    svg += `<text transform="translate(${{cx}},${{height - 16}}) rotate(-52)" text-anchor="end" font-size="9" fill="#555">${{d.label}}</text>`;
  }});
  svg += `</svg></div>`;
  return svg;
}}

function renderSmallMultiples() {{
  const host = document.getElementById("smallMultiples");
  host.innerHTML = METRICS.map(svgSmallMultiple).join("");
}}

function renderHeatmap() {{
  const wrap = document.getElementById("heatmapWrap");
  let html = `<table><thead><tr><th>Combination</th>`;
  METRICS.forEach(metric => {{
    html += `<th>${{METRIC_LABELS[metric]}}</th>`;
  }});
  html += `</tr></thead><tbody>`;

  rows.forEach(row => {{
    html += `<tr><td><span class="chip"><span class="dot" style="background:${{CARD_COLORS[String(row.cardinality)]}}"></span><span>${{row.condition_label}}</span></span></td>`;
    METRICS.forEach(metric => {{
      const stat = row.metrics[metric];
      if (!stat) {{
        html += `<td class="metric-cell" style="background: repeating-linear-gradient(-45deg, rgba(0,0,0,0.04), rgba(0,0,0,0.04) 8px, transparent 8px, transparent 16px);"><span class="metric-main">Pending</span><span class="metric-sub">dataset-level only</span></td>`;
      }} else {{
        html += `<td class="metric-cell" style="background:${{colorForMetricValue(metric, stat.mean)}}"><span class="metric-main">${{stat.mean.toFixed(3)}}</span><span class="metric-sub">± ${{stat.std.toFixed(3)}}</span></td>`;
      }}
    }});
    html += `</tr>`;
  }});
  html += `</tbody></table>`;
  wrap.innerHTML = html;
}}

function renderRanking(metric) {{
  const ranking = document.getElementById("ranking");
  const statRows = rows.map(row => {{
    const stat = row.metrics[metric];
    return {{
      ...row,
      stat,
      score: stat ? stat.mean : null,
    }};
  }});
  const filtered = statRows.filter(r => r.stat);
  filtered.sort((a, b) => HIGHER_IS_BETTER[metric] ? b.score - a.score : a.score - b.score);
  const [min, max] = metricRange(metric);
  ranking.innerHTML = filtered.map((row, idx) => {{
    const frac = Math.max(0, Math.min(1, (row.score - min) / (max - min || 1)));
    const stdFrac = row.stat.std / (max - min || 1);
    const tickPos = Math.max(0, Math.min(100, (frac + stdFrac) * 100));
    return `<div class="rank-row">
      <div>${{idx + 1}}</div>
      <div><span class="chip"><span class="dot" style="background:${{CARD_COLORS[String(row.cardinality)]}}"></span><span>${{row.condition_label}}</span></span></div>
      <div class="rank-bar">
        <div class="rank-fill" style="width:${{(frac * 100).toFixed(1)}}%"></div>
        <div class="rank-tick" style="left:${{tickPos.toFixed(1)}}%"></div>
      </div>
      <div style="text-align:right;font-variant-numeric:tabular-nums">${{row.score.toFixed(3)}} ± ${{row.stat.std.toFixed(3)}}</div>
    </div>`;
  }}).join("");
}}

function initSelector() {{
  const select = document.getElementById("metricSelect");
  METRICS.forEach(metric => {{
    const opt = document.createElement("option");
    opt.value = metric;
    opt.textContent = METRIC_LABELS[metric];
    if (metric === "pq") opt.selected = true;
    if (metric === "fid") opt.disabled = true;
    select.appendChild(opt);
  }});
  select.addEventListener("change", () => renderRanking(select.value));
  renderRanking(select.value);
}}

renderSmallMultiples();
renderHeatmap();
initSelector();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML proposal for dataset-level ablation metrics.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "inference_output/cache",
        help="Parent directory containing per-tile cache folders with metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "inference_output/ablation_dataset_metrics_proposal.html",
        help="HTML output path",
    )
    args = parser.parse_args()

    payload = _build_payload(args.cache_root.resolve())
    args.output.write_text(_html(payload), encoding="utf-8")
    print(f"Wrote proposal HTML → {args.output}")


if __name__ == "__main__":
    main()
