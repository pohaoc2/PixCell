#!/usr/bin/env python3
"""
Generate a self-contained HTML ablation grid visualization.

Output: <cache_dir>/ablation_grid.html
All images are base64-embedded (with lime-green cell-mask contour baked in).
Grid is pre-sorted by PQ (cosine as tiebreaker); hover tooltips remain.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3.ablation_vis_utils import (
    FOUR_GROUP_ORDER,
    condition_metric_key,
    ordered_subset_condition_tuples,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_MANIFEST_TO_SCORE_KEY: dict[str, str] = {
    "cell_identity": "cell_types",
    "cell_types": "cell_types",
    "cell_state": "cell_state",
    "vasculature": "vasculature",
    "microenv": "microenv",
}

_GROUP_LABEL: dict[str, str] = {
    "cell_types": "CT",
    "cell_state": "CS",
    "vasculature": "Vas",
    "microenv": "Env",
}

_COLOR_BY_CARD: dict[int, str] = {
    1: "#009E73",
    2: "#0072B2",
    3: "#D55E00",
    4: "#9B59B6",
}
COLOR_REF = "#999999"
COLOR_INACTIVE = "#CCCCCC"

METRIC_COLORS = {
    "cosine": "#111111",
    "lpips":  "#111111",
    "aji":    "#111111",
    "pq":     "#111111",
}
METRIC_LABELS = {
    "cosine": "Cosine",
    "lpips": "LPIPS",
    "aji": "AJI",
    "pq": "PQ",
}

ALL4CH_KEY = condition_metric_key(FOUR_GROUP_ORDER)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _b64_raw(path: Path) -> str:
    """Plain base64 data URI (no overlay)."""
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _b64_with_contour(img_path: Path, cell_mask: np.ndarray) -> str:
    """Load image, bake lime-green cell-mask contour, return base64 data URI."""
    img_arr = np.array(Image.open(img_path).convert("RGB"))
    h, w = img_arr.shape[:2]

    mask = cell_mask
    if mask.shape != (h, w):
        mask_pil = Image.fromarray(
            (np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L"
        )
        mask = np.array(mask_pil.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_arr)
    ax.contour(mask, levels=[0.5], colors=["lime"], linewidths=0.7, alpha=0.85)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{data}"


def _encode_image(img_path: Path, cell_mask: np.ndarray | None) -> str | None:
    if not img_path.exists():
        return None
    if cell_mask is not None:
        return _b64_with_contour(img_path, cell_mask)
    return _b64_raw(img_path)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _normalize_active_groups(groups: list[str]) -> list[str]:
    return [_MANIFEST_TO_SCORE_KEY.get(g, g) for g in groups]


def _condition_key_from_canonical(active: list[str]) -> str:
    return condition_metric_key(tuple(active))


def _load_metrics(cache_dir: Path) -> dict[str, dict[str, float | None]]:
    """Load per-condition metrics. Falls back to uni_cosine_scores.json."""
    metrics_path = cache_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            d = json.load(f)
        return d.get("per_condition", {})

    cosine_path = cache_dir / "uni_cosine_scores.json"
    if cosine_path.exists():
        with open(cosine_path) as f:
            raw = json.load(f)
        per_cond = raw.get("per_condition", {})
        return {k: {"cosine": v, "lpips": None, "aji": None, "pq": None}
                for k, v in per_cond.items()}
    return {}


def _load_cell_mask(cache_dir: Path, manifest: dict) -> np.ndarray | None:
    """Load the reference cell mask used for contour overlay."""
    rel = manifest.get("cell_mask_path")
    if not rel:
        return None
    path = cache_dir / rel
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _build_cells(
    cache_dir: Path,
    orion_root: Path,
    all4ch_image: Path,
    metrics: dict[str, dict],
    cell_mask: np.ndarray | None,
) -> list[dict]:
    tile_id = cache_dir.name

    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)

    cells: list[dict] = []

    for section in manifest["sections"]:
        for entry in section["entries"]:
            canonical = _normalize_active_groups(entry["active_groups"])
            key = _condition_key_from_canonical(canonical)
            m = metrics.get(key, {})
            img_path = cache_dir / entry["image_path"]
            cells.append({
                "key": key,
                "active_groups": canonical,
                "cardinality": len(canonical),
                "border_color": _COLOR_BY_CARD[len(canonical)],
                "image_b64": _encode_image(img_path, cell_mask),
                "cosine": m.get("cosine"),
                "lpips": m.get("lpips"),
                "aji": m.get("aji"),
                "pq": m.get("pq"),
                "is_ref": False,
                "is_all4ch": False,
            })

    # All-4-ch
    m = metrics.get(ALL4CH_KEY, {})
    cells.append({
        "key": ALL4CH_KEY,
        "active_groups": list(FOUR_GROUP_ORDER),
        "cardinality": 4,
        "border_color": _COLOR_BY_CARD[4],
        "image_b64": _encode_image(all4ch_image, cell_mask),
        "cosine": m.get("cosine"),
        "lpips": m.get("lpips"),
        "aji": m.get("aji"),
        "pq": m.get("pq"),
        "is_ref": False,
        "is_all4ch": True,
    })

    # Real H&E — no metrics, always last
    he_path = orion_root / "he" / f"{tile_id}.png"
    cells.append({
        "key": "__ref__",
        "active_groups": [],
        "cardinality": 0,
        "border_color": COLOR_REF,
        "image_b64": _b64_raw(he_path) if he_path.exists() else None,
        "cosine": None,
        "lpips": None,
        "aji": None,
        "pq": None,
        "is_ref": True,
        "is_all4ch": False,
    })

    return cells


def _sort_cells(cells: list[dict]) -> list[dict]:
    """Sort by PQ desc, cosine desc as tiebreaker; Real H&E always last."""
    ref = [c for c in cells if c["is_ref"]]
    rest = [c for c in cells if not c["is_ref"]]

    def _key(c: dict):
        pq = c["pq"]
        co = c["cosine"]
        return (
            0 if pq is not None else 1,       # nulls last
            -(pq if pq is not None else 0),
            0 if co is not None else 1,
            -(co if co is not None else 0),
            c["key"],
        )

    rest.sort(key=_key)
    return rest + ref


# ── HTML generation ───────────────────────────────────────────────────────────

def _generate_html(cells: list[dict], tile_id: str) -> str:
    cells_json = json.dumps(cells)
    four_group_order = list(FOUR_GROUP_ORDER)
    group_labels = {k: v for k, v in _GROUP_LABEL.items()}
    color_by_card = {str(k): v for k, v in _COLOR_BY_CARD.items()}

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ablation Grid — {tile_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f5f5f5;
    color: #222;
    padding: 24px;
  }}
  h1 {{
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 4px;
    color: #333;
  }}
  .subtitle {{
    font-size: 0.78rem;
    color: #777;
    margin-bottom: 20px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    max-width: 1100px;
  }}
  .cell {{
    background: #fff;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    cursor: default;
    transition: box-shadow 0.15s, transform 0.15s;
  }}
  .cell:hover {{
    box-shadow: 0 4px 16px rgba(0,0,0,0.14);
    transform: translateY(-2px);
    z-index: 10;
    position: relative;
  }}
  .cell.best-cell {{
    background: #FFFBE6;
  }}
  /* Dot row */
  .dot-row {{
    display: flex;
    gap: 6px;
    justify-content: center;
    align-items: flex-end;
    margin-bottom: 6px;
  }}
  .dot-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
  }}
  .dot-col-label {{
    font-size: 0.58rem;
    font-weight: 600;
    color: #555;
    line-height: 1;
  }}
  .dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }}
  .dot.inactive {{
    background: transparent;
    border: 1.5px solid {COLOR_INACTIVE};
  }}
  /* Image area */
  .img-wrap {{
    width: 100%;
    aspect-ratio: 1;
    border-radius: 4px;
    overflow: hidden;
    border-width: 3px;
    border-style: solid;
  }}
  .img-wrap.ref-border {{
    border-style: dashed;
    border-color: {COLOR_REF};
  }}
  .img-wrap img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }}
  .img-placeholder {{
    width: 100%;
    height: 100%;
    background: #f0f0f0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    color: #aaa;
  }}
  /* Metric bars */
  .metrics {{
    margin-top: 7px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }}
  .metric-row {{
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  .metric-label {{
    font-size: 0.55rem;
    font-weight: 600;
    width: 92px;
    flex-shrink: 0;
  }}
  .bar-bg {{
    flex: 1;
    height: 5px;
    background: #efefef;
    border-radius: 3px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 3px;
  }}
  .bar-bg.placeholder {{
    background: repeating-linear-gradient(
      90deg,
      #d0d0d0 0px, #d0d0d0 4px,
      transparent 4px, transparent 8px
    );
  }}
  .metric-val {{
    font-size: 0.6rem;
    color: #111;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
  }}
  /* Tooltip */
  .tooltip {{
    display: none;
    position: fixed;
    background: rgba(20,20,20,0.92);
    color: #fff;
    font-size: 0.73rem;
    padding: 10px 14px;
    border-radius: 8px;
    pointer-events: none;
    z-index: 1000;
    max-width: 240px;
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
  }}
  .tooltip.visible {{ display: block; }}
  .tooltip-title {{ font-weight: 700; font-size: 0.8rem; margin-bottom: 4px; }}
  .tooltip-metric {{
    display: flex; align-items: center; gap: 6px; margin: 2px 0;
  }}
  .tooltip-metric-bar {{
    flex: 1; height: 4px; background: #444; border-radius: 2px; overflow: hidden;
  }}
  .tooltip-metric-fill {{ height: 100%; border-radius: 2px; }}
  .tooltip-metric-val {{ width: 36px; text-align: right; font-size: 0.68rem; color: #ccc; }}
  /* Legend */
  .legend {{
    display: flex; gap: 16px; flex-wrap: wrap;
    margin-top: 20px; max-width: 1100px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.75rem; color: #555; }}
  .legend-dot {{ width: 11px; height: 11px; border-radius: 50%; }}
  .legend-dash {{ width: 18px; border-top: 2.5px dashed {COLOR_REF}; }}
</style>
</head>
<body>

<h1>Ablation Grid — Tile {tile_id}</h1>
<p class="subtitle">Sorted by PQ (cosine fallback). Real H&amp;E pinned last. Hover for details.</p>

<div class="grid" id="grid"></div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#009E73"></div>1-ch</div>
  <div class="legend-item"><div class="legend-dot" style="background:#0072B2"></div>2-ch</div>
  <div class="legend-item"><div class="legend-dot" style="background:#D55E00"></div>3-ch</div>
  <div class="legend-item"><div class="legend-dot" style="background:#9B59B6"></div>4-ch (All)</div>
  <div class="legend-item"><div class="legend-dash"></div>Real H&amp;E</div>
  <span style="margin-left:12px;font-size:0.72rem;color:#999">
    Bars: shared black fill on a literal 0..1 scale &nbsp;|&nbsp; Dashed = not yet computed
  </span>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const CELLS = {cells_json};
const FOUR_GROUP_ORDER = {json.dumps(four_group_order)};
const GROUP_LABELS = {json.dumps(group_labels)};
const COLOR_BY_CARD = {json.dumps(color_by_card)};
const METRIC_COLORS = {json.dumps(METRIC_COLORS)};
const METRIC_LABELS = {json.dumps(METRIC_LABELS)};
const COLOR_REF = "{COLOR_REF}";
const COLOR_INACTIVE = "{COLOR_INACTIVE}";

function computeNorm(metric) {{
  return {{min: 0, max: 1}};
}}

function normalizeVal(val, norm, metric) {{
  if (val == null) return null;
  let v = norm.max === norm.min ? 1.0 : (val - norm.min) / (norm.max - norm.min);
  return Math.max(0, Math.min(1, v));
}}

function renderGrid() {{
  const norm = {{}};
  ["cosine","lpips","aji","pq"].forEach(m => norm[m] = computeNorm(m));

  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  CELLS.forEach((cell, idx) => {{
    const isBest = idx === 0 && !cell.is_ref;
    const isFirst = idx === 0;
    const card = cell.cardinality;

    const el = document.createElement("div");
    el.className = "cell" + (isBest ? " best-cell" : "");

    // Dot row — labels only on first cell
    const dotRow = document.createElement("div");
    dotRow.className = "dot-row";
    if (!cell.is_ref) {{
      FOUR_GROUP_ORDER.forEach(g => {{
        const col = document.createElement("div");
        col.className = "dot-col";

        if (isFirst) {{
          const lbl = document.createElement("div");
          lbl.className = "dot-col-label";
          lbl.textContent = GROUP_LABELS[g] || g;
          col.appendChild(lbl);
        }}

        const dot = document.createElement("div");
        const active = cell.active_groups.includes(g);
        dot.className = "dot " + (active ? "active" : "inactive");
        if (active) dot.style.background = COLOR_BY_CARD[String(card)];
        col.appendChild(dot);

        dotRow.appendChild(col);
      }});
    }} else {{
      // spacer to keep image aligned
      dotRow.style.height = isFirst ? "28px" : "16px";
    }}
    el.appendChild(dotRow);

    // Image
    const imgWrap = document.createElement("div");
    imgWrap.className = "img-wrap" + (cell.is_ref ? " ref-border" : "");
    if (!cell.is_ref) imgWrap.style.borderColor = cell.border_color;
    if (cell.image_b64) {{
      const img = document.createElement("img");
      img.src = cell.image_b64;
      img.alt = cell.key;
      imgWrap.appendChild(img);
    }} else {{
      const ph = document.createElement("div");
      ph.className = "img-placeholder";
      ph.textContent = "no image";
      imgWrap.appendChild(ph);
    }}
    el.appendChild(imgWrap);

    // Metric bars
    if (!cell.is_ref) {{
      const metricsDiv = document.createElement("div");
      metricsDiv.className = "metrics";
      ["cosine","lpips","aji","pq"].forEach(m => {{
        const row = document.createElement("div");
        row.className = "metric-row";

        const label = document.createElement("div");
        label.className = "metric-label";
        label.style.color = METRIC_COLORS[m];
        label.textContent = METRIC_LABELS[m];
        row.appendChild(label);

        const val = cell[m];
        const barBg = document.createElement("div");
        if (val != null) {{
          barBg.className = "bar-bg";
          const fill = document.createElement("div");
          fill.className = "bar-fill";
          fill.style.background = METRIC_COLORS[m];
          fill.style.width = `${{(normalizeVal(val, norm[m], m) * 100).toFixed(1)}}%`;
          barBg.appendChild(fill);
        }} else {{
          barBg.className = "bar-bg placeholder";
        }}
        row.appendChild(barBg);

        const valDiv = document.createElement("div");
        valDiv.className = "metric-val";
        valDiv.textContent = val != null ? val.toFixed(4) : "—";
        row.appendChild(valDiv);

        metricsDiv.appendChild(row);
      }});
      el.appendChild(metricsDiv);
    }}

    el.addEventListener("mouseenter", e => showTooltip(e, cell, norm));
    el.addEventListener("mousemove", moveTooltip);
    el.addEventListener("mouseleave", hideTooltip);

    grid.appendChild(el);
  }});
}}

const tooltip = document.getElementById("tooltip");

function showTooltip(e, cell, norm) {{
  const groups = cell.active_groups.map(g => GROUP_LABELS[g] || g);
  let html = `<div class="tooltip-title">${{cell.is_ref ? "Real H&E" : cell.key}}</div>`;
  if (groups.length) html += `<div style="color:#aaa;font-size:0.68rem;margin-bottom:6px">${{groups.join("+")}}</div>`;

  if (!cell.is_ref) {{
    ["cosine","lpips","aji","pq"].forEach(m => {{
      const val = cell[m];
      html += `<div class="tooltip-metric">
        <span style="width:120px;font-size:0.65rem;color:${{METRIC_COLORS[m]}};font-weight:600">${{METRIC_LABELS[m]}}</span>
        <div class="tooltip-metric-bar">
          ${{val != null
            ? `<div class="tooltip-metric-fill" style="background:${{METRIC_COLORS[m]}};width:${{(normalizeVal(val,norm[m],m)*100).toFixed(1)}}%"></div>`
            : `<div style="height:100%;background:repeating-linear-gradient(90deg,#555 0,#555 3px,transparent 3px,transparent 6px)"></div>`
          }}
        </div>
        <span class="tooltip-metric-val">${{val != null ? val.toFixed(4) : "—"}}</span>
      </div>`;
    }});
  }}

  tooltip.innerHTML = html;
  tooltip.classList.add("visible");
  moveTooltip(e);
}}

function moveTooltip(e) {{
  const pad = 16;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + 260 > window.innerWidth) x = e.clientX - 260 - pad;
  if (y + tooltip.offsetHeight > window.innerHeight) y = e.clientY - tooltip.offsetHeight - pad;
  tooltip.style.left = x + "px";
  tooltip.style.top = y + "px";
}}

function hideTooltip() {{ tooltip.classList.remove("visible"); }}

renderGrid();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--orion-root", required=True, type=Path)
    parser.add_argument("--all4ch-image", type=Path, default=None)
    parser.add_argument("--output-name", default="ablation_grid")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    orion_root = args.orion_root.resolve()
    all4ch_image = (args.all4ch_image or cache_dir / "all" / "generated_he.png").resolve()

    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)

    cell_mask = _load_cell_mask(cache_dir, manifest)
    metrics = _load_metrics(cache_dir)
    cells = _build_cells(cache_dir, orion_root, all4ch_image, metrics, cell_mask)
    cells = _sort_cells(cells)

    html = _generate_html(cells, cache_dir.name)
    out_path = cache_dir / f"{args.output_name}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
