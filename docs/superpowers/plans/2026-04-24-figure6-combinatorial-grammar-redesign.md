# Figure 6 — Combinatorial Grammar — Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current dense `figures/pngs/09_combinatorial_grammar.png` with a redesigned 3-panel figure (reference + pixel-diff grid, L2 heatmap, horizontal-bar case studies) plus a new SI figure showing raw H&E grids for 4 representative anchors. Generate the 15 missing anchor reference images.

**Architecture:** New helper package `src/paper_figures/fig_combinatorial_grammar_panels/` (one module per panel + a `_shared.py` for selection/diff math). New SI renderer `src/paper_figures/fig_combinatorial_grammar_si.py`. New standalone reference generator `src/a3_combinatorial_sweep/generate_references.py` that reuses A3's existing inference helpers (`_load_generation_runtime`, `_load_anchor_ctrl`, `_load_anchor_uni`, `_make_generation_noise`, `_render_generated_image`) and writes to `inference_output/paired_ablation/ablation_results/<anchor>/all/generated_he.png`. Existing `fig_combinatorial_grammar.py` is rewritten as a thin orchestrator.

**Tech Stack:** Python 3, matplotlib, NumPy, PIL, pytest. Inference via existing PixCell ControlNet stack (no new model code). Conda envs: `he-multiplex` for tests/figures, `pixcell` for GPU inference (the reference-generation step only).

**Spec:** `docs/superpowers/specs/2026-04-24-figure6-combinatorial-grammar-redesign-design.md`

---

## File structure

**New:**
- `src/a3_combinatorial_sweep/generate_references.py`
- `src/paper_figures/fig_combinatorial_grammar_panels/__init__.py`
- `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py`
- `src/paper_figures/fig_combinatorial_grammar_panels/_diff_grid.py`
- `src/paper_figures/fig_combinatorial_grammar_panels/_l2_heatmap.py`
- `src/paper_figures/fig_combinatorial_grammar_panels/_case_studies.py`
- `src/paper_figures/fig_combinatorial_grammar_si.py`
- `tests/test_a3_generate_references.py`
- `tests/test_fig_combinatorial_grammar.py`

**Modified:**
- `src/paper_figures/fig_combinatorial_grammar.py` — rewrite as thin orchestrator
- `src/paper_figures/main.py` — add SI renderer call

**Output PNGs (regenerated):**
- `figures/pngs/09_combinatorial_grammar.png`
- `figures/pngs/SI_09_combinatorial_grammar_anchors.png`

---

## Task 1: Shared utilities — anchor selection + diff math

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/__init__.py`
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py`
- Create: `tests/test_fig_combinatorial_grammar.py`

- [ ] **Step 1: Write the failing test for `compute_pixel_diff`**

Create `tests/test_fig_combinatorial_grammar.py`:

```python
"""Tests for figure 6 redesign helpers and panel renderers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.paper_figures.fig_combinatorial_grammar_panels import _shared


def _write_rgb(path: Path, value: int, *, hot_box: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((32, 32, 3), value, dtype=np.uint8)
    if hot_box:
        arr[8:24, 8:24, 0] = min(255, value + 100)
    Image.fromarray(arr).save(path)


def test_compute_pixel_diff_shape_and_nonneg():
    ref = np.full((32, 32, 3), 100, dtype=np.uint8)
    cond = np.full((32, 32, 3), 130, dtype=np.uint8)
    diff = _shared.compute_pixel_diff(cond, ref)
    assert diff.shape == (32, 32)
    assert diff.dtype == np.float32
    assert np.all(diff >= 0)
    assert np.isclose(diff.mean(), 30.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py::test_compute_pixel_diff_shape_and_nonneg -v
```

Expected: FAIL with `ModuleNotFoundError: src.paper_figures.fig_combinatorial_grammar_panels`.

- [ ] **Step 3: Create empty package init**

Create `src/paper_figures/fig_combinatorial_grammar_panels/__init__.py`:

```python
"""Panel renderers and shared helpers for figure 6 (combinatorial grammar)."""
```

- [ ] **Step 4: Implement `_shared.compute_pixel_diff`**

Create `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py`:

```python
"""Shared helpers for figure 6 panel renderers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


STATES: tuple[str, ...] = ("prolif", "nonprolif", "dead")
LEVELS: tuple[str, ...] = ("low", "mid", "high")


def compute_pixel_diff(condition_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """Mean absolute per-channel difference, returned as a 2D float32 map."""
    cond = np.asarray(condition_rgb, dtype=np.float32)
    ref = np.asarray(reference_rgb, dtype=np.float32)
    if cond.shape != ref.shape:
        raise ValueError(f"shape mismatch: cond={cond.shape}, ref={ref.shape}")
    return np.mean(np.abs(cond - ref), axis=-1).astype(np.float32)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py::test_compute_pixel_diff_shape_and_nonneg -v
```

Expected: PASS.

- [ ] **Step 6: Add tests for CSV readers, condition ID, residual lookup**

Append to `tests/test_fig_combinatorial_grammar.py`:

```python
def test_condition_id_format():
    assert _shared.condition_id("prolif", "low", "high") == "prolif_low_high"


def test_residual_lookup_extracts_all_residual_columns():
    rows = [
        {
            "cell_state": "prolif",
            "oxygen_label": "low",
            "glucose_label": "low",
            "residual_l2_norm": "1.5",
            "residual_mean_cell_size": "-0.4",
            "n_anchors": "20",
        }
    ]
    lookup = _shared.residual_lookup(rows)
    key = ("prolif", "low", "low")
    assert key in lookup
    assert lookup[key]["residual_l2_norm"] == 1.5
    assert lookup[key]["residual_mean_cell_size"] == -0.4
    assert "n_anchors" not in lookup[key]
```

- [ ] **Step 7: Run tests, verify failures**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py -v
```

Expected: 2 fails (`condition_id`, `residual_lookup` missing).

- [ ] **Step 8: Implement `condition_id` and `residual_lookup`**

Append to `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py`:

```python
def condition_id(state: str, oxygen_label: str, glucose_label: str) -> str:
    return f"{state}_{oxygen_label}_{glucose_label}"


def residual_lookup(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    """Map (state, oxygen_label, glucose_label) -> {residual_*: float}."""
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["cell_state"]), str(row["oxygen_label"]), str(row["glucose_label"]))
        parsed: dict[str, float] = {}
        for name, value in row.items():
            if name.startswith("residual_") and value not in (None, ""):
                parsed[name] = float(value)
        out[key] = parsed
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))
```

- [ ] **Step 9: Run tests, verify pass**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py -v
```

Expected: 3 PASS.

- [ ] **Step 10: Add tests for representative anchor + sweep magnitude + SI selection**

Append:

```python
def _make_signature_rows(anchors_to_n_conditions: dict[str, int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    states = ("prolif", "nonprolif", "dead")
    levels = ("low", "mid", "high")
    for anchor, n_conds in anchors_to_n_conditions.items():
        i = 0
        for state in states:
            for o in levels:
                for g in levels:
                    if i >= n_conds:
                        break
                    rows.append({
                        "anchor_id": anchor,
                        "cell_state": state,
                        "oxygen_label": o,
                        "glucose_label": g,
                        "mean_cell_size": str(10.0 + i + (0 if anchor == "a0" else 5.0)),
                        "nuclear_density": str(0.1 * i),
                        "nucleus_area_median": str(20.0 + i),
                        "nucleus_area_iqr": str(2.0 + i),
                        "hematoxylin_burden": str(0.5),
                        "hematoxylin_ratio": str(0.5),
                        "eosin_ratio": str(0.5),
                        "glcm_contrast": str(1.0),
                        "glcm_homogeneity": str(0.8),
                    })
                    i += 1
    return rows


def test_pick_representative_anchor_returns_max_coverage():
    rows = _make_signature_rows({"a0": 27, "a1": 20, "a2": 27})
    assert _shared.pick_representative_anchor(rows) == "a0"


def test_compute_anchor_sweep_magnitude_returns_per_anchor_float():
    rows = _make_signature_rows({"a0": 27, "a1": 27})
    mags = _shared.compute_anchor_sweep_magnitude(rows)
    assert set(mags.keys()) == {"a0", "a1"}
    assert all(isinstance(v, float) for v in mags.values())


def test_select_si_anchors_returns_four_distinct_with_representative_first():
    anchors = {f"a{i}": 27 for i in range(20)}
    rows = _make_signature_rows(anchors)
    representative = "a0"
    picks = _shared.select_si_anchors(
        rows,
        representative_id=representative,
        reference_exists_fn=lambda _aid: True,
    )
    assert len(picks) == 4
    assert picks[0] == representative
    assert len(set(picks)) == 4


def test_select_si_anchors_skips_anchors_missing_reference():
    anchors = {f"a{i}": 27 for i in range(20)}
    rows = _make_signature_rows(anchors)
    blocked = {"a5", "a10"}
    picks = _shared.select_si_anchors(
        rows,
        representative_id="a0",
        reference_exists_fn=lambda aid: aid not in blocked,
    )
    assert all(aid not in blocked for aid in picks)
```

- [ ] **Step 11: Run tests, verify failures**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py -v
```

Expected: 4 fails (`pick_representative_anchor`, `compute_anchor_sweep_magnitude`, `select_si_anchors` missing).

- [ ] **Step 12: Implement anchor-selection helpers**

Append to `src/paper_figures/fig_combinatorial_grammar_panels/_shared.py`:

```python
MORPHOLOGY_METRICS: tuple[str, ...] = (
    "nuclear_density",
    "eosin_ratio",
    "hematoxylin_ratio",
    "hematoxylin_burden",
    "mean_cell_size",
    "nucleus_area_median",
    "nucleus_area_iqr",
    "glcm_contrast",
    "glcm_homogeneity",
)


def pick_representative_anchor(signature_rows: list[dict[str, str]]) -> str:
    """Anchor with the most rows in morphological_signatures.csv (most complete sweep)."""
    counts: dict[str, int] = {}
    for row in signature_rows:
        anchor = str(row["anchor_id"])
        counts[anchor] = counts.get(anchor, 0) + 1
    if not counts:
        raise ValueError("morphological_signatures rows are empty")
    max_count = max(counts.values())
    tied = sorted(anchor for anchor, count in counts.items() if count == max_count)
    return tied[0]


def compute_anchor_sweep_magnitude(signature_rows: list[dict[str, str]]) -> dict[str, float]:
    """Per-anchor proxy for sweep response: sum of variance across MORPHOLOGY_METRICS."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in signature_rows:
        grouped.setdefault(str(row["anchor_id"]), []).append(row)
    out: dict[str, float] = {}
    for anchor, rows in grouped.items():
        total = 0.0
        for metric in MORPHOLOGY_METRICS:
            values = [float(r[metric]) for r in rows if r.get(metric) not in (None, "")]
            if len(values) < 2:
                continue
            total += float(np.var(values, ddof=0))
        out[anchor] = total
    return out


def select_si_anchors(
    signature_rows: list[dict[str, str]],
    *,
    representative_id: str,
    reference_exists_fn,
) -> list[str]:
    """Return [representative, low, mid, high] by sweep magnitude.

    Excludes any anchor for which reference_exists_fn returns False.
    Falls back to next-ranked candidate when a pick is unavailable.
    """
    mags = compute_anchor_sweep_magnitude(signature_rows)
    eligible = sorted(
        ((aid, mag) for aid, mag in mags.items()
         if aid != representative_id and reference_exists_fn(aid)),
        key=lambda pair: pair[1],
    )
    if not eligible:
        return [representative_id] if reference_exists_fn(representative_id) else []
    n = len(eligible)
    target_indices = [0, n // 2, n - 1]
    picks: list[str] = []
    if reference_exists_fn(representative_id):
        picks.append(representative_id)
    used: set[str] = {representative_id}
    for idx in target_indices:
        for offset in range(n):
            probe = (idx + offset) % n
            candidate = eligible[probe][0]
            if candidate not in used:
                picks.append(candidate)
                used.add(candidate)
                break
    return picks
```

- [ ] **Step 13: Run tests, verify pass**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py -v
```

Expected: 7 PASS.

- [ ] **Step 14: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/ tests/test_fig_combinatorial_grammar.py
git commit -m "feat(fig6): add shared helpers for combinatorial grammar redesign"
```

---

## Task 2: Reference generator script — TDD with mocked inference

**Files:**
- Create: `src/a3_combinatorial_sweep/generate_references.py`
- Create: `tests/test_a3_generate_references.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_a3_generate_references.py`:

```python
"""Tests for the reference generator that backfills original-TME H&E renders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.a3_combinatorial_sweep import generate_references as gr


def _write_anchor_list(path: Path, anchors: list[str]) -> None:
    path.write_text("\n".join(anchors) + "\n", encoding="utf-8")


def _make_existing_reference(output_root: Path, anchor_id: str) -> Path:
    target = output_root / anchor_id / "all" / "generated_he.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(target)
    return target


def test_target_path_uses_ablation_results_layout(tmp_path: Path):
    target = gr.target_path(tmp_path, "10240_11008")
    assert target == tmp_path / "10240_11008" / "all" / "generated_he.png"


def test_plan_skips_anchors_with_existing_reference(tmp_path: Path):
    output_root = tmp_path / "ablation_results"
    _make_existing_reference(output_root, "have")
    anchors_path = tmp_path / "anchors.txt"
    _write_anchor_list(anchors_path, ["have", "missing"])

    plan = gr.plan_missing_anchors(anchors_path=anchors_path, output_root=output_root)
    assert plan == ["missing"]


def test_run_invokes_render_only_for_missing_anchors(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "ablation_results"
    _make_existing_reference(output_root, "have")
    anchors_path = tmp_path / "anchors.txt"
    _write_anchor_list(anchors_path, ["have", "missing_a", "missing_b"])

    rendered: list[str] = []

    def _fake_render_and_save(anchor_id: str, output_path: Path, **_kwargs) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(output_path)
        rendered.append(anchor_id)
        return output_path

    monkeypatch.setattr(gr, "render_and_save_reference", _fake_render_and_save)

    summary = gr.run(
        anchors_path=anchors_path,
        output_root=output_root,
        config_path=tmp_path / "config.py",
        checkpoint_dir=tmp_path / "ckpt",
        data_root=tmp_path / "data",
        device="cpu",
    )

    assert sorted(rendered) == ["missing_a", "missing_b"]
    assert summary["skipped"] == ["have"]
    assert sorted(summary["generated"]) == ["missing_a", "missing_b"]
    assert summary["failed"] == []


def test_run_logs_failures_without_aborting(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "ablation_results"
    anchors_path = tmp_path / "anchors.txt"
    _write_anchor_list(anchors_path, ["a", "b"])

    def _fake_render_and_save(anchor_id: str, output_path: Path, **_kwargs) -> Path:
        if anchor_id == "a":
            raise RuntimeError("boom")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 50, dtype=np.uint8)).save(output_path)
        return output_path

    monkeypatch.setattr(gr, "render_and_save_reference", _fake_render_and_save)

    summary = gr.run(
        anchors_path=anchors_path,
        output_root=output_root,
        config_path=tmp_path / "config.py",
        checkpoint_dir=tmp_path / "ckpt",
        data_root=tmp_path / "data",
        device="cpu",
    )

    assert summary["generated"] == ["b"]
    assert summary["failed"] == ["a"]
```

- [ ] **Step 2: Run tests, verify failures**

```bash
conda run -n he-multiplex pytest tests/test_a3_generate_references.py -v
```

Expected: 4 fails (`generate_references` module missing).

- [ ] **Step 3: Implement generator script**

Create `src/a3_combinatorial_sweep/generate_references.py`:

```python
"""Generate original-TME reference H&E images for A3 anchors.

Writes to ``inference_output/paired_ablation/ablation_results/<anchor>/all/generated_he.png``,
matching the existing paired-ablation convention. Idempotent: skips anchors that
already have a reference. Failures on individual anchors are logged but do not
abort the batch.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def target_path(output_root: Path, anchor_id: str) -> Path:
    return Path(output_root) / anchor_id / "all" / "generated_he.png"


def read_anchor_list(anchors_path: Path) -> list[str]:
    return [
        line.strip()
        for line in Path(anchors_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def plan_missing_anchors(*, anchors_path: Path, output_root: Path) -> list[str]:
    anchors = read_anchor_list(anchors_path)
    return [aid for aid in anchors if not target_path(output_root, aid).is_file()]


def render_and_save_reference(
    anchor_id: str,
    output_path: Path,
    *,
    config_path: Path,
    checkpoint_dir: Path,
    data_root: Path,
    device: str = "cuda",
    guidance_scale: float = 2.5,
    num_steps: int = 20,
    seed: int = 42,
) -> Path:
    """Render the anchor's H&E from its ORIGINAL unmodified TME channels.

    Reuses the A3 inference helpers but skips ``_build_condition_ctrl``: feeds
    ``base_ctrl`` directly into ``_render_generated_image`` so all 4 channel
    groups carry their real values.
    """
    from src.a3_combinatorial_sweep.main import (
        _load_anchor_ctrl,
        _load_anchor_uni,
        _load_generation_runtime,
        _make_generation_noise,
        _render_generated_image,
        _save_image,
    )

    models, runtime_config, scheduler, exp_channels_dir, feat_dir = _load_generation_runtime(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
        device=device,
        num_steps=num_steps,
    )
    active_channels = list(runtime_config.data.active_channels)
    base_ctrl = _load_anchor_ctrl(
        anchor_id,
        active_channels=active_channels,
        image_size=runtime_config.image_size,
        exp_channels_dir=exp_channels_dir,
    )
    uni_embeds = _load_anchor_uni(anchor_id, feat_dir=feat_dir)
    fixed_noise = _make_generation_noise(
        config=runtime_config,
        scheduler=scheduler,
        device=device,
        seed=seed,
    )
    generated = _render_generated_image(
        base_ctrl,
        models=models,
        config=runtime_config,
        scheduler=scheduler,
        uni_embeds=uni_embeds,
        device=device,
        guidance_scale=guidance_scale,
        fixed_noise=fixed_noise,
        seed=seed,
    )
    return _save_image(generated, output_path)


def run(
    *,
    anchors_path: Path,
    output_root: Path,
    config_path: Path,
    checkpoint_dir: Path,
    data_root: Path,
    device: str = "cuda",
    guidance_scale: float = 2.5,
    num_steps: int = 20,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Process every anchor in ``anchors_path``; return a per-anchor outcome summary."""
    anchors = read_anchor_list(anchors_path)
    skipped: list[str] = []
    generated: list[str] = []
    failed: list[str] = []
    for anchor_id in anchors:
        out_path = target_path(output_root, anchor_id)
        if out_path.is_file():
            skipped.append(anchor_id)
            LOGGER.info("skip %s (reference already exists)", anchor_id)
            continue
        try:
            render_and_save_reference(
                anchor_id,
                out_path,
                config_path=config_path,
                checkpoint_dir=checkpoint_dir,
                data_root=data_root,
                device=device,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed,
            )
            generated.append(anchor_id)
            LOGGER.info("generated reference for %s -> %s", anchor_id, out_path)
        except Exception:
            LOGGER.exception("reference generation failed for %s", anchor_id)
            failed.append(anchor_id)
    return {"skipped": skipped, "generated": generated, "failed": failed}


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_cli(argv)
    summary = run(
        anchors_path=args.anchors,
        output_root=args.output_root,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, verify pass**

```bash
conda run -n he-multiplex pytest tests/test_a3_generate_references.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/a3_combinatorial_sweep/generate_references.py tests/test_a3_generate_references.py
git commit -m "feat(a3): add reference generator for missing anchor original-TME renders"
```

---

## Task 3: Run reference generator on real data

This task uses GPU. Run inside the `pixcell` conda env.

- [ ] **Step 1: Locate the latest checkpoint dir**

```bash
ls -d checkpoints/pixcell_controlnet_exp/checkpoints/step_* 2>/dev/null | sort | tail -1
```

Expected: prints one path, e.g. `checkpoints/pixcell_controlnet_exp/checkpoints/step_NNNNNN`. Save as `$CKPT`.

- [ ] **Step 2: Verify how many references are already on disk**

```bash
A3_ANCHORS=src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt
ABL_DIR=inference_output/paired_ablation/ablation_results
have=0; missing=0
while IFS= read -r anchor; do
  [ -z "$anchor" ] && continue
  if [ -f "$ABL_DIR/$anchor/all/generated_he.png" ]; then
    have=$((have+1))
  else
    missing=$((missing+1))
  fi
done < "$A3_ANCHORS"
echo "have=$have missing=$missing"
```

Expected: `have=5 missing=15` (or whatever the current state is — the script is idempotent).

- [ ] **Step 3: Run the generator**

```bash
conda run -n pixcell python -m src.a3_combinatorial_sweep.generate_references \
  --anchors src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt \
  --output-root inference_output/paired_ablation/ablation_results \
  --config configs/config_controlnet_exp.py \
  --checkpoint-dir <PASTE $CKPT FROM STEP 1> \
  --data-root data/orion-crc33 \
  --device cuda
```

Expected: prints a JSON summary with `skipped` (existing) + `generated` (newly created) + `failed` (empty). Runtime on T4: ~5-10 min for 15 missing anchors.

- [ ] **Step 4: Verify all 20 anchors covered**

```bash
A3_ANCHORS=src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt
ABL_DIR=inference_output/paired_ablation/ablation_results
missing=0
while IFS= read -r anchor; do
  [ -z "$anchor" ] && continue
  [ -f "$ABL_DIR/$anchor/all/generated_he.png" ] || { echo "MISSING $anchor"; missing=$((missing+1)); }
done < "$A3_ANCHORS"
echo "missing=$missing"
```

Expected: `missing=0`.

- [ ] **Step 5: Commit no code (data only — generated PNGs may or may not be tracked)**

If `inference_output/` is gitignored, no commit. Otherwise:

```bash
git status inference_output/paired_ablation/ablation_results/ | head
```

If files are tracked, ask the user before committing 15 PNGs. Otherwise note "data step complete" in PR description.

---

## Task 4: Panel A renderer — diff grid + reference inset

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/_diff_grid.py`
- Modify: `tests/test_fig_combinatorial_grammar.py`

- [ ] **Step 1: Write failing test for `render_panel_a`**

Append to `tests/test_fig_combinatorial_grammar.py`:

```python
def _write_sweep_tile(generated_root: Path, anchor: str, state: str, o: str, g: str, value: int) -> None:
    arr = np.full((16, 16, 3), value, dtype=np.uint8)
    out = generated_root / anchor / f"{state}_{o}_{g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out)


def _populate_anchor_sweep(generated_root: Path, anchor: str) -> None:
    states = ("prolif", "nonprolif", "dead")
    levels = ("low", "mid", "high")
    base = 100
    i = 0
    for state in states:
        for o in levels:
            for g in levels:
                _write_sweep_tile(generated_root, anchor, state, o, g, base + i)
                i += 1


def test_render_panel_a_creates_axes_for_grid_plus_reference(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.paper_figures.fig_combinatorial_grammar_panels._diff_grid import render_panel_a

    generated_root = tmp_path / "generated"
    ablation_root = tmp_path / "ablation_results"
    anchor = "anchor_x"
    _populate_anchor_sweep(generated_root, anchor)
    ref_path = ablation_root / anchor / "all" / "generated_he.png"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)).save(ref_path)

    fig = plt.figure(figsize=(6, 4))
    outer = fig.add_gridspec(1, 1)
    render_panel_a(
        fig,
        outer[0, 0],
        anchor_id=anchor,
        generated_root=generated_root,
        reference_path=ref_path,
    )
    fig.canvas.draw()
    n_axes = len(fig.axes)
    plt.close(fig)
    # 1 outer + 27 grid + 1 reference inset + 1 colorbar = 30
    assert n_axes >= 29
```

- [ ] **Step 2: Run test, verify failure**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py::test_render_panel_a_creates_axes_for_grid_plus_reference -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_diff_grid.render_panel_a`**

Create `src/paper_figures/fig_combinatorial_grammar_panels/_diff_grid.py`:

```python
"""Panel A — reference H&E inset + 3x9 pixel-diff grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tools.ablation_report.shared import INK, plt

from . import _shared

STATES = _shared.STATES
LEVELS = _shared.LEVELS


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.05, 1.02, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=13, fontweight="bold", color=INK,
    )


def _draw_dashed_border(ax: plt.Axes) -> None:
    rect = Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        transform=ax.transAxes,
        fill=False, linestyle="--", linewidth=0.8,
        edgecolor="#9A9A9A",
    )
    ax.add_patch(rect)


def render_panel_a(
    fig: plt.Figure,
    subgrid,
    *,
    anchor_id: str,
    generated_root: Path,
    reference_path: Path,
) -> None:
    """Render reference inset above a 3x9 diff heatmap grid; share a colorbar."""
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "A")
    _draw_dashed_border(outer_ax)

    reference_rgb = _shared.load_rgb(reference_path)

    diffs: dict[tuple[int, int], np.ndarray] = {}
    vmax = 0.0
    for state_idx, state in enumerate(STATES):
        for o_idx, o in enumerate(LEVELS):
            for g_idx, g in enumerate(LEVELS):
                col = o_idx * len(LEVELS) + g_idx
                cond_path = generated_root / anchor_id / f"{_shared.condition_id(state, o, g)}.png"
                cond_rgb = _shared.load_rgb(cond_path)
                diff = _shared.compute_pixel_diff(cond_rgb, reference_rgb)
                diffs[(state_idx, col)] = diff
                vmax = max(vmax, float(diff.max()))

    inner = subgrid.subgridspec(
        4, 10,
        height_ratios=[1.1, 1.0, 1.0, 1.0],
        width_ratios=[1.0] * 9 + [0.08],
        hspace=0.06, wspace=0.04,
    )

    ref_ax = fig.add_subplot(inner[0, 0])
    ref_ax.imshow(reference_rgb)
    ref_ax.set_xticks([])
    ref_ax.set_yticks([])
    for spine in ref_ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#6A6A6A")
    ref_ax.set_title("reference (original TME)", fontsize=7, color=INK, pad=2.0)

    grid_axes: list[plt.Axes] = []
    last_im = None
    for state_idx, state in enumerate(STATES):
        for col in range(9):
            ax = fig.add_subplot(inner[state_idx + 1, col])
            im = ax.imshow(
                diffs[(state_idx, col)],
                cmap="magma", vmin=0.0, vmax=max(vmax, 1e-6),
                aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.25)
                spine.set_edgecolor("#8A8A8A")
            if col == 0:
                ax.set_ylabel(state, fontsize=6.5, color=INK)
            grid_axes.append(ax)
            last_im = im

    cbar_ax = fig.add_subplot(inner[1:4, 9])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6, colors=INK)
    cbar.set_label("|cond − ref| (mean abs RGB)", fontsize=6.5, color=INK)
```

- [ ] **Step 4: Run test, verify pass**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py::test_render_panel_a_creates_axes_for_grid_plus_reference -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/_diff_grid.py tests/test_fig_combinatorial_grammar.py
git commit -m "feat(fig6): panel A renderer — reference inset + 3x9 diff grid"
```

---

## Task 5: Panel B renderer — 3×9 L2 heatmap with cell text

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/_l2_heatmap.py`
- Modify: `tests/test_fig_combinatorial_grammar.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_fig_combinatorial_grammar.py`:

```python
def _make_residual_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    states = ("prolif", "nonprolif", "dead")
    levels = ("low", "mid", "high")
    i = 0
    for state in states:
        for o in levels:
            for g in levels:
                rows.append({
                    "cell_state": state,
                    "oxygen_label": o,
                    "glucose_label": g,
                    "residual_l2_norm": str(1.0 + i),
                    "residual_mean_cell_size": str(0.3 - 0.01 * i),
                    "residual_nucleus_area_median": str(0.2 + 0.01 * i),
                    "residual_nucleus_area_iqr": str(-0.05 - 0.005 * i),
                    "residual_nuclear_density": str(0.001 * i),
                    "residual_hematoxylin_burden": str(0.001),
                    "residual_hematoxylin_ratio": str(0.0005),
                    "residual_eosin_ratio": str(-0.001),
                    "residual_glcm_contrast": str(0.0001),
                    "residual_glcm_homogeneity": str(0.00005),
                })
                i += 1
    return rows


def test_render_panel_b_creates_heatmap_axes(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.paper_figures.fig_combinatorial_grammar_panels._l2_heatmap import render_panel_b

    fig = plt.figure(figsize=(6, 3))
    outer = fig.add_gridspec(1, 1)
    render_panel_b(fig, outer[0, 0], residual_rows=_make_residual_rows())
    fig.canvas.draw()
    # outer + heatmap + colorbar
    assert len(fig.axes) >= 3
    plt.close(fig)
```

- [ ] **Step 2: Run, verify failure**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py::test_render_panel_b_creates_heatmap_axes -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `_l2_heatmap.render_panel_b`**

Create `src/paper_figures/fig_combinatorial_grammar_panels/_l2_heatmap.py`:

```python
"""Panel B — 3x9 residual-L2 heatmap with per-cell numeric labels."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tools.ablation_report.shared import INK, plt

from . import _shared

STATES = _shared.STATES
LEVELS = _shared.LEVELS


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.05, 1.02, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=13, fontweight="bold", color=INK,
    )


def _draw_dashed_border(ax: plt.Axes) -> None:
    rect = Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        transform=ax.transAxes,
        fill=False, linestyle="--", linewidth=0.8,
        edgecolor="#9A9A9A",
    )
    ax.add_patch(rect)


def render_panel_b(fig: plt.Figure, subgrid, *, residual_rows: list[dict[str, str]]) -> None:
    """Render the 3x9 residual L2 heatmap, sharing the (state, oxygen/glucose) axes with panel A."""
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "B")
    _draw_dashed_border(outer_ax)

    inner = subgrid.subgridspec(1, 1)
    ax = fig.add_subplot(inner[0, 0])

    lookup = _shared.residual_lookup(residual_rows)
    matrix = np.zeros((len(STATES), len(LEVELS) * len(LEVELS)), dtype=np.float64)
    for state_idx, state in enumerate(STATES):
        for o_idx, o in enumerate(LEVELS):
            for g_idx, g in enumerate(LEVELS):
                col = o_idx * len(LEVELS) + g_idx
                residuals = lookup.get((state, o, g), {})
                matrix[state_idx, col] = float(residuals.get("residual_l2_norm", 0.0))

    vmax = float(matrix.max()) if matrix.size else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels(STATES, fontsize=6.5, color=INK)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        [f"{o}/{g}" for o in LEVELS for g in LEVELS],
        rotation=30, ha="right", fontsize=6, color=INK,
    )
    ax.set_title(
        "interaction magnitude — residual L2 norm "
        "(sweep levels: low=0.50, mid=0.75, high=1.00)",
        fontsize=7, loc="left", color=INK, pad=4.0,
    )
    ax.grid(False)

    threshold = 0.5 * vmax
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value < threshold else "black"
            ax.text(j, i, f"{value:.2g}", ha="center", va="center",
                    fontsize=6.5, color=text_color)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6, colors=INK)
```

- [ ] **Step 4: Run, verify pass**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py::test_render_panel_b_creates_heatmap_axes -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/_l2_heatmap.py tests/test_fig_combinatorial_grammar.py
git commit -m "feat(fig6): panel B renderer — L2 heatmap with cell text labels"
```

---

## Task 6: Panel C renderer — 3 horizontal-bar case studies

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_panels/_case_studies.py`
- Modify: `tests/test_fig_combinatorial_grammar.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_fig_combinatorial_grammar.py`:

```python
def test_select_case_rows_returns_low_mid_high():
    from src.paper_figures.fig_combinatorial_grammar_panels._case_studies import select_case_rows

    rows = _make_residual_rows()
    cases = select_case_rows(rows)
    labels = [label for label, _ in cases]
    assert labels == ["lowest", "median", "highest"]
    l2_values = [float(row["residual_l2_norm"]) for _, row in cases]
    assert l2_values == sorted(l2_values)


def test_render_panel_c_creates_three_subplots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.paper_figures.fig_combinatorial_grammar_panels._case_studies import render_panel_c

    fig = plt.figure(figsize=(4, 8))
    outer = fig.add_gridspec(1, 1)
    render_panel_c(fig, outer[0, 0], residual_rows=_make_residual_rows())
    fig.canvas.draw()
    # outer + 3 case subplots
    assert len(fig.axes) >= 4
    plt.close(fig)
```

- [ ] **Step 2: Run, verify failures**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py -k "case" -v
```

Expected: 2 fails.

- [ ] **Step 3: Implement `_case_studies`**

Create `src/paper_figures/fig_combinatorial_grammar_panels/_case_studies.py`:

```python
"""Panel C — three horizontal-bar case studies (lowest/median/highest L2)."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle

from tools.ablation_report.shared import INK, SOFT_GRID, plt

from . import _shared


_METRIC_LABELS: dict[str, str] = {
    "residual_nuclear_density": "nuclear density",
    "residual_mean_cell_size": "mean cell size",
    "residual_nucleus_area_median": "nucleus area median",
    "residual_nucleus_area_iqr": "nucleus area IQR",
    "residual_hematoxylin_burden": "hematoxylin burden",
    "residual_hematoxylin_ratio": "hematoxylin ratio",
    "residual_eosin_ratio": "eosin ratio",
    "residual_glcm_contrast": "GLCM contrast",
    "residual_glcm_homogeneity": "GLCM homogeneity",
}


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.05, 1.02, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=13, fontweight="bold", color=INK,
    )


def _draw_dashed_border(ax: plt.Axes) -> None:
    rect = Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        transform=ax.transAxes,
        fill=False, linestyle="--", linewidth=0.8,
        edgecolor="#9A9A9A",
    )
    ax.add_patch(rect)


def select_case_rows(residual_rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    """Return [(lowest, row), (median, row), (highest, row)] by residual_l2_norm."""
    if not residual_rows:
        return []
    sorted_rows = sorted(
        residual_rows,
        key=lambda row: float(row.get("residual_l2_norm", 0.0) or 0.0),
    )
    n = len(sorted_rows)
    indices = [0, n // 2, n - 1]
    labels = ["lowest", "median", "highest"]
    out: list[tuple[str, dict[str, str]]] = []
    used: set[int] = set()
    for label, idx in zip(labels, indices, strict=True):
        if idx in used:
            continue
        used.add(idx)
        out.append((label, sorted_rows[idx]))
    return out


def render_panel_c(fig: plt.Figure, subgrid, *, residual_rows: list[dict[str, str]]) -> None:
    """Render 3 stacked horizontal-bar case-study subplots."""
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    _panel_label(outer_ax, "C")
    _draw_dashed_border(outer_ax)
    outer_ax.text(
        0.0, 1.01,
        "Signed residuals — lowest / median / highest L2 (sorted per-row by |residual|)",
        transform=outer_ax.transAxes,
        fontsize=7.5, ha="left", va="bottom", color=INK,
    )

    cases = select_case_rows(residual_rows)
    if not cases:
        return

    inner = subgrid.subgridspec(len(cases), 1, hspace=0.55)
    for row_idx, (case_label, case_row) in enumerate(cases):
        ax = fig.add_subplot(inner[row_idx, 0])
        state = str(case_row["cell_state"])
        o = str(case_row["oxygen_label"])
        g = str(case_row["glucose_label"])
        l2 = float(case_row.get("residual_l2_norm", 0.0) or 0.0)

        ranked: list[tuple[float, str, float]] = []
        for key, label in _METRIC_LABELS.items():
            value = case_row.get(key)
            if value in ("", None):
                continue
            v = float(value)
            ranked.append((abs(v), label, v))
        ranked.sort(reverse=True)
        labels = [label for _, label, _ in ranked]
        values = [v for _, _, v in ranked]
        y = np.arange(len(labels), dtype=np.float64)

        ax.barh(y, values, height=0.66, color="#4C78A8", edgecolor="black", linewidth=0.5)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5, color=INK)
        ax.invert_yaxis()  # most-extreme metric at top
        ax.tick_params(axis="x", labelsize=6.5, colors=INK)
        ax.set_title(
            f"{case_label}: {state}, O2={o}, glucose={g}, L2={l2:.3g}",
            fontsize=7, loc="left", color=INK,
        )
        ax.grid(axis="x", color=SOFT_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
```

- [ ] **Step 4: Run, verify pass**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py -k "case" -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_panels/_case_studies.py tests/test_fig_combinatorial_grammar.py
git commit -m "feat(fig6): panel C renderer — horizontal-bar case studies"
```

---

## Task 7: Rewrite main figure orchestrator

**Files:**
- Modify: `src/paper_figures/fig_combinatorial_grammar.py`
- Modify: `tests/test_fig_combinatorial_grammar.py`

- [ ] **Step 1: Write failing end-to-end test for `build_combinatorial_grammar_figure`**

Append to `tests/test_fig_combinatorial_grammar.py`:

```python
def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_save_combinatorial_grammar_figure_renders_main_png(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")

    from src.paper_figures.fig_combinatorial_grammar import save_combinatorial_grammar_figure

    generated_root = tmp_path / "generated"
    ablation_root = tmp_path / "ablation_results"
    representative = "anchor_0"
    _populate_anchor_sweep(generated_root, representative)

    ref = ablation_root / representative / "all" / "generated_he.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)).save(ref)

    sig_rows = _make_signature_rows({representative: 27})
    res_rows = _make_residual_rows()
    sig_path = tmp_path / "signatures.csv"
    res_path = tmp_path / "residuals.csv"
    _write_csv(sig_path, sig_rows)
    _write_csv(res_path, res_rows)

    out_png = tmp_path / "fig.png"
    result = save_combinatorial_grammar_figure(
        out_png=out_png,
        generated_root=generated_root,
        signatures_csv=sig_path,
        residuals_csv=res_path,
        ablation_root=ablation_root,
        dpi=80,
    )
    assert result == out_png
    assert out_png.is_file()
    with Image.open(out_png) as img:
        assert img.width > 400
        assert img.height > 200
```

- [ ] **Step 2: Run, verify failure**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py::test_save_combinatorial_grammar_figure_renders_main_png -v
```

Expected: FAIL (signature mismatch — current `save_combinatorial_grammar_figure` doesn't accept `ablation_root`).

- [ ] **Step 3: Rewrite `fig_combinatorial_grammar.py`**

Replace the entire contents of `src/paper_figures/fig_combinatorial_grammar.py`:

```python
"""Figure 6: Combinatorial Grammar — Emergent Signatures (redesigned)."""
from __future__ import annotations

from pathlib import Path

from tools.ablation_report.shared import plt

from src.paper_figures.fig_combinatorial_grammar_panels import _shared
from src.paper_figures.fig_combinatorial_grammar_panels._case_studies import render_panel_c
from src.paper_figures.fig_combinatorial_grammar_panels._diff_grid import render_panel_a
from src.paper_figures.fig_combinatorial_grammar_panels._l2_heatmap import render_panel_b


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A3_OUT = ROOT / "src" / "a3_combinatorial_sweep" / "out"
DEFAULT_GENERATED_ROOT = DEFAULT_A3_OUT / "generated"
DEFAULT_SIGNATURES_CSV = DEFAULT_A3_OUT / "morphological_signatures.csv"
DEFAULT_RESIDUALS_CSV = DEFAULT_A3_OUT / "additive_model_residuals.csv"
DEFAULT_ABLATION_ROOT = ROOT / "inference_output" / "paired_ablation" / "ablation_results"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs" / "09_combinatorial_grammar.png"


def _reference_path(ablation_root: Path, anchor_id: str) -> Path:
    return Path(ablation_root) / anchor_id / "all" / "generated_he.png"


def build_combinatorial_grammar_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    residuals_csv: Path = DEFAULT_RESIDUALS_CSV,
    ablation_root: Path = DEFAULT_ABLATION_ROOT,
) -> plt.Figure:
    generated_root = Path(generated_root)
    signatures_csv = Path(signatures_csv)
    residuals_csv = Path(residuals_csv)
    ablation_root = Path(ablation_root)

    if not signatures_csv.is_file():
        raise FileNotFoundError(f"missing signatures csv: {signatures_csv}")
    if not residuals_csv.is_file():
        raise FileNotFoundError(f"missing residuals csv: {residuals_csv}")

    signature_rows = _shared.read_csv(signatures_csv)
    residual_rows = _shared.read_csv(residuals_csv)
    anchor_id = _shared.pick_representative_anchor(signature_rows)

    reference_path = _reference_path(ablation_root, anchor_id)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"missing reference render for representative anchor {anchor_id}: {reference_path}\n"
            "Run: python -m src.a3_combinatorial_sweep.generate_references "
            "--anchors src/a3_combinatorial_sweep/anchors_k20_t1_medoid.txt "
            f"--output-root {ablation_root} ..."
        )

    fig = plt.figure(figsize=(12.0, 9.0), facecolor="white")
    outer = fig.add_gridspec(
        2, 2,
        width_ratios=[1.6, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.22, hspace=0.30,
    )

    render_panel_a(
        fig, outer[0, 0],
        anchor_id=anchor_id,
        generated_root=generated_root,
        reference_path=reference_path,
    )
    render_panel_b(fig, outer[1, 0], residual_rows=residual_rows)
    render_panel_c(fig, outer[:, 1], residual_rows=residual_rows)

    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.06, top=0.95)
    return fig


def save_combinatorial_grammar_figure(
    *,
    out_png: Path = DEFAULT_OUT_PNG,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    residuals_csv: Path = DEFAULT_RESIDUALS_CSV,
    ablation_root: Path = DEFAULT_ABLATION_ROOT,
    dpi: int = 300,
) -> Path:
    fig = build_combinatorial_grammar_figure(
        generated_root=generated_root,
        signatures_csv=signatures_csv,
        residuals_csv=residuals_csv,
        ablation_root=ablation_root,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png
```

- [ ] **Step 4: Run, verify pass**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar.py tests/test_fig_combinatorial_grammar.py
git commit -m "refactor(fig6): rewrite main orchestrator using panel modules"
```

---

## Task 8: SI figure — 4-anchor raw H&E grids

**Files:**
- Create: `src/paper_figures/fig_combinatorial_grammar_si.py`
- Modify: `tests/test_fig_combinatorial_grammar.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_fig_combinatorial_grammar.py`:

```python
def test_save_combinatorial_grammar_si_figure_renders(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")

    from src.paper_figures.fig_combinatorial_grammar_si import save_combinatorial_grammar_si_figure

    generated_root = tmp_path / "generated"
    ablation_root = tmp_path / "ablation_results"
    anchors = [f"a{i}" for i in range(20)]
    for aid in anchors:
        _populate_anchor_sweep(generated_root, aid)
        ref = ablation_root / aid / "all" / "generated_he.png"
        ref.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)).save(ref)

    sig_rows = _make_signature_rows({aid: 27 for aid in anchors})
    sig_path = tmp_path / "signatures.csv"
    _write_csv(sig_path, sig_rows)

    out_png = tmp_path / "si.png"
    result = save_combinatorial_grammar_si_figure(
        out_png=out_png,
        generated_root=generated_root,
        signatures_csv=sig_path,
        ablation_root=ablation_root,
        dpi=80,
    )
    assert result == out_png
    assert out_png.is_file()
```

- [ ] **Step 2: Run, verify failure**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py::test_save_combinatorial_grammar_si_figure_renders -v
```

Expected: FAIL.

- [ ] **Step 3: Implement SI renderer**

Create `src/paper_figures/fig_combinatorial_grammar_si.py`:

```python
"""SI Figure for figure 6 — raw H&E sweep grids for 4 representative anchors."""
from __future__ import annotations

from pathlib import Path

from matplotlib.patches import Rectangle

from tools.ablation_report.shared import INK, plt

from src.paper_figures.fig_combinatorial_grammar_panels import _shared


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A3_OUT = ROOT / "src" / "a3_combinatorial_sweep" / "out"
DEFAULT_GENERATED_ROOT = DEFAULT_A3_OUT / "generated"
DEFAULT_SIGNATURES_CSV = DEFAULT_A3_OUT / "morphological_signatures.csv"
DEFAULT_ABLATION_ROOT = ROOT / "inference_output" / "paired_ablation" / "ablation_results"
DEFAULT_OUT_PNG = ROOT / "figures" / "pngs" / "SI_09_combinatorial_grammar_anchors.png"

STATES = _shared.STATES
LEVELS = _shared.LEVELS


def _draw_anchor_subgrid(
    fig: plt.Figure,
    subgrid,
    *,
    anchor_id: str,
    generated_root: Path,
    title: str,
) -> None:
    outer_ax = fig.add_subplot(subgrid)
    outer_ax.axis("off")
    outer_ax.text(
        0.0, 1.02, title,
        transform=outer_ax.transAxes,
        ha="left", va="bottom",
        fontsize=8, color=INK,
    )

    inner = subgrid.subgridspec(3, 9, hspace=0.04, wspace=0.04)
    for state_idx, state in enumerate(STATES):
        for o_idx, o in enumerate(LEVELS):
            for g_idx, g in enumerate(LEVELS):
                col = o_idx * len(LEVELS) + g_idx
                tile_path = generated_root / anchor_id / f"{_shared.condition_id(state, o, g)}.png"
                ax = fig.add_subplot(inner[state_idx, col])
                ax.imshow(_shared.load_rgb(tile_path))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.25)
                    spine.set_edgecolor("#8A8A8A")
                if col == 0:
                    ax.set_ylabel(state, fontsize=6.0, color=INK)
                if state_idx == 0:
                    ax.set_title(f"{o}/{g}", fontsize=5.5, pad=1.2, color=INK)


def build_combinatorial_grammar_si_figure(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    ablation_root: Path = DEFAULT_ABLATION_ROOT,
) -> plt.Figure:
    generated_root = Path(generated_root)
    signatures_csv = Path(signatures_csv)
    ablation_root = Path(ablation_root)

    if not signatures_csv.is_file():
        raise FileNotFoundError(f"missing signatures csv: {signatures_csv}")
    signature_rows = _shared.read_csv(signatures_csv)
    representative = _shared.pick_representative_anchor(signature_rows)

    def _has_reference(aid: str) -> bool:
        return (ablation_root / aid / "all" / "generated_he.png").is_file()

    picks = _shared.select_si_anchors(
        signature_rows,
        representative_id=representative,
        reference_exists_fn=_has_reference,
    )
    mags = _shared.compute_anchor_sweep_magnitude(signature_rows)

    fig = plt.figure(figsize=(14.0, 10.0), facecolor="white")
    outer = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.20)

    role_labels = {0: "representative", 1: "low magnitude", 2: "mid magnitude", 3: "high magnitude"}
    for idx, anchor_id in enumerate(picks):
        if idx >= 4:
            break
        row, col = divmod(idx, 2)
        title = (
            f"{role_labels.get(idx, '')} — anchor {anchor_id} "
            f"(sweep magnitude={mags.get(anchor_id, 0.0):.3g})"
        )
        _draw_anchor_subgrid(
            fig, outer[row, col],
            anchor_id=anchor_id,
            generated_root=generated_root,
            title=title,
        )

    fig.text(0.02, 0.97, "Figure S6", fontsize=10, color=INK, ha="left", va="top", fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.94)
    return fig


def save_combinatorial_grammar_si_figure(
    *,
    out_png: Path = DEFAULT_OUT_PNG,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    signatures_csv: Path = DEFAULT_SIGNATURES_CSV,
    ablation_root: Path = DEFAULT_ABLATION_ROOT,
    dpi: int = 300,
) -> Path:
    fig = build_combinatorial_grammar_si_figure(
        generated_root=generated_root,
        signatures_csv=signatures_csv,
        ablation_root=ablation_root,
    )
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png
```

- [ ] **Step 4: Run, verify pass**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_combinatorial_grammar_si.py tests/test_fig_combinatorial_grammar.py
git commit -m "feat(fig6): add SI figure with raw H&E grids for 4 anchors"
```

---

## Task 9: Wire SI renderer into `paper_figures/main.py`

**Files:**
- Modify: `src/paper_figures/main.py`

- [ ] **Step 1: Read current main.py to find the exact line to insert near**

```bash
grep -n "save_combinatorial_grammar_figure\|09_combinatorial_grammar" src/paper_figures/main.py
```

Expected: shows the import line and the existing `save_combinatorial_grammar_figure(out_png=PNG_DIR / "09_combinatorial_grammar.png")` call.

- [ ] **Step 2: Add SI import**

In `src/paper_figures/main.py`, change:

```python
from src.paper_figures.fig_combinatorial_grammar import save_combinatorial_grammar_figure
```

to:

```python
from src.paper_figures.fig_combinatorial_grammar import save_combinatorial_grammar_figure
from src.paper_figures.fig_combinatorial_grammar_si import save_combinatorial_grammar_si_figure
```

- [ ] **Step 3: Add the SI render call immediately after the main render call**

After the line:

```python
        save_combinatorial_grammar_figure(out_png=PNG_DIR / "09_combinatorial_grammar.png")
```

add:

```python
        save_combinatorial_grammar_si_figure(out_png=PNG_DIR / "SI_09_combinatorial_grammar_anchors.png")
```

(preserve any surrounding try/except — match the existing pattern around fig 6.)

- [ ] **Step 4: Run main.py to confirm it imports cleanly**

```bash
conda run -n he-multiplex python -c "from src.paper_figures import main as _; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/main.py
git commit -m "feat(fig6): wire SI renderer into paper_figures.main"
```

---

## Task 10: Render real PNGs and visual review

- [ ] **Step 1: Render the main fig from real data**

```bash
conda run -n he-multiplex python -c "
from src.paper_figures.fig_combinatorial_grammar import save_combinatorial_grammar_figure
print(save_combinatorial_grammar_figure())
"
```

Expected: prints the path `figures/pngs/09_combinatorial_grammar.png`. No exceptions.

- [ ] **Step 2: Render the SI fig from real data**

```bash
conda run -n he-multiplex python -c "
from src.paper_figures.fig_combinatorial_grammar_si import save_combinatorial_grammar_si_figure
print(save_combinatorial_grammar_si_figure())
"
```

Expected: prints `figures/pngs/SI_09_combinatorial_grammar_anchors.png`. No exceptions.

- [ ] **Step 3: Inspect both PNGs with the Read tool**

Use Claude's Read tool on:
- `figures/pngs/09_combinatorial_grammar.png`
- `figures/pngs/SI_09_combinatorial_grammar_anchors.png`

Sanity-check:
- Main: panel A reference inset visible top-left; 3×9 diff grid below; panel B 3×9 heatmap with numeric L2 in each cell; panel C 3 horizontal-bar subplots on the right; no clipped titles; dashed borders on panels.
- SI: 4 anchor grids in 2×2 layout; titles include anchor IDs and sweep magnitudes; tile borders soft gray.

If anything is clipped or overlapping, adjust `subplots_adjust` margins or `wspace`/`hspace` in the relevant renderer and re-render.

- [ ] **Step 4: Commit the regenerated PNGs**

```bash
git status figures/pngs/09_combinatorial_grammar.png figures/pngs/SI_09_combinatorial_grammar_anchors.png
git add figures/pngs/09_combinatorial_grammar.png figures/pngs/SI_09_combinatorial_grammar_anchors.png
git commit -m "chore(fig6): regenerate main + SI PNGs from redesigned renderer"
```

- [ ] **Step 5: Run the full focused test suite once more**

```bash
conda run -n he-multiplex pytest tests/test_fig_combinatorial_grammar.py tests/test_a3_generate_references.py -v
```

Expected: all PASS.

---

## Self-review notes

- Spec coverage: §Goals 1 (panel A diff) → Task 4; 2 (panel B share row-axis) → Task 5; 3 (panel C horizontal bars) → Task 6; 4 (SI 4 anchors) → Task 8; 5 (Fig-5 polish) → applied across Tasks 4/5/6 (dashed borders, INK colors, height-matched colorbars, no suptitle, inline panel titles).
- Reference generation: Task 2 (script + tests) + Task 3 (run on real data).
- Wiring: Task 9.
- Regeneration + visual review: Task 10.
- Function names consistent: `render_panel_a/b/c`, `save_combinatorial_grammar_figure`, `save_combinatorial_grammar_si_figure`, `compute_pixel_diff`, `pick_representative_anchor`, `compute_anchor_sweep_magnitude`, `select_si_anchors`, `target_path`, `plan_missing_anchors`, `render_and_save_reference`, `run`.
- All test code shown verbatim; all implementation code shown verbatim; no "similar to Task N" placeholders.
