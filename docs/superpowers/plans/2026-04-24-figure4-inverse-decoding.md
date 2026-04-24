# Figure 4 — Inverse Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `figures/pngs/07_inverse_decoding.png` — a two-panel figure showing (a) T1 linear probe R² for 3–4 encoders with 95% CI, and (b) T2 CODEX marker MLP probe R² as diverging bars.

**Architecture:** New helper `run_encoder_probe_to_csv()` in the existing `src/a1_probe_encoders/main.py` runs any embeddings file through the shared CV pipeline and writes a standardised CSV. A new standalone figure module `src/paper_figures/fig_inverse_decoding.py` reads those CSVs and renders both panels. `src/paper_figures/main.py` calls it.

**Tech Stack:** Python 3.10+, numpy, matplotlib, torchvision (ResNet-50), scikit-learn (Ridge CV), conda env `he-multiplex` for probe runs, `pixcell` for ResNet-50 extraction.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/a1_probe_encoders/main.py` | Modify | Add `run_encoder_probe_to_csv()` and `extract_resnet50_embeddings()` + `run_resnet50_worker()` |
| `src/paper_figures/fig_inverse_decoding.py` | Create | Load CSVs, render both panels, return `Figure` |
| `src/paper_figures/main.py` | Modify | Call `build_inverse_decoding_figure()`, save `07_inverse_decoding.png` |
| `tests/test_encoder_probe_csv.py` | Create | Unit tests for `run_encoder_probe_to_csv()` and `extract_resnet50_embeddings()` |
| `tests/test_fig_inverse_decoding.py` | Create | Unit tests for figure builder with mock CSV data |

---

## Task 1 — `run_encoder_probe_to_csv()` helper

Adds a reusable function to `src/a1_probe_encoders/main.py` that runs the standard 5-fold CV Ridge probe on **any** `.npy` embeddings file and writes a CSV with columns `target, r2_mean, r2_sd, n_valid_folds`. Used in Tasks 3 and 4.

**Files:**
- Modify: `src/a1_probe_encoders/main.py`
- Create: `tests/test_encoder_probe_csv.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_encoder_probe_csv.py
"""Tests for run_encoder_probe_to_csv."""
from __future__ import annotations
import csv
import json
import numpy as np
import pytest
from pathlib import Path
from src.a1_probe_encoders.main import run_encoder_probe_to_csv


def _write_fake_cv_splits(path: Path, tile_ids: list[str]) -> None:
    from src._tasklib.tile_ids import tile_ids_sha1
    n = len(tile_ids)
    mid = n // 2
    splits = [
        {"train_idx": list(range(mid)), "test_idx": list(range(mid, n))},
        {"train_idx": list(range(mid, n)), "test_idx": list(range(mid))},
    ]
    path.write_text(
        json.dumps({
            "version": 1,
            "tile_count": n,
            "tile_ids_sha1": tile_ids_sha1(tile_ids),
            "block_size_px": 2048,
            "n_splits": 2,
            "splits": splits,
        }),
        encoding="utf-8",
    )


def _write_fake_manifest(cv_path: Path, target_names: list[str]) -> None:
    cv_path.with_name("manifest.json").write_text(
        json.dumps({"target_names": target_names}), encoding="utf-8"
    )


def test_run_encoder_probe_to_csv_creates_correct_schema(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_tiles, n_dim, n_targets = 20, 8, 3
    target_names = ["cell_density", "prolif_frac", "immune_frac"]
    tile_ids = [f"tile_{i:04d}" for i in range(n_tiles)]

    embeddings = rng.standard_normal((n_tiles, n_dim)).astype(np.float32)
    targets = rng.random((n_tiles, n_targets)).astype(np.float32)

    emb_path = tmp_path / "embeddings.npy"
    tgt_path = tmp_path / "targets.npy"
    ids_path = tmp_path / "tile_ids.txt"
    cv_path  = tmp_path / "cv_splits.json"
    out_csv  = tmp_path / "probe_results.csv"

    np.save(emb_path, embeddings)
    np.save(tgt_path, targets)
    ids_path.write_text("\n".join(tile_ids), encoding="utf-8")
    _write_fake_cv_splits(cv_path, tile_ids)
    _write_fake_manifest(cv_path, target_names)

    result = run_encoder_probe_to_csv(
        emb_path,
        targets_path=tgt_path,
        tile_ids_path=ids_path,
        cv_splits_path=cv_path,
        output_csv_path=out_csv,
    )

    assert result == out_csv
    assert out_csv.is_file()
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(rows) == n_targets
    assert set(rows[0].keys()) == {"target", "r2_mean", "r2_sd", "n_valid_folds"}
    targets_found = {r["target"] for r in rows}
    assert targets_found == set(target_names)
    for row in rows:
        assert row["n_valid_folds"] == "2"
        float(row["r2_mean"])   # must be parseable
        float(row["r2_sd"])
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n he-multiplex python -m pytest tests/test_encoder_probe_csv.py::test_run_encoder_probe_to_csv_creates_correct_schema -v
```

Expected: `FAILED` — `ImportError: cannot import name 'run_encoder_probe_to_csv'`

- [ ] **Step 3: Add `run_encoder_probe_to_csv()` to `src/a1_probe_encoders/main.py`**

Insert after the `summarize_probe_results` function (around line 152):

```python
def run_encoder_probe_to_csv(
    embeddings_path: "str | Path",
    *,
    targets_path: "str | Path",
    tile_ids_path: "str | Path",
    cv_splits_path: "str | Path",
    output_csv_path: "str | Path",
) -> Path:
    """Run 5-fold Ridge probe on any embeddings file; write target/r2_mean/r2_sd/n_valid_folds CSV.

    Reuses the shared CV splits so results are comparable across encoders.
    """
    tile_ids = load_tile_ids(tile_ids_path)
    targets = _load_targets(targets_path)
    if targets.shape[0] != len(tile_ids):
        raise ValueError("targets row count does not match tile_ids count")
    target_names = _load_target_names(targets_path, cv_splits_path)
    embeddings = np.asarray(np.load(embeddings_path), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(tile_ids):
        raise ValueError(
            f"embeddings shape {embeddings.shape} incompatible with {len(tile_ids)} tiles"
        )
    splits = load_cv_splits(tile_ids, cv_splits_path)
    fold_scores, _, _ = run_cv_regression(embeddings, targets, splits)
    rows = summarize_probe_results(fold_scores, target_names)

    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["target", "r2_mean", "r2_sd", "n_valid_folds"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target": row["target"],
                    "r2_mean": float(row["r2_mean"]),
                    "r2_sd": float(row["r2_sd"]),
                    "n_valid_folds": int(row["n_valid_folds"]),
                }
            )
    return out_path
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n he-multiplex python -m pytest tests/test_encoder_probe_csv.py::test_run_encoder_probe_to_csv_creates_correct_schema -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/a1_probe_encoders/main.py tests/test_encoder_probe_csv.py
git commit -m "feat: add run_encoder_probe_to_csv helper for multi-encoder CV probe"
```

---

## Task 2 — ResNet-50 feature extractor

Adds `extract_resnet50_embeddings()` and `run_resnet50_worker()` to `src/a1_probe_encoders/main.py`.

**Files:**
- Modify: `src/a1_probe_encoders/main.py`
- Modify: `tests/test_encoder_probe_csv.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_encoder_probe_csv.py`:

```python
def test_extract_resnet50_embeddings_shape(tmp_path: Path) -> None:
    """ResNet-50 extractor returns (N, 2048) float32 for a small batch of synthetic PNGs."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.a1_probe_encoders.main import extract_resnet50_embeddings
    from PIL import Image

    he_dir = tmp_path / "he"
    he_dir.mkdir()
    tile_ids = [f"00{i:03d}_0000" for i in range(4)]
    for tid in tile_ids:
        img = Image.fromarray(
            (np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        )
        img.save(he_dir / f"{tid}.png")

    embeddings = extract_resnet50_embeddings(he_dir, tile_ids, device="cpu", batch_size=2)
    assert embeddings.shape == (4, 2048)
    assert embeddings.dtype == np.float32
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n pixcell python -m pytest tests/test_encoder_probe_csv.py::test_extract_resnet50_embeddings_shape -v
```

Expected: `FAILED` — `ImportError: cannot import name 'extract_resnet50_embeddings'`

- [ ] **Step 3: Add `extract_resnet50_embeddings()` and `run_resnet50_worker()` to `src/a1_probe_encoders/main.py`**

Add after the `_VIRCHOW_BATCH_SIZE` constant at the top:

```python
_RESNET50_BATCH_SIZE = 64
```

Add after `run_virchow_worker()` (around line 683):

```python
def extract_resnet50_embeddings(
    he_dir: "str | Path",
    tile_ids: "list[str]",
    *,
    device: str = "cuda",
    batch_size: int = _RESNET50_BATCH_SIZE,
) -> np.ndarray:
    """Extract ImageNet ResNet-50 global-avg-pool features; returns (N, 2048) float32."""
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    resolved = _resolve_device(device)
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
    backbone = tv_models.resnet50(weights=weights)
    # Drop the classification head; keep up to global avg pool → (N, 2048, 1, 1)
    encoder = torch.nn.Sequential(*list(backbone.children())[:-1]).to(resolved)
    encoder.eval()

    outputs: list[np.ndarray] = []
    for batch_ids in _iter_batches(list(tile_ids), max(1, batch_size)):
        images = [
            _load_rgb_image(_find_tile_image_path(he_dir, tid))
            for tid in batch_ids
        ]
        tensors = torch.stack(
            [transform(img.convert("RGB")) for img in images], dim=0
        ).to(resolved)
        with torch.inference_mode():
            feats = encoder(tensors).flatten(1)  # (B, 2048)
        outputs.append(feats.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def run_resnet50_worker(config: ProbeEncodersConfig) -> Path:
    output_dir = ensure_directory(config.out_dir)
    tile_ids = load_tile_ids(config.tile_ids_path)
    embeddings = extract_resnet50_embeddings(
        config.he_dir, tile_ids, device=config.device
    )
    output_path = output_dir / "resnet50_embeddings.npy"
    np.save(output_path, embeddings)
    return output_path
```

Also add `"resnet50"` to the `--worker` choices and the dispatch block in `main()`:

```python
# in parser.add_argument("--worker", ...)
parser.add_argument("--worker", choices=("raw_cnn", "virchow", "compare", "resnet50"), default=None)

# in the dispatch block
if args.worker == "resnet50":
    run_resnet50_worker(config)
    return 0
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
conda run -n pixcell python -m pytest tests/test_encoder_probe_csv.py::test_extract_resnet50_embeddings_shape -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/a1_probe_encoders/main.py tests/test_encoder_probe_csv.py
git commit -m "feat: add ResNet-50 feature extractor and worker to probe_encoders"
```

---

## Task 3 — Re-run Virchow2 probe with fold-level scores

No new code — uses `run_encoder_probe_to_csv()` from Task 1 on the existing `virchow_embeddings.npy`. Run in `he-multiplex` (CPU-only — embeddings already exist, just Ridge regression).

**Files:** none (data-only task)

- [ ] **Step 1: Verify virchow embeddings exist**

```bash
ls -lh src/a1_probe_encoders/out/virchow_embeddings.npy
python -c "import numpy as np; e = np.load('src/a1_probe_encoders/out/virchow_embeddings.npy'); print(e.shape, e.dtype)"
```

Expected: `(10379, 2560) float32` (or similar — Virchow2 output dim)

- [ ] **Step 2: Run probe**

```bash
conda run -n he-multiplex python - <<'EOF'
from pathlib import Path
from src.a1_probe_encoders.main import run_encoder_probe_to_csv

run_encoder_probe_to_csv(
    Path("src/a1_probe_encoders/out/virchow_embeddings.npy"),
    targets_path=Path("src/a1_mask_targets/out/targets_T1.npy"),
    tile_ids_path=Path("src/a1_mask_targets/out/tile_ids.txt"),
    cv_splits_path=Path("src/a1_probe_linear/out/cv_splits.json"),
    output_csv_path=Path("src/a1_probe_encoders/out/virchow2_linear_probe_results.csv"),
)
print("Done")
EOF
```

Expected: prints `Done`, file appears.

- [ ] **Step 3: Spot-check output**

```bash
head -5 src/a1_probe_encoders/out/virchow2_linear_probe_results.csv
```

Expected: header row `target,r2_mean,r2_sd,n_valid_folds` then rows like `cell_density,0.966...,0.00...,5`

- [ ] **Step 4: Commit data artifact reference**

```bash
git add src/a1_probe_encoders/out/virchow2_linear_probe_results.csv
git commit -m "data: add Virchow2 5-fold linear probe results with fold SD"
```

---

## Task 4 — Extract ResNet-50 embeddings and run probe

Run the `resnet50` worker (needs GPU / `pixcell` env for fast extraction), then probe in `he-multiplex`.

**Files:** none (data-only task)

- [ ] **Step 1: Extract ResNet-50 embeddings (GPU)**

```bash
conda run -n pixcell python -m src.a1_probe_encoders.main \
  --worker resnet50 \
  --he-dir data/orion-crc33/he_pngs \
  --targets-path src/a1_mask_targets/out/targets_T1.npy \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --cv-splits-path src/a1_probe_linear/out/cv_splits.json \
  --out-dir src/a1_probe_encoders/out \
  --device cuda
```

Expected: `src/a1_probe_encoders/out/resnet50_embeddings.npy` created (~80 MB for 10 379 tiles × 2048 floats).

Verify shape:
```bash
python -c "import numpy as np; e = np.load('src/a1_probe_encoders/out/resnet50_embeddings.npy'); print(e.shape, e.dtype)"
```

Expected: `(10379, 2048) float32`

- [ ] **Step 2: Run linear probe**

```bash
conda run -n he-multiplex python - <<'EOF'
from pathlib import Path
from src.a1_probe_encoders.main import run_encoder_probe_to_csv

run_encoder_probe_to_csv(
    Path("src/a1_probe_encoders/out/resnet50_embeddings.npy"),
    targets_path=Path("src/a1_mask_targets/out/targets_T1.npy"),
    tile_ids_path=Path("src/a1_mask_targets/out/tile_ids.txt"),
    cv_splits_path=Path("src/a1_probe_linear/out/cv_splits.json"),
    output_csv_path=Path("src/a1_probe_encoders/out/resnet50_linear_probe_results.csv"),
)
print("Done")
EOF
```

- [ ] **Step 3: Spot-check output**

```bash
head -5 src/a1_probe_encoders/out/resnet50_linear_probe_results.csv
```

Expected: `cell_density` R² roughly 0.82–0.90 (consistent with `cnn_r2` column in `encoder_comparison.csv`).

- [ ] **Step 4: Commit**

```bash
git add src/a1_probe_encoders/out/resnet50_linear_probe_results.csv
git commit -m "data: add ResNet-50 ImageNet 5-fold linear probe results"
```

> **Note on REMEDIS:** REMEDIS weights (BiT-L ResNet-50x4) require access request from Google Research. If available, run the same two steps substituting `remedis_embeddings.npy` and `remedis_linear_probe_results.csv`. The figure code (Task 5) accepts it as an optional path. Skip for now.

---

## Task 5 — Build `fig_inverse_decoding.py`

Creates `src/paper_figures/fig_inverse_decoding.py` with two rendering functions and tests them with mock CSV data.

**Files:**
- Create: `src/paper_figures/fig_inverse_decoding.py`
- Create: `tests/test_fig_inverse_decoding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fig_inverse_decoding.py
"""Tests for the Figure 4 inverse-decoding panel builder."""
from __future__ import annotations
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from pathlib import Path


def _write_t1_csv(path: Path, targets: list[tuple[str, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target", "r2_mean", "r2_sd", "n_valid_folds"])
        w.writeheader()
        for name, mean, sd in targets:
            w.writerow({"target": name, "r2_mean": mean, "r2_sd": sd, "n_valid_folds": 5})


def _write_t2_csv(path: Path, markers: list[tuple[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target", "r2_mean", "r2_sd", "n_valid_folds"])
        w.writeheader()
        for name, mean in markers:
            w.writerow({"target": name, "r2_mean": mean, "r2_sd": 0.05, "n_valid_folds": 5})


_T1_TARGETS = [
    ("cell_density", 0.953, 0.007),
    ("prolif_frac", 0.863, 0.035),
    ("nonprolif_frac", 0.826, 0.023),
    ("glucose_mean", 0.821, 0.020),
    ("oxygen_mean", 0.810, 0.022),
    ("healthy_frac", 0.710, 0.013),
    ("cancer_frac", 0.669, 0.016),
    ("vasculature_frac", 0.509, 0.022),
    ("immune_frac", 0.495, 0.042),
    ("dead_frac", -0.135, 0.107),
]

_T2_MARKERS = [
    ("PD-1", 0.364), ("E-cadherin", 0.238), ("CD45RO", 0.094),
    ("Ki67", 0.050), ("CD3e", 0.045), ("Pan-CK", 0.033), ("CD45", 0.003),
    ("CD4", -0.002), ("CD163", -0.035), ("CD68", -0.042), ("SMA", -0.052),
    ("CD20", -0.142), ("CD31", -0.144), ("CD8a", -0.191), ("FOXP3", -1.060),
    # these should be filtered out:
    ("Hoechst", 0.355), ("AF1", 0.004), ("Argo550", -2.7), ("PD-L1", -0.076),
]


def test_build_inverse_decoding_figure_returns_figure_with_two_axes(tmp_path: Path) -> None:
    from src.paper_figures.fig_inverse_decoding import build_inverse_decoding_figure

    uni_csv = tmp_path / "uni.csv"
    vir_csv = tmp_path / "virchow.csv"
    rnet_csv = tmp_path / "resnet50.csv"
    t2_csv = tmp_path / "t2.csv"

    _write_t1_csv(uni_csv, _T1_TARGETS)
    _write_t1_csv(vir_csv, [(t, m * 0.98, s) for t, m, s in _T1_TARGETS])
    _write_t1_csv(rnet_csv, [(t, m * 0.90, s) for t, m, s in _T1_TARGETS])
    _write_t2_csv(t2_csv, _T2_MARKERS)

    fig = build_inverse_decoding_figure(
        uni_t1_csv=uni_csv,
        virchow_t1_csv=vir_csv,
        resnet50_t1_csv=rnet_csv,
        t2_mlp_csv=t2_csv,
    )

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_non_typing_markers_excluded(tmp_path: Path) -> None:
    from src.paper_figures.fig_inverse_decoding import load_t2_data

    t2_csv = tmp_path / "t2.csv"
    _write_t2_csv(t2_csv, _T2_MARKERS)
    markers = load_t2_data(t2_csv)
    names = [m["marker"] for m in markers]
    assert "Hoechst" not in names
    assert "Argo550" not in names
    assert "PD-L1" not in names
    assert "AF1" not in names
    assert len(markers) == 15


def test_t1_panel_sorted_by_uni_r2_descending(tmp_path: Path) -> None:
    from src.paper_figures.fig_inverse_decoding import load_t1_data

    uni_csv = tmp_path / "uni.csv"
    _write_t1_csv(uni_csv, _T1_TARGETS)
    targets = load_t1_data({"UNI-2h": uni_csv})
    r2_values = [t["encoders"]["UNI-2h"]["r2_mean"] for t in targets]
    assert r2_values == sorted(r2_values, reverse=True)
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
conda run -n he-multiplex python -m pytest tests/test_fig_inverse_decoding.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'src.paper_figures.fig_inverse_decoding'`

- [ ] **Step 3: Create `src/paper_figures/fig_inverse_decoding.py`**

```python
"""Figure 4 — Inverse Decoding: What H&E Encodes.

Two-panel vertical bar chart:
  (a) T1 mask-fraction targets — grouped bars for 2–4 encoders, 95% CI on UNI-2h
  (b) T2 CODEX marker mean intensities — diverging bars, UNI-2h MLP
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# ── constants ────────────────────────────────────────────────────────────────

_T_CRIT_4DF = 2.776  # t_{0.975, df=4} for 5-fold CV 95% CI

_NON_TYPING: frozenset[str] = frozenset({"Hoechst", "AF1", "Argo550", "PD-L1"})

_T1_DISPLAY_LABELS: dict[str, str] = {
    "cell_density":    "density",
    "prolif_frac":     r"$f_{\mathrm{prolif}}$",
    "nonprolif_frac":  r"$f_{\mathrm{nonprolif}}$",
    "glucose_mean":    "glucose",
    "oxygen_mean":     r"O$_2$",
    "healthy_frac":    r"$f_{\mathrm{healthy}}$",
    "cancer_frac":     r"$f_{\mathrm{cancer}}$",
    "vasculature_frac": r"$f_{\mathrm{vasc}}$",
    "immune_frac":     r"$f_{\mathrm{immune}}$",
    "dead_frac":       r"$f_{\mathrm{dead}}$",
}

_ENCODER_COLORS: dict[str, str] = {
    "UNI-2h":    "#2c7bb6",
    "Virchow2":  "#d7191c",
    "REMEDIS":   "#555555",
    "ResNet-50": "#aaaaaa",
}
_ENCODER_DASHED: frozenset[str] = frozenset({"REMEDIS", "ResNet-50"})

_T2_CATEGORY_COLORS: dict[str, str] = {
    "immune_signaling": "#8e44ad",
    "epithelial":       "#e67e22",
    "proliferation":    "#16a085",
    "immune_structural": "#7f8c8d",
}

_T2_MARKER_CATEGORIES: dict[str, str] = {
    "PD-1":       "immune_signaling",
    "E-cadherin": "epithelial",
    "Pan-CK":     "epithelial",
    "Ki67":       "proliferation",
}

_T2_AXIS_CAP = -0.30   # FOXP3 displayed capped at this value


# ── data loading ─────────────────────────────────────────────────────────────

def _read_probe_csv(path: Path) -> dict[str, dict[str, float]]:
    """Read target→{r2_mean, r2_sd, n_valid_folds} from a probe CSV."""
    result: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            result[row["target"]] = {
                "r2_mean": float(row["r2_mean"]),
                "r2_sd": float(row.get("r2_sd", "nan")),
                "n_valid_folds": float(row.get("n_valid_folds", "nan")),
            }
    return result


def load_t1_data(
    encoder_csvs: dict[str, Path],
) -> list[dict[str, Any]]:
    """Return targets sorted by UNI-2h R² descending.

    Each entry: {"target": str, "label": str, "encoders": {name: {r2_mean, r2_sd}}}
    """
    all_data: dict[str, dict[str, dict[str, float]]] = {
        name: _read_probe_csv(path)
        for name, path in encoder_csvs.items()
        if path is not None and Path(path).is_file()
    }
    if "UNI-2h" not in all_data:
        raise ValueError("UNI-2h CSV is required")

    uni_data = all_data["UNI-2h"]
    targets_ordered = sorted(
        uni_data.keys(),
        key=lambda t: uni_data[t]["r2_mean"],
        reverse=True,
    )
    result = []
    for target in targets_ordered:
        result.append(
            {
                "target": target,
                "label": _T1_DISPLAY_LABELS.get(target, target),
                "encoders": {
                    name: data.get(target, {"r2_mean": float("nan"), "r2_sd": float("nan")})
                    for name, data in all_data.items()
                },
            }
        )
    return result


def load_t2_data(t2_mlp_csv: Path) -> list[dict[str, Any]]:
    """Return non-typing markers sorted by R² descending.

    Each entry: {"marker": str, "r2_mean": float, "category": str, "capped": bool}
    """
    raw = _read_probe_csv(t2_mlp_csv)
    markers = []
    for marker, vals in raw.items():
        if marker in _NON_TYPING:
            continue
        category = _T2_MARKER_CATEGORIES.get(marker, "immune_structural")
        r2 = vals["r2_mean"]
        capped = r2 < _T2_AXIS_CAP
        markers.append(
            {
                "marker": marker,
                "r2_mean": r2,
                "r2_display": max(r2, _T2_AXIS_CAP),
                "category": category,
                "capped": capped,
            }
        )
    return sorted(markers, key=lambda m: m["r2_mean"], reverse=True)


# ── panel renderers ───────────────────────────────────────────────────────────

def _draw_panel_a(
    ax: plt.Axes,
    targets: list[dict[str, Any]],
    encoder_order: list[str],
) -> None:
    """Grouped vertical bars, 95% CI on UNI-2h only."""
    n_enc = len(encoder_order)
    bar_w = min(0.18, 0.75 / n_enc)
    x_base = np.arange(len(targets))

    for enc_idx, enc_name in enumerate(encoder_order):
        color = _ENCODER_COLORS.get(enc_name, "#888888")
        dashed = enc_name in _ENCODER_DASHED
        x_pos = x_base + (enc_idx - (n_enc - 1) / 2) * bar_w
        heights = [
            t["encoders"].get(enc_name, {}).get("r2_mean", float("nan"))
            for t in targets
        ]
        lw = 0.6 if dashed else 0.0
        ls = "--" if dashed else "-"
        ax.bar(
            x_pos, heights, width=bar_w,
            color=color, alpha=0.80 if dashed else 1.0,
            linewidth=lw, linestyle=ls, edgecolor="black" if dashed else "none",
            label=enc_name,
            zorder=2,
        )
        # 95% CI only on UNI-2h
        if enc_name == "UNI-2h":
            for xi, t in zip(x_pos, targets):
                sd = t["encoders"].get("UNI-2h", {}).get("r2_sd", float("nan"))
                r2 = t["encoders"].get("UNI-2h", {}).get("r2_mean", float("nan"))
                if np.isfinite(sd) and np.isfinite(r2):
                    half = _T_CRIT_4DF * sd
                    ax.errorbar(
                        xi, r2, yerr=half,
                        fmt="none", ecolor="black",
                        elinewidth=0.9, capsize=2.5, capthick=0.9,
                        zorder=3,
                    )

    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [t["label"] for t in targets],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("R²", fontsize=9)
    ax.set_ylim(-0.20, 1.05)
    ax.yaxis.grid(True, linewidth=0.4, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, colors="black")
    ax.yaxis.label.set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.legend(
        fontsize=7.5, frameon=False,
        loc="upper right", ncol=2,
    )
    ax.set_title("(a) T1 targets · linear probe R²", fontsize=9, loc="left", pad=4)


def _draw_panel_b(ax: plt.Axes, markers: list[dict[str, Any]]) -> None:
    """Vertical diverging bars; value labels at bar tips."""
    x_pos = np.arange(len(markers))
    for xi, m in enumerate(markers):
        color = _T2_CATEGORY_COLORS[m["category"]]
        r2_disp = m["r2_display"]
        ax.bar([xi], [r2_disp], width=0.65, color=color, zorder=2)
        # value label
        label = f"{m['r2_mean']:.3f}" + ("*" if m["capped"] else "")
        va = "bottom" if r2_disp >= 0 else "top"
        offset = 0.012 if r2_disp >= 0 else -0.012
        ax.text(xi, r2_disp + offset, label, ha="center", va=va, fontsize=6.0, color="black")

    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [m["marker"] for m in markers],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("R²", fontsize=9)
    ax.set_ylim(_T2_AXIS_CAP - 0.10, 0.50)
    ax.yaxis.grid(True, linewidth=0.4, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, colors="black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # category legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=c, label=lbl.replace("_", " "))
        for lbl, c in _T2_CATEGORY_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper right")
    if any(m["capped"] for m in markers):
        ax.text(
            0.02, 0.02,
            f"* axis capped at {_T2_AXIS_CAP}",
            transform=ax.transAxes, fontsize=6.5, color="black", va="bottom",
        )
    ax.set_title("(b) T2 CODEX markers · MLP probe R²", fontsize=9, loc="left", pad=4)


# ── public API ────────────────────────────────────────────────────────────────

def build_inverse_decoding_figure(
    *,
    uni_t1_csv: Path,
    virchow_t1_csv: Path | None = None,
    resnet50_t1_csv: Path | None = None,
    remedis_t1_csv: Path | None = None,
    t2_mlp_csv: Path,
    figsize: tuple[float, float] = (10.0, 4.2),
    dpi: int = 300,
) -> plt.Figure:
    """Render Figure 4 and return the Figure object (caller saves it)."""
    encoder_csvs: dict[str, Path] = {"UNI-2h": Path(uni_t1_csv)}
    encoder_order = ["UNI-2h"]
    for name, path in [
        ("Virchow2", virchow_t1_csv),
        ("REMEDIS", remedis_t1_csv),
        ("ResNet-50", resnet50_t1_csv),
    ]:
        if path is not None and Path(path).is_file():
            encoder_csvs[name] = Path(path)
            encoder_order.append(name)

    targets = load_t1_data(encoder_csvs)
    markers = load_t2_data(Path(t2_mlp_csv))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=figsize,
        facecolor="white",
        gridspec_kw={"width_ratios": [len(targets), len(markers)]},
    )
    _draw_panel_a(ax_a, targets, encoder_order)
    _draw_panel_b(ax_b, markers)
    fig.tight_layout(pad=1.2)
    return fig
```

- [ ] **Step 4: Run tests**

```bash
conda run -n he-multiplex python -m pytest tests/test_fig_inverse_decoding.py -v
```

Expected: all 3 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/paper_figures/fig_inverse_decoding.py tests/test_fig_inverse_decoding.py
git commit -m "feat: add Figure 4 inverse-decoding panel builder"
```

---

## Task 6 — Wire into `src/paper_figures/main.py`

**Files:**
- Modify: `src/paper_figures/main.py`

- [ ] **Step 1: Add import and path constants**

At the top of `src/paper_figures/main.py`, add:

```python
from src.paper_figures.fig_inverse_decoding import build_inverse_decoding_figure
```

Add constants after the existing `PAIRED_REFERENCE_ROOT` block:

```python
PROBE_ENCODERS_OUT = ROOT / "src" / "a1_probe_encoders" / "out"
T1_UNI_CSV      = ROOT / "src" / "a1_probe_linear" / "out" / "linear_probe_results.csv"
T1_VIRCHOW_CSV  = PROBE_ENCODERS_OUT / "virchow2_linear_probe_results.csv"
T1_RESNET50_CSV = PROBE_ENCODERS_OUT / "resnet50_linear_probe_results.csv"
T1_REMEDIS_CSV  = PROBE_ENCODERS_OUT / "remedis_linear_probe_results.csv"
T2_MLP_CSV      = ROOT / "src" / "a1_codex_targets" / "probe_out" / "t2_mlp" / "mlp_probe_results.csv"
```

- [ ] **Step 2: Add the figure call inside `main()`**

After the last `build_representative_ablation_grid` call, add:

```python
    if T2_MLP_CSV.is_file():
        fig_inv = build_inverse_decoding_figure(
            uni_t1_csv=T1_UNI_CSV,
            virchow_t1_csv=T1_VIRCHOW_CSV if T1_VIRCHOW_CSV.is_file() else None,
            resnet50_t1_csv=T1_RESNET50_CSV if T1_RESNET50_CSV.is_file() else None,
            remedis_t1_csv=T1_REMEDIS_CSV if T1_REMEDIS_CSV.is_file() else None,
            t2_mlp_csv=T2_MLP_CSV,
        )
        save_figure_png(fig_inv, PNG_DIR / "07_inverse_decoding.png")
    else:
        print("Skipping 07_inverse_decoding.png — T2 MLP CSV not found at", T2_MLP_CSV)
```

- [ ] **Step 3: Run the full pipeline**

```bash
conda run -n he-multiplex python -m src.paper_figures.main
```

Expected: prints `07_inverse_decoding.png` in the saved list, file exists at `figures/pngs/07_inverse_decoding.png`.

Spot-check:
```bash
ls -lh figures/pngs/07_inverse_decoding.png
python -c "from PIL import Image; img = Image.open('figures/pngs/07_inverse_decoding.png'); print(img.size)"
```

Expected: image size roughly 3000 × 1260 (10 inch × 4.2 inch at 300 DPI).

- [ ] **Step 4: Commit**

```bash
git add src/paper_figures/main.py figures/pngs/07_inverse_decoding.png
git commit -m "feat: wire Figure 4 inverse-decoding into paper figures pipeline"
```

---

## Self-review notes

- **Spec coverage:** All 5 spec steps covered. REMEDIS is optional with clear `if file.is_file()` guard.
- **Type consistency:** `run_encoder_probe_to_csv` in Task 1 matches usage in Tasks 3 & 4. `load_t1_data` / `load_t2_data` used in tests match signatures in implementation. `build_inverse_decoding_figure` signature in Task 5 matches call in Task 6.
- **No placeholders:** All code blocks are complete.
- **Dependent ordering:** Tasks 3 & 4 (data) must precede Task 6 (wiring) for full CSV set. Tasks 5 & 6 can proceed with partial data — `is_file()` guards handle missing CSVs gracefully.
