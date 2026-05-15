# a4 Appearance Probes + Shared Tile Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `src/a4_uni_probe` to fit Ridge probes for 8 tissue appearance attrs (H/E stain, Haralick texture) on real H&E tiles, and re-run all sweep/null experiments with a single shared 30-tile set so cross-attr comparisons are on identical tissue.

**Architecture:** Add `build_appearance_label_matrix` to `labels.py`, wire it into `build_label_matrix` via an optional `he_dir` param, extend `_select_sweep_attrs` with an `attr_pool` filter, and add `--fixed-tile-ids` / `--attr-pool` / `--he-dir` CLI args. All generation commands use existing sharding (two terminals, 15 tiles each).

**Tech Stack:** Python 3.11+, scikit-learn Ridge, skimage (rgb2hed, GLCM), numpy, pytest. All GPU inference via existing `edit.py` pipeline.

---

## File Map

| File | Change |
|---|---|
| `src/a4_uni_probe/labels.py` | Add `APPEARANCE_ATTR_NAMES`, `build_appearance_label_matrix`, extend `ALL_ATTR_NAMES` and `build_label_matrix` |
| `src/a4_uni_probe/edit.py` | Extend `_select_sweep_attrs` with `attr_pool` param; add `--fixed-tile-ids` branch in `run_sweep` and `run_null` |
| `src/a4_uni_probe/main.py` | Add `--he-dir` to `probe`; add `--fixed-tile-ids` and `--attr-pool` to `sweep` and `null` |
| `src/a4_uni_probe/probe.py` | Pass `he_dir=args.he_dir` to `build_label_matrix` |
| `tests/test_a4_labels.py` | Add tests for `APPEARANCE_ATTR_NAMES`, `build_appearance_label_matrix`, extended `build_label_matrix` |
| `tests/test_a4_edit.py` | Add test for `--fixed-tile-ids` tile bypass and `attr_pool` filter |
| `inference_output/a1_concat/a4_uni_probe/shared_tiles.json` | Generated once (Task 5) |

---

## Task 1: Add appearance label constants and builder to `labels.py`

**Files:**
- Modify: `src/a4_uni_probe/labels.py`
- Test: `tests/test_a4_labels.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_a4_labels.py`:

```python
from src.a4_uni_probe.labels import (
    APPEARANCE_ATTR_NAMES,
    build_appearance_label_matrix,
)


def _write_he_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def test_appearance_attr_names_are_correct_set():
    expected = {
        "h_mean", "e_mean",
        "texture_h_contrast", "texture_h_homogeneity", "texture_h_energy",
        "texture_e_contrast", "texture_e_homogeneity", "texture_e_energy",
    }
    assert set(APPEARANCE_ATTR_NAMES) == expected
    assert len(APPEARANCE_ATTR_NAMES) == 8


def test_build_appearance_label_matrix_returns_finite_values(tmp_path: Path):
    he_dir = tmp_path / "he"
    rgb = np.full((32, 32, 3), [180, 140, 170], dtype=np.uint8)
    _write_he_png(he_dir / "0_0.png", rgb)
    mat = build_appearance_label_matrix(["0_0"], he_dir)
    assert mat.shape == (1, len(APPEARANCE_ATTR_NAMES))
    assert np.all(np.isfinite(mat))


def test_build_appearance_label_matrix_missing_he_gives_nan(tmp_path: Path):
    he_dir = tmp_path / "he"
    he_dir.mkdir()
    mat = build_appearance_label_matrix(["missing_tile"], he_dir)
    assert mat.shape == (1, len(APPEARANCE_ATTR_NAMES))
    assert np.all(np.isnan(mat))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ec2-user/PixCell
pytest tests/test_a4_labels.py::test_appearance_attr_names_are_correct_set tests/test_a4_labels.py::test_build_appearance_label_matrix_returns_finite_values tests/test_a4_labels.py::test_build_appearance_label_matrix_missing_he_gives_nan -v
```

Expected: FAIL with `ImportError: cannot import name 'APPEARANCE_ATTR_NAMES'`

- [ ] **Step 3: Add `APPEARANCE_ATTR_NAMES` and `build_appearance_label_matrix` to `labels.py`**

At the top of `src/a4_uni_probe/labels.py`, add the import:

```python
from src.a4_uni_probe.appearance_metrics import appearance_row_for_image
```

After the `MORPHOLOGY_ATTR_NAMES` tuple, add:

```python
APPEARANCE_ATTR_NAMES = (
    "h_mean",
    "e_mean",
    "texture_h_contrast",
    "texture_h_homogeneity",
    "texture_h_energy",
    "texture_e_contrast",
    "texture_e_homogeneity",
    "texture_e_energy",
)
```

After the existing `ALL_ATTR_NAMES` line, add the function:

```python
def build_appearance_label_matrix(
    tile_ids: list[str],
    he_dir: str | Path,
) -> np.ndarray:
    he_root = Path(he_dir)
    rows: list[list[float]] = []
    for tile_id in tile_ids:
        he_path = he_root / f"{tile_id}.png"
        if he_path.is_file():
            row = appearance_row_for_image(he_path)
            rows.append([float(row.get(f"appearance.{name}", float("nan"))) for name in APPEARANCE_ATTR_NAMES])
        else:
            rows.append([float("nan")] * len(APPEARANCE_ATTR_NAMES))
    return np.asarray(rows, dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_a4_labels.py::test_appearance_attr_names_are_correct_set tests/test_a4_labels.py::test_build_appearance_label_matrix_returns_finite_values tests/test_a4_labels.py::test_build_appearance_label_matrix_missing_he_gives_nan -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/a4_uni_probe/labels.py tests/test_a4_labels.py
git commit -m "feat(a4): add APPEARANCE_ATTR_NAMES and build_appearance_label_matrix"
```

---

## Task 2: Extend `ALL_ATTR_NAMES` and `build_label_matrix` with `he_dir`

**Files:**
- Modify: `src/a4_uni_probe/labels.py`
- Test: `tests/test_a4_labels.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_a4_labels.py`:

```python
def test_build_label_matrix_includes_appearance_attrs_when_he_dir_given(tmp_path: Path):
    exp_root = _make_channel_major_exp_channels(tmp_path)
    cellvit_root = tmp_path / "cellvit_real"
    cellvit_root.mkdir()
    (cellvit_root / "0_0.json").write_text(
        json.dumps({
            "tile_area": 65536.0,
            "nuclei": [{"area": 100.0, "eccentricity": 0.5, "intensity_h": 0.4, "intensity_e": 0.2}],
        }),
        encoding="utf-8",
    )
    he_dir = tmp_path / "he"
    rgb = np.full((32, 32, 3), [180, 140, 170], dtype=np.uint8)
    _write_he_png(he_dir / "0_0.png", rgb)

    labels, attr_names = build_label_matrix(["0_0"], exp_root, cellvit_root, he_dir=he_dir)
    assert labels.shape == (1, len(ALL_ATTR_NAMES))
    assert "h_mean" in attr_names
    assert "texture_h_contrast" in attr_names
    h_mean_idx = attr_names.index("h_mean")
    assert np.isfinite(labels[0, h_mean_idx])


def test_build_label_matrix_appearance_nan_when_no_he_dir(tmp_path: Path):
    exp_root = _make_channel_major_exp_channels(tmp_path)
    cellvit_root = tmp_path / "cellvit_real"
    cellvit_root.mkdir()
    (cellvit_root / "0_0.json").write_text(
        json.dumps({
            "tile_area": 65536.0,
            "nuclei": [{"area": 100.0, "eccentricity": 0.5, "intensity_h": 0.4, "intensity_e": 0.2}],
        }),
        encoding="utf-8",
    )
    labels, attr_names = build_label_matrix(["0_0"], exp_root, cellvit_root)
    assert labels.shape == (1, len(ALL_ATTR_NAMES))
    h_mean_idx = attr_names.index("h_mean")
    assert np.isnan(labels[0, h_mean_idx])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_a4_labels.py::test_build_label_matrix_includes_appearance_attrs_when_he_dir_given tests/test_a4_labels.py::test_build_label_matrix_appearance_nan_when_no_he_dir -v
```

Expected: FAIL — `build_label_matrix` does not accept `he_dir` and `ALL_ATTR_NAMES` does not include appearance attrs.

- [ ] **Step 3: Update `ALL_ATTR_NAMES` and `build_label_matrix`**

In `src/a4_uni_probe/labels.py`, replace:

```python
ALL_ATTR_NAMES = CHANNEL_ATTR_NAMES + MORPHOLOGY_ATTR_NAMES
```

with:

```python
ALL_ATTR_NAMES = CHANNEL_ATTR_NAMES + MORPHOLOGY_ATTR_NAMES + APPEARANCE_ATTR_NAMES
```

Replace the `build_label_matrix` signature and body:

```python
def build_label_matrix(
    tile_ids: list[str],
    exp_channels_root: str | Path,
    cellvit_real_dir: str | Path,
    *,
    he_dir: str | Path | None = None,
    resolution: int = 256,
) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    cellvit_root = Path(cellvit_real_dir)
    for tile_id in tile_ids:
        channel_row = compute_channel_attributes(exp_channels_root, tile_id, resolution=resolution)
        morph_row = compute_morphology_attributes_from_cellvit(cellvit_root / f"{tile_id}.json")
        row = {**channel_row, **morph_row}
        rows.append([float(row[attr_name]) for attr_name in CHANNEL_ATTR_NAMES + MORPHOLOGY_ATTR_NAMES])
    base_labels = np.asarray(rows, dtype=np.float32)

    if he_dir is not None:
        appearance_labels = build_appearance_label_matrix(tile_ids, he_dir)
    else:
        appearance_labels = np.full((len(tile_ids), len(APPEARANCE_ATTR_NAMES)), float("nan"), dtype=np.float32)

    return np.concatenate([base_labels, appearance_labels], axis=1), list(ALL_ATTR_NAMES)
```

- [ ] **Step 4: Run all a4 label tests**

```bash
pytest tests/test_a4_labels.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/a4_uni_probe/labels.py tests/test_a4_labels.py
git commit -m "feat(a4): extend ALL_ATTR_NAMES and build_label_matrix with appearance attrs"
```

---

## Task 3: Wire `--he-dir` into `probe` stage

**Files:**
- Modify: `src/a4_uni_probe/main.py`
- Modify: `src/a4_uni_probe/probe.py`

- [ ] **Step 1: Add `--he-dir` to probe subparser in `main.py`**

In `src/a4_uni_probe/main.py`, after the existing `DEFAULT_HE_DIR` constant (already defined as `DEFAULT_DATA_ROOT / "he"`), add to the `p_probe` subparser block:

```python
p_probe.add_argument("--he-dir", type=Path, default=DEFAULT_HE_DIR)
```

- [ ] **Step 2: Pass `he_dir` in `probe.py`**

In `src/a4_uni_probe/probe.py`, inside `run_probe`, replace:

```python
labels, attr_names = build_label_matrix(tile_ids, args.exp_channels_dir, args.cellvit_real_dir)
```

with:

```python
labels, attr_names = build_label_matrix(
    tile_ids, args.exp_channels_dir, args.cellvit_real_dir,
    he_dir=getattr(args, "he_dir", None),
)
```

- [ ] **Step 3: Verify probe CLI accepts the new arg**

```bash
python -m src.a4_uni_probe.main probe --help | grep he-dir
```

Expected: line containing `--he-dir`

- [ ] **Step 4: Commit**

```bash
git add src/a4_uni_probe/main.py src/a4_uni_probe/probe.py
git commit -m "feat(a4): wire --he-dir into probe stage for appearance label fitting"
```

---

## Task 4: Add `--fixed-tile-ids` and `--attr-pool` to sweep/null

**Files:**
- Modify: `src/a4_uni_probe/edit.py`
- Modify: `src/a4_uni_probe/main.py`
- Test: `tests/test_a4_edit.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_a4_edit.py`:

```python
import json
from pathlib import Path
from src.a4_uni_probe.edit import _select_sweep_attrs
from src.a4_uni_probe.labels import APPEARANCE_ATTR_NAMES, MORPHOLOGY_ATTR_NAMES


def test_select_sweep_attrs_morphology_pool(tmp_path: Path):
    csv_path = tmp_path / "probe_results.csv"
    import csv as _csv
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["attr", "delta_r2_uni_minus_tme"])
        w.writeheader()
        w.writerow({"attr": "eccentricity_mean", "delta_r2_uni_minus_tme": "0.23"})
        w.writerow({"attr": "h_mean", "delta_r2_uni_minus_tme": "0.50"})
        w.writerow({"attr": "nuclear_area_mean", "delta_r2_uni_minus_tme": "0.10"})
    attrs = _select_sweep_attrs(csv_path, top_k=2, attr_pool="morphology")
    assert "h_mean" not in attrs
    assert "eccentricity_mean" in attrs


def test_select_sweep_attrs_appearance_pool(tmp_path: Path):
    csv_path = tmp_path / "probe_results.csv"
    import csv as _csv
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["attr", "delta_r2_uni_minus_tme"])
        w.writeheader()
        w.writerow({"attr": "eccentricity_mean", "delta_r2_uni_minus_tme": "0.23"})
        w.writerow({"attr": "h_mean", "delta_r2_uni_minus_tme": "0.50"})
        w.writerow({"attr": "e_mean", "delta_r2_uni_minus_tme": "0.30"})
    attrs = _select_sweep_attrs(csv_path, top_k=2, attr_pool="appearance")
    assert "eccentricity_mean" not in attrs
    assert "h_mean" in attrs
    assert "e_mean" in attrs
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_a4_edit.py::test_select_sweep_attrs_morphology_pool tests/test_a4_edit.py::test_select_sweep_attrs_appearance_pool -v
```

Expected: FAIL — `_select_sweep_attrs` does not accept `attr_pool`

- [ ] **Step 3: Update `_select_sweep_attrs` in `edit.py`**

In `src/a4_uni_probe/edit.py`, add this import at the top if not present:

```python
import json
```

Replace `_select_sweep_attrs`:

```python
def _select_sweep_attrs(probe_csv: Path, top_k: int, attr_pool: str = "morphology") -> list[str]:
    from src.a4_uni_probe.labels import APPEARANCE_ATTR_NAMES, MORPHOLOGY_ATTR_NAMES
    pool = APPEARANCE_ATTR_NAMES if attr_pool == "appearance" else MORPHOLOGY_ATTR_NAMES
    rows: list[tuple[str, float]] = []
    with probe_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            attr = row["attr"]
            if attr not in pool:
                continue
            rows.append((attr, float(row["delta_r2_uni_minus_tme"])))
    rows.sort(key=lambda item: (float("-inf") if not np.isfinite(item[1]) else item[1]), reverse=True)
    return [attr for attr, _ in rows[:top_k]]
```

- [ ] **Step 4: Add `--fixed-tile-ids` branch in `run_sweep`**

In `src/a4_uni_probe/edit.py`, inside `run_sweep`, replace the per-attr tile selection:

```python
        selected_tile_ids = _shard_tile_ids(
            _select_sweep_tiles(labels_npz, attr, args.k_tiles, args.seed),
            tile_shard_index,
            tile_shard_count,
        )
```

with:

```python
        if getattr(args, "fixed_tile_ids", None) is not None:
            _fixed = json.loads(Path(args.fixed_tile_ids).read_text(encoding="utf-8"))
            _all_tile_ids = _fixed["tile_ids"]
        else:
            _all_tile_ids = _select_sweep_tiles(labels_npz, attr, args.k_tiles, args.seed)
        selected_tile_ids = _shard_tile_ids(_all_tile_ids, tile_shard_index, tile_shard_count)
```

Also replace the `attrs = _select_sweep_attrs(...)` call in `run_sweep`:

```python
    attrs = _select_sweep_attrs(probe_csv, args.top_k_attrs, attr_pool=getattr(args, "attr_pool", "morphology"))
```

- [ ] **Step 5: Add `--fixed-tile-ids` branch in `run_null`**

Same pattern in `run_null`. Replace the per-attr tile selection:

```python
        selected_tile_ids = _shard_tile_ids(
            _select_sweep_tiles(labels_npz, attr, args.k_tiles, args.seed),
            tile_shard_index,
            tile_shard_count,
        )
```

with:

```python
        if getattr(args, "fixed_tile_ids", None) is not None:
            _fixed = json.loads(Path(args.fixed_tile_ids).read_text(encoding="utf-8"))
            _all_tile_ids = _fixed["tile_ids"]
        else:
            _all_tile_ids = _select_sweep_tiles(labels_npz, attr, args.k_tiles, args.seed)
        selected_tile_ids = _shard_tile_ids(_all_tile_ids, tile_shard_index, tile_shard_count)
```

And:

```python
    attrs = _select_sweep_attrs(probe_csv, args.top_k_attrs, attr_pool=getattr(args, "attr_pool", "morphology"))
```

- [ ] **Step 6: Add CLI args to `main.py`**

In `src/a4_uni_probe/main.py`, add to both `p_sweep` and `p_null` subparser blocks:

```python
p_sweep.add_argument("--fixed-tile-ids", type=Path, default=None)
p_sweep.add_argument("--attr-pool", choices=["morphology", "appearance"], default="morphology")
```

```python
p_null.add_argument("--fixed-tile-ids", type=Path, default=None)
p_null.add_argument("--attr-pool", choices=["morphology", "appearance"], default="morphology")
```

- [ ] **Step 7: Run all a4 edit tests**

```bash
pytest tests/test_a4_edit.py -v
```

Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add src/a4_uni_probe/edit.py src/a4_uni_probe/main.py tests/test_a4_edit.py
git commit -m "feat(a4): add --fixed-tile-ids and --attr-pool to sweep/null pipeline"
```

---

## Task 5: Generate `shared_tiles.json`

**Files:**
- Create: `inference_output/a1_concat/a4_uni_probe/shared_tiles.json`

- [ ] **Step 1: Generate and save the shared tile set**

```bash
python3 - <<'EOF'
import json
import numpy as np

data = np.load("inference_output/a1_concat/a4_uni_probe/labels.npz", allow_pickle=True)
tile_ids = data["tile_ids"].tolist()
attr_names = data["attr_names"].tolist()
labels = data["labels"]

morph_attrs = ["eccentricity_mean", "nuclear_area_mean", "nuclei_density"]
morph_idx = [attr_names.index(a) for a in morph_attrs]
valid_mask = np.all(np.isfinite(labels[:, morph_idx]), axis=1)
valid_tile_ids = [tid for tid, v in zip(tile_ids, valid_mask.tolist()) if v]

rng = np.random.default_rng(42)
chosen = rng.choice(len(valid_tile_ids), size=30, replace=False)
chosen_ids = [valid_tile_ids[i] for i in sorted(chosen.tolist())]

out = {"tile_ids": chosen_ids, "seed": 42, "n": 30, "pool_size": len(valid_tile_ids)}
with open("inference_output/a1_concat/a4_uni_probe/shared_tiles.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {len(chosen_ids)} tiles")
print(chosen_ids[:5], "...")
EOF
```

Expected output: `Saved 30 tiles` followed by 5 tile IDs.

- [ ] **Step 2: Commit**

```bash
git add inference_output/a1_concat/a4_uni_probe/shared_tiles.json
git commit -m "data(a4): generate shared 30-tile set for unified sweep/null experiments"
```

---

## Task 6: Re-run `probe` stage

No code changes — run the updated pipeline to get appearance `w` vectors.

- [ ] **Step 1: Run probe**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main probe \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --features-dir data/orion-crc33/features \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --cellvit-real-dir src/a4_uni_probe/out/cellvit \
  --he-dir data/orion-crc33/he
```

Expected: completes without error. Check outputs:

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('inference_output/a1_concat/a4_uni_probe/probe_results.csv')
print(df[['attr','delta_r2_uni_minus_tme']].to_string())
"
```

Expected: 14 rows for existing attrs + 8 new appearance attrs (h_mean, e_mean, texture_*).

- [ ] **Step 2: Verify appearance probe directions exist**

```bash
ls inference_output/a1_concat/a4_uni_probe/probe_directions/ | grep -E "h_mean|e_mean|texture"
```

Expected: 8 `.npy` files for appearance attrs.

- [ ] **Step 3: Commit updated probe outputs**

```bash
git add inference_output/a1_concat/a4_uni_probe/probe_results.csv \
        inference_output/a1_concat/a4_uni_probe/probe_results.json \
        inference_output/a1_concat/a4_uni_probe/probe_directions/
git commit -m "data(a4): refit probes with appearance attrs, add 8 new probe directions"
```

---

## Task 7: Run sweep — morphology attrs (two terminals)

- [ ] **Step 1: Terminal 1 — morphology sweep, shard 0**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main sweep \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
  --config-path checkpoints/concat_95470_0/config.py \
  --data-root data/orion-crc33 \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --alphas -1 0 1 \
  --k-tiles 30 \
  --top-k-attrs 3 \
  --attr-pool morphology \
  --fixed-tile-ids inference_output/a1_concat/a4_uni_probe/shared_tiles.json \
  --tile-shard-index 0 \
  --tile-shard-count 2
```

- [ ] **Step 2: Terminal 2 — morphology sweep, shard 1**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main sweep \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
  --config-path checkpoints/concat_95470_0/config.py \
  --data-root data/orion-crc33 \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --alphas -1 0 1 \
  --k-tiles 30 \
  --top-k-attrs 3 \
  --attr-pool morphology \
  --fixed-tile-ids inference_output/a1_concat/a4_uni_probe/shared_tiles.json \
  --tile-shard-index 1 \
  --tile-shard-count 2
```

- [ ] **Step 3: Verify morphology sweep outputs**

After both terminals finish:

```bash
python3 -c "
from pathlib import Path
for attr_dir in sorted(Path('inference_output/a1_concat/a4_uni_probe/sweep').iterdir()):
    pngs = list(attr_dir.rglob('*.png'))
    print(attr_dir.name, len(pngs), 'PNGs')
"
```

Expected: 3 attr dirs, each with 180 PNGs (30 tiles × 2 dirs × 3 alphas).

---

## Task 8: Run sweep — appearance attrs (two terminals)

- [ ] **Step 1: Terminal 1 — appearance sweep, shard 0**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main sweep \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
  --config-path checkpoints/concat_95470_0/config.py \
  --data-root data/orion-crc33 \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --alphas -1 0 1 \
  --k-tiles 30 \
  --top-k-attrs 3 \
  --attr-pool appearance \
  --fixed-tile-ids inference_output/a1_concat/a4_uni_probe/shared_tiles.json \
  --tile-shard-index 0 \
  --tile-shard-count 2
```

- [ ] **Step 2: Terminal 2 — appearance sweep, shard 1**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main sweep \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
  --config-path checkpoints/concat_95470_0/config.py \
  --data-root data/orion-crc33 \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --alphas -1 0 1 \
  --k-tiles 30 \
  --top-k-attrs 3 \
  --attr-pool appearance \
  --fixed-tile-ids inference_output/a1_concat/a4_uni_probe/shared_tiles.json \
  --tile-shard-index 1 \
  --tile-shard-count 2
```

- [ ] **Step 3: Verify appearance sweep outputs**

```bash
python3 -c "
from pathlib import Path
for attr_dir in sorted(Path('inference_output/a1_concat/a4_uni_probe/sweep').iterdir()):
    pngs = list(attr_dir.rglob('*.png'))
    print(attr_dir.name, len(pngs), 'PNGs')
"
```

Expected: 6 attr dirs total (3 morph + 3 appearance), each 180 PNGs.

---

## Task 9: Run null — morphology attrs (two terminals)

- [ ] **Step 1: Terminal 1 — morphology null, shard 0**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main null \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
  --config-path checkpoints/concat_95470_0/config.py \
  --data-root data/orion-crc33 \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --k-tiles 30 \
  --top-k-attrs 3 \
  --attr-pool morphology \
  --fixed-tile-ids inference_output/a1_concat/a4_uni_probe/shared_tiles.json \
  --full-null-root inference_output/a1_concat/a2_decomposition/generated \
  --tile-shard-index 0 \
  --tile-shard-count 2
```

- [ ] **Step 2: Terminal 2 — morphology null, shard 1**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main null \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
  --config-path checkpoints/concat_95470_0/config.py \
  --data-root data/orion-crc33 \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --k-tiles 30 \
  --top-k-attrs 3 \
  --attr-pool morphology \
  --fixed-tile-ids inference_output/a1_concat/a4_uni_probe/shared_tiles.json \
  --full-null-root inference_output/a1_concat/a2_decomposition/generated \
  --tile-shard-index 1 \
  --tile-shard-count 2
```

- [ ] **Step 3: Verify morphology null outputs**

```bash
python3 -c "
from pathlib import Path
for attr_dir in sorted(Path('inference_output/a1_concat/a4_uni_probe/null').iterdir()):
    pngs = list(attr_dir.rglob('*.png'))
    print(attr_dir.name, len(pngs), 'PNGs')
"
```

Expected: 3 attr dirs, each with 60 PNGs (30 tiles × 2 conditions).

---

## Task 10: Run null — appearance attrs (two terminals)

- [ ] **Step 1: Terminal 1 — appearance null, shard 0**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main null \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
  --config-path checkpoints/concat_95470_0/config.py \
  --data-root data/orion-crc33 \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --k-tiles 30 \
  --top-k-attrs 3 \
  --attr-pool appearance \
  --fixed-tile-ids inference_output/a1_concat/a4_uni_probe/shared_tiles.json \
  --full-null-root inference_output/a1_concat/a2_decomposition/generated \
  --tile-shard-index 0 \
  --tile-shard-count 2
```

- [ ] **Step 2: Terminal 2 — appearance null, shard 1**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main null \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --checkpoint-dir checkpoints/concat_95470_0/checkpoints/step_0002600 \
  --config-path checkpoints/concat_95470_0/config.py \
  --data-root data/orion-crc33 \
  --exp-channels-dir data/orion-crc33/exp_channels \
  --k-tiles 30 \
  --top-k-attrs 3 \
  --attr-pool appearance \
  --fixed-tile-ids inference_output/a1_concat/a4_uni_probe/shared_tiles.json \
  --full-null-root inference_output/a1_concat/a2_decomposition/generated \
  --tile-shard-index 1 \
  --tile-shard-count 2
```

- [ ] **Step 3: Verify appearance null outputs**

```bash
python3 -c "
from pathlib import Path
for attr_dir in sorted(Path('inference_output/a1_concat/a4_uni_probe/null').iterdir()):
    pngs = list(attr_dir.rglob('*.png'))
    print(attr_dir.name, len(pngs), 'PNGs')
"
```

Expected: 6 attr dirs total (3 morph + 3 appearance), each 60 PNGs.

---

## Task 11: Post-processing — appearance metrics + figures

- [ ] **Step 1: Run appearance metrics on all sweep and null outputs**

```bash
cd /home/ec2-user/PixCell
python -m src.a4_uni_probe.main appearance \
  --out-dir inference_output/a1_concat/a4_uni_probe \
  --data-root data/orion-crc33
```

Expected: creates `appearance_sweep_summary.csv` and `appearance_null_summary.csv` in the out dir.

- [ ] **Step 2: Verify summary CSVs have entries for all 6 attrs**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('inference_output/a1_concat/a4_uni_probe/appearance_sweep_summary.csv')
print('Sweep attrs:', df['attr'].unique().tolist())
df2 = pd.read_csv('inference_output/a1_concat/a4_uni_probe/appearance_null_summary.csv')
print('Null attrs:', df2['attr'].unique().tolist())
"
```

Expected: 6 attrs in each (3 morph + 3 appearance).

- [ ] **Step 3: Re-run figures**

```bash
python -m src.a4_uni_probe.main figures \
  --out-dir inference_output/a1_concat/a4_uni_probe
```

Expected: `figures/panel_*.png` regenerated without error.

- [ ] **Step 4: Final commit**

```bash
git add inference_output/a1_concat/a4_uni_probe/appearance_sweep_summary.csv \
        inference_output/a1_concat/a4_uni_probe/appearance_null_summary.csv \
        inference_output/a1_concat/a4_uni_probe/figures/
git commit -m "data(a4): post-processing appearance metrics and figures for shared-tile experiment"
```
