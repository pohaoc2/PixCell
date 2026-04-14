# LOO SSIM Publication Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `render_loo_ssim_figure()` to `tools/vis/leave_one_out_diff.py` — a 3-row × 5-column publication figure (full H&E / cell inset / SSIM loss) alongside the existing pixel-diff figure.

**Architecture:** All new code lives in one existing file. Three pure helpers (`ssim_loss_map`, `_select_inset_region`, `_draw_inset_marker`) feed into the main rendering function. The CLI gains a `--ssim` flag that writes `leave_one_out_ssim.png` next to the existing `leave_one_out_diff.png`.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, Pillow, scikit-image (`skimage.metrics.structural_similarity`), mpl_toolkits (already present via matplotlib).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `requirements-ci.txt` | Modify | Add `scikit-image` |
| `requirements.txt` | Modify | Add `scikit-image>=0.21` |
| `tools/vis/leave_one_out_diff.py` | Modify | Add helpers + `render_loo_ssim_figure()` + CLI flag |
| `tests/test_leave_one_out_diff.py` | Modify | Add tests for each new helper + smoke test |

---

## Task 1: Install scikit-image dependency

**Files:**
- Modify: `requirements-ci.txt`
- Modify: `requirements.txt`

- [ ] **Step 1: Add scikit-image to requirements-ci.txt**

Open `requirements-ci.txt` and add after the `Pillow` line:
```
scikit-image>=0.21
```

- [ ] **Step 2: Add scikit-image to requirements.txt**

Open `requirements.txt` and add after the `lpips` line (in the Vision section):
```
scikit-image>=0.21
```

- [ ] **Step 3: Install**

```bash
pip install scikit-image>=0.21
```

Expected: installs without error. Verify:
```bash
python -c "from skimage.metrics import structural_similarity; print('ok')"
```
Expected output: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements-ci.txt requirements.txt
git commit -m "chore: add scikit-image dependency for SSIM loss computation"
```

---

## Task 2: `ssim_loss_map()` helper

**Files:**
- Modify: `tools/vis/leave_one_out_diff.py` (add after `_load_cell_mask_array`)
- Modify: `tests/test_leave_one_out_diff.py` (add at end)

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_leave_one_out_diff.py`:

```python
# ── ssim_loss_map ─────────────────────────────────────────────────────────────

def test_ssim_loss_map_identical_images_returns_near_zero() -> None:
    from tools.vis.leave_one_out_diff import ssim_loss_map

    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = ssim_loss_map(img, img)

    assert result.shape == (32, 32)
    assert result.dtype == np.float32
    assert float(result.max()) < 0.05  # near-zero for identical images


def test_ssim_loss_map_different_images_has_nonzero_loss() -> None:
    from tools.vis.leave_one_out_diff import ssim_loss_map

    img_a = np.zeros((32, 32, 3), dtype=np.uint8)
    img_b = np.full((32, 32, 3), 200, dtype=np.uint8)
    result = ssim_loss_map(img_a, img_b)

    assert float(result.mean()) > 0.1


def test_ssim_loss_map_values_in_0_1() -> None:
    from tools.vis.leave_one_out_diff import ssim_loss_map

    rng = np.random.default_rng(42)
    img_a = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    img_b = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    result = ssim_loss_map(img_a, img_b)

    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_leave_one_out_diff.py::test_ssim_loss_map_identical_images_returns_near_zero tests/test_leave_one_out_diff.py::test_ssim_loss_map_different_images_has_nonzero_loss tests/test_leave_one_out_diff.py::test_ssim_loss_map_values_in_0_1 -v
```

Expected: FAIL with `ImportError: cannot import name 'ssim_loss_map'`

- [ ] **Step 3: Implement `ssim_loss_map()`**

Add after `_load_cell_mask_array` in `tools/vis/leave_one_out_diff.py`:

```python
def ssim_loss_map(img_all: np.ndarray, img_drop: np.ndarray, *, win_size: int = 11) -> np.ndarray:
    """Return H×W float32 SSIM structural loss in [0, 1].

    Args:
        img_all: H×W×3 uint8 baseline (all channels) image.
        img_drop: H×W×3 uint8 leave-one-out image.
        win_size: SSIM window size; auto-clamped to image size (must be odd).

    Returns:
        H×W float32 array where 0 = identical structure, 1 = maximum loss.
    """
    from skimage.metrics import structural_similarity as _ssim

    gray_all = img_all.mean(axis=2).astype(np.float64)
    gray_drop = img_drop.mean(axis=2).astype(np.float64)
    H, W = gray_all.shape
    actual_win = min(win_size, H, W)
    if actual_win % 2 == 0:
        actual_win -= 1
    actual_win = max(actual_win, 3)
    _, ssim_full = _ssim(gray_all, gray_drop, full=True, win_size=actual_win, data_range=255)
    return np.clip(1.0 - ssim_full, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_leave_one_out_diff.py::test_ssim_loss_map_identical_images_returns_near_zero tests/test_leave_one_out_diff.py::test_ssim_loss_map_different_images_has_nonzero_loss tests/test_leave_one_out_diff.py::test_ssim_loss_map_values_in_0_1 -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add tools/vis/leave_one_out_diff.py tests/test_leave_one_out_diff.py
git commit -m "feat: add ssim_loss_map helper for structural diff computation"
```

---

## Task 3: `_select_inset_region()` helper

**Files:**
- Modify: `tools/vis/leave_one_out_diff.py` (add after `ssim_loss_map`)
- Modify: `tests/test_leave_one_out_diff.py` (add at end)

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_leave_one_out_diff.py`:

```python
# ── _select_inset_region ──────────────────────────────────────────────────────

def test_select_inset_region_picks_highest_loss_region() -> None:
    from tools.vis.leave_one_out_diff import _select_inset_region

    loss = np.zeros((128, 128), dtype=np.float32)
    loss[64:80, 64:80] = 1.0  # hot spot

    y, x = _select_inset_region(loss, crop=64, stride=8)

    # The selected crop must overlap with the hot spot
    assert y <= 64 and y + 64 >= 80, f"y={y} misses hot spot rows 64–80"
    assert x <= 64 and x + 64 >= 80, f"x={x} misses hot spot cols 64–80"


def test_select_inset_region_returns_valid_crop_bounds() -> None:
    from tools.vis.leave_one_out_diff import _select_inset_region

    rng = np.random.default_rng(7)
    loss = rng.random((256, 256)).astype(np.float32)

    y, x = _select_inset_region(loss, crop=64, stride=8)

    assert 0 <= y <= 256 - 64, f"y={y} out of bounds"
    assert 0 <= x <= 256 - 64, f"x={x} out of bounds"


def test_select_inset_region_uniform_loss_returns_origin() -> None:
    """Uniform loss: first window wins (top-left corner)."""
    from tools.vis.leave_one_out_diff import _select_inset_region

    loss = np.ones((128, 128), dtype=np.float32)
    y, x = _select_inset_region(loss, crop=64, stride=8)

    assert y == 0
    assert x == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_leave_one_out_diff.py::test_select_inset_region_picks_highest_loss_region tests/test_leave_one_out_diff.py::test_select_inset_region_returns_valid_crop_bounds tests/test_leave_one_out_diff.py::test_select_inset_region_uniform_loss_returns_origin -v
```

Expected: FAIL with `ImportError: cannot import name '_select_inset_region'`

- [ ] **Step 3: Implement `_select_inset_region()`**

Add after `ssim_loss_map` in `tools/vis/leave_one_out_diff.py`:

```python
def _select_inset_region(
    loss_mean: np.ndarray,
    crop: int = 64,
    stride: int = 8,
) -> tuple[int, int]:
    """Return (y, x) top-left of the crop with highest mean SSIM loss.

    Slides a ``crop × crop`` window (step ``stride``) over ``loss_mean`` and
    picks the position whose window mean is highest.  The first window wins on
    ties (top-left bias).

    Args:
        loss_mean: H×W float32 mean SSIM loss map (average across conditions).
        crop: Crop side length in pixels.
        stride: Sliding-window stride in pixels.

    Returns:
        ``(y, x)`` top-left corner of the selected crop (both ≥ 0).
    """
    H, W = loss_mean.shape
    best_score: float = -1.0
    best_yx: tuple[int, int] = (0, 0)
    for y in range(0, H - crop + 1, stride):
        for x in range(0, W - crop + 1, stride):
            score = float(loss_mean[y : y + crop, x : x + crop].mean())
            if score > best_score:
                best_score = score
                best_yx = (y, x)
    return best_yx
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_leave_one_out_diff.py::test_select_inset_region_picks_highest_loss_region tests/test_leave_one_out_diff.py::test_select_inset_region_returns_valid_crop_bounds tests/test_leave_one_out_diff.py::test_select_inset_region_uniform_loss_returns_origin -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add tools/vis/leave_one_out_diff.py tests/test_leave_one_out_diff.py
git commit -m "feat: add _select_inset_region helper for auto inset crop selection"
```

---

## Task 4: `render_loo_ssim_figure()` main renderer

**Files:**
- Modify: `tools/vis/leave_one_out_diff.py` (add constants + function before `render_loo_cache`)
- Modify: `tests/test_leave_one_out_diff.py` (add smoke test)

- [ ] **Step 1: Write the failing smoke test**

First add the helper to produce 128×128 synthetic images (required because `_select_inset_region` needs images ≥ 64px and SSIM works better on larger images). Add to the bottom of `tests/test_leave_one_out_diff.py`:

```python
# ── render_loo_ssim_figure ────────────────────────────────────────────────────

def _make_cache_large(tmp_path: Path, resolution: int = 128) -> Path:
    """Like _make_cache but with resolution×resolution images for SSIM tests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    group_names = ("cell_types", "cell_state", "vasculature", "microenv")

    all_img = np.full((resolution, resolution, 3), 200, dtype=np.uint8)
    (tmp_path / "all").mkdir()
    Image.fromarray(all_img).save(tmp_path / "all" / "generated_he.png")

    (tmp_path / "triples").mkdir()
    entries_triples = []
    for i, omit in enumerate(group_names):
        active = [g for g in group_names if g != omit]
        val = 80 + i * 30
        img = np.full((resolution, resolution, 3), val, dtype=np.uint8)
        fname = f"{i + 1:02d}_{'__'.join(active)}.png"
        Image.fromarray(img).save(tmp_path / "triples" / fname)
        entries_triples.append({
            "active_groups": active,
            "condition_label": f"triples_{i}",
            "image_label": f"lbl_{i}",
            "image_path": f"triples/{fname}",
        })

    manifest = {
        "version": 1,
        "tile_id": "large_tile",
        "group_names": list(group_names),
        "sections": [
            {"title": "3 active groups", "subset_size": 3, "entries": entries_triples},
            {
                "title": "4 active groups",
                "subset_size": 4,
                "entries": [{
                    "active_groups": list(group_names),
                    "condition_label": "all",
                    "image_label": "all",
                    "image_path": "all/generated_he.png",
                }],
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_render_loo_ssim_figure_produces_png(tmp_path) -> None:
    from tools.vis.leave_one_out_diff import render_loo_ssim_figure

    cache = _make_cache_large(tmp_path / "cache")
    out_path = tmp_path / "leave_one_out_ssim.png"
    render_loo_ssim_figure(cache, out_path=out_path)

    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_render_loo_ssim_figure_with_cell_mask(tmp_path) -> None:
    from tools.vis.leave_one_out_diff import render_loo_ssim_figure

    cache = _make_cache_large(tmp_path / "cache")
    # Write a cell mask
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 255
    Image.fromarray(mask).save(cache / "cell_mask.png")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    manifest["cell_mask_path"] = "cell_mask.png"
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out_path = tmp_path / "ssim_with_mask.png"
    render_loo_ssim_figure(cache, out_path=out_path)

    assert out_path.is_file()
    assert out_path.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_leave_one_out_diff.py::test_render_loo_ssim_figure_produces_png tests/test_leave_one_out_diff.py::test_render_loo_ssim_figure_with_cell_mask -v
```

Expected: FAIL with `ImportError: cannot import name 'render_loo_ssim_figure'`

- [ ] **Step 3: Add module-level color constants**

Add these constants after the existing `COLOR_REF` / `COLOR_BASELINE` lines near the top of `tools/vis/leave_one_out_diff.py`:

```python
# Per-group highlight colours for the SSIM figure
_LOO_SSIM_HIGHLIGHT: dict[str, str] = {
    "cell_state": "#ff6644",
    "microenv": "#ddaa00",
}
_LOO_SSIM_INSET_TEAL = "#00ccaa"
_LOO_SSIM_NEUTRAL = "#555555"
```

- [ ] **Step 4: Implement `render_loo_ssim_figure()`**

Add the full function before `render_loo_cache` in `tools/vis/leave_one_out_diff.py`:

```python
def render_loo_ssim_figure(
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    out_path: Path,
    crop_size: int = 64,
) -> None:
    """Save the LOO SSIM publication figure (3 rows × 5 columns).

    Row 0 — Generated H&E (full tile, 256×256) with teal inset-region marker
             on the all-channels panel.
    Row 1 — Cell inset: 64×64 crop (same region every column), upsampled 4×
             with nearest-neighbour. Region is auto-selected as the window
             with the highest mean SSIM loss across all four drop conditions.
    Row 2 — SSIM structural loss map (cell-masked, globally normalised).
             All-channels column is shown as a black "0 (baseline)" panel.

    Args:
        cache_dir: Ablation cache directory containing ``manifest.json``.
        orion_root: Optional ORION dataset root (unused in output; reserved).
        style_mapping: Optional tile-id remapping (unused in output; reserved).
        out_path: Destination PNG path.
        crop_size: Side length of the inset crop in pixels (default 64).
    """
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cache_dir = Path(cache_dir)
    manifest = load_manifest(cache_dir)
    sections = manifest["sections"]
    group_names = tuple(manifest["group_names"])
    tile_id = str(manifest["tile_id"])
    cell_mask = _load_cell_mask_array(cache_dir, manifest)

    # ── Load images ──────────────────────────────────────────────────────────
    all_entry = _find_all_entry(sections, len(group_names))
    img_all = _load_rgb_float32(cache_dir / all_entry["image_path"]).astype(np.uint8)

    loo_images: list[np.ndarray] = []
    for group in FOUR_GROUP_ORDER:
        entry = find_loo_entry(sections, group)
        loo_images.append(
            _load_rgb_float32(cache_dir / entry["image_path"]).astype(np.uint8)
        )

    # ── SSIM loss maps ───────────────────────────────────────────────────────
    raw_ssim: list[np.ndarray] = [ssim_loss_map(img_all, img) for img in loo_images]

    if cell_mask is not None:
        H, W = img_all.shape[:2]
        if cell_mask.shape != (H, W):
            cell_mask = np.asarray(
                Image.fromarray((np.clip(cell_mask, 0, 1) * 255).astype(np.uint8)).resize(
                    (W, H), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        binary = (cell_mask > 0.5).astype(np.float32)
        raw_ssim = [m * binary for m in raw_ssim]

    global_max = max(float(m.max()) for m in raw_ssim)
    if global_max > 0.0:
        ssim_norm: list[np.ndarray] = [
            np.clip(m / global_max, 0.0, 1.0).astype(np.float32) for m in raw_ssim
        ]
    else:
        ssim_norm = [np.zeros_like(m) for m in raw_ssim]

    # ── Inset region selection ────────────────────────────────────────────────
    loss_mean = np.stack(ssim_norm).mean(axis=0).astype(np.float32)
    iy, ix = _select_inset_region(loss_mean, crop=crop_size, stride=8)

    def _crop_upsample(img: np.ndarray) -> np.ndarray:
        H_out, W_out = img.shape[:2]
        crop = img[iy : iy + crop_size, ix : ix + crop_size]
        return np.asarray(
            Image.fromarray(crop).resize((W_out, H_out), Image.NEAREST),
            dtype=np.uint8,
        )

    # ── Column data ───────────────────────────────────────────────────────────
    he_images = [img_all] + loo_images
    inset_images = [_crop_upsample(img) for img in he_images]
    # SSIM row: all-channels column is zeros (no diff to itself)
    ssim_display = [np.zeros(img_all.shape[:2], dtype=np.float32)] + ssim_norm

    col_groups = [None] + list(FOUR_GROUP_ORDER)  # type: ignore[list-item]
    col_labels = ["All channels"] + [
        f"Drop {g.replace('_', ' ').title()}" for g in FOUR_GROUP_ORDER
    ]
    col_title_colors = ["#cccccc"] + [
        _LOO_SSIM_HIGHLIGHT.get(g, "#888888") for g in FOUR_GROUP_ORDER
    ]
    col_title_weights = ["normal"] + [
        "bold" if g in _LOO_SSIM_HIGHLIGHT else "normal" for g in FOUR_GROUP_ORDER
    ]
    inset_colors = [_LOO_SSIM_INSET_TEAL] + [
        _LOO_SSIM_HIGHLIGHT.get(g, _LOO_SSIM_NEUTRAL) for g in FOUR_GROUP_ORDER
    ]
    inset_lw = [2.0] + [2.0 if g in _LOO_SSIM_HIGHLIGHT else 1.0 for g in FOUR_GROUP_ORDER]

    # ── Figure layout ─────────────────────────────────────────────────────────
    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot4", ["#000000", "#ff4400", "#ffff00", "#ffffff"]
    )

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(
        f"Leave-one-out group diff (SSIM) — tile {tile_id}", fontsize=12, y=0.985
    )

    for col in range(5):
        # Row 0: full H&E
        ax0 = axes[0, col]
        ax0.imshow(he_images[col])
        if cell_mask is not None:
            _maybe_contour_cell_mask(ax0, cell_mask, he_images[col].shape[:2])
        if col == 0:
            ax0.add_patch(
                Rectangle(
                    (ix, iy),
                    crop_size,
                    crop_size,
                    linewidth=2,
                    edgecolor=_LOO_SSIM_INSET_TEAL,
                    facecolor="none",
                )
            )
        ax0.set_title(
            col_labels[col],
            fontsize=9,
            color=col_title_colors[col],
            fontweight=col_title_weights[col],
        )
        ax0.set_xticks([])
        ax0.set_yticks([])
        if col == 0:
            ax0.set_ylabel("Generated H&E", fontsize=10)

        # Row 1: cell inset
        ax1 = axes[1, col]
        ax1.imshow(inset_images[col])
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(inset_colors[col])
            spine.set_linewidth(inset_lw[col])
        ax1.set_xticks([])
        ax1.set_yticks([])
        if col == 0:
            ax1.set_ylabel("Cell inset\n(auto-selected)", fontsize=10)

        # Row 2: SSIM loss
        ax2 = axes[2, col]
        im = ax2.imshow(ssim_display[col], cmap=hot_cmap, vmin=0.0, vmax=1.0)
        if col == 0:
            ax2.text(
                0.5,
                0.5,
                "0\n(baseline)",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#777777",
            )
        ax2.set_xticks([])
        ax2.set_yticks([])
        if col == 0:
            ax2.set_ylabel("SSIM loss\n(cell-masked)", fontsize=10)

    # ── Colorbar under last column ────────────────────────────────────────────
    divider = make_axes_locatable(axes[2, 4])
    cbar_ax = divider.append_axes("bottom", size="8%", pad=0.20)
    sm = matplotlib.cm.ScalarMappable(
        cmap=hot_cmap, norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=8, pad=1)
    cbar.set_label("SSIM structural loss (globally normalized)", fontsize=8, labelpad=3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_leave_one_out_diff.py::test_render_loo_ssim_figure_produces_png tests/test_leave_one_out_diff.py::test_render_loo_ssim_figure_with_cell_mask -v
```

Expected: 2 PASSED

- [ ] **Step 6: Run full test file to check no regressions**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_leave_one_out_diff.py -v
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add tools/vis/leave_one_out_diff.py tests/test_leave_one_out_diff.py
git commit -m "feat: add render_loo_ssim_figure — 3-row SSIM publication figure"
```

---

## Task 5: Wire into `render_loo_cache()` and CLI

**Files:**
- Modify: `tools/vis/leave_one_out_diff.py` — update `render_loo_cache`, `render_loo_cache_root`, `main()`

- [ ] **Step 1: Update `render_loo_cache()` to accept `--ssim` flag**

Find `render_loo_cache` and update the signature and body:

```python
def render_loo_cache(
    cache_dir: Path,
    *,
    orion_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    out_path: Path | None = None,
    stats_path: Path | None = None,
    ssim: bool = False,
) -> tuple[Path, Path]:
    """Render one cache dir and return figure/stats paths.

    When ``ssim=True``, also writes ``leave_one_out_ssim.png`` alongside the
    existing pixel-diff figure.
    """
    cache_dir = Path(cache_dir)
    out_path = Path(out_path) if out_path is not None else cache_dir / "leave_one_out_diff.png"
    stats_path = (
        Path(stats_path) if stats_path is not None else out_path.with_name("leave_one_out_diff_stats.json")
    )

    diffs = compute_loo_diffs(cache_dir)
    save_loo_stats(diffs, stats_path)
    render_loo_diff_figure(
        diffs,
        cache_dir,
        orion_root=orion_root,
        style_mapping=style_mapping,
        out_path=out_path,
    )

    if ssim:
        ssim_path = out_path.with_name("leave_one_out_ssim.png")
        render_loo_ssim_figure(
            cache_dir,
            orion_root=orion_root,
            style_mapping=style_mapping,
            out_path=ssim_path,
        )

    return out_path, stats_path
```

- [ ] **Step 2: Update `render_loo_cache_root()` to forward the flag**

Find `render_loo_cache_root` and add `ssim: bool = False` to the signature, forwarding it into each `render_loo_cache` call:

```python
def render_loo_cache_root(
    cache_root: Path,
    *,
    orion_root: Path | None = None,
    style_mapping: dict[str, str] | None = None,
    out_root: Path | None = None,
    workers: int = 1,
    show_progress: bool = True,
    ssim: bool = False,
) -> list[tuple[Path, Path]]:
    """Render leave-one-out figures for every cache under cache_root."""
    cache_root = Path(cache_root)
    cache_dirs = _find_cache_dirs(cache_root)
    if not cache_dirs:
        raise FileNotFoundError(f"No manifest.json files found under {cache_root}")

    worker_count = max(1, int(workers))

    def _resolve_outputs(cache_dir: Path) -> tuple[Path, Path]:
        if out_root is None:
            out_path = cache_dir / "leave_one_out_diff.png"
            stats_path = cache_dir / "leave_one_out_diff_stats.json"
        else:
            rel = cache_dir.relative_to(cache_root)
            out_path = Path(out_root) / rel / "leave_one_out_diff.png"
            stats_path = Path(out_root) / rel / "leave_one_out_diff_stats.json"
        return out_path, stats_path

    if worker_count == 1:
        rendered: list[tuple[Path, Path]] = []
        iterator = _progress(
            cache_dirs,
            total=len(cache_dirs),
            desc="Rendering LOO",
            disable=not show_progress,
        )
        for cache_dir in iterator:
            out_path, stats_path = _resolve_outputs(cache_dir)
            rendered.append(
                render_loo_cache(
                    cache_dir,
                    orion_root=orion_root,
                    style_mapping=style_mapping,
                    out_path=out_path,
                    stats_path=stats_path,
                    ssim=ssim,
                )
            )
        return rendered

    future_to_cache: dict[Any, Path] = {}
    rendered: list[tuple[Path, Path]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for cache_dir in cache_dirs:
            out_path, stats_path = _resolve_outputs(cache_dir)
            future = executor.submit(
                render_loo_cache,
                cache_dir,
                orion_root=orion_root,
                style_mapping=style_mapping,
                out_path=out_path,
                stats_path=stats_path,
                ssim=ssim,
            )
            future_to_cache[future] = cache_dir

    iterator = _progress(
        as_completed(future_to_cache),
        total=len(future_to_cache),
        desc=f"Rendering LOO ({worker_count} workers)",
        disable=not show_progress,
    )
    completed: dict[Path, tuple[Path, Path]] = {}
    for future in iterator:
        cache_dir = future_to_cache[future]
        completed[cache_dir] = future.result()

    for cache_dir in cache_dirs:
        rendered.append(completed[cache_dir])
    return rendered
```

- [ ] **Step 3: Add `--ssim` to `main()`**

Find the `main()` function and add the flag after `--no-progress`:

```python
parser.add_argument(
    "--ssim",
    action="store_true",
    help="Also render the SSIM structural-loss figure (leave_one_out_ssim.png)",
)
```

Then pass `ssim=args.ssim` to both `render_loo_cache(...)` and `render_loo_cache_root(...)` calls inside `main()`.

- [ ] **Step 4: Smoke-test the CLI end-to-end**

```bash
cd /home/ec2-user/PixCell && python -m pytest tests/test_leave_one_out_diff.py -v
```

Expected: all tests pass (existing + new).

- [ ] **Step 5: Commit**

```bash
git add tools/vis/leave_one_out_diff.py
git commit -m "feat: wire --ssim flag into render_loo_cache and CLI"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|-----------------|------|
| 3-row × 5-col layout | Task 4 |
| Row 1: full H&E with teal inset marker | Task 4 |
| Row 2: fixed 64×64 crop, 4× upsampled, nearest-neighbour | Task 4 |
| Inset: auto-selected from max mean SSIM loss | Task 3 + Task 4 |
| Row 3: SSIM loss, cell-masked, globally normalized | Task 2 + Task 4 |
| All-channels SSIM panel = black "0 (baseline)" | Task 4 |
| Colormap: hot (black→red→yellow→white) | Task 4 |
| Colorbar: one subplot wide, under last column | Task 4 |
| ★ highlights on cell_state and microenv headers | Task 4 |
| Inset border colours per column | Task 4 |
| Output: `leave_one_out_ssim.png` | Task 5 |
| CLI `--ssim` flag | Task 5 |
| scikit-image dependency | Task 1 |

All requirements covered. No gaps.

**Placeholder scan:** No TBDs, TODOs, or vague steps found.

**Type consistency:** `ssim_loss_map` returns `np.ndarray` (float32 H×W); `_select_inset_region` takes `np.ndarray` float32 and returns `tuple[int, int]`; both are consumed correctly in `render_loo_ssim_figure`. `render_loo_cache` returns `tuple[Path, Path]` — unchanged signature shape.
