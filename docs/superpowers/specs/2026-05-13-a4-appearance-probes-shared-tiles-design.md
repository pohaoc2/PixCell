# Design: a4 Appearance Probes + Shared Tile Set

**Date:** 2026-05-13
**Scope:** Extend `src/a4_uni_probe` to probe tissue appearance attrs (H&E stain, texture) with Ridge regression, and unify all null/sweep experiments to a single shared 30-tile set so cross-attr comparisons are on identical tissue.

---

## Background and Motivation

The current a4 pipeline fits Ridge probes only for morphological attrs (eccentricity, nuclear area, nuclei density) and TME channel attrs. Appearance metrics (H/E stain statistics, Haralick texture) were only measured as side-effects on morphology-targeted edits, not probed directly. Two problems:

1. **Appearance not targeted:** the `targeted` direction in appearance sweep/null is the morphology probe direction, not an appearance probe direction. Cross-attr comparison of appearance effects is confounded.
2. **Different tiles per attr:** `_select_sweep_tiles` stratifies by each attr's value, producing disjoint tile sets per attr. Appearance differences across attrs may reflect tile differences, not attr-specific effects.

---

## Approach: Extend a4 In-Place (Option A)

Extend the existing `src/a4_uni_probe` pipeline. Old null/sweep outputs for morphology attrs (which used different tiles) are superseded and overwritten with shared-tile results.

---

## Design

### 1. Appearance Labels on Real H&E Tiles

**File:** `src/a4_uni_probe/labels.py`

Add:
- `APPEARANCE_ATTR_NAMES` — 8 attrs:
  `h_mean, e_mean, texture_h_contrast, texture_h_homogeneity, texture_h_energy, texture_e_contrast, texture_e_homogeneity, texture_e_energy`
  Excluded: `h_std`, `e_std` (redundant with mean for probing purposes), `stain_vector_angle_deg` (requires paired reference, introduces self-referential dependency).
- `build_appearance_label_matrix(tile_ids, he_dir)` — calls `appearance_row_for_image` from `appearance_metrics.py` for each tile's real H&E image at `data/orion-crc33/he/{tile_id}.png`.
- Extend `ALL_ATTR_NAMES` to append `APPEARANCE_ATTR_NAMES`.
- Extend `build_label_matrix` to call `build_appearance_label_matrix` and concatenate columns.

Probes are fit on all ~10,273 valid tiles (those with valid morphology labels and existing H&E files). The shared tile set is only used for generation, not probe fitting.

**New CLI arg** in `main.py probe` subparser:
```
--he-dir PATH    path to real H&E tiles (default: data/orion-crc33/he)
```

### 2. Shared Tile Set

**File:** `inference_output/a1_concat/a4_uni_probe/shared_tiles.json`

Generated once: 30 tile IDs random-sampled from the ~10,273 valid pool, `seed=42`. Format:
```json
{"tile_ids": ["10752_20480", ...], "seed": 42, "n": 30}
```

Used by all null and sweep runs. Not regenerated unless explicitly replaced.

### 3. Pipeline Changes — Fixed Tile Arg

**File:** `src/a4_uni_probe/edit.py`

New argument added to both `run_null` and `run_sweep`:
```
--fixed-tile-ids PATH   JSON file with shared tile list;
                        bypasses per-attr stratification when provided
```

When `--fixed-tile-ids` is set: skip `_select_sweep_tiles`, load tile IDs from the JSON file directly. All other logic (sharding, direction loading, generation, metrics) unchanged.

**File:** `src/a4_uni_probe/main.py` — add `--fixed-tile-ids` and `--he-dir` to `sweep` and `null` subparsers.

### 4. Probe Stage

Re-run `probe` stage after label extension. Outputs:
- `probe_results.csv` — now includes all 8 appearance attrs with UNI R², TME R², and ΔR²
- `probe_directions/{attr}_uni_direction.npy` — new `w` files for each appearance attr

Top-3 appearance attrs by ΔR² are selected for sweep and null (same `_select_sweep_attrs` logic, `top_k=3`).

### 5. Experiments

All experiments use `--fixed-tile-ids shared_tiles.json`. Sweep uses `--alphas -1 0 1` (3 values, down from 5).

| Experiment | Attrs | Tiles | Dirs | Alphas | Images |
|---|---|---|---|---|---|
| Sweep — morphology | 3 | 30 | 2 (targeted + random) | 3 | 540 |
| Sweep — appearance | top 3 by ΔR² | 30 | 2 | 3 | 540 |
| Null — morphology | 3 | 30 | 2 (targeted + random) | — | 180 |
| Null — appearance | top 3 by ΔR² | 30 | 2 | — | 180 |
| **Total** | | | | | **1,440** |

### 6. Parallelism

Two terminals per experiment run, each handling half the tiles:
```
# Terminal 1
python -m src.a4_uni_probe.main sweep ... --tile-shard-index 0 --tile-shard-count 2

# Terminal 2
python -m src.a4_uni_probe.main sweep ... --tile-shard-index 1 --tile-shard-count 2
```

Each process uses ~6GB VRAM; fits within 15GB total. Sharding already implemented in `edit.py` — no new code needed.

### 7. Post-Generation

After generation, re-run the `appearance` subcommand (already implemented in `appearance_metrics.py`) to add stain/texture readouts to sweep and null metrics CSVs. Then re-run `figures` to regenerate panels A–E (and extend to cover appearance attrs).

---

## Execution Order

1. Extend `labels.py` with appearance attrs and `build_appearance_label_matrix`
2. Add `--he-dir` to `probe` subparser in `main.py`
3. Add `--fixed-tile-ids` to `sweep` and `null` subparsers
4. Generate `shared_tiles.json`
5. Re-run `probe` stage → new `probe_results.csv` + appearance `w` files
6. Run `sweep` (morph + appearance) in two terminal pairs
7. Run `null` (morph + appearance) in two terminal pairs
8. Run `appearance` subcommand on all new outputs
9. Re-run `figures`

---

## Files Changed

| File | Change |
|---|---|
| `src/a4_uni_probe/labels.py` | Add `APPEARANCE_ATTR_NAMES`, `build_appearance_label_matrix`, extend `ALL_ATTR_NAMES` and `build_label_matrix` |
| `src/a4_uni_probe/main.py` | Add `--he-dir` to `probe`, `--fixed-tile-ids` to `sweep` and `null` |
| `src/a4_uni_probe/edit.py` | Add `--fixed-tile-ids` handling in `run_sweep` and `run_null` |
| `inference_output/.../shared_tiles.json` | New — generated once |
| `inference_output/.../probe_results.csv` | Overwritten with appearance attrs added |
| `inference_output/.../probe_directions/` | New `w` files for 8 appearance attrs |
| `inference_output/.../sweep/` | Overwritten for morph attrs; new dirs for appearance attrs |
| `inference_output/.../null/` | Overwritten for morph attrs; new dirs for appearance attrs |

---

## Out of Scope

- Lasso / stability selection (discussed but deferred — distributed representations favor Ridge)
- Subspace nulling (k>1 directions) — future work
- Re-running full-UNI-null (`tme_only.png`) baseline for new tiles — can be added later
- New figure panels for appearance probe results — handled by existing `figures` subcommand with minor extension
