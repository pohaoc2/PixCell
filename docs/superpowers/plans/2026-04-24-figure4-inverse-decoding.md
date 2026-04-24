# Figure 4 — Inverse Decoding Current Plan

**Goal:** Produce `figures/pngs/07_inverse_decoding.png` with the intended encoder set for this project:
- `UNI-2h`
- `Virchow2`
- `CTransPath`

`REMEDIS` is intentionally out of scope for this plan.

## Current status

### Done

- `run_encoder_probe_to_csv()` exists in `src/a1_probe_encoders/main.py`
- `src/paper_figures/fig_inverse_decoding.py` exists
- `src/paper_figures/main.py` is wired to build `07_inverse_decoding.png`
- focused tests for the helper and figure builder pass
- `src/a1_probe_encoders/out/virchow2_linear_probe_results.csv` exists
- `figures/pngs/07_inverse_decoding.png` exists, but does not yet include CTransPath results
- local CTransPath weights are staged at:
  - `pretrained_models/ctranspath/config.json`
  - `pretrained_models/ctranspath/model.safetensors`

### Remaining blocker

- CTransPath extraction is not complete yet.
- The loader in `src/a1_probe_encoders/main.py` has been partially adapted to current `timm`, but the real end-to-end extraction still needs finishing and validation.

## Remaining work

### Task 1 — Finish CTransPath extraction

Run the `ctranspath` worker in the `pixcell` environment with GPU access.

Use the real paths in this workspace:

```bash
/home/ec2-user/miniconda3/envs/pixcell/bin/python -m src.a1_probe_encoders.main \
  --worker ctranspath \
  --he-dir data/orion-crc33/he \
  --targets-path src/a1_mask_targets/out/mask_targets_T1.npy \
  --tile-ids-path src/a1_mask_targets/out/tile_ids.txt \
  --cv-splits-path src/a1_probe_linear/out/cv_splits.json \
  --out-dir src/a1_probe_encoders/out \
  --device cuda
```

Expected output:

```text
src/a1_probe_encoders/out/ctranspath_embeddings.npy
```

Validation:

```bash
python -c "import numpy as np; e = np.load('src/a1_probe_encoders/out/ctranspath_embeddings.npy'); print(e.shape, e.dtype)"
```

Expected:

- `10379` rows
- embedding width likely `768`
- `float32`

### Task 2 — Run the linear probe on CTransPath embeddings

```bash
/home/ec2-user/miniconda3/envs/he-multiplex/bin/python -c "from pathlib import Path; from src.a1_probe_encoders.main import run_encoder_probe_to_csv; out = run_encoder_probe_to_csv(Path('src/a1_probe_encoders/out/ctranspath_embeddings.npy'), targets_path=Path('src/a1_mask_targets/out/mask_targets_T1.npy'), tile_ids_path=Path('src/a1_mask_targets/out/tile_ids.txt'), cv_splits_path=Path('src/a1_probe_linear/out/cv_splits.json'), output_csv_path=Path('src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv')); print(out)"
```

Expected output:

```text
src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv
```

Spot-check:

```bash
head -5 src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv
```

Expected header:

```text
target,r2_mean,r2_sd,n_valid_folds
```

### Task 3 — Rebuild Figure 4 with CTransPath included

```bash
conda run -n he-multiplex python -m src.paper_figures.main
```

Expected:

- `figures/pngs/07_inverse_decoding.png` updated
- the T1 panel includes `CTransPath` when `src/a1_probe_encoders/out/ctranspath_linear_probe_results.csv` exists

Spot-check:

```bash
ls -lh figures/pngs/07_inverse_decoding.png
python -c "from PIL import Image; img = Image.open('figures/pngs/07_inverse_decoding.png'); print(img.size)"
```

## Relevant files

- `src/a1_probe_encoders/main.py`
- `src/paper_figures/fig_inverse_decoding.py`
- `src/paper_figures/main.py`
- `tests/test_encoder_probe_csv.py`
- `tests/test_fig_inverse_decoding.py`

## Notes

- Inside the sandbox, `torch.cuda.is_available()` may be `False`; use unsandboxed GPU execution when running the real CTransPath extraction.
- Future CTransPath runs should use `pretrained_models/ctranspath/` instead of re-downloading weights.
- `REMEDIS` has been removed from this plan and should not be part of the remaining Figure 4 work.
