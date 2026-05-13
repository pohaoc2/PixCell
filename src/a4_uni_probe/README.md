The a4 UNI probe pipeline writes cached labels and features during `probe`, then reuses the saved UNI directions for `sweep`, `null`, and `figures`. Generated-tile morphology metrics stay `NaN` until an offline CellViT pass writes JSON sidecars beside the generated PNGs; rerun `python -m src.a4_uni_probe.main figures` after those sidecars land.

If `python -m src.a4_uni_probe.main appearance` has already written `appearance_sweep_summary.csv` and `appearance_null_summary.csv`, the `figures` command will also render complete appearance bar-chart panels for all stain and texture metrics, not just the original morphology summaries.

## Shared-Tile Sweep/Null Runs

The shared-tile appearance experiment uses a fixed 30-tile manifest at `inference_output/a1_concat/a4_uni_probe/shared_tiles.json`. Use `--fixed-tile-ids` plus `--attr-pool` so morphology and appearance runs compare the same tissue.

Probe refit with appearance labels:

```bash
python -m src.a4_uni_probe.main probe \
	--out-dir inference_output/a1_concat/a4_uni_probe \
	--features-dir data/orion-crc33/features \
	--exp-channels-dir data/orion-crc33/exp_channels \
	--cellvit-real-dir src/a4_uni_probe/out/cellvit \
	--he-dir data/orion-crc33/he
```

Sweep helper:

```bash
bash inference_output/a1_concat/a4_uni_probe/sweep_shards.sh morphology
bash inference_output/a1_concat/a4_uni_probe/sweep_shards.sh appearance
```

Null helper:

```bash
bash inference_output/a1_concat/a4_uni_probe/null_shards.sh morphology
bash inference_output/a1_concat/a4_uni_probe/null_shards.sh appearance
```

Generation count for the shared-tile experiment:

- Sweep, morphology: 3 attrs × 30 tiles × 2 directions × 3 alphas = 540 generations.
- Sweep, appearance: 3 attrs × 30 tiles × 2 directions × 3 alphas = 540 generations.
- Null, morphology: 3 attrs × 30 tiles × 2 conditions = 180 generations.
- Null, appearance: 3 attrs × 30 tiles × 2 conditions = 180 generations.

Total new a4 generations: 1,440 PNGs. The `full_uni_null` comparison in null summaries is read from existing `a2_decomposition/generated/*/tme_only.png` outputs and does not add generation work here.
