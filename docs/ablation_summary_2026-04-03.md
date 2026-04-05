# Channel Ablation Summary

This note summarizes the cached Stage 3 ablation results across `1000` tile-level `metrics.json` files under `inference_output/cache/*/metrics.json`, plus the existing figures:

- [dataset_metrics_filtered.png](/home/ec2-user/PixCell/figures/dataset_metrics_filtered.png)
- [leave_one_out_diff.png](/home/ec2-user/PixCell/inference_output/cache/512_9728/leave_one_out_diff.png)
- [exp1_microenv cancer sweep](/home/ec2-user/PixCell/inference_output/channel_sweep/exp1_microenv/cancer_12800_16384.png)
- [exp2_cell_type_relabeling.png](/home/ec2-user/PixCell/inference_output/channel_sweep/exp2_cell_type_relabeling.png)
- [exp3_cell_state_relabeling.png](/home/ec2-user/PixCell/inference_output/channel_sweep/exp3_cell_state_relabeling.png)

## 1. Average performance by number of active groups


| groups | cosine | lpips | aji   | pq    | fid    |
| ------ | ------ | ----- | ----- | ----- | ------ |
| 1g     | 0.548  | 0.419 | 0.208 | 0.115 | 65.118 |
| 2g     | 0.565  | 0.395 | 0.347 | 0.241 | 62.967 |
| 3g     | 0.570  | 0.380 | 0.456 | 0.362 | 64.330 |
| 4g     | 0.564  | 0.372 | 0.539 | 0.476 | 68.477 |


Interpretation:

- Adding more groups improves `LPIPS`, `AJI`, and `PQ` almost monotonically.
- `FID` does **not** improve monotonically. It is best at `2g`, worsens at `3g`, and is worst at `4g`.

## 2. Best and worst conditions


| metric | best condition                               | value  | worst condition                              | value  |
| ------ | -------------------------------------------- | ------ | -------------------------------------------- | ------ |
| cosine | `cell_state+vasculature`                     | 0.580  | `cell_types`                                 | 0.525  |
| lpips  | `cell_state+cell_types+microenv+vasculature` | 0.372  | `cell_types`                                 | 0.446  |
| aji    | `cell_state+cell_types+microenv+vasculature` | 0.539  | `cell_types`                                 | 0.062  |
| pq     | `cell_state+cell_types+microenv+vasculature` | 0.476  | `cell_types`                                 | 0.010  |
| fid    | `cell_types+microenv`                        | 60.091 | `cell_state+cell_types+microenv+vasculature` | 68.477 |


Interpretation:

- The full `4g` model is strongest for structure-aware metrics (`LPIPS`, `AJI`, `PQ`).
- The full `4g` model is **not** strongest for `FID`.
- `cell_types` alone is weak across nearly every metric.

## 3. Average effect of adding each group

Positive deltas mean "adding this group tends to help."


| added group   | delta lpips (better +) | delta aji | delta pq | delta fid (better +) |
| ------------- | ---------------------- | --------- | -------- | -------------------- |
| `cell_types`  | +0.002                 | +0.007    | +0.006   | +0.245               |
| `cell_state`  | +0.031                 | +0.214    | +0.203   | -1.090               |
| `vasculature` | +0.002                 | +0.024    | +0.027   | +0.328               |
| `microenv`    | +0.036                 | +0.228    | +0.254   | -0.501               |


Interpretation:

- `cell_state` and `microenv` are the dominant contributors to `AJI/PQ/LPIPS`.
- Those same groups tend to hurt `FID` on average.
- `cell_types` and `vasculature` are weaker refiners.

## 4. Presence-vs-absence summary

This table compares all conditions where a group is present versus absent.


| group present? | cosine delta | lpips delta | aji delta | pq delta | fid delta |
| -------------- | ------------ | ----------- | --------- | -------- | --------- |
| `cell_types`   | -0.003       | +0.005      | -0.032    | -0.028   | +0.471    |
| `cell_state`   | +0.024       | -0.030      | +0.201    | +0.184   | +0.665    |
| `vasculature`  | -0.006       | +0.005      | -0.016    | -0.007   | +0.329    |
| `microenv`     | +0.010       | -0.034      | +0.213    | +0.234   | -0.058    |


Notes:

- For `LPIPS`, lower is better, so a negative delta is an improvement.
- For `FID`, lower is better, so a positive delta means the group makes `FID` worse.

## 5. Figure-based evidence

### Why "pixel diff is visible" but H&E "looks similar"

Evidence:

- [leave_one_out_diff.png](/home/ec2-user/PixCell/inference_output/cache/512_9728/leave_one_out_diff.png)
- [leave_one_out_diff_stats.json](/home/ec2-user/PixCell/inference_output/cache/512_9728/leave_one_out_diff_stats.json)

Key numbers from the leave-one-out example:


| removed group | mean diff | max diff | pct pixels > 10 |
| ------------- | --------- | -------- | --------------- |
| `cell_types`  | 1.870     | 124.921  | 2.29%           |
| `cell_state`  | 8.752     | 144.978  | 24.28%          |
| `vasculature` | 2.106     | 114.607  | 2.74%           |
| `microenv`    | 6.960     | 255.000  | 19.99%          |


Interpretation:

- `cell_state` and `microenv` create real, localized changes.
- The global tissue layout and stain family stay similar, so the H&E panels can still look qualitatively close.

### Why `cell_state` and `microenv` look most influential

Evidence:

- [exp2_cell_type_relabeling.png](/home/ec2-user/PixCell/inference_output/channel_sweep/exp2_cell_type_relabeling.png)
- [exp3_cell_state_relabeling.png](/home/ec2-user/PixCell/inference_output/channel_sweep/exp3_cell_state_relabeling.png)
- [exp1_microenv cancer sweep](/home/ec2-user/PixCell/inference_output/channel_sweep/exp1_microenv/cancer_12800_16384.png)

Interpretation:

- Cell-state relabeling changes nuclei phenotype and texture visibly.
- Microenvironment sweeps shift local contrast, stain density, and neighborhood appearance.
- These are strong conditional edits, even when the coarse tissue scene stays fixed.

## 6. Answer to the FID question

Does adding more groups generate "worse" results?

Not in a blanket sense.

- If "better" means more condition-faithful and biologically structured outputs, then more groups help: `PQ`, `AJI`, and `LPIPS` improve strongly from `1g -> 4g`.
- If "better" means lower dataset-level `FID`, then more groups do **not** help, and `4g` is worst.

This is most consistent with a metric-mismatch story:

- `FID` rewards matching the overall population distribution of real H&E.
- Adding `cell_state` and `microenv` makes outputs more condition-sensitive and more structurally distinct.
- That added specificity improves local faithfulness, but can move the generated image distribution away from the average real H&E distribution used by `FID`.

In short:

- more groups improve **controllability / semantic faithfulness**
- more groups do not necessarily improve **global unconditional realism**

