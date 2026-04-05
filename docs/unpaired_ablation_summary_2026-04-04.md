# Unpaired Channel Ablation Summary

This note summarizes the cached Stage 3 ablation results across `1000` tile-level `metrics.json` files under `/home/ec2-user/PixCell/inference_output/unpaired_ablation/ablation_results`.

For this unpaired setup, `exp_channels/` stay attached to the layout tile while `he/` and `features/` are remapped to a different style tile via [unpaired_mapping.json](/home/ec2-user/PixCell/inference_output/unpaired_ablation/data/orion-crc33-unpaired/metadata/unpaired_mapping.json).

Evidence files:

- [dataset_metrics_filtered.png](/home/ec2-user/PixCell/inference_output/unpaired_ablation/dataset_metrics_filtered.png)
- [29952_46080 ablation grid](/home/ec2-user/PixCell/inference_output/unpaired_ablation/ablation_results/29952_46080/ablation_grid.png)
- [29952_46080 leave-one-out diff](/home/ec2-user/PixCell/inference_output/unpaired_ablation/leave_one_out/29952_46080/leave_one_out_diff.png)
- [29952_46080 leave-one-out stats](/home/ec2-user/PixCell/inference_output/unpaired_ablation/leave_one_out/29952_46080/leave_one_out_diff_stats.json)
- [channel_sweep/cache/exp1_microenv/cancer_22528_34304/o2_0.00__glucose_0.00.png](/home/ec2-user/PixCell/inference_output/unpaired_ablation/channel_sweep/cache/exp1_microenv/cancer_22528_34304/o2_0.00__glucose_0.00.png)
- [channel_sweep/cache/exp1_microenv/cancer_22528_34304/o2_0.00__glucose_0.25.png](/home/ec2-user/PixCell/inference_output/unpaired_ablation/channel_sweep/cache/exp1_microenv/cancer_22528_34304/o2_0.00__glucose_0.25.png)
- [channel_sweep/cache/exp1_microenv/cancer_22528_34304/o2_0.00__glucose_0.50.png](/home/ec2-user/PixCell/inference_output/unpaired_ablation/channel_sweep/cache/exp1_microenv/cancer_22528_34304/o2_0.00__glucose_0.50.png)

## 1. Average performance by number of active groups


| groups | cosine | lpips | aji   | pq    | fid    | style_hed |
| ------ | ------ | ----- | ----- | ----- | ------ | --------- |
| 1g     | 0.526  | 0.450 | 0.135 | 0.056 | 65.670 | 0.072     |
| 2g     | 0.519  | 0.452 | 0.239 | 0.137 | 63.641 | 0.068     |
| 3g     | 0.502  | 0.456 | 0.335 | 0.232 | 64.411 | 0.067     |
| 4g     | 0.476  | 0.461 | 0.421 | 0.338 | 66.927 | 0.070     |


Interpretation:

- Adding more groups worsens `cosine` almost monotonically.
- Adding more groups worsens `lpips` almost monotonically.
- Adding more groups improves `aji` almost monotonically.
- Adding more groups improves `pq` almost monotonically.
- `fid` is best at `2g` and weakest at `4g`.
- `style_hed` is best at `3g` and weakest at `1g`.

## 2. Best and worst conditions


| metric    | best condition                               | value  | worst condition                   | value  |
| --------- | -------------------------------------------- | ------ | --------------------------------- | ------ |
| cosine    | `cell_state`                                 | 0.533  | `cell_state+microenv+vasculature` | 0.475  |
| lpips     | `microenv+vasculature`                       | 0.446  | `cell_state+microenv`             | 0.461  |
| aji       | `cell_state+cell_types+microenv+vasculature` | 0.421  | `cell_types`                      | 0.036  |
| pq        | `cell_state+cell_types+microenv+vasculature` | 0.338  | `cell_types`                      | 0.003  |
| fid       | `cell_types+microenv`                        | 59.824 | `cell_types`                      | 69.459 |
| style_hed | `cell_types+microenv`                        | 0.062  | `cell_types`                      | 0.080  |


Interpretation:

- The full `4g` model is strongest for `aji`, `pq`.
- `cell_types` is weak across multiple metrics.

## 3. Average effect of adding each group

Positive deltas mean "adding this group tends to help."


| added group   | delta cosine | delta lpips (better +) | delta aji | delta pq | delta fid (better +) | delta style_hed (better +) |
| ------------- | ------------ | ---------------------- | --------- | -------- | -------------------- | -------------------------- |
| `cell_types`  | 0.001        | -0.000                 | 0.005     | 0.004    | 0.419                | 0.001                      |
| `cell_state`  | -0.022       | -0.011                 | 0.178     | 0.150    | -1.088               | 0.003                      |
| `vasculature` | -0.002       | 0.001                  | 0.019     | 0.019    | -0.020               | -0.000                     |
| `microenv`    | -0.034       | -0.003                 | 0.191     | 0.189    | 1.409                | 0.003                      |


Interpretation:

- `microenv` is the strongest positive contributor to `aji`, `pq`.
- `microenv` helps `fid`, `style_hed` most on average, but slightly hurts `lpips`.
- Conditions containing `microenv` have the largest positive shift in `aji`, `pq`.

## 4. Presence-vs-absence summary

This table compares all conditions where a group is present versus absent.


| group present? | cosine delta | lpips delta | aji delta | pq delta | fid delta | style_hed delta |
| -------------- | ------------ | ----------- | --------- | -------- | --------- | --------------- |
| `cell_types`   | 0.003        | 0.000       | -0.024    | -0.016   | -0.257    | -0.000          |
| `cell_state`   | -0.018       | -0.010      | 0.164     | 0.133    | -0.909    | 0.003           |
| `vasculature`  | -0.000       | 0.001       | -0.010    | -0.002   | -0.608    | -0.002          |
| `microenv`     | -0.031       | -0.002      | 0.177     | 0.172    | 1.880     | 0.004           |


Notes:

- For `cosine`, higher is better, so a positive delta is an improvement.
- For `lpips`, lower is better, so a positive delta means the group helps.
- For `aji`, higher is better, so a positive delta is an improvement.
- For `pq`, higher is better, so a positive delta is an improvement.
- For `fid`, lower is better, so a positive delta means the group helps.
- For `style_hed`, lower is better, so a positive delta means the group helps.

## 5. Figure-based evidence

Representative leave-one-out tile: `29952_46080` (chosen as the closest tile to the dataset-average leave-one-out profile).


| removed group | mean diff | max diff | pct pixels > 10 |
| ------------- | --------- | -------- | --------------- |
| `cell_types`  | 4.658     | 133.438  | 11.510          |
| `cell_state`  | 19.839    | 232.283  | 46.915          |
| `vasculature` | 5.026     | 116.678  | 12.311          |
| `microenv`    | 19.428    | 239.148  | 44.788          |


Interpretation:

- `cell_state` and `microenv` create the largest localized pixel changes in leave-one-out diffs.
- The global stain family can still look similar even when those local deltas are large.

## 6. Answer to the FID question

Does adding more groups generate "worse" results?

Not in a blanket sense.

- If 'better' means stronger structure-aware metrics, then more groups help: `aji`, `pq` rise from `1g -> 4g`.
- `fid` is not best at `4g`; its best average is `2g`.
- `style_hed` is not best at `4g`; its best average is `3g`.
- `lpips` is not best at `4g`; its best average is `1g`.

