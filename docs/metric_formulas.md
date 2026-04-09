# Metric Formulas

This note summarizes the nine metrics currently used in the PixCell ablation reports and the formulas implemented in this repo.

## Notation

- `x_ref`, `x_gen`: reference and generated H&E images
- `phi(.)`: feature extractor / embedding function
- `gt_inst`, `pred_inst`: ground-truth and predicted instance masks
- `gt_fg`, `pred_fg`: foreground-vs-background binary masks derived from the instance masks
- `TP`, `TN`, `FP`, `FN`: true/false positive/negative pixel counts
- `mu_r`, `Sigma_r`: mean and covariance of reference features
- `mu_g`, `Sigma_g`: mean and covariance of generated features

## 1. FVD

`FVD` in the report is a Fréchet distance between dataset-level feature distributions:

`FVD = ||mu_r - mu_g||_2^2 + Tr(Sigma_r + Sigma_g - 2 * (Sigma_r Sigma_g)^(1/2))`

- The same Fréchet formula is used in [`tools/compute_fid.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/compute_fid.py).
- In this repo, the backend can be UNI (`fud`), Virchow-2 (`fvd`), or Inception (`fid`).
- The ablation HTML report is now labeling this family as `FVD`.
- Statistics are computed over the dataset for each condition, not per tile.
- Lower is better.

## 2. Cosine Similarity

UNI cosine similarity between flattened reference and generated embeddings:

`cosine = (a . b) / ((||a||_2 + 1e-8) (||b||_2 + 1e-8))`

- Implemented in [`tools/stage3/uni_cosine_similarity.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/stage3/uni_cosine_similarity.py).
- Higher is better.

## 3. LPIPS

This repo uses LPIPS with the AlexNet backbone:

`LPIPS(x_ref, x_gen) = sum_l mean_{h,w}( || w_l odot (fhat_l(x_ref)[h,w] - fhat_l(x_gen)[h,w]) ||_2^2 )`

- `fhat_l` denotes normalized deep features at layer `l`.
- `w_l` are the learned LPIPS channel weights.
- The implementation calls the `lpips` package directly in [`tools/compute_ablation_metrics.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/compute_ablation_metrics.py).
- Lower is better.

## 4. AJI

Aggregated Jaccard Index over matched instances:

`AJI = (sum_k |G_k cap P_{m(k)}|) / (sum_k |G_k cup P_{m(k)}| + sum_{u in unmatched GT} |G_u| + sum_{v in unmatched Pred} |P_v|)`

- Matching is chosen by Hungarian assignment on the pairwise IoU matrix.
- Pairs with zero IoU are ignored.
- Implemented in [`tools/compute_ablation_metrics.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/compute_ablation_metrics.py).
- Higher is better.

## 5. PQ

Panoptic Quality is computed from matched instances at IoU `>= 0.5`:

`SQ = (1 / TP) sum_{matched} IoU_i`

`RQ = TP / (TP + 0.5 FP + 0.5 FN)`

`PQ = SQ * RQ`

- Matching is chosen by Hungarian assignment on the IoU matrix, then thresholded at `0.5`.
- Implemented in [`tools/compute_ablation_metrics.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/compute_ablation_metrics.py).
- Higher is better.

## 6. DICE

Pixel-level foreground Dice score:

`DICE = 2 TP / (2 TP + FP + FN)`

- Foreground means `instance_id > 0`.
- Implemented in [`tools/compute_ablation_metrics.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/compute_ablation_metrics.py).
- Higher is better.

## 7. IoU

Pixel-level foreground intersection-over-union:

`IoU = TP / (TP + FP + FN)`

- Foreground means `instance_id > 0`.
- Implemented in [`tools/compute_ablation_metrics.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/compute_ablation_metrics.py).
- Higher is better.

## 8. Accuracy

Pixel-level binary accuracy on foreground-vs-background masks:

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

- Foreground means `instance_id > 0`.
- Implemented in [`tools/compute_ablation_metrics.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/compute_ablation_metrics.py).
- Higher is better.

## 9. HED Style Score

The HED style score compares stain-channel mean and standard deviation differences in HED space over the union tissue mask:

`HED_score = sum_{c in {H,E}} ( |mu_ref,c - mu_gen,c| + |sigma_ref,c - sigma_gen,c| )`

- Only the Hematoxylin and Eosin channels are used.
- The mask is `tissue_mask_ref OR tissue_mask_gen`.
- Implemented in [`tools/compute_ablation_metrics.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/compute_ablation_metrics.py) and mirrored locally in [`tools/render_ablation_html_report.py`](/home/pohaoc2/UW/bagherilab/PixCell/tools/render_ablation_html_report.py).
- Lower is better.
