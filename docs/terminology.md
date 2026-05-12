# PixCell Conditioning Terminology

Canonical terms for the two ControlNet conditioning sources. Replaces the older "semantic vs spatial" framing, which mislabels both modalities — TME channels carry biological **identity** (semantic) per pixel, and UNI embeds tissue **appearance** (also semantic in the everyday sense). The real axis is **global appearance prior** vs **spatially-resolved biological maps**.

## Two conditioning sources

| Term | What it is | Modality | Encodes |
|---|---|---|---|
| **Appearance prior** (UNI) | Global pooled embedding from UNI-2h, a pathology foundation model pretrained on H&E. One vector per patch, no spatial grid. | H&E image | Stain profile, tissue texture, nuclear morphology distribution, cell density regime, fine appearance cues |
| **Structural / biological layout** (TME channels) | Per-pixel maps from CODEX/segmentation: cell-type masks, cell-state masks, vasculature, microenv (oxygen, glucose). | Spatial-omics channels | Cell identity per pixel, vessel geometry, microenv gradients — explicit hand-defined biology |

## Use these terms, not these

| Avoid | Prefer | Why |
|---|---|---|
| "semantic features" (for UNI) | "appearance prior" or "appearance embedding" | TME channels are also semantic — they label cell identity per pixel. The UNI-vs-channels split is not semantic-vs-non-semantic. |
| "spatial features" (for channels) | "structural layout" or "biological layout maps" | UNI also contains implicit spatial information (densities, regional patterns). The channels-vs-UNI split is not spatial-vs-non-spatial. |
| "style features" (for UNI) | "appearance prior" | "Style" suggests purely cosmetic / stain-only; UNI carries biology too. |
| "content features" (for channels) | "structural layout" | Imported from style-transfer literature; misleading for biological maps. |

## How they differ, precisely

- **Granularity**: UNI is one global vector; channels are per-pixel rasters.
- **Modality of origin**: UNI is learned from H&E pixels; channels come from spatial-omics segmentation.
- **Information overlap**: both encode coarse cell-type identity. Their **residual difference** is the load-bearing claim: UNI carries appearance-level morphology and texture that channels cannot represent (see `2026-05-12-uni-probe-semantic-ablation-design.md`).

## Paper / docstring usage

- First reference: "appearance prior (UNI)" / "structural layout (TME channels)".
- After that: "UNI" / "TME" is fine.
- Do not write "semantic UNI features" or "spatial TME features" — the modifiers are misleading and conflict with this document.
