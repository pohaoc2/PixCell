# Parameter-to-Phenotype Mapping via Generative H&E Translation

**Date:** 2026-04-06  
**Status:** Draft

---

## Scientific Question

> Do different simulation parameter combinations reproduce the distinct spatial TME phenotypes observed in real CRC tissue?

Two sub-claims:

- **Fidelity**: some parameter combinations produce TME configurations that fall within the distribution of real CRC regional archetypes in UNI feature space.
- **Coverage**: the simulation parameter space spans the full diversity of regional archetypes present in real CRC, or reveals which archetypes it cannot reproduce.

---

## Motivation

CRC tumors are spatially heterogeneous. Different regions of the same tumor have distinct cell compositions and dynamics: immune cells (CD8+ T cells, macrophages) concentrate at the invasive margin, tumor cells occupy the core under hypoxic/nutrient-deprived conditions, and desmoplastic stroma separates regions. No single simulation run can model the entire tissue; each run corresponds to a local spatial context governed by a specific parameter regime.

PixCell bridges the representation gap: it translates simulation spatial maps (cell type/state masks, oxygen/glucose, vasculature) into H&E patches, placing simulation outputs and real H&E in the same UNI feature space for direct comparison.

---

## Inputs


| Input                            | Description                                                                                                                            |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Simulation outputs Y_1, ..., Y_N | N parameter combinations, each producing: oxygen/glucose concentration maps, cell type/state masks, vasculature map (all 256x256 .png) |
| CRC-07 H&E tiles                 | ~10k tiles (256x256) from patient CRC-07 in the ORION-CRC dataset                                                                      |
| Style-patient H&E tiles          | H&E tiles from a different CRC patient (not CRC-07), used only for style-conditioned visualization                                     |


---

## Step 1: Regional Archetype Discovery (CRC-07)

1. Compute UNI embeddings for all ~10k CRC-07 tiles.
2. **Check first**: inspect ORION-CRC dataset for existing pathologist annotations or published region masks (tumor core, stroma, invasive margin, necrosis, immune-infiltrated regions). If annotations exist, use them to define or validate archetypes directly rather than relying solely on k-means.
3. If no annotations: sweep K=3–7 using k-means on UNI embeddings. Score each K with silhouette score. Select the largest K where every cluster medoid is biologically interpretable (each cluster can be assigned a tissue-region name).
4. Biological upper bound: K ≤ 6. Expected archetypes for CRC: tumor core, invasive margin, immune-infiltrated stroma, desmoplastic stroma, necrosis.
5. Output: K cluster centroids + one representative medoid tile per cluster.

**K selection rule:** *"K was selected as the largest value ≤ 6 for which all cluster representatives were biologically interpretable by visual inspection."*

---

## Step 2: Simulation-to-Archetype Matching (Quantitative Arm)

1. For each Y_N, generate G_N using **TME-only mode** (null UNI embedding — no style reference). This eliminates any X confounding; UNI(G_N) reflects Y_N's spatial structure only.
2. Compute UNI(G_N) for all N.
3. Assign each G_N to its nearest archetype: `archetype(N) = argmin_k dist(UNI(G_N), centroid_k)`.
4. For each archetype k, find the best-matched parameter combination: `N_k* = argmin_{N: archetype(N)=k} dist(UNI(G_N), centroid_k)`.
5. Report coverage: which archetypes have matched parameter combinations, which do not. Gaps = TME contexts the simulation cannot reproduce.

---

## Step 3: Style-Conditioned Visualization (Qualitative Arm)

For each matched pair (archetype k, best parameter N_k*):

1. Select X_k = the tile from the **style patient** (different CRC patient, not CRC-07) that is nearest to centroid_k in UNI space. This gives a context-matched style reference without using the target patient.
2. Generate G_visual_k = f(X_k, Y_N_k*) using style-conditioned PixCell inference.
3. Show side-by-side: real CRC-07 medoid of archetype k vs G_visual_k.

**Generalizability demonstration:** model trained on ORION-CRC, style reference from a second patient, simulation spatial maps from neither — yet produces realistic H&E for each regional context.

**Prerequisite check:** confirm the style patient has tiles covering all K archetypes (i.e., its UNI distribution overlaps with CRC-07's clusters). If a style patient lacks a particular TME context, use the closest available tile and note the limitation.

---

## Step 4: Parameter Space Interpretation

- Plot parameter combinations in 2D (e.g., PCA or UMAP of parameter vectors), colored by assigned archetype.
- Identify parameter regimes that govern each TME context (e.g., low oxygen + high necrosis rate → tumor core; high immune infiltration → invasive margin).
- If any archetype has no matched parameters: interpret as a biological gap in the simulation model.

---

## Figure Structure


| Figure | Content                                                                                                                                                              |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fig 1  | UMAP of CRC-07 UNI embeddings colored by cluster; one medoid tile per cluster with biological label                                                                  |
| Fig 2  | UMAP overlay: CRC-07 tiles (gray background) + G_N points colored by archetype assignment — shows where simulation outputs land relative to real tissue distribution |
| Fig 3  | Side-by-side per archetype: real CRC-07 medoid | style-conditioned G_visual_k — shows qualitative realism and style generalizability                                 |
| Fig 4  | Parameter space colored by archetype assignment — reveals which parameter regimes govern which TME context                                                           |


---

## Key Design Decisions


| Decision                                   | Rationale                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| TME-only for quantitative comparison       | Eliminates X confounding; UNI(G_N) reflects Y_N structure only                             |
| Style-conditioned for visualization        | Demonstrates cross-patient generalizability; X from a different patient avoids circularity |
| X_k selected nearest to archetype centroid | Context-matched style reference; principled, not cherry-picked                             |
| K bounded by biological interpretability   | Ensures clusters map to named tissue regions; prevents over-segmentation                   |
| Check for existing annotations first       | Annotations are authoritative; k-means is fallback only                                    |


---

## Open Questions

- **Annotations**: ORION-CRC pathologist-level annotations for CRC-07 need web search verification. If they exist, use them to define or validate K rather than relying solely on k-means.
- **Style patients**: 33 patients available (ORION-CRC33). For each archetype k, select the patient (excluding CRC-07) whose UNI distribution best covers centroid_k — different archetypes may use different style patients.
- **N=1024**: Sufficient for ≤5–6 swept parameters with Latin hypercube sampling. Fix biologically constrained parameters; sweep only uncertain ones. Confirm parameter dimensionality before finalizing N.
- **Simulation model**: ARCADE (Bagheri lab) preferred — outputs (cell type/state masks, oxygen/glucose maps, vasculature) map directly to PixCell channel groups with no conversion. PhysiCell is an alternative but requires an output translation step. Final choice TBD.
