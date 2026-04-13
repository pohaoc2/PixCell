# PixCell-ControlNet: Simulation-to-Experiment TME Mapping

A multi-channel ControlNet for mapping tumor microenvironment (TME) simulation outputs to experimental H&E histology, using paired ORION-CRC multiplexed imaging data.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pohaoc2/PixCell/blob/main/notebook/multichannel_controlnet.ipynb)
[![Open Stage 3 Paired Ablation Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pohaoc2/PixCell/blob/main/notebook/stage3_paired_ablation_a100_colab.ipynb)
[![CI](https://github.com/pohaoc2/PixCell/actions/workflows/ci.yml/badge.svg)](https://github.com/pohaoc2/PixCell/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pohaoc2/PixCell/branch/main/graph/badge.svg)](https://codecov.io/gh/pohaoc2/PixCell)

---

## Overview

This project fine-tunes [PixCell-256](https://huggingface.co/StonyBrook-CVLab/PixCell-256) with a ControlNet that accepts multi-channel spatial TME maps (cell type, cell state, vasculature, oxygen, glucose) and generates realistic H&E patches that match the simulated spatial layout.

**Training data:** Paired H&E + CODEX multiplexed protein imaging from the [ORION-CRC dataset](https://github.com/labsyspharm/orion-crc/tree/main).

**Two inference modes:**
- **Style-conditioned**: reference H&E UNI embedding + TME channels
- **TME-only**: null UNI embedding (CFG dropout trained at 15%)

**Key papers:**
- PixCell: [arXiv:2506.05127](https://arxiv.org/abs/2506.05127)
- ORION-CRC: [labsyspharm/orion-crc](https://github.com/labsyspharm/orion-crc/tree/main)

---

## Contents

- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Stage Guides](#stage-guides)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Installation

Python >= 3.9, PyTorch >= 2.0.1, CUDA 11.7+

```bash
bash setup.sh
```

Dependency note:

- Primary dependency file is `requirements.txt`.
- `requirements_nommcv.txt` is now a compatibility alias to `requirements.txt`.
- `mmcv==1.7.0` is installed separately (not via requirements) because it needs:
  `pip install mmcv==1.7.0 --no-build-isolation`

---

## Pipeline Overview

The pipeline is divided into four sequential stages:

```text
Stage 0: Model Setup          stage0_setup.py
         ↓ pretrained weights ready
Stage 1: Feature Extraction   stage1_extract_features.py
         ↓ UNI embeddings + VAE latents cached
Stage 2: Training             stage2_train.py
         ↓ ControlNet + TME module checkpoint
Stage 3: Inference            stage3_inference.py
         → experimental-like H&E from simulation channels
```

**Training** uses paired experimental data (ORION-CRC H&E + CODEX multichannel).  
**Inference** accepts unpaired simulation channels and generates realistic H&E.

---

## Stage Guides

The stage-specific runbooks now live in separate files:

| Guide | Covers |
|------|--------|
| [`stage1.md`](stage1.md) | Stage 0 setup, pretrained weights, feature extraction, experimental data layout |
| [`stage2.md`](stage2.md) | ControlNet + TME training, configs, checkpoints, TensorBoard monitoring |
| [`stage3.md`](stage3.md) | Inference, validation, ablations, metrics, visualization and reporting tools |

Additional workflow references:

- [`ablation_cli.md`](ablation_cli.md): full paired + unpaired ablation workflow reference

---

## Citation

If you use this work, please cite:

```bibtex
@article{yellapragada2025pixcell,
  title={PixCell: A generative foundation model for digital histopathology images},
  author={Yellapragada, Srikar and Graikos, Alexandros and Li, Zilinghan and Triaridis, Kostas
          and Belagali, Varun and Kapse, Saarthak and Nandi, Tarak Nath and Madduri, Ravi K
          and Prasanna, Prateek and Kurc, Tahsin and others},
  journal={arXiv preprint arXiv:2506.05127},
  year={2025}
}
```

---

## Acknowledgements

Built on [PixArt-Sigma](https://github.com/PixArt-alpha/PixArt-sigma), [HuggingFace Diffusers](https://github.com/huggingface/diffusers), and the [ORION-CRC](https://github.com/labsyspharm/orion-crc/tree/main) dataset.
