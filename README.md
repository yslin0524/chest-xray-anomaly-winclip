# FourierCLIP — Zero-Shot Chest X-Ray Anomaly Detection with Frequency-Aware CLIP

> **Individual contribution extracted from a group course project — EECS 545 Machine Learning, University of Michigan (2025).**
> Full project team: Puzhu Wang, Haina Jiang, Yi-Chen Wang, Yi-Sheng Lin.
> This repository contains the components I personally designed and implemented.

---

## Overview

Chest X-ray anomaly detection is a clinically critical but label-scarce problem. Collecting large labeled medical datasets is expensive, making zero-shot and few-shot approaches particularly attractive for real-world deployment.

This project extends **WinCLIP** — a sliding-window CLIP inference framework originally designed for industrial defect detection — to the medical imaging domain. I integrated a **Frequency Encoder** that transforms input images via 2D FFT and extracts frequency-domain features through a lightweight CNN, fusing them with CLIP's spatial embeddings to improve sensitivity to fine-grained pathological textures (e.g., pulmonary fibrosis, ground-glass opacities). I also introduced a **multi-prompt mean scoring strategy** that replaces WinCLIP's original single averaged text vector with the full set of 611 disease-specific prompt embeddings, significantly improving anomaly discrimination.

---

## My Contribution

This repository was extracted from a group project. The following components are my individual work:

| Component | File | Description |
|-----------|------|-------------|
| Frequency Encoder | `FrequencyEncoder.py` | 3-layer CNN over FFT real+imaginary maps |
| FFT-fused WinCLIP model | `WinCLIP/model.py` | `encode_image` FFT fusion, bug-fixed anomaly scoring |
| Medical text prompts | `WinCLIP/ad_prompts.py` | 8 normal + 47 abnormal state prompts x 13 templates |
| Multi-prompt mean scoring | `WinCLIP/model.py` | `WinClipAD_NoFFT_MultiPrompt` subclass |
| Evaluation pipeline | `eval_WinCLIP.py`, `run_winclip.py` | End-to-end inference and metric computation |
| Experiment notebook | `WinCLIP.ipynb` | All 8 model variants, AUROC + macro F1 comparison |

---

## System / Method Overview

```
Input Image (240x240 RGB)
        |
        +---------------------------+
        |                          |
   CLIP ViT Encoder           FrequencyEncoder
   (spatial features)          +- torch.fft.fft2
        |                      +- concat real + imag -> (B, 6, H, W)
        |                      +- Conv2d x 3 + BN + ReLU
        |                      +- AdaptiveAvgPool2d(1)
        |                      +- Linear -> (B, 640)
        |                          |
        +---- element-wise add + L2 normalize ----+
                                                  |
                                          Fused Feature (B, 640)
                                                  |
                         +------------------------+
                         |                        |
               Text Feature Gallery         Visual Gallery
               (611 abnormal prompts         (k-shot normal
                + 104 normal prompts)         image features)
                         |                        |
               Textual Anomaly Score    Visual Anomaly Score
                         |                        |
                         +---- Harmonic Mean ------+
                                      |
                              Anomaly Map (15x15)
                                      |
                         Upsample -> (400x400)
                                      |
                            Top-50 Mean -> Image Score
                                      |
                                   AUROC
```

**Key design choices:**
- **FFT fusion:** Real and imaginary parts of 2D FFT are concatenated as a 6-channel map and processed by a 3-layer CNN. The resulting 640-dim frequency vector is added element-wise to CLIP's spatial feature and re-normalized — a lightweight, parameter-efficient fusion.
- **Multi-prompt mean scoring:** Instead of collapsing 611 abnormal prompt embeddings into one averaged vector (which dilutes disease-specific signals), each window patch is scored against all 611 vectors and the mean similarity is taken. This preserves the diversity of disease descriptions.
- **Top-50 mean aggregation:** The final image-level score uses the mean of the top-50 anomaly map values, more stable than max pooling while still focusing on the most anomalous regions.
- **Macro F1 evaluation:** Due to severe class imbalance in the dataset (86% abnormal), standard binary F1 is uninformative. All F1 scores reported here use `average='macro'` to equally weight normal and abnormal classes.

---

## Key Features

- **Zero-shot inference** — no training required; uses CLIP's pretrained vision-language alignment directly on chest X-rays
- **Frequency-domain augmentation** — FFT encoder captures high-frequency texture patterns (fibrosis, ground-glass) that CLIP's spatial branch alone misses
- **611 medical text prompts** — covers 47 distinct chest pathologies across 13 radiological templates (e.g., "a frontal chest x-ray of a patient with pleural effusion")
- **Multi-prompt mean strategy** — replaces WinCLIP's single averaged abnormal text vector with mean similarity across all individual prompt vectors, improving AUROC by +0.067
- **Backbone ablation** — experiments across ViT-B-32 and ViT-B-16-plus-240, with and without FFT fusion

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA T4 via Google Colab)

```
torch>=2.0
torchvision
open_clip_torch
scikit-learn
pandas
numpy
tqdm
Pillow
opencv-python
```

---

## Installation

```bash
git clone https://github.com/yslin0524/chest-xray-anomaly-winclip.git
cd chest-xray-anomaly-winclip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch scikit-learn pandas numpy tqdm Pillow opencv-python
```

---

## Usage

### Running the Colab notebook

Open `WinCLIP.ipynb` in Google Colab. The notebook is self-contained and runs all 8 model variants sequentially. Set the paths at the top of the setup cell:

```python
IMAGE_FOLDER = "/content/images/chest-xrays-indiana-university/images/valid_uid_images"
CSV_PATH     = "/content/images/.../final_clean_dataset.csv"
PROJ_PATH    = "/content/images/.../indiana_projections.csv"
```

Then run all cells in order. Results are printed at the end of each model cell, and a summary table is printed by the final evaluation cell.

### Using FrequencyEncoder standalone

```python
from FrequencyEncoder import FrequencyEncoder
import torch

encoder = FrequencyEncoder(in_channels=3, out_dim=640).to("cuda")
x = torch.randn(4, 3, 240, 240).to("cuda")
freq_features = encoder(x)   # (4, 640)
```

---

## Results

All results are on the **Indiana University Chest X-ray test set** (1,621 samples: 1,389 abnormal / 232 normal), using the same split as the PaDiM baseline for fair comparison. F1 scores use `average='macro'`.

### Main comparison

| Model | AUROC | Macro F1 |
|-------|-------|----------|
| PaDiM (baseline) | 0.619 | 0.4805 |
| Zero-shot CLIP (ViT-B-32) | 0.5962 | 0.5394 |
| WinCLIP | 0.5040 | 0.5031 |
| WinCLIP + Multi-Prompt Mean | 0.5715 | 0.5279 |
| CLIP (full project) | 0.6960 | 0.7552 |
| **FourierCLIP (full project)** | **0.7544** | **0.8182** |

### Ablation: FFT encoder effect by backbone

| Model | AUROC | Macro F1 |
|-------|-------|----------|
| Zero-shot CLIP (ViT-B-32) | 0.5962 | 0.5394 |
| Zero-shot CLIP + FFT (ViT-B-32) | 0.6037 | 0.5442 |
| Zero-shot CLIP (ViT-B-16-plus-240) | 0.5973 | 0.5280 |
| Zero-shot CLIP + FFT (ViT-B-16-plus-240) | 0.5807 | 0.5177 |

### Ablation: Multi-prompt mean vs. original WinCLIP

| Model | AUROC | Macro F1 |
|-------|-------|----------|
| WinCLIP (no FFT) | 0.5040 | 0.5031 |
| WinCLIP + FFT | 0.5034 | 0.5052 |
| WinCLIP + Multi-Prompt Mean | 0.5715 | 0.5279 |
| WinCLIP + FFT + Multi-Prompt Mean | 0.5715 | 0.5294 |

**Key findings:**
- Multi-prompt mean improves WinCLIP AUROC by +0.067, confirming that averaging 611 prompts into a single vector loses discriminative information.
- FFT fusion improves ViT-B-32 (+0.0075 AUROC) but slightly hurts ViT-B-16-plus-240 (-0.0166), suggesting the randomly initialized FrequencyEncoder is sensitive to backbone embedding geometry and requires task-specific training to be consistently beneficial.
- WinCLIP's sliding-window mechanism underperforms simple zero-shot CLIP on chest X-rays, likely because chest pathologies are global rather than locally bounded — a fundamental mismatch with WinCLIP's industrial-defect design.

---

## Dataset

**Indiana University Chest X-ray Collection**
- 3,955 radiology reports paired with 7,470 chest X-ray images (frontal + lateral)
- Labels derived from the `impression` field of radiology reports
- Download: [Kaggle — Indiana University Chest X-Rays](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

**Preprocessing steps:**
1. Filter to patients with exactly one frontal and one lateral view
2. Restore anonymized tokens (e.g., `XXXX`) using context-based AI fill-in
3. Derive binary labels: `normal=1` in the report -> label 0; otherwise label 1
4. Final dataset: 3,208 valid patients; test split of 1,621 from `test_padim.csv` (shared with PaDiM baseline)

Place extracted images at:
```
images/chest-xrays-indiana-university/images/valid_uid_images/
```

---

## Citation

```bibtex
@inproceedings{jeong2023winclip,
  title     = {WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation},
  author    = {Jeong, Jongheon and Zou, Yang and Kim, Taewan and Zhang, Dongqing
               and Ravichandran, Avinash and Dabeer, Onkar},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2023}
}

@misc{fouriercliip2025,
  title  = {FourierCLIP: Fourier-Aware Vision-Language Models for
            Chest X-Ray Anomaly Detection},
  author = {Wang, Puzhu and Jiang, Haina and Wang, Yi-Chen and Lin, Yi-Sheng},
  note   = {EECS 545 Machine Learning Final Project,
            University of Michigan, 2025},
  url    = {https://github.com/wpuzhu/EECS545-Final-Project}
}
```

---

## Acknowledgements

This work builds on the [WinCLIP](https://github.com/caoyunkang/WinClip) open-source implementation and uses [OpenCLIP](https://github.com/mlfoundations/open_clip) for backbone loading. Dataset provided by the Indiana University Network of Care.
