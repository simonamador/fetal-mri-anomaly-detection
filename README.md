# Fetal MRI Anomaly Detection
 
**Conditional deep generative normative modeling for structural and developmental anomaly detection in the fetal brain**
 
[![NeuroImage](https://img.shields.io/badge/NeuroImage-2025-blue)](https://doi.org/10.1016/j.neuroimage.2025.121442)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neuroimage.2025.121442-green)](https://doi.org/10.1016/j.neuroimage.2025.121442)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[Carlos Simon Amador Izaguirre](https://github.com/simonamador) •
[Sungmin You](https://github.com/VictorSungminYou) •
[Guillermo Tafoya Milo](https://github.com/GuillermoTafoya) •
[NeuroIm Lab](https://labs.childrenshospital.org/neuroim)
 
*Fetal Neonatal Neuroimaging and Developmental Science Center, Boston Children's Hospital / Harvard Medical School*
 
---
 
## Overview
 
Fetal brain anomalies - including ventriculomegaly, polymicrogyria, lissencephaly, macrocephaly, and microcephaly - are difficult to detect reliably through visual inspection of MRI alone, due to rapid morphological changes across gestational ages and high intra-/inter-observer variability.
 
This repository contains the **research and development code** for a two-stage unsupervised anomaly detection framework applied to fetal brain MRI:
 
1. **GA-VAE** - a gestational-age-conditioned variational autoencoder that learns normative fetal brain appearance and produces a pixel-wise anomaly map from the reconstruction error.
2. **In-Painting Refinement** - an AOT-GAN-based inpainting stage that masks detected anomalous regions and refines the normative reconstruction, yielding a cleaner final anomaly map.
The framework requires **no anomalous labels during training** - it learns entirely from typically developing (TD) fetuses and can generalize to a wide range of unspecified pathologies.
 
> **Note on relationship to the paper:** This repo contains the iterative development framework (**GA-VAE + AOT-GANN pipeline**, shown in `/assets/new_framework.png`) built as part of the broader research effort. The published model - CCVAEGAN - extends these ideas with cyclic consistency training and a multi-task discriminator. The official paper implementation is maintained at: [VictorSungminYou/Fetal-Brain-AnomalyDetection](https://github.com/VictorSungminYou/Fetal-Brain-AnomalyDetection).
 
---
 
## Publication
 
**You S., Gondova A., Amador Izaguirre C.S., Tafoya Milo G., Jeong S., Lee H.J., Tarui T., Rollins C.K., Yun H.J., Grant P.E., Im K.**
*Conditional deep generative normative modeling for structural and developmental anomaly detection in the fetal brain.*
**NeuroImage**, 319, 121442 (2025).
[https://doi.org/10.1016/j.neuroimage.2025.121442](https://doi.org/10.1016/j.neuroimage.2025.121442)
 
```bibtex
@article{you2025fetal,
  title     = {Conditional deep generative normative modeling for structural and developmental anomaly detection in the fetal brain},
  author    = {You, Sungmin and Gondova, Andrea and Amador Izaguirre, Carlos Simon and Tafoya Milo, Guillermo and Jeong, Seungyoon and Lee, Han-Jui and Tarui, Tomo and Rollins, Caitlin K. and Yun, Hyuk Jin and Grant, P. Ellen and Im, Kiho},
  journal   = {NeuroImage},
  volume    = {319},
  pages     = {121442},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/j.neuroimage.2025.121442}
}
```
 
---
 
## Key Results (Published CCVAEGAN)
 
### Mixed-site cohort - reconstruction quality (TD test set, n=85)
 
| Model | MAE ↓ | MSE ↓ | SSIM ↑ | MS-SSIM ↑ |
|---|---|---|---|---|
| VAEGAN | 0.117 ± 0.012 | 0.025 ± 0.005 | 0.515 ± 0.060 | 0.647 ± 0.050 |
| CVAEGAN | 0.103 ± 0.011 | 0.020 ± 0.004 | 0.575 ± 0.064 | 0.690 ± 0.050 |
| Cycle-VAEGAN | 0.083 ± 0.014 | 0.015 ± 0.005 | 0.704 ± 0.096 | 0.772 ± 0.066 |
| **CCVAEGAN** | **0.057 ± 0.012** | **0.006 ± 0.003** | **0.814 ± 0.071** | **0.856 ± 0.052** |
 
All differences vs. CCVAEGAN significant (paired t-test, p < 0.001).
 
### Mixed-site cohort - anomaly detection AUROC (TD n=85 vs. anomalies n=146)
 
| Anomaly score | VAEGAN | CVAEGAN | Cycle-VAEGAN | **CCVAEGAN** |
|---|---|---|---|---|
| MAE | 0.900 | 0.933 | 0.898 | **0.997** |
| MSE | 0.914 | 0.934 | 0.898 | **0.996** |
| 1 − SSIM | 0.803 | 0.958 | 0.863 | **0.995** |
| 1 − MS-SSIM | 0.855 | 0.953 | 0.760 | **0.996** |
 
### Per-pathology AUROC (CCVAEGAN, mixed-site)
 
| Condition | AUROC |
|---|---|
| Polymicrogyria | 1.000 |
| Macrocephaly | 1.000 |
| Microcephaly | 1.000 |
| Lissencephaly | 1.000 |
| Ventriculomegaly | 0.990 |
 
---
 
## Framework Architecture
 
![Pipeline diagram](/assets/new_framework.png)
 
The pipeline consists of two stages.
 
### Stage 1 - GA-VAE (normative reconstruction)
 
A gestational-age-conditioned variational autoencoder trained exclusively on typically developing fetal brain MRI.
 
```
Inputs: MRI slice (158x158) + Gestational Age scalar
        |
        v
+----------------------------------+
|  Encoder                         |
|  4x [ Conv2D 3x3, stride 2       |
|        LeakyReLU, BatchNorm ]    |
+---------------+------------------+
                |
                v  z ~ N(mu, e^sigma)  <-- GA conditioning
                |
+---------------+------------------+
|  Decoder                         |
|  4x [ TransConv 3x3, stride 2    |
|        LeakyReLU, BatchNorm ]    |
+---------------+------------------+
                |
         MRI Reconstruction (x_hat)
                |
                v
      Anomaly score (per pixel):
      anomaly = |eq(x_hat) - eq(x_bar)| * loss_per(x_bar, x_hat)
```
 
The gestational age is injected at the bottleneck, conditioning the decoder to generate a normative reference appropriate for the fetus's developmental stage. The anomaly map is derived from a combination of equivariant feature differences and a perceptual reconstruction loss.
 
### Stage 2 - In-Painting Refinement (AOT-GANN)
 
Anomalous regions identified in Stage 1 are masked and inpainted by an AOT-GAN, producing a refined normative reconstruction and a cleaner final anomaly map.
 
```
Anomaly map (Stage 1)
        |
        v
  Mask generation:
  mask = binarize(
           gaussfilter(
             binarize(anomaly, p95)
           ), 0.1)
        |
        v
  AOT-GANN (inpainting)
  ---------------------
  2x [ Conv, LeakyReLU ]
        |
        v
  Refined reconstruction --> Final anomaly map
```
 
The mask is computed by binarizing the initial anomaly map at the 95th percentile, applying Gaussian smoothing, and re-binarizing at 0.1 - isolating the most anomalous regions while suppressing noise. AOT-GAN then fills these regions using surrounding normative tissue context.
 
---
 
## Dataset
 
Data from four sites with IRB approval at Boston Children's Hospital:
 
| Site | Scanner | Subjects |
|---|---|---|
| Boston Children's Hospital (BCH) | Siemens 3T Skyra | TD + anomalies |
| Taipei Veterans General Hospital (TVGH) | GE 1.5T | TD + anomalies |
| Tufts Medical Center (TMC) | Philips 1.5T | TD + anomalies |
| dHCP (public) | Philips 3T | TD only |
 
**Anomaly types:** ventriculomegaly (n=111), polymicrogyria (n=15), macrocephaly (n=11), microcephaly (n=6), lissencephaly (n=3).
 
**GA range:** 19.1–38.7 weeks. Input: central 30 axial slices per subject, 158×158 px, min-max normalized.
 
> Clinical data is not publicly available due to institutional policy. Access may be requested via the corresponding author (kiho.im@childrens.harvard.edu). The [dHCP dataset](https://biomedia.github.io/dHCP-release-notes/) is publicly available.
 
---
 
## Installation
 
```bash
git clone https://github.com/simonamador/Anomaly-Detection.git
cd Anomaly-Detection
pip install -r requirements.txt
```
 
**Requirements:**
 
```
torch>=1.12.0
torchvision>=0.13.0
numpy
scipy
scikit-image
matplotlib
nibabel
```
 
Tested on Python 3.8, PyTorch 1.12, CUDA 11.6, Nvidia RTX A6000.
 
---
 
## Usage
 
### Preprocessing
 
MRI volumes should be motion-corrected 3D reconstructions (0.86 mm isotropic, aligned to standard anatomical planes). Extract the central 30 axial slices and apply min-max normalization per slice.
 
### Training
 
```bash
python train_framework.py \
  --data_dir /path/to/td_train_slices \
  --epochs 2000 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --latent_dim 512 \
  --output_dir ./checkpoints
```
 
### Evaluation
 
```bash
python validation.py \
  --data_dir /path/to/td_test_slices \
  --checkpoint ./checkpoints/best.pth
```
 
### Inference
 
```bash
python main.py \
  --data_dir /path/to/clinical_slices \
  --checkpoint ./checkpoints/best.pth \
  --ga_value 27.04 \
  --output_dir ./anomaly_maps
```
 
---
 
## Repository Structure
 
```
Anomaly-Detection/
├── assets/              # Architecture diagram (new_framework.png) and figures
├── data/                # Data loading and preprocessing utilities
├── models/              # GA-VAE and AOT-GANN model definitions
├── utils/               # Loss functions, metrics, visualization
├── old_framework/       # Earlier VAE experiments preceding the GA-VAE pipeline
├── main.py              # Inference and anomaly map generation
├── train_framework.py   # Training loop
├── validation.py        # Reconstruction quality evaluation
└── requirements.txt
```
 
---
 
## Limitations
 
- Requires spatial alignment of input slices to standard anatomical planes - not always available directly from clinical single-shot sequences.
- Generation quality decreases at older GAs, reflecting higher morphological complexity of later-stage cortical folding.
- Anomaly scores are mildly sensitive to image quality (noise, blur), though this does not affect TD vs. anomaly discrimination in practice.
- Generalizability is bounded by the imaging protocols seen during training; out-of-distribution protocols may require domain adaptation.
---
 
## Funding
 
Supported by the National Institute of Neurological Disorders and Stroke (R01NS114087), National Institute of Biomedical Imaging and Bioengineering (R01EB031170, R01EB032708), and Eunice Kennedy Shriver National Institute of Child Health and Human Development (R01HD100009).
 
---
 
## License
 
MIT License. See [LICENSE](LICENSE).
