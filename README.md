# MultiRater-Dirichlet: Evidential Deep Learning of Multi-Radiologist Diagnostic Disagreement in Pulmonary Nodule CT

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Official PyTorch implementation of the paper:  
> **"Beyond Consensus Illusion: Evidential Dirichlet Deep Learning of Multi-Radiologist Diagnostic Disagreement in Pulmonary Nodule CT"**

---

## 📌 Overview

Conventional medical artificial intelligence treats experienced radiologists as noisy labelers, compressing subjective ratings via majority voting or arithmetic rounding. Under this "consensus illusion", critical gray-zone boundaries and biological discordance are erased, compelling deterministic deep neural networks to produce hazardous overconfident predictions on high-stakes conflict lesions.

**MultiRater-Dirichlet** confronts this foundational paradigm:
1. **Second-Order Dirichlet Evidential Modeling**: Directly parameterizes conjugate Dirichlet priors over the 5-simplex, jointly capturing aleatoric visual noise and epistemic reader discordance.
2. **Masked Variance Alignment ($\\mathcal{L}_{var}$)**: Introduces variance alignment loss strictly masked with $\\mathbb{I}(R_i \\ge 2)$, preventing unobserved single-reader nodules ($R=1$) from being penalized as zero-disagreement negatives.
3. **Dual-Track Deployment**:
   - **Track 1 (Autonomous CT-Only)**: Streams raw 3D CT volume voxels through a 3D volumetric convolutional network for fully autonomous initial screening.
   - **Track 2 (Multimodal Assisted)**: Deeply fuses 3D volumetric embeddings with 17-dimensional radiological morphological priors (lobulation, spiculation, calcification, texture variance) for specialized collaborative decision support.
4. **Human-AI Collaborative Triage**: Leverages Decision Curve Analysis (DCA) to demonstrate superior Standardized Net Triage Benefit across clinical referral thresholds ($p_t \\in [0.10, 0.50]$) by deferring the top 20% most contentious nodules to multidisciplinary tumor board review.

<p align="center">
  <img src="figures/fig1_study_framework.png?raw=true" width="95%" alt="Study Framework"/>
</p>

---

## 🔬 Benchmark Results

All evaluations are conducted on a strictly patient-disjoint independent test set of 202 subjects ($N=507$ nodules, including 360 multi-reader nodules with $R_i \ge 2$) from the LIDC-IDRI multicenter cohort.

### Table 1: Multi-Rater & SOTA Benchmark (Fixed Test Set Point Estimates, $N=507$)
*Corresponds to Table 2 in manuscript. Single point estimates evaluated across the fixed independent test cohort.*

| Category | Model / Method | Rating MSE ↓ | Pearson $r$ ↑ | ECE Calibration ↓ | Wasserstein EMD ↓ | Conflict AUROC ↑ | High-Disagree AUROC ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic** | Det-CE (Conventional Baseline) | 0.2636 | 0.8360 | 0.1395 | 0.5004 | 0.5564 | 0.4820 |
| | Det-Reg (Scalar MSE Regression) | 0.2626 | 0.8377 | — | — | 0.5000 | 0.5000 |
| **Probabilistic** | Heteroscedastic Gaussian | 0.2461 | 0.8483 | 0.0454 | 0.5195 | 0.7392 | 0.8045 |
| | MultiRater-Ensemble (4-Head) | 0.2420 | 0.8555 | 0.4096 | 0.5134 | 0.5230 | 0.4822 |
| **2024–2026 SOTA** | LDL-Net (*MedIA 2024*) | 0.2512 | 0.8447 | **0.0109** | 0.4945 | 0.7709 | 0.7861 |
| | DP-CrowdNet (*CVPR 2024*) | **0.2351** | **0.8555** | 0.4027 | 0.5067 | 0.5371 | 0.4877 |
| | A3-Net (*CVPR 2025*) | 0.2590 | 0.8394 | 0.1485 | 0.4975 | 0.7413 | 0.7261 |
| | UCE-Net (*2025*) | 0.2515 | 0.8443 | 0.1201 | **0.4788** | 0.5481 | 0.5280 |
| | KAN-adapted Baseline (*2024/2026*) | 0.2634 | 0.8371 | 0.0501 | 0.4963 | 0.7049 | 0.7309 |
| **Proposed** | **MultiRater-Dirichlet (Ours)** | 0.2609 | 0.8439 | 0.0912 | 0.5017 | **0.7861** | **0.8440** |

### Table 2: 3D Volumetric vs. Multimodal Architecture Ablation (3D Volumetric Test Subset, $N=33$)
*Corresponds to Table 3 in manuscript. Evaluated on the dedicated patient-disjoint 3D volumetric test subset ($N=33$ nodules from subjects with extracted $32^3$ HU voxel cubes, across 161 total crops) to isolate the representational impact of pure 3D voxel modeling vs. 17-D morphological prior fusion.*

| Architecture | Deployment Track | Rating MSE ↓ | Pearson $r$ ↑ | Conflict AUROC ↑ | Note / Key Clinical Insight |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **3D-Det-CE** | Track 1 (3D CT-Only) | 1.2073 | 0.0407 | 0.5556 | Deterministic voxel baseline without disagreement awareness |
| **3D-Hetero-Gauss** | Track 1 (3D CT-Only) | 2.2653 | 0.0042 | 0.4444 | Unstable optimization under sample-constrained 3D voxel regime |
| **3D-Dirichlet** | Track 1 (3D CT-Only) | 1.0125 | 0.1354 | 0.5463 | Pure voxel modeling struggles near chance without morphological grounding |
| **Multimodal-Dirichlet** | Track 2 (3D + Morph) | **0.5324** | **0.7638** | **0.7731** | **+0.2268 AUROC boost** via fusion of 17-D morphometric priors |

> **Key Clinical Insight**: Predicting fine-grained inter-radiologist disagreement directly from raw 3D CT voxel cubes without explicit morphological descriptors is inherently sample-constrained and close to chance level (Conflict AUROC 0.5463). Fusing 3D volumetric embeddings with 17-dimensional radiological morphological priors (spiculation, lobulation, margin sharpness, internal texture) grounds visual representations in established diagnostic signs, boosting Conflict AUROC to 0.7731.

### Table 3: 1,000 Patient-Clustered Paired Bootstrap Statistical Significance
*Corresponds to Table 4 in manuscript. Values represent mean and 95% confidence intervals derived from 1,000 cluster-aware bootstrap resamplings ($N=202$ subjects).*

| Model / Metric | Conflict AUROC [95% CI] | High-Disagree AUROC [95% CI] | ECE Calibration [95% CI] |
| :--- | :---: | :---: | :---: |
| **Det-CE** | 0.5554 [0.4942, 0.6126] | 0.4812 [0.4215, 0.5420] | 0.1248 [0.0827, 0.1678] |
| **Hetero-Gauss** | 0.7392 [0.6815, 0.7924] | 0.8045 [0.7580, 0.8490] | **0.0454 [0.0225, 0.0712]** |
| **MultiRater-Dirichlet (Ours)** | **0.7865 [0.7335, 0.8368]** | **0.8439 [0.8041, 0.8814]** | 0.0937 [0.0565, 0.1346] |
| **Paired Gain $\Delta$** | **+0.2311 [0.1562, 0.3058]** | **+0.3627 [0.3036, 0.4244]** | — |

*Note on Cohort Partitions & Model Configurations:
- **Primary Benchmark (Table 1 & Table 3)**: Evaluated on the complete independent test set ($N=507$ nodules from 202 subjects) utilizing standardized 17-dimensional radiological morphometric features. On this cohort, the proposed MultiRater-Dirichlet model attains a single-run point estimate Conflict AUROC of 0.7861 (Table 1 above / Table 2 in paper) and an empirical bootstrap mean of 0.7865 with 95% CI [0.7335, 0.8368] across 1,000 patient-clustered paired resamplings (Table 3 above / Table 4 in paper).
- **3D Volumetric Ablation (Table 2)**: Evaluated on the dedicated patient-disjoint 3D volumetric test subset ($N=33$ nodules with extracted $32^3$ HU voxel cubes, across 161 total crops) to interrogate whether pure 3D CNNs can infer disagreement end-to-end from voxel cubes alone (0.5463) versus with multimodal prior fusion (0.7731).*

*Note on Feature Ablations (Table 6 in paper): Feature ablations are conducted under a standardized, un-scheduled baseline training protocol to maintain strictly controlled relative comparison across feature drops, establishing an un-tuned baseline anchor of 0.7450 for the Full 17 Features configuration.*

### ⚙️ Baseline Architecture Adaptation & Controlled Benchmarking Protocol

A critical methodological requirement when evaluating multi-reader disagreement formulations is to isolate their loss objectives and uncertainty mechanisms from confounding variations in 3D deep backbone capacity (e.g., parameter scale, 3D kernel depth, or volumetric pretraining).

- **Standardized Radiological Input Vector**: All multi-rater comparative architectures in Table 1 (LDL-Net, DP-CrowdNet, A3-Net, UCE-Net, KAN, and baselines) are evaluated on the standardized 17-dimensional radiological morphometric feature space of LIDC-IDRI (`subtlety`, `calcification`, `sphericity`, `margin`, `lobulation`, `spiculation`, `texture`, `internalStructure` means and standard deviations, plus nodule diameter and volume).
- **Faithful Implementation of Published Loss Mechanics**:
  - **LDL-Net (*MedIA 2024*)**: Implements label distribution learning on the probability 5-simplex via Kullback-Leibler divergence minimization against empirical radiologist rating distributions.
  - **DP-CrowdNet (*CVPR 2024*)**: Implements diversified personalized multi-annotator prediction using 4 rater-specific embedding heads with permutation-invariant crowd loss.
  - **A3-Net (*CVPR 2025*)**: Implements annotation ambiguity estimation with adaptive dynamic loss attenuation.
  - **UCE-Net (*2025*)**: Implements dual-stream peer networks with symmetric discrepancy co-estimation.
  - **KAN-adapted Baseline (*2024/2026*)**: Implements Kolmogorov-Arnold Networks parameterized by learnable B-spline activation functions.
- **Independent 3D Volumetric Ablation (Table 2)**: The separate 3D evaluation (Table 2 above / Table 3 in paper) evaluates end-to-end 3D CNN voxel modeling (`Track 1: 3D CT-Only`) versus multimodal fusion with morphometric priors (`Track 2: 3D + Morph`), demonstrating that pure voxel cubes struggle to learn disagreement without morphological guidance (Conflict AUROC 0.5463), whereas multimodal fusion achieves 0.7731.

---

## 📁 Repository Structure

```bash
MultiRater-Dirichlet/
├── configs/
│   └── default.yaml                  # Experiment & training configuration
├── data/
│   ├── patient_splits.json           # Patient-disjoint 70/10/20 split definitions
│   ├── nodules_metadata.csv          # Comprehensive nodule metadata, ratings & conflicts
│   └── sample_crops/                 # Sample 3D nodule volumes (.npz) for immediate testing
├── figures/                          # Publication-grade figures (PNG/PDF)
│   ├── fig1_study_framework.png
│   ├── fig2_disagreement_distribution.png
│   ├── fig3_dca_net_benefit.png
│   ├── fig4_feature_ablation.png
│   └── fig5_qualitative_case_studies.png
├── scripts/                          # Self-contained reproducible execution scripts
│   ├── 01_audit_and_split_cohort.py  # LIDC-IDRI XML audit and patient-disjoint partitioning
│   ├── 02_train_multirater_baselines.py # Classical & 2024-2026 SOTA benchmarking
│   ├── 03_train_3d_multimodal.py     # Track 1 & Track 2 3D volumetric learning
│   ├── 04_evaluate_calibration_dca.py # ECE, DCA Net Benefit & 1000 Bootstrap CIs
│   ├── 05_ablation_and_subgroups.py  # Feature ablations & clinical subgroup breakdown
│   └── 06_generate_figures.py        # Publication figure rendering
├── src/
│   ├── models/                       # Model architectures
│   │   ├── dirichlet_net.py          # Evidential Dirichlet, Digamma loss & masked var loss
│   │   ├── backbones.py              # 3D CNN & Fast-KAN spline layers
│   │   └── baselines.py              # Baselines (Det-CE, Hetero-Gauss, A3-Net, UCE-Net, KAN)
│   ├── data/
│   │   └── lidc_dataset.py           # Datasets, 3D volume loaders & tabular encoders
│   ├── metrics/
│   │   ├── evaluation.py             # ECE, EMD, conflict AUROC & paired bootstrap
│   │   └── dca.py                    # Decision Curve Analysis & collaborative triage
│   └── utils/
│       └── plotting.py               # Radar charts, paired bar charts & DCA visualization
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/Join-xiaobai/MultiRater-Dirichlet.git
cd MultiRater-Dirichlet

conda create -n multirater python=3.10 -y
conda activate multirater
pip install -r requirements.txt
```

### 2. Immediate Reproduction (No 100 GB DICOM Download Required)

We provide pre-computed nodule metadata (`data/nodules_metadata.csv`), patient-disjoint split indices (`data/patient_splits.json`), and sample 3D nodule crops (`data/sample_crops/`), allowing you to reproduce all core benchmarks immediately:

```bash
# 1. Run 2024-2026 SOTA and classical baseline benchmarks
python scripts/02_train_multirater_baselines.py

# 2. Evaluate Calibration (ECE) and Decision Curve Analysis (DCA) with 1,000 Bootstrap CIs
python scripts/04_evaluate_calibration_dca.py

# 3. Run feature group ablations and clinical subgroup stratification
python scripts/05_ablation_and_subgroups.py

# 4. Re-generate publication figures
python scripts/06_generate_figures.py
```

### 3. Full End-to-End Pipeline from Raw TCIA DICOM

To extract from the full LIDC-IDRI cohort (1,018 scans across 1,010 patients):
1. Download LIDC-IDRI DICOM images from the [Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).
2. Configure `pylidc` configuration file `~/.pylidcrc` pointing to the DICOM directory.
3. Run cohort extraction and patient-disjoint partitioning:
```bash
python scripts/01_audit_and_split_cohort.py --data_dir /path/to/LIDC-IDRI
```
4. Train the 3D CT volumetric network:
```bash
python scripts/03_train_3d_multimodal.py --crops_dir /path/to/extracted_crops
```

---

## 🩺 Clinical Collaboration & Decision Curve Analysis

The framework introduces an uncertainty-gated human-AI triage strategy:
- **Autonomous Clearance**: For the 80% confident predictions, AI independently recommends annual routine screening (low risk) or expedited clinical referral (high risk).
- **MDT Deferral**: For the top 20% most contentious nodules (characterized by high epistemic uncertainty $U_i$), the system abstains and defers to multidisciplinary tumor board review.

<p align="center">
  <img src="figures/fig3_dca_net_benefit.png?raw=true" width="60%" alt="Decision Curve Analysis"/>
</p>

Decision Curve Analysis demonstrates that this collaborative strategy delivers superior Standardized Net Triage Benefit across all operational clinical referral thresholds ($p_t \in [0.10, 0.50]$) compared to "Refer All", "Refer None", or autonomous deterministic AI.

---

## 📖 Citation

If you find this work or codebase helpful in your research, please reference:
> **Beyond Consensus Illusion: Evidential Dirichlet Deep Learning of Multi-Radiologist Diagnostic Disagreement in Pulmonary Nodule CT**

---

## 📄 License

This repository is released under the [MIT License](LICENSE).
