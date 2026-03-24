# Glaucoma Detection & Structural Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

> **Final Year B.E. Project** — An end-to-end glaucoma screening system combining classical Machine Learning, Deep Learning classification, U-Net segmentation, Grad-CAM explainability, and rigorous statistical evaluation.

---

## 🚀 Project Overview

Glaucoma is the leading cause of irreversible blindness worldwide, typically diagnosed by clinically assessing deformations in the optic disc and optic cup as quantified by the Cup-to-Disc Ratio (CDR). Early, automated detection is critical for scalable screening in resource-limited settings.

This project provides a **fully reproducible research pipeline** spanning data exploration, classical and deep feature extraction, CNN classification, deep segmentation, explainability heatmaps, and statistically sound evaluation — all designed to be compared in a publication-quality format.

### Supported Datasets

| Dataset | Size | Task |
|---|---|---|
| **ACRIMA** | 705 fundus images | Classification |
| **DRISHTI-GS1** | 101 images + GT masks | Segmentation / CDR |
| **RIM-ONE DL** | 313 images | Classification |
| **EyePACS-AIROGS-light-v2** | Large-scale | Classification |

---

## ✅ Current Achievements

### Phase 1 — Data Exploration & Visualisation
- Loaded and validated all four datasets through a unified data loader.
- Used **proxy segmentation** (color-channel thresholding) to generate initial Cup-to-Disc Ratio estimates and compared them against DRISHTI-GS1 ground truth annotations.
- Generated rich exploratory visualisations: class distribution, color feature correlation maps, CDR proxy vs. GT scatter plots, and annotated fundus sample grids.

---

### Phase 2 — Feature Engineering & Classical ML Baseline

Extracted clinically motivated handcrafted features:
- **Texture:** Local Binary Patterns (LBP)
- **Structure:** CDR proxy, morphological descriptors
- **Colour / Intensity:** Channel statistics, PCA projections

Evaluated on ACRIMA + RIM-ONE DL combined split (70:15:15):

| Model | AUC | Sensitivity | Specificity | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.7126 | 0.670 | 0.649 | 0.652 | 0.659 |
| **SVM (RBF)** | **0.7888** | **0.756** | **0.682** | **0.718** | **0.717** |
| Random Forest | 0.7674 | 0.714 | 0.689 | 0.695 | 0.701 |

> Best classical baseline: **SVM (RBF)** with `AUC = 0.7888`.

---

### Phase 3 — Deep Learning Classification (CNN)

Trained a **ResNet-18** transfer-learning pipeline with:
- Two-stage fine-tuning (frozen backbone → full unfreeze)
- Mixed-precision training (`torch.amp`) for ~30% speedup on RTX GPUs
- Early stopping on validation AUC
- TensorBoard-logged loss, AUC, and learning rate curves

| Model | AUC | Sensitivity | Specificity | F1 | Accuracy |
|---|---|---|---|---|---|
| **CNN (ResNet-18)** | **0.9445** | **0.932** | **0.816** | **0.874** | **0.871** |

> Deep learning outperforms the best classical baseline by **+15.6% AUC**.

---

### Phase 4 — Deep Segmentation (U-Net)

Replaced proxy CDR heuristics with a **dedicated U-Net** trained on DRISHTI-GS1 ground truth masks for pixel-accurate optic disc and cup delineation.

| Target | Dice | IoU |
|---|---|---|
| **Optic Disc** | **0.9677** | **0.9378** |
| **Optic Cup** | **0.8794** | **0.7931** |

- CDR values derived from U-Net segmentations now correlate closely with expert annotations.
- Segmentation training curves and sample result overlays saved to `outputs/figures/`.

---

### Phase 5 — Explainability & Trust (Grad-CAM)

Applied **Gradient-weighted Class Activation Mapping** to the trained ResNet-18 to validate that predictions are anatomically grounded:
- Heatmaps confirm model attention is localized to the **optic nerve head** region — not background or image artifacts.
- Computed per-image **focus scores** to quantitatively verify saliency concentration (`outputs/results/gradcam_focus_scores.csv`).
- Publication figure generated: `outputs/figures/gradcam_publication_figure.png`.

---

### Phase 6 — Rigorous Statistical Evaluation

`evaluation/final_eval.py` implements paper-grade statistical analysis:
- **Bootstrap 95% Confidence Intervals** (2000 resamples) for all metrics
- **DeLong's Test** for pairwise AUC comparison between models
- **McNemar's Test** for classifier agreement analysis
- **Cross-dataset performance breakdown** (per-cohort breakdown table)
- **Publication-ready results table** with CI-annotated cells

---

### Project Notebooks

| Notebook | Content |
|---|---|
| `01_data_exploration.ipynb` | Dataset loading, CDR proxy, visualisations |
| `02_feature_engineering.ipynb` | LBP, PCA, colour features |
| `03_classical_ml.ipynb` | SVM, RF, LR benchmarks |
| `04_cnn_training.ipynb` | ResNet-18 two-stage fine-tuning |
| `05_explainability.ipynb` | Grad-CAM heatmaps & focus score analysis |
| `06_segmentation.ipynb` | U-Net training, CDR from masks, Dice/IoU |

---

## 🔮 Proposed Enhancements (Future Work)

The following enhancements are proposed to extend the system into a more complete, publication-ready and clinically deployable product:

### 🧠 Model Architecture Upgrades
- **Vision Transformers (ViT / Swin-T):** Replace ResNet-18 with self-attention based models that capture long-range dependencies across the optic disc boundary — expected to particularly improve **specificity**.
- **EfficientNet / ConvNeXt:** Lightweight yet high-performing CNN backbones for resource-constrained clinical hardware.
- **Multi-task Learning Head:** Jointly optimise the classification and CDR regression objectives on a single shared encoder, reducing the two-stage pipeline into one unified model.
- **Ensemble & Stacking:** Fuse probabilities from CNN and classical ML into a meta-classifier for robustness on edge cases.

### 🌐 Cross-Dataset Generalisation
- **Domain Adaptation:** Implement adversarial domain adaptation or histogram normalisation to mitigate camera-hardware bias introduced when training on one dataset (e.g., RIM-ONE DL) and testing on another (e.g., ACRIMA).
- **Leave-One-Dataset-Out Evaluation:** Rigorous protocol to measure how well the model generalises across unseen acquisition environments — critical for real-world clinical deployment.

### ⚖️ Class Imbalance & Hard Cases
- **Focal Loss:** Down-weight easy negatives to focus gradients on difficult borderline positives.
- **Hard-Negative Mining:** Explicitly identify and oversample images where the model is most confused, improving calibration in the critical high-sensitivity operating region.

### 📊 Enhanced Explainability
- **SHAP Values:** Feature attribution for the tabular classical ML models to rank which handcrafted features (CDR, LBP, colour) matter most clinically.
- **Integrated Gradients / LIME:** Model-agnostic post-hoc explainers for the CNN, complementing Grad-CAM.
- **Patient-level Report Generation:** Auto-generate a structured PDF explainability report per patient combining the fundus image, Grad-CAM overlay, predicted CDR, confidence score, and recommendation.

### 🚀 Deployment & MLOps
- **FastAPI + Streamlit / Gradio Web App:** A drag-and-drop web interface for clinicians — upload a fundus image, instantly receive classification, CDR, Grad-CAM overlay, and confidence score.
- **ONNX Export:** Convert the trained PyTorch model to ONNX format for framework-agnostic, cross-platform inference (edge devices, IoT cameras).
- **MLflow / Weights & Biases:** Migrate from CSV logging to full experiment tracking with parameter logging, metric visualisation and model registry.
- **Docker Containerisation:** Package the inference API + model into a reproducible Docker image for clinic or cloud deployment.

### 🩺 Clinical Integration
- **Longitudinal Tracking:** Store and compare CDR measurements across multiple patient visits to identify progressive structural deterioration — significant clinically even if a single-visit screen is negative.
- **Uncertainty Quantification:** Monte Carlo Dropout or Deep Ensembles to produce calibrated prediction confidence — flagging images where the model is uncertain for expert review rather than making binary decisions.

---

## 🛠 Setup & Installation

The project uses `uv` for fast, reproducible dependency management.

```bash
# 1. Install uv
pip install uv

# 2. Create virtual environment
uv venv

# 3. Install all dependencies (CUDA-enabled PyTorch included)
uv sync

# 4. Configure paths
copy .env.example .env
# Edit .env: set GLAUCOMA_DATA_DIR to your local datasets root folder
```

> **CUDA Note:** `pyproject.toml` is pre-configured with PyTorch CUDA 12.4 (`cu124`). If your driver version differs, update the index URL accordingly. Run `nvidia-smi` to verify your CUDA version.

---

## 📁 Project Structure

```
glaucoma-detection-project/
├── config.py                  # Central config — paths, hyperparameters, device
├── main.py
├── data/
│   └── dataset_loader.py      # Unified loader for all 4 datasets
├── features/
│   └── feature_extractor.py   # Handcrafted feature extraction (LBP, CDR, colour)
├── models/
│   ├── classical_ml.py        # SVM, RF, LR training & serialisation
│   ├── cnn_model.py           # GlaucomaResNet definition & optimizer builder
│   ├── trainer.py             # Two-stage CNN Trainer with AUC-based early stopping
│   ├── unet.py                # U-Net architecture for optic disc/cup segmentation
│   └── seg_trainer.py         # Segmentation training loop (Dice loss)
├── explainability/
│   └── gradcam.py             # Grad-CAM heatmap generation & focus scoring
├── evaluation/
│   └── final_eval.py          # Bootstrap CI, DeLong's test, McNemar's test
├── notebooks/                 # Step-by-step experimental notebooks (01–06)
├── outputs/
│   ├── figures/               # All plots, heatmaps, segmentation overlays
│   ├── results/               # Metric CSVs, ROC data, feature cache
│   └── logs/                  # TensorBoard event files
├── pyproject.toml             # uv dependency specification
└── .env.example               # Environment variable template
├── datasets/
│   ├── glaucoma        
│   │   ├── ACRIMA          
│   │   ├── RIM-ONE DL            
│   │   ├── EyePACS-AIROGS          
│   │   └── RIM-ONE_DL         
```
