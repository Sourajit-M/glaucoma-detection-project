# Glaucoma Detection & Structural Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg)](https://pytorch.org/)

An end-to-end Machine Learning and Deep Learning system designed to identify glaucoma from retinal fundus images and conduct structural optic disc/cup analysis. This project rigorously compares classical feature-engineering techniques against modern Convolutional Neural Network architectures alongside robust Model Explainability metrics.

---

## 🚀 Project Overview

Glaucoma is a top cause of irreversible blindness, primarily diagnosed by assessing the physiological deformations in the optic disc and optic cup. This project provides a fully reproducible pipeline from data exploration and feature extraction to training baseline models, advanced deep learning architectures, and visual explainability mapping.

**Supported Datasets:**
- **ACRIMA** (705 classification images)
- **DRISHTI-GS1** (101 images, segmentation masks + Cup-to-Disc Ratios)
- **RIM-ONE DL** (313 classification images)
- **EyePACS-AIROGS-light-v2**

---

## ✅ Current Achievements

As of the current phase, the primary experimental pipelines across baseline modeling, deep vision classification, and anatomical segmentation are complete.

### 1. Data Exploration & Visualisation
- **Structural Assessment:** Initiated with proxy segmentation methods to isolate Optic Cup/Disc regions and approximate the diagnostic Cup-to-Disc Ratio (CDR).
- **Comprehensive Visualisations:** Documented class distributions, color feature correlations, and structural variances (e.g., `outputs/figures/sample_images.png`).

### 2. Feature Engineering & Classical ML
- Extracted domain-specific numerical features (e.g., Local Binary Patterns for textures, PCA component visualizations, explicit color mappings).
- Built and evaluated robust baseline models over these crafted features:
  - **Support Vector Machine (RBF):** `AUC = 0.7888`
  - **Random Forest:** `AUC = 0.7674`

### 3. Deep Learning Classification Pipeline
- Developed an end-to-end vision pipeline leveraging Transfer Learning via **ResNet18**.
- Outperformed classical ML benchmarks significantly, demonstrating deep representation superiority.
  - **Best CNN Model Performance:**
    - **AUC:** `0.9445`
    - **Sensitivity:** `0.9320`
    - **Accuracy:** `0.8714`

### 4. Deep Learning Segmentation (U-Net)
- Developed and trained a **U-Net** architecture specifically for the precise anatomical tracing of the Optic Disc and Cup (eliminating the need for proxy thresholding).
- Correlated exact Cup-to-Disc Ratios (CDR) reliably against Ground Truth annotations.
- **Top Segmentation Performance (DRISHTI-GS1):**
  - **Optic Disc:** `Dice = 0.967`, `IoU = 0.937`
  - **Optic Cup:** `Dice = 0.879`, `IoU = 0.793`

### 5. Explainability & Trust (XAI)
- **Grad-CAM Integration:** Generated visual explainer heatmaps using Gradient-weighted Class Activation Mapping (`outputs/figures/gradcam_publication_figure.png`).
- Successfully proved and validated that the deep residual architecture focuses accurately on anatomical pathology (e.g., localized optic nerve head deformations) rather than extraneous dataset artifacts.

All analytical processes are tracked sequentially in the `notebooks/` directory:
- `01_data_exploration.ipynb`
- `02_feature_engineering.ipynb`
- `03_classical_ml.ipynb`
- `04_cnn_training.ipynb`
- `05_explainability.ipynb`
- `06_segmentation.ipynb`

---

## 🔮 Further Enhancements

While highly accurate pipelines have been established, enhancements into robust clinical application include:

### Advanced Modeling & Architecture
- **More Sophisticated ViTs:** Upgrade from `ResNet18` to architectures such as `EfficientNet`, `ConvNeXt`, or Vision Transformers (`ViT`) specifically tracking higher specificity.
- **Ensemble Techniques:** Fuse classical texture/morphological features with deep CNN embeddings in a meta-classifier setup to improve statistical robustness.
- **SHAP values:** Quantify feature value attribution for the tabular baseline models.

### Real-World Robustness
- **Cross-Dataset Generalisation:** Implement multi-domain evaluations (e.g., train on `RIM-ONE DL`, test strictly on `ACRIMA`) to ensure the CNN doesn't overfit to an isolated camera hardware domain.
- **Hard-Mining & Class Imbalance:** Implement focal loss or hard-negative mining to improve generalization across challenging border cases.

### Deployment & Tooling
- **Web Interface / REST API:** Package the classification and segmentation models in a `FastAPI` + `Streamlit/Gradio` application, enabling drag-and-drop clinical analysis.
- **MLOps / Experiment Tracking:** Migrate tracking outputs from native saving to robust automated workflows such as MLflow or Weights & Biases (W&B).

---

## 🛠 Setup & Installation

The project isolates dependencies securely using `uv` (a modern Python packaging tool) mimicking structural standards.

1. **Install uv** (if not installed):
   ```bash
   pip install uv
   ```
2. **Setup virtual environment:**
   ```bash
   uv venv
   ```
3. **Install Dependencies:**
   ```bash
   uv sync
   ```
4. **Environment Setup:**
   Duplicate `.env.example` to `.env` and fill `GLAUCOMA_DATA_DIR` referencing your absolute system path containing the root dataset folders.
