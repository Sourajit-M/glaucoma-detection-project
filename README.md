# Glaucoma Detection & Structural Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg)](https://pytorch.org/)

An end-to-end Machine Learning and Deep Learning system designed to identify glaucoma from retinal fundus images and conduct structural optic disc/cup analysis. This project rigorously compares classical feature-engineering techniques against modern Convolutional Neural Network architectures.

---

## 🚀 Project Overview

Glaucoma is a top cause of irreversible blindness, primarily diagnosed by assessing the physiological deformations in the optic disc and optic cup. This project provides a fully reproducible pipeline from data exploration and feature extraction to training baseline models and advanced, state-of-the-art deep learning architectures.

**Supported Datasets:**
- **ACRIMA** (705 classification images)
- **DRISHTI-GS1** (101 images, segmentation masks + Cup-to-Disc Ratios)
- **RIM-ONE DL** (313 classification images)
- **EyePACS-AIROGS-light-v2**

---

## ✅ Current Achievements

As of the current phase, the primary experimental pipeline is complete, yielding strong empirical results comparing shallow and deep learning models. 

### 1. Data Exploration & Visualisation
- **Structural Assessment:** Utilised proxy segmentation methods to isolate Optic Cup/Disc regions and approximate the diagnostic Cup-to-Disc Ratio (CDR).
- **Comprehensive Visualisations:** Documented class distributions, color feature correlations, and structural variances (e.g., `outputs/figures/sample_images.png`, `outputs/figures/cdr_proxy_vs_gt.png`).

### 2. Feature Engineering & Classical ML
- Extracted domain-specific numerical features (e.g., Local Binary Patterns for textures, PCA component visualizations, explicit color mappings).
- Built and evaluated robust baseline models over these crafted features:
  - **Support Vector Machine (RBF):** `AUC = 0.7888`
  - **Random Forest:** `AUC = 0.7674`

### 3. Deep Learning Training Pipeline
- Developed an end-to-end vision pipeline leveraging Transfer Learning via **ResNet18**.
- Outperformed classical ML benchmarks significantly, demonstrating deep representation superiority.
  - **Best CNN Model Performance:**
    - **AUC:** `0.9445`
    - **Sensitivity:** `0.9320`
    - **Accuracy:** `0.8714`
- Validated with exhaustive tracking (ROC curves, confusion matrices, and test metric logging).

All analytical processes are tracked sequentially in the `notebooks/` directory:
- `01_data_exploration.ipynb`
- `02_feature_engineering.ipynb`
- `03_classical_ml.ipynb`
- `04_cnn_training.ipynb`

---

## 🔮 Further Enhancements

While we have achieved highly accurate binary classification, the following expansions are on the roadmap to make this pipeline a fully clinical-grade tool:

### Advanced Modeling & Architecture
- **More Sophisticated CNNs & ViTs:** Upgrade from `ResNet18` to more advanced architectures such as `EfficientNet`, `ConvNeXt`, or Vision Transformers (`ViT`) for improved accuracy, specifically focusing on raising specificity.
- **Deep Segmentation Pipelines:** Implement a dedicated deep learning segmentation architecture (e.g., `U-Net`, `MAnet`) fine-tuned specifically for precise anatomical tracing of the Optic Disc and Cup, completely replacing proxy heuristics.
- **Ensemble Techniques:** Fuse classical texture/morphological features with deep CNN embeddings in a meta-classifier setup to improve robustness.

### Explainability & Trust (XAI)
- **Grad-CAM Integration:** Introduce visual explainers using Gradient-weighted Class Activation Mapping to highlight regions the model relies on (ensuring it targets anatomical damage rather than dataset artifacts).
- **SHAP values:** Quantify feature value attribution for tabular baseline models.

### Real-World Robustness
- **Cross-Dataset Generalisation:** Implement multi-domain evaluations (e.g., train on `RIM-ONE DL`, test strictly on `ACRIMA`) to ensure the model doesn't overfit to a single camera/hardware artifact domain.
- **Hard-Mining & Class Imbalance:** Implement focal loss or hard-negative mining to improve generalization across challenging border cases.

### Deployment & Tooling
- **Web Interface / REST API:** Package the best-performing models in a `FastAPI` + `Streamlit/Gradio` application, allowing drag-and-drop inference for end users/clinicians.
- **MLOps / Experiment Tracking:** Migrate outputs from rudimentary CSV logging to robust automated trackers such as MLflow or Weights & Biases (W&B).

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
