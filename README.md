# Glaucoma Detection & Structural Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

> **Final Year B.E. Project** — An end-to-end glaucoma screening system combining classical Machine Learning, Deep Learning classification, U-Net segmentation, Grad-CAM explainability, rigorous statistical evaluation, and a production-ready REST API backend.

---

## 🚀 Project Overview

Glaucoma is the leading cause of irreversible blindness worldwide, typically diagnosed by clinically assessing deformations in the optic disc and optic cup as quantified by the Cup-to-Disc Ratio (CDR). Early, automated detection is critical for scalable screening in resource-limited settings.

This project provides a **fully reproducible research pipeline** spanning data exploration, classical and deep feature extraction, CNN classification, deep segmentation, explainability heatmaps, statistically sound evaluation — and a deployable **FastAPI backend** serving real-time inference via ONNX-exported models.

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

### Phase 7 — Production REST API (FastAPI + ONNX) ✨ New

A complete **FastAPI backend** (`webapp/backend/`) serving real-time glaucoma screening through a single REST endpoint. The pipeline is:

```
Upload fundus image → CNN (ONNX) inference → Grad-CAM overlay
                   → U-Net disc/cup segmentation → CDR computation → JSON response
```

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model load status & readiness check |
| `POST` | `/predict` | Full inference: classification + Grad-CAM + CDR |
| `GET` | `/results/metrics` | Pre-computed evaluation metrics for all models |

**`POST /predict` response includes:**
- `prediction` — `"glaucoma"` or `"normal"`
- `probability` — model confidence score (0–1)
- `confidence` — `"high"` / `"medium"` / `"low"` tier
- `cdr` — Cup-to-Disc Ratio from U-Net segmentation
- `cdr_risk` — `"elevated"` / `"borderline"` / `"normal"`
- `processing_time_ms` — end-to-end latency in milliseconds
- `images` — base64-encoded: original, Grad-CAM overlay, disc mask, cup mask, segmentation overlay
- `clinical_note` — mandatory disclaimer for non-clinical use

**Technical stack:**
- **FastAPI** with async lifespan model loading
- **ONNX Runtime** for framework-agnostic, fast CPU/GPU inference
- **Pydantic v2** schemas for strict request/response validation
- **CORS** middleware configured for frontend integration
- **Docker** support via `Dockerfile`

**Input validation guards:**
- File extension whitelist (`.png`, `.jpg`, `.jpeg`, `.bmp`)
- 10 MB file size limit
- Minimum image resolution check (100×100 px)
- Corrupted / non-decodable image detection

**Test suite** (`pytest`) — all passing ✅:

| Test | Description |
|---|---|
| `test_predict_valid_image` | 200 response on valid PNG |
| `test_predict_response_schema` | All fields present with correct types |
| `test_predict_invalid_extension` | 400 on `.pdf` upload |
| `test_predict_too_large` | 400 on > 10 MB file |
| `test_predict_corrupted_image` | 400 on non-decodable bytes |
| `test_predict_tiny_image` | 400 on image < 100×100 |
| `test_metrics_endpoint` | `/results/metrics` returns all 6 models + ablation + segmentation |

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

### 🖥️ Frontend & Full-Stack Completion
- **React / Next.js Frontend:** Complete the `webapp/frontend/` with a clinical-grade drag-and-drop UI — upload a fundus image and receive an interactive dashboard showing the Grad-CAM overlay, segmentation masks, CDR gauge, and confidence breakdown.
- **Real-time CDR Trend Chart:** Visualise longitudinal CDR measurements across multiple sessions per patient.

### 🩺 Clinical Integration
- **Longitudinal Tracking:** Store and compare CDR measurements across multiple patient visits to identify progressive structural deterioration — significant clinically even if a single-visit screen is negative.
- **Uncertainty Quantification:** Monte Carlo Dropout or Deep Ensembles to produce calibrated prediction confidence — flagging images where the model is uncertain for expert review rather than making binary decisions.

### 🚀 MLOps
- **MLflow / Weights & Biases:** Migrate from CSV logging to full experiment tracking with parameter logging, metric visualisation and model registry.
- **CI/CD Pipeline:** GitHub Actions workflow to auto-run the pytest suite on every push and validate ONNX model loading.

---

## 🛠 Setup & Installation

### Research Pipeline (model training / notebooks)

```bash
# 1. Install uv
pip install uv

# 2. Create virtual environment & install dependencies (CUDA PyTorch included)
uv venv && uv sync

# 3. Configure data path
copy .env.example .env
# Edit .env: set GLAUCOMA_DATA_DIR to your local datasets root folder
```

> **CUDA Note:** `pyproject.toml` is pre-configured with PyTorch CUDA 12.4 (`cu124`). Run `nvidia-smi` to verify your driver version.

### FastAPI Backend

```bash
cd webapp/backend

# Install backend dependencies
pip install -r requirements.txt

# Copy and configure environment
copy .env.example .env

# Run the API server
uvicorn app.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### Running Tests

```bash
cd webapp/backend
uv run pytest tests/ -v
```

### Docker

```bash
cd webapp/backend
docker build -t glaucoma-api .
docker run -p 8000:8000 glaucoma-api
```

---

## 📁 Project Structure

```
glaucoma-detection-project/
├── config.py                        # Central config — paths, hyperparameters, device
├── main.py
├── data/
│   └── dataset_loader.py            # Unified loader for all 4 datasets
├── features/
│   └── feature_extractor.py         # Handcrafted features (LBP, CDR, colour)
├── models/
│   ├── classical_ml.py              # SVM, RF, LR training & serialisation
│   ├── cnn_model.py                 # GlaucomaResNet definition & optimizer builder
│   ├── trainer.py                   # Two-stage CNN Trainer (AUC early stopping)
│   ├── unet.py                      # U-Net architecture for disc/cup segmentation
│   └── seg_trainer.py               # Segmentation training loop (Dice loss)
├── explainability/
│   └── gradcam.py                   # Grad-CAM heatmap generation & focus scoring
├── evaluation/
│   └── final_eval.py                # Bootstrap CI, DeLong's test, McNemar's test
├── notebooks/                       # Step-by-step experimental notebooks (01–06)
├── outputs/
│   ├── figures/                     # Plots, heatmaps, segmentation overlays
│   ├── results/                     # Metric CSVs, ROC data, feature cache
│   └── logs/                        # TensorBoard event files
├── webapp/
│   ├── backend/                     # FastAPI inference server
│   │   ├── app/
│   │   │   ├── main.py              # FastAPI app + lifespan model loading
│   │   │   ├── core/
│   │   │   │   ├── config.py        # Pydantic settings
│   │   │   │   └── model_loader.py  # ONNX model registry
│   │   │   ├── routers/
│   │   │   │   ├── health.py        # GET /health
│   │   │   │   ├── predict.py       # POST /predict
│   │   │   │   └── results.py       # GET /results/metrics
│   │   │   ├── schemas/             # Pydantic request/response models
│   │   │   ├── services/            # CNN inference, Grad-CAM, segmentation
│   │   │   └── utils/               # Image I/O, validation, encoding
│   │   ├── tests/                   # pytest suite (7 tests, all passing)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── frontend/                    # (in progress)
├── datasets/
│   └── glaucoma/
│       ├── ACRIMA/
│       ├── DRISHTI-GS1/
│       ├── RIM-ONE_DL/
│       └── EyePACS-AIROGS/
├── pyproject.toml                   # uv dependency specification
└── .env.example                     # Environment variable template
```
