# GlaucomaDetect

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-9%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

> An interpretable hybrid deep learning system for glaucoma detection from retinal
> fundus images — combining ResNet18 transfer learning, U-Net structural segmentation,
> Grad-CAM explainability, and a production-ready REST API with a clinical dashboard.

**AUC 0.947 · Sensitivity 0.930 · Disc Dice 0.968 · Cup Dice 0.879**

---

## Overview

Glaucoma is the second leading cause of irreversible blindness worldwide. Detection
relies on assessing the optic disc and cup in retinal fundus photographs — a task that
is specialist-dependent, time-consuming, and inaccessible at scale.

This project delivers a **fully reproducible end-to-end pipeline** from raw fundus
images to a deployed REST API, covering:

- Classical ML baselines (LR, SVM-RBF, Random Forest) with 45-dimensional handcrafted features
- Deep learning classification (ResNet18, EfficientNet-B0) with two-stage transfer learning
- U-Net segmentation for optic disc and cup — enabling accurate CDR computation
- Grad-CAM explainability with quantitative disc focus scoring
- Ablation study identifying CNN + CDR as the optimal fusion strategy
- Statistical significance testing (DeLong's AUC test, McNemar's test, bootstrap CI)
- Production FastAPI backend serving all models as ONNX with sub-5s CPU inference
- React clinical dashboard with monochromatic design, Grad-CAM viewer, and metrics dashboard

---

## Results

### Classification — held-out test set (n = 1,050)

| Model | AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.713 | 0.670 | 0.649 | 0.652 |
| Random Forest | 0.767 | 0.714 | 0.689 | 0.695 |
| SVM-RBF | 0.789 | 0.766 | 0.682 | 0.718 |
| EfficientNet-B0 | 0.929 | 0.904 | 0.785 | 0.854 |
| CNN ResNet18 | 0.945 | 0.932 | 0.816 | 0.874 |
| **CNN + CDR (proposed)** | **0.947** | **0.930** | **0.802** | **0.865** |

CNN significantly outperforms SVM-RBF: Z = 10.11, p < 0.0001 (DeLong's test);
χ² = 98.94, p < 0.0001 (McNemar's test).

### Ablation study

| Fusion variant | AUC |
|---|---|
| CNN only | 0.9445 |
| CNN + SVM | 0.9282 |
| CNN + SVM + CDR | 0.9290 |
| **CNN + CDR** | **0.9468** |

CDR from U-Net adds complementary structural information (+0.002 AUC).
SVM features are subsumed by CNN learned representations and degrade performance.

### Segmentation — DRISHTI-GS1 (n = 50)

| Structure | Dice | IoU |
|---|---|---|
| Optic disc | 0.9677 | 0.9378 |
| Optic cup | 0.8794 | 0.7931 |

---

## Datasets

| Dataset | Images | Task |
|---|---|---|
| [ACRIMA](https://figshare.com/s/c2d31f850af14c5b5232) | 705 | Classification |
| [RIM-ONE DL](https://github.com/miag-ull/rim-one-dl) | 485 | Classification |
| [EyePACS-AIROGS-light-v2](https://airogs.grand-challenge.org/) | 3,540 | Classification |
| [DRISHTI-GS1](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) | 50 | Segmentation |

---

## Project Structure

```
glaucoma-detection-project/
├── config.py                     ← Central config: paths, hyperparameters, device
├── export_onnx.py                ← Export PyTorch models to ONNX for serving
│
├── data/
│   └── dataset_loader.py         ← Unified loader for all 4 datasets
├── features/
│   └── feature_extractor.py      ← Handcrafted features: LBP, CDR proxy, colour
├── models/
│   ├── classical_ml.py           ← SVM, RF, LR with GridSearchCV
│   ├── cnn_model.py              ← GlaucomaResNet + GlaucomaEfficientNet
│   ├── trainer.py                ← Two-stage CNN trainer (mixed precision, AUC early stopping)
│   ├── unet.py                   ← U-Net with DiceBCE loss + CDR computation
│   ├── seg_trainer.py            ← Segmentation training loop
│   └── ensemble.py               ← HybridEnsemble: CNN + CDR late fusion
├── explainability/
│   └── gradcam.py                ← Grad-CAM + focus score
├── evaluation/
│   └── final_eval.py             ← Bootstrap CI, DeLong's test, McNemar's test
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_ml.ipynb
│   ├── 04_cnn_training.ipynb
│   ├── 05_explainability.ipynb
│   ├── 05b_efficientnet.ipynb
│   ├── 06_segmentation.ipynb
│   ├── 07_final_evaluation.ipynb
│   └── 08_ensemble.ipynb
│
├── outputs/
│   ├── models/                   ← Saved .pth and .onnx model files
│   ├── figures/                  ← All publication figures (300 DPI)
│   └── results/                  ← Metric CSVs, ROC data, feature cache
│
├── webapp/
│   ├── docker-compose.yml
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py           ← FastAPI app + lifespan startup
│   │   │   ├── core/             ← Config, model registry
│   │   │   ├── routers/          ← /health, /predict, /results/metrics
│   │   │   ├── schemas/          ← Pydantic request/response models
│   │   │   ├── services/         ← CNN inference, Grad-CAM, segmentation
│   │   │   └── utils/            ← Image validation, preprocessing, encoding
│   │   ├── data/metrics.json     ← Pre-computed research metrics
│   │   ├── models/               ← .onnx model files (gitignored)
│   │   ├── tests/                ← pytest suite (9 tests)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── frontend/
│       ├── src/
│       │   ├── pages/            ← Predict, Dashboard, About
│       │   ├── components/       ← Navbar, UploadZone, ResultCard, HeatmapViewer
│       │   ├── hooks/            ← usePrediction (TanStack Query)
│       │   ├── lib/              ← Axios API client
│       │   └── types/            ← TypeScript interfaces
│       ├── Dockerfile
│       ├── nginx.conf
│       └── vite.config.ts
│
├── pyproject.toml                ← uv dependency spec (CUDA PyTorch)
└── .env.example
```

---

## Setup

### Prerequisites

- Python 3.11+, [uv](https://docs.astral.sh/uv/), Node.js 20+, pnpm
- NVIDIA GPU with CUDA 12.4 recommended for training (inference runs on CPU)

### Research pipeline

```bash
# Install dependencies via uv (includes CUDA PyTorch)
uv venv && uv sync

# Configure data paths
copy .env.example .env
# Set GLAUCOMA_DATA_DIR to your datasets root folder

# Run notebooks in order
uv run jupyter notebook
```

### Backend (development)

```bash
cd webapp/backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### Frontend (development)

```bash
cd webapp/frontend
pnpm install
pnpm dev
# App: http://localhost:5173
```

### Docker (full stack)

```bash
# First export models to ONNX and copy to webapp/backend/models/
uv run python export_onnx.py

cd webapp
docker compose up --build
# App:  http://localhost:5173
# API:  http://localhost:8000/docs
```

### Tests

```bash
cd webapp/backend
pytest tests/ -v
# 9 passed in ~0.4s
```

---

## API Reference

### `POST /predict`

Upload a retinal fundus image for full analysis.

**Request:** `multipart/form-data` with field `file` (.jpg / .png / .bmp, max 10 MB)

**Response:**
```json
{
  "prediction":         "glaucoma",
  "probability":        0.8734,
  "confidence":         "high",
  "cdr":                0.712,
  "cdr_risk":           "elevated",
  "processing_time_ms": 1240,
  "images": {
    "original":             "<base64_png>",
    "heatmap_overlay":      "<base64_png>",
    "disc_mask":            "<base64_png>",
    "cup_mask":             "<base64_png>",
    "segmentation_overlay": "<base64_png>"
  },
  "clinical_note": "This result is generated by a research model..."
}
```

### `GET /health`
Returns model load status. Used by Docker health check.

### `GET /results/metrics`
Returns pre-computed evaluation metrics for all models.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Training | Python 3.11, PyTorch 2.3, CUDA 12.4 |
| Classical ML | scikit-learn, NumPy, OpenCV |
| Segmentation | segmentation-models-pytorch |
| Serving | FastAPI, ONNX Runtime, Uvicorn |
| Validation | Pydantic v2 |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Charts | Recharts |
| HTTP | TanStack Query, Axios |
| Container | Docker, nginx |
| Package mgr | uv (Python), pnpm (Node) |

---

## Deployment

### Railway (recommended)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

cd webapp
railway up
```

Set environment variables in Railway dashboard:
- Backend service: contents of `backend/.env`
- Frontend service: `VITE_API_URL=https://your-backend.railway.app`

### Render

Create two services from `webapp/`:
- **Backend** — Docker, root `./backend`, port 8000
- **Frontend** — Docker, root `./frontend`, port 80

Set `VITE_API_URL` in the frontend service environment to the backend URL.

---

## Limitations

- U-Net trained on 50 DRISHTI images — larger segmentation datasets would improve CDR accuracy
- Binary classification only — severity grading (early/moderate/advanced) not yet implemented
- No prospective clinical validation — research use only
- Cross-dataset distribution shift not explicitly modelled

---

## Future Work

- Vision Transformer backbone (ViT-B/16) for improved long-range disc boundary modelling
- Multi-task learning: joint classification and CDR regression on a shared encoder
- Domain adaptation for cross-device generalisation
- Longitudinal CDR tracking across patient visits
- Monte Carlo Dropout for calibrated uncertainty quantification
- GitHub Actions CI/CD — auto-run pytest on every push

---

## Citation

If you use this work, please cite:

```
Majumder S., Bairagi K., Biswas E., Dutta R., Jha P.K. (2025).
An Interpretable Hybrid Deep Learning System for Glaucoma Detection
Using Retinal Fundus Images with Structural Analysis and Grad-CAM Explainability.
Narula Institute of Technology.
```

---

## Acknowledgements

Datasets: ACRIMA, RIM-ONE DL, EyePACS-AIROGS, DRISHTI-GS1 teams for
making their data publicly available for research.

---

> **Disclaimer:** This system is a research tool and is not intended for clinical
> diagnosis. Always consult a qualified ophthalmologist for medical decisions.