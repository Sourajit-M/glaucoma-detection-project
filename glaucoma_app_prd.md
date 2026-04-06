# Product Requirements Document
## GlaucomaDetect — Web Application

**Version:** 1.0  
**Project:** ML-Based Glaucoma Detection & Structural Analysis System  
**Stack:** FastAPI (backend) · React + Vite (frontend)  
**Deployment:** Docker → Railway / Render (free tier)

---

## 1. Product Overview

### 1.1 Purpose
A production-grade web application that exposes the trained glaucoma detection
pipeline as an interactive tool. Users upload a retinal fundus image and receive
a prediction, probability score, Grad-CAM heatmap, CDR value, and segmentation
overlay — all in a single request. A separate dashboard page presents the
research results from the paper.

### 1.2 Target users
- Ophthalmology researchers evaluating the model
- ML engineers reviewing the portfolio project
- Technical recruiters / interviewers
- Conference paper reviewers wanting a live demo

### 1.3 Success metrics
- Inference + response time < 5 seconds on free-tier CPU
- All 3 pages functional and mobile-responsive
- Passes Docker build and runs locally with `docker compose up`
- Zero broken imports, zero 500 errors on valid input

---

## 2. Repository Structure

```
glaucoma-app/
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI app entry point
│   │   ├── routers/
│   │   │   ├── predict.py       ← POST /predict
│   │   │   ├── health.py        ← GET /health
│   │   │   └── results.py       ← GET /results/metrics
│   │   ├── services/
│   │   │   ├── inference.py     ← model loading + prediction
│   │   │   ├── gradcam.py       ← Grad-CAM heatmap generation
│   │   │   └── segmentation.py  ← U-Net CDR computation
│   │   ├── schemas/
│   │   │   └── prediction.py    ← Pydantic request/response models
│   │   ├── core/
│   │   │   ├── config.py        ← env-based settings (pydantic-settings)
│   │   │   └── model_loader.py  ← singleton model registry
│   │   └── utils/
│   │       └── image.py         ← preprocess, validate, encode helpers
│   ├── models/                  ← .onnx model weight files (gitignored)
│   ├── tests/
│   │   ├── test_predict.py
│   │   └── test_health.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Predict.tsx      ← Page 1: upload + results
│   │   │   ├── Dashboard.tsx    ← Page 2: research metrics
│   │   │   └── About.tsx        ← Page 3: how it works
│   │   ├── components/
│   │   │   ├── UploadZone.tsx
│   │   │   ├── ResultCard.tsx
│   │   │   ├── HeatmapViewer.tsx
│   │   │   ├── MetricCard.tsx
│   │   │   ├── RocChart.tsx
│   │   │   └── Navbar.tsx
│   │   ├── hooks/
│   │   │   └── usePrediction.ts ← React Query mutation
│   │   ├── types/
│   │   │   └── api.ts           ← TypeScript interfaces
│   │   ├── lib/
│   │   │   └── api.ts           ← axios client
│   │   └── main.tsx
│   ├── public/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── package.json
│
├── docker-compose.yml
└── README.md
```

---

## 3. Backend PRD

### 3.1 Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Framework | FastAPI 0.111+ | Async, auto OpenAPI docs, Pydantic native |
| ML runtime | ONNX Runtime 1.18 | 3–5× faster than PyTorch on CPU, no GPU needed |
| Image processing | OpenCV + Pillow | Same as training pipeline |
| Validation | Pydantic v2 | Type-safe request/response schemas |
| Config | pydantic-settings | `.env` file → typed Settings object |
| Testing | pytest + httpx | Async-compatible test client |
| Server | Uvicorn | ASGI, production-grade |
| Container | Docker (python:3.11-slim) | Reproducible deployment |

### 3.2 Environment variables (`.env`)

```
# Model paths (relative to /app/models/)
CNN_MODEL_PATH=glaucoma_resnet18.onnx
DISC_MODEL_PATH=disc_unet.onnx
CUP_MODEL_PATH=cup_unet.onnx

# Inference settings
IMAGE_SIZE=224
SEG_IMAGE_SIZE=256
CDR_THRESHOLD=0.65
CONFIDENCE_THRESHOLD=0.5

# Server
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=http://localhost:5173,https://your-frontend.railway.app

# Limits
MAX_FILE_SIZE_MB=10
ALLOWED_EXTENSIONS=jpg,jpeg,png,bmp
```

### 3.3 API Endpoints

---

#### `GET /health`

Returns service status and loaded model names. Used by Docker health check
and frontend to verify backend connectivity.

**Response 200:**
```json
{
  "status": "healthy",
  "models_loaded": ["cnn", "disc_unet", "cup_unet"],
  "version": "1.0.0"
}
```

---

#### `POST /predict`

Core endpoint. Accepts a fundus image, runs the full inference pipeline,
returns prediction + Grad-CAM + CDR + segmentation masks.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | Yes | Fundus image (.jpg/.png/.bmp) |
| `return_heatmap` | `bool` | No (default: true) | Include Grad-CAM overlay |
| `return_masks` | `bool` | No (default: true) | Include segmentation masks |

**Validation rules:**
- File size ≤ `MAX_FILE_SIZE_MB`
- Extension in `ALLOWED_EXTENSIONS`
- Image must be readable by OpenCV (not corrupted)
- Minimum image dimensions: 100×100 px

**Processing pipeline (in order):**
1. Read and validate image bytes
2. Preprocess: CLAHE → circular mask → resize to 224×224 → normalise
3. CNN inference (ONNX): → `cnn_probability`, `prediction`
4. If `return_heatmap`: Grad-CAM on layer4 → heatmap overlay PNG
5. If `return_masks`: U-Net disc + cup inference → CDR computation
6. Encode images as base64 strings
7. Return structured JSON response

**Response 200:**
```json
{
  "prediction": "glaucoma",
  "probability": 0.8734,
  "confidence": "high",
  "cdr": 0.712,
  "cdr_risk": "elevated",
  "processing_time_ms": 1240,
  "images": {
    "original": "<base64_png>",
    "heatmap_overlay": "<base64_png>",
    "disc_mask": "<base64_png>",
    "cup_mask": "<base64_png>",
    "segmentation_overlay": "<base64_png>"
  },
  "clinical_note": "CDR of 0.71 exceeds the 0.65 clinical threshold. This is a screening tool only — not a clinical diagnosis."
}
```

**Confidence levels:**
- `probability >= 0.80` → `"high"`
- `probability >= 0.60` → `"medium"`
- `probability < 0.60` → `"low"`

**CDR risk levels:**
- `cdr >= 0.65` → `"elevated"`
- `cdr >= 0.50` → `"borderline"`
- `cdr < 0.50` → `"normal"`

**Error responses:**

| Code | Condition |
|---|---|
| 400 | Invalid file type, file too large, corrupted image |
| 422 | Pydantic validation error (bad form fields) |
| 500 | Model inference failure (with safe error message) |
| 503 | Models not yet loaded (startup in progress) |

---

#### `GET /results/metrics`

Returns the pre-computed research metrics for the Dashboard page.
Static data — loaded from a JSON file at startup, not computed at request time.

**Response 200:**
```json
{
  "dataset_info": {
    "total_images": 4730,
    "test_set_size": 1050,
    "datasets": ["ACRIMA", "RIM-ONE DL", "EyePACS-AIROGS"]
  },
  "models": [
    {
      "name": "Logistic Regression",
      "type": "classical_ml",
      "auc": 0.713, "sensitivity": 0.670, "specificity": 0.649, "f1": 0.652
    },
    {
      "name": "SVM-RBF",
      "type": "classical_ml",
      "auc": 0.789, "sensitivity": 0.766, "specificity": 0.682, "f1": 0.718
    },
    {
      "name": "Random Forest",
      "type": "classical_ml",
      "auc": 0.767, "sensitivity": 0.714, "specificity": 0.689, "f1": 0.695
    },
    {
      "name": "EfficientNet-B0",
      "type": "deep_learning",
      "auc": 0.929, "sensitivity": 0.904, "specificity": 0.785, "f1": 0.854
    },
    {
      "name": "CNN ResNet18",
      "type": "deep_learning",
      "auc": 0.945, "sensitivity": 0.932, "specificity": 0.816, "f1": 0.874
    },
    {
      "name": "CNN + CDR (proposed)",
      "type": "hybrid",
      "auc": 0.947, "sensitivity": 0.930, "specificity": 0.802, "f1": 0.865
    }
  ],
  "ablation": [
    {"variant": "CNN only", "auc": 0.9445},
    {"variant": "CNN + SVM", "auc": 0.9282},
    {"variant": "CNN + SVM + CDR", "auc": 0.9290},
    {"variant": "CNN + CDR", "auc": 0.9468}
  ],
  "segmentation": [
    {"structure": "Optic disc", "dice": 0.9677, "iou": 0.9378},
    {"structure": "Optic cup",  "dice": 0.8794, "iou": 0.7931}
  ],
  "roc_curves": {
    "fpr": {"CNN + CDR": [...], "SVM-RBF": [...], "CNN ResNet18": [...]},
    "tpr": {"CNN + CDR": [...], "SVM-RBF": [...], "CNN ResNet18": [...]}
  }
}
```

---

### 3.4 Model loading strategy

Models are loaded once at startup into a singleton `ModelRegistry` object
and held in memory. This avoids the 2–3 second ONNX load time on every request.

```
Startup sequence:
  1. Read Settings from .env
  2. Load CNN ONNX model  → registry["cnn"]
  3. Load disc U-Net ONNX → registry["disc_unet"]
  4. Load cup U-Net ONNX  → registry["cup_unet"]
  5. Mark service as ready
  6. /health returns "healthy" only after step 4 completes
```

If any model file is missing, the service starts in degraded mode:
- `/health` returns `"degraded"` with a list of missing models
- `/predict` returns 503 with a clear message

### 3.5 ONNX export requirement

Before deploying, the three PyTorch models must be exported to ONNX:

```python
# CNN ResNet18
torch.onnx.export(model, dummy_input, "glaucoma_resnet18.onnx",
                  input_names=["image"],
                  output_names=["logits"],
                  dynamic_axes={"image": {0: "batch_size"}},
                  opset_version=17)

# U-Net (same pattern for disc and cup)
torch.onnx.export(unet_model, seg_dummy, "disc_unet.onnx",
                  input_names=["image"],
                  output_names=["mask_logits"],
                  dynamic_axes={"image": {0: "batch_size"}},
                  opset_version=17)
```

### 3.6 Non-functional requirements

| Requirement | Target |
|---|---|
| Response time (P95) | < 5s on 2-core free tier CPU |
| Max image size | 10 MB |
| Concurrent requests | Handle 5 simultaneous (Uvicorn default workers) |
| Memory footprint | < 1 GB total (all 3 ONNX models loaded) |
| CORS | Frontend origin whitelisted via `ALLOWED_ORIGINS` env var |
| Error messages | Never expose stack traces or internal paths in responses |
| Clinical disclaimer | Every prediction response includes a disclaimer string |

### 3.7 Testing requirements

| Test | What it covers |
|---|---|
| `test_health_ok` | GET /health returns 200 with correct fields |
| `test_predict_valid_image` | POST /predict with valid PNG returns 200 + correct schema |
| `test_predict_invalid_extension` | .pdf file returns 400 |
| `test_predict_too_large` | 11 MB file returns 400 |
| `test_predict_corrupted_image` | Random bytes returns 400 |
| `test_predict_schema` | Response matches PredictionResponse Pydantic model |
| `test_metrics_endpoint` | GET /results/metrics returns correct model count |

---

## 4. Frontend PRD

### 4.1 Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Framework | React 18 + TypeScript | Type safety, component model |
| Build | Vite 5 | Fast HMR, small bundle |
| Styling | Tailwind CSS v3 | Utility-first, no custom CSS files needed |
| Data fetching | TanStack Query v5 | Caching, loading states, error handling |
| HTTP client | Axios | Interceptors for base URL config |
| Charts | Recharts | React-native, composable ROC curves |
| Routing | React Router v6 | Client-side navigation |
| File upload | react-dropzone | Drag-and-drop with validation |
| Icons | Lucide React | Consistent icon set |

### 4.2 Pages

---

#### Page 1 — Predict (`/`)

The primary page. Layout: two-column on desktop, stacked on mobile.

**Left column — Upload panel:**
- Drag-and-drop zone with a dashed border, camera icon, and "Drop a fundus image or click to browse" label
- Accepted formats badge: JPG · PNG · BMP · max 10 MB
- On file select: image thumbnail preview replaces the drop zone
- "Analyse image" primary button — disabled until a file is selected
- Loading state: spinner + "Analysing…" text + estimated time (3–5s)

**Right column — Results panel** (visible only after successful response):
- Verdict card: large "GLAUCOMA" (red) or "NORMAL" (green) text + probability percentage + confidence badge
- CDR card: numeric CDR value + colour-coded risk level pill (red/amber/green)
- Image viewer tabs: Original | Grad-CAM overlay | Segmentation overlay
  - Tabs switch the displayed image without re-fetching
- Clinical disclaimer banner (amber) at bottom of results

**Error states:**
- Invalid file type: inline error below dropzone
- File too large: inline error below dropzone
- API error (500/503): error card with "Try again" button
- No models loaded (503): "Service starting up, please wait 30 seconds and retry"

---

#### Page 2 — Research dashboard (`/dashboard`)

Static data from `GET /results/metrics`. No user input.

**Section 1 — Summary metric cards (row of 4):**
- Best AUC: 0.947
- Best Sensitivity: 0.932
- Test set size: 1,050
- Datasets used: 3

**Section 2 — Model comparison table:**
- Sortable columns: Model, Type, AUC, Sensitivity, Specificity, F1
- Proposed model row highlighted
- Type badge: Classical ML (gray) / Deep Learning (blue) / Hybrid (teal)

**Section 3 — ROC curve chart:**
- Multi-line Recharts LineChart
- One line per model, colour-coded consistently with the paper figures
- Proposed model rendered thicker and in a distinct colour
- X-axis: 1 − Specificity, Y-axis: Sensitivity
- Diagonal reference line (random classifier)

**Section 4 — Ablation study bar chart:**
- Horizontal bar chart, 4 bars
- CNN + CDR bar highlighted in the proposed model colour
- Y-axis shows AUC values 0.88 → 0.96 (zoomed for visibility)

**Section 5 — Segmentation metrics cards:**
- Optic disc: Dice 0.968, IoU 0.938
- Optic cup: Dice 0.879, IoU 0.793

---

#### Page 3 — About (`/about`)

Editorial layout, no interactive elements.

**Content:**
- Project title and one-paragraph abstract summary
- Pipeline diagram (static SVG image exported from the architecture diagram)
- "How it works" — 4-step numbered list (Preprocessing → CNN → Segmentation → Fusion)
- Dataset credits with links to original dataset papers
- GitHub repository link (prominent button)
- Paper link (if published) or "Paper submitted to [venue]" placeholder
- Tech stack badges row: Python · FastAPI · PyTorch · React · Docker

---

### 4.3 Shared components

| Component | Props | Behaviour |
|---|---|---|
| `Navbar` | `activePage` | Top nav with 3 links + GitHub icon button |
| `MetricCard` | `label, value, subtitle?` | Gray surface card, 24px number |
| `ResultCard` | `prediction, probability, cdr` | Verdict display with colour coding |
| `HeatmapViewer` | `images: {original, heatmap, segmentation}` | Tabbed image switcher |
| `UploadZone` | `onFile, maxSizeMB, accept` | Drag-and-drop with validation |
| `RocChart` | `data: RocData[]` | Recharts multi-line ROC plot |
| `ErrorBanner` | `message, onRetry?` | Amber warning card |

### 4.4 API client configuration

```typescript
// src/lib/api.ts
const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
  timeout: 30000,   // 30s for slow CPU inference
})
```

The `VITE_API_URL` environment variable is set at build time for production.
For local development it defaults to `http://localhost:8000`.

### 4.5 Non-functional requirements

| Requirement | Target |
|---|---|
| Initial load time | < 2s on 4G (Vite production bundle ≤ 500 KB) |
| Mobile responsive | Fully functional on 375px viewport |
| Accessibility | All interactive elements keyboard-navigable, alt text on all images |
| Loading feedback | Every async operation has a visible loading indicator |
| Error recovery | Every error state has a retry path |
| Image preview | Thumbnail shown before submission, not after |

---

## 5. Docker Setup

### `docker-compose.yml`
```yaml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file: ./backend/.env
    volumes:
      - ./backend/models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    ports:
      - "5173:80"
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      backend:
        condition: service_healthy
```

### Backend `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend `Dockerfile`
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

---

## 6. Build Order

Build the backend first, in this sequence:

| Phase | Work |
|---|---|
| B1 | ONNX export script — convert all 3 models |
| B2 | `core/config.py` — pydantic-settings with `.env` |
| B3 | `core/model_loader.py` — singleton model registry |
| B4 | `utils/image.py` — preprocessing helpers |
| B5 | `services/inference.py` — CNN ONNX inference |
| B6 | `services/gradcam.py` — Grad-CAM on ONNX (no autograd; use numpy) |
| B7 | `services/segmentation.py` — U-Net ONNX + CDR |
| B8 | `schemas/prediction.py` — Pydantic response models |
| B9 | `routers/health.py` + `routers/results.py` |
| B10 | `routers/predict.py` — wire everything together |
| B11 | `main.py` — FastAPI app, CORS, lifespan |
| B12 | Tests — pytest suite |
| B13 | Dockerfile + local docker test |
| F1–F10 | Frontend (after backend passes all tests) |

---

## 7. Out of Scope (v1.0)

- User authentication or session management
- Result history / database storage
- Batch image processing
- Model retraining via the UI
- Mobile native app
- Real-time video fundus analysis
