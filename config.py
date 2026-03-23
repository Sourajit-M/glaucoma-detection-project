"""
config.py
=========
Central configuration for the Glaucoma Detection project.

HOW PATHS WORK
──────────────
Paths are never hardcoded here. They are loaded from a .env file
that lives at the project root and is specific to each machine.

Setup (one-time):
    1. Copy .env.example  →  .env
    2. Set GLAUCOMA_DATA_DIR in .env to your local datasets folder
    3. Optionally set GLAUCOMA_OUTPUT_DIR (defaults to <project>/outputs/)

Your .env (example):
    GLAUCOMA_DATA_DIR=d:\\Machine Learning\\glaucoma detection project\\datasets\\glaucoma
    GLAUCOMA_OUTPUT_DIR=          ← leave blank to use default

The .env file is in .gitignore — it is never committed.
Anyone cloning the repo just fills in their own .env.
"""

import os
from pathlib import Path

# ── Load .env ────────────────────────────────────────────────────────
# Use python-dotenv if available; fall back to manual parsing so the
# project works even before dependencies are installed.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    # Manual fallback: parse KEY=VALUE lines from .env
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

# ─────────────────────────────────────────────────────────────────────
# PROJECT ROOT  (absolute path to this file's directory)
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────────────────────────────
# DATA DIRECTORY  —  read from .env, no fallback on purpose
# (a missing path should fail loudly, not silently use a wrong folder)
# ─────────────────────────────────────────────────────────────────────
_raw_data_dir = os.getenv("GLAUCOMA_DATA_DIR", "").strip()

if not _raw_data_dir:
    raise EnvironmentError(
        "\n\nGLAUCOMA_DATA_DIR is not set.\n"
        "Steps to fix:\n"
        "  1. Copy .env.example  →  .env\n"
        "  2. Set GLAUCOMA_DATA_DIR in .env to your datasets folder\n"
        "     Example:\n"
        r"       GLAUCOMA_DATA_DIR=d:\Machine Learning\glaucoma detection project\datasets\glaucoma"
        "\n"
    )

ROOT_DATA_DIR = Path(_raw_data_dir)

if not ROOT_DATA_DIR.exists():
    raise FileNotFoundError(
        f"\n\nGLAUCOMA_DATA_DIR does not exist on disk:\n  {ROOT_DATA_DIR}\n"
        "Check the path in your .env file.\n"
    )

# ─────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY  —  defaults to <project_root>/outputs/
# ─────────────────────────────────────────────────────────────────────
_raw_output_dir = os.getenv("GLAUCOMA_OUTPUT_DIR", "").strip()
OUTPUT_DIR = Path(_raw_output_dir) if _raw_output_dir else PROJECT_ROOT / "outputs"

MODELS_DIR     = OUTPUT_DIR / "models"
FIGURES_DIR    = OUTPUT_DIR / "figures"
RESULTS_DIR    = OUTPUT_DIR / "results"
LOGS_DIR       = OUTPUT_DIR / "logs"
FEATURES_CACHE = RESULTS_DIR / "features_cache.pkl"

# Create output subdirs if they don't exist yet
for _d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────────────────────────────
DATASETS = {

    # ------------------------------------------------------------------
    # ACRIMA  (705 images; label encoded in filename: _g_ = glaucoma)
    # Layout:  ACRIMA/database/<images>
    # ------------------------------------------------------------------
    "ACRIMA": {
        "root":       ROOT_DATA_DIR / "ACRIMA" / "database",
        "label_mode": "filename",
        "task":       ["classification"],
    },

    # ------------------------------------------------------------------
    # DRISHTI-GS1  (101 images; segmentation + CDR)
    # Layout:  DRISHTI-GS1/Drishti_GS_trainingData/{Images, GT}
    #                      Drishti_GS_testingData/{Images, GT}
    # ------------------------------------------------------------------
    "DRISHTI": {
        "train_images": ROOT_DATA_DIR / "DRISHTI-GS1" / "Drishti_GS_trainingData" / "Images",
        "train_gt":     ROOT_DATA_DIR / "DRISHTI-GS1" / "Drishti_GS_trainingData" / "GT",
        "test_images":  ROOT_DATA_DIR / "DRISHTI-GS1" / "Drishti_GS_testingData"  / "Images",
        "test_gt":      ROOT_DATA_DIR / "DRISHTI-GS1" / "Drishti_GS_testingData"  / "GT",
        "task":         ["segmentation", "cdr"],
    },

    # ------------------------------------------------------------------
    # RIM-ONE DL  (313 images; label from folder name)
    # Layout:  RIM-ONE_DL/Train/{Glaucoma, Normal}
    #                     Test/{Glaucoma, Normal}
    # ------------------------------------------------------------------
    "RIMONE": {
        "train_root": ROOT_DATA_DIR / "RIM-ONE_DL" / "Train",
        "test_root":  ROOT_DATA_DIR / "RIM-ONE_DL" / "Test",
        "label_mode": "folder",
        "task":       ["classification"],
    },

    # ------------------------------------------------------------------
    # EyePACS-AIROGS-light-v2  (label from subfolder: RG / NRG)
    # Layout:  EyePACS-AIROGS/train/{RG, NRG}
    #                         validation/{RG, NRG}
    #                         test/{RG, NRG}
    #                         metadata.csv
    # ------------------------------------------------------------------
    "AIROGS": {
        "train_root":      ROOT_DATA_DIR / "EyePACS-AIROGS" / "train",
        "val_root":        ROOT_DATA_DIR / "EyePACS-AIROGS" / "validation",
        "test_root":       ROOT_DATA_DIR / "EyePACS-AIROGS" / "test",
        "metadata_csv":    ROOT_DATA_DIR / "EyePACS-AIROGS" / "metadata.csv",
        "label_mode":      "folder",
        "positive_folder": "RG",
        "task":            ["classification"],
    },
}

# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE     = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────
SEED                = 42
BATCH_SIZE          = 16
NUM_EPOCHS          = 50
LEARNING_RATE       = 1e-4
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 7

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ─────────────────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────
CNN_BACKBONE  = "resnet18"   # options: resnet18 | resnet50 | efficientnet_b0
NUM_CLASSES   = 2
DROPOUT_RATE  = 0.5
USE_PRETRAINED = True

# ─────────────────────────────────────────────────────────────────────
# SEGMENTATION  (U-Net on DRISHTI)
# ─────────────────────────────────────────────────────────────────────
SEG_IMAGE_SIZE = (256, 256)
SEG_BATCH_SIZE = 8
SEG_EPOCHS     = 30
SEG_LR         = 1e-3

# ─────────────────────────────────────────────────────────────────────
# CLINICAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────
CDR_GLAUCOMA_THRESHOLD = 0.65   # CDR >= this value → likely glaucoma

# ─────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────
import torch

def _resolve_device() -> str:
    """
    Returns the best available device and prints a clear diagnostic.
    If CUDA is not detected despite having a GPU, the message explains why.
    """
    if torch.cuda.is_available():
        return "cuda"

    import warnings
    warnings.warn(
        "\n[config] WARNING: CUDA not available — training will be slow on CPU.\n"
        "  Likely cause: PyTorch was installed without CUDA support.\n"
        "  Fix: run  `uv sync`  — pyproject.toml already points torch at\n"
        "  the CUDA 12.1 index. If your CUDA is 11.8, change cu121 → cu118\n"
        "  in pyproject.toml first. Run `nvidia-smi` to check your version.\n",
        stacklevel=2,
    )
    return "cpu"

DEVICE = _resolve_device()

# ─────────────────────────────────────────────────────────────────────
# SANITY PRINT  (run  `python config.py`  to verify your setup)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"ROOT_DATA_DIR : {ROOT_DATA_DIR}")
    print(f"OUTPUT_DIR    : {OUTPUT_DIR}")

    if torch.cuda.is_available():
        name     = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_ver = torch.version.cuda
        print(f"DEVICE        : cuda  →  {name}  ({vram_gb:.1f} GB VRAM, CUDA {cuda_ver})")
    else:
        print("DEVICE        : cpu   (see warning above)")