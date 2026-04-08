import cv2
import numpy as np
import base64
from pathlib import Path
from fastapi import HTTPException

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────
def validate_upload(filename: str, file_bytes: bytes) -> None:
    """
    Raises HTTPException 400 if the file fails any validation check.
    Call this before doing anything else with the uploaded file.
    """
    # 1. Extension check
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' not allowed. "
                    f"Accepted: {', '.join(settings.allowed_extensions_set)}",
        )

    # 2. Size check
    if len(file_bytes) > settings.max_file_size_bytes:
        size_mb = len(file_bytes) / 1024 ** 2
        raise HTTPException(
            status_code=400,
            detail=f"File size {size_mb:.1f} MB exceeds limit "
                    f"of {settings.max_file_size_mb} MB",
        )

    # 3. Readability check — can OpenCV actually decode this?
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="File could not be read as an image. "
                    "It may be corrupted or not a valid image file.",
        )

    # 4. Minimum dimensions
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({w}×{h}px). Minimum size is 100×100px.",
        )


# ─────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────
def load_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode bytes → BGR numpy array.
    Always call validate_upload() before this.
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR, uint8


# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING  (mirrors your training pipeline exactly)
# ─────────────────────────────────────────────────────────────────────
def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab   = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_circular_mask(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    out = img_rgb.copy()
    out[mask == 0] = 0
    return out


def preprocess_for_cnn(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR image → (1, 3, 224, 224) float32 numpy array ready for ONNX.
    Mirrors the training preprocessing pipeline exactly.
    """
    size = settings.image_size

    img_bgr = apply_clahe(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = apply_circular_mask(img_rgb)
    img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)

    img = img_rgb.astype(np.float32) / 255.0

    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std

    # HWC → CHW → add batch dimension
    return img.transpose(2, 0, 1)[np.newaxis, :]    # (1, 3, 224, 224)


def preprocess_for_unet(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR image → (1, 3, 256, 256) float32 numpy array ready for ONNX.
    """
    size = settings.seg_image_size

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)

    img  = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std

    return img.transpose(2, 0, 1)[np.newaxis, :]    # (1, 3, 256, 256)


# ─────────────────────────────────────────────────────────────────────
# ENCODING  (numpy array → base64 PNG string for JSON responses)
# ─────────────────────────────────────────────────────────────────────
def encode_image(img: np.ndarray) -> str:
    """
    Encode a numpy image (any dtype) as a base64 PNG string.
    The frontend decodes this with:
        <img src={`data:image/png;base64,${encoded}`} />
    """
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def bgr_to_rgb_encoded(img_bgr: np.ndarray) -> str:
    """Convenience — converts BGR to RGB then base64 encodes."""
    return encode_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))