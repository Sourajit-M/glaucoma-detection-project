import cv2
import numpy as np
import onnxruntime as ort


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_segmentation(
    disc_session: ort.InferenceSession,
    cup_session:  ort.InferenceSession,
    image_bgr:    np.ndarray,           # original BGR image, uint8
    seg_size:     int = 256,
) -> dict:
    """
    Run disc and cup U-Nets, compute CDR, build overlay.

    Args:
        disc_session: Loaded disc U-Net ONNX session
        cup_session:  Loaded cup U-Net ONNX session
        image_bgr:    Original BGR image (any size)
        seg_size:     U-Net input size (256)

    Returns dict with:
        cdr:              float — cup-to-disc ratio
        cdr_risk:         str  — "elevated", "borderline", or "normal"
        disc_mask:        np.ndarray (256, 256) uint8 binary mask
        cup_mask:         np.ndarray (256, 256) uint8 binary mask
        overlay_bgr:      np.ndarray (256, 256, 3) uint8 colour overlay
    """
    # ── Step 1: preprocess for U-Net ─────────────────────────────
    img_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_rgb  = cv2.resize(img_rgb, (seg_size, seg_size),
                        interpolation=cv2.INTER_LANCZOS4)

    img      = img_rgb.astype(np.float32) / 255.0
    mean     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std      = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img      = (img - mean) / std
    tensor   = img.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, 256, 256)

    # ── Step 2: run both U-Nets ───────────────────────────────────
    disc_logits = disc_session.run(None, {"image": tensor})[0]  # (1,1,256,256)
    cup_logits  = cup_session.run( None, {"image": tensor})[0]  # (1,1,256,256)

    # Remove batch + channel dims → (256, 256)
    disc_prob = sigmoid(disc_logits[0, 0])
    cup_prob  = sigmoid(cup_logits[0, 0])

    # ── Step 3: threshold to binary masks ────────────────────────
    disc_mask = (disc_prob >= 0.5).astype(np.uint8) * 255
    cup_mask  = (cup_prob  >= 0.5).astype(np.uint8) * 255

    # ── Step 4: compute CDR ───────────────────────────────────────
    disc_area = float((disc_mask > 0).sum())
    cup_area  = float((cup_mask  > 0).sum())

    if disc_area < 1.0:
        cdr = 0.5   # fallback — no disc detected
    else:
        cup_area = min(cup_area, disc_area)   # anatomical constraint
        cdr = float(np.sqrt(cup_area / disc_area))
        cdr = round(float(np.clip(cdr, 0.0, 1.0)), 3)

    # ── Step 5: CDR risk level ────────────────────────────────────
    if cdr >= 0.65:
        cdr_risk = "elevated"
    elif cdr >= 0.50:
        cdr_risk = "borderline"
    else:
        cdr_risk = "normal"

    # ── Step 6: build colour overlay ─────────────────────────────
    # Start with the resized RGB image, paint disc green, cup red
    overlay = img_rgb.copy()
    overlay[disc_mask > 0] = [0,   200,  0]    # green = disc
    overlay[cup_mask  > 0] = [200,  50, 50]    # red   = cup
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    return {
        "cdr":         cdr,
        "cdr_risk":    cdr_risk,
        "disc_mask":   disc_mask,
        "cup_mask":    cup_mask,
        "overlay_bgr": overlay_bgr,
    }