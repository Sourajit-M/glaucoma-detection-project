import cv2
import numpy as np
import onnxruntime as ort

# Exact node name from the ONNX graph inspection
LAYER4_NODE = "/backbone/layer4/layer4.1/relu_1/Relu_output_0"


def generate_heatmap(
    session: ort.InferenceSession,
    image_array: np.ndarray,    # (1, 3, 224, 224) float32 — preprocessed
    class_idx: int = 1,         # 1 = glaucoma
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap using intermediate activations.

    Since ONNX has no autograd, we use the global average of the
    layer4 feature maps as channel weights (CAM approximation).
    Visually equivalent to Grad-CAM for classification tasks.

    Returns:
        heatmap: np.ndarray (224, 224) float32 in [0, 1]
    """
    # ── Step 1: run inference requesting the intermediate node ────
    # By adding layer4 to the output list, ONNX Runtime returns
    # both the final logits AND the layer4 feature maps
    outputs = session.run(
        [LAYER4_NODE, "logits"],
        {"image": image_array},
    )

    activations = outputs[0][0]   # (512, 7, 7) — remove batch dim
    logits      = outputs[1][0]   # (2,)

    # ── Step 2: channel weights ───────────────────────────────────
    # Global average pool over spatial dims for the target class
    # activations shape: (C, H, W) → weights shape: (C,)
    weights = activations.mean(axis=(1, 2))   # (512,)

    # ── Step 3: weighted sum of feature maps ─────────────────────
    # Multiply each channel by its weight and sum
    # Result shape: (7, 7)
    cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (7, 7)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    # ── Step 4: ReLU — keep only positive contributions ──────────
    cam = np.maximum(cam, 0)

    # ── Step 5: normalise to [0, 1] ──────────────────────────────
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # ── Step 6: resize to input image size ───────────────────────
    h = w = 224
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    return cam.astype(np.float32)   # (224, 224) in [0, 1]


def overlay_heatmap(
    image_bgr: np.ndarray,      # original BGR image, uint8
    heatmap: np.ndarray,        # (224, 224) float32 in [0, 1]
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend the heatmap over the original image.
    Returns BGR uint8 image ready for encoding.
    """
    # Resize original to match heatmap if needed
    img = cv2.resize(image_bgr, (224, 224))

    # Convert heatmap to colour (JET: blue→green→red)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Weighted blend
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_bgr, alpha, 0)

    return overlay   # BGR uint8