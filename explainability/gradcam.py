"""
explainability/gradcam.py
==========================
Grad-CAM (Gradient-weighted Class Activation Mapping) for GlaucomaResNet.

What Grad-CAM does
──────────────────
Grad-CAM computes the gradient of the predicted class score with respect
to the final convolutional feature maps. Regions that strongly influence
the prediction get high activation — producing a heatmap that shows
WHERE the model is looking.

For glaucoma detection, a clinically correct model should highlight the
optic disc / cup region. This is what we verify and show in the paper.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"

Usage
─────
    from explainability.gradcam import GradCAM, overlay_heatmap

    gcam    = GradCAM(model)
    heatmap = gcam.generate(image_tensor, class_idx=1)  # 1 = glaucoma
    overlay = overlay_heatmap(original_image_rgb, heatmap)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE, IMAGE_SIZE


# ─────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM implementation for GlaucomaResNet (ResNet18 backbone).

    Hooks into the last convolutional layer (layer4[-1]) to capture
    activations and gradients during the forward/backward pass.

    Args:
        model:      GlaucomaResNet instance (on DEVICE, eval mode)
        target_layer: Which layer to hook. Defaults to ResNet18's
                      final conv block (layer4[-1].conv2), which
                      produces the most spatially informative maps.
    """

    def __init__(self, model, target_layer: str = "layer4"):
        self.model  = model
        self.model.eval()

        self._activations = None
        self._gradients   = None

        # Resolve target layer from backbone
        layer = getattr(model.backbone, target_layer)
        # Hook the last BasicBlock in the layer
        target_block = layer[-1]

        self._fwd_hook = target_block.register_forward_hook(self._save_activations)
        self._bwd_hook = target_block.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int = 1,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for one image.

        Args:
            image_tensor: Preprocessed tensor (1, 3, H, W) on DEVICE
            class_idx:    Class to explain. 1 = glaucoma (default), 0 = normal.

        Returns:
            heatmap: np.ndarray (H, W) in [0, 1], same spatial size as input
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(image_tensor)

        # Backward pass for the target class
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM: global average pool the gradients → channel weights
        # activations: (1, C, H, W)  gradients: (1, C, H, W)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)   # keep only positive contributions

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to input spatial size
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        return cam.astype(np.float32)

    def remove_hooks(self):
        """Call after you are done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────────────────────────────
# HEATMAP OVERLAY
# ─────────────────────────────────────────────────────────────────────
def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original RGB fundus image.

    Args:
        image_rgb: Original RGB image uint8 (H, W, 3) — NOT normalised
        heatmap:   Grad-CAM output (H, W) in [0, 1]
        alpha:     Heatmap opacity (0 = invisible, 1 = full heatmap)
        colormap:  OpenCV colormap. JET (blue→red) is standard for papers.
                   COLORMAP_MAGMA is a good alternative for colourblind accessibility.

    Returns:
        RGB overlay image uint8 (H, W, 3)
    """
    # Convert heatmap to a uint8 BGR colourmap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb   = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original
    image_f    = image_rgb.astype(np.float32)
    heatmap_f  = heatmap_rgb.astype(np.float32)
    overlay_f  = (1 - alpha) * image_f + alpha * heatmap_f
    return np.clip(overlay_f, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# BATCH GENERATION
# ─────────────────────────────────────────────────────────────────────
def generate_gradcam_grid(
    model,
    df_samples: "pd.DataFrame",
    class_idx: int = 1,
    n_cols: int = 4,
) -> tuple:
    """
    Generate Grad-CAM heatmaps and overlays for a batch of images.

    Args:
        model:      Trained GlaucomaResNet (best checkpoint loaded)
        df_samples: DataFrame rows with [image_path, label]
        class_idx:  Class to explain (1 = glaucoma)
        n_cols:     Images per row in the output grid

    Returns:
        (originals, heatmaps, overlays, predictions, probabilities)
        Each is a list of np.ndarray, one per sample.
    """
    from torchvision import transforms
    from config import NORMALIZE_MEAN, NORMALIZE_STD

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    gcam = GradCAM(model)

    originals, heatmaps, overlays = [], [], []
    predictions, probabilities    = [], []

    for _, row in df_samples.iterrows():
        # Load original (for overlay — NOT normalised)
        img_bgr = cv2.imread(str(row["image_path"]))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

        # Preprocessed tensor for model input
        tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

        # Grad-CAM
        heatmap = gcam.generate(tensor, class_idx=class_idx)

        # Prediction
        with torch.no_grad():
            logits = model(tensor)
            prob   = torch.softmax(logits, dim=1)[0, 1].item()
            pred   = int(prob >= 0.5)

        originals.append(img_rgb)
        heatmaps.append(heatmap)
        overlays.append(overlay_heatmap(img_rgb, heatmap))
        predictions.append(pred)
        probabilities.append(prob)

    gcam.remove_hooks()
    return originals, heatmaps, overlays, predictions, probabilities


# ─────────────────────────────────────────────────────────────────────
# OPTIC DISC FOCUS SCORE
# ─────────────────────────────────────────────────────────────────────
def compute_disc_focus_score(heatmap: np.ndarray, margin: float = 0.35) -> float:
    """
    Quantifies how much the Grad-CAM heatmap concentrates on the
    central region (where the optic disc is after preprocessing).

    After circular mask + resize, the optic disc is roughly centred.
    We define a central square of `margin` fractional radius and compute
    what fraction of total heatmap activation falls inside it.

    A score > 0.5 means the model is predominantly attending to the
    optic disc region — clinically correct behaviour.

    Args:
        heatmap: Grad-CAM output (H, W) in [0, 1]
        margin:  Fraction of image width/height defining the central zone

    Returns:
        float in [0, 1] — fraction of activation in the central zone
    """
    h, w   = heatmap.shape
    mh, mw = int(h * margin), int(w * margin)
    centre = heatmap[mh: h - mh, mw: w - mw]

    total  = heatmap.sum()
    if total < 1e-8:
        return 0.0
    return float(centre.sum() / total)
