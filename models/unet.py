"""
models/unet.py
==============
Lightweight U-Net for optic disc and cup segmentation.

Architecture
────────────
Standard U-Net encoder-decoder with skip connections.
Encoder uses pretrained ResNet18 backbone (via segmentation-models-pytorch)
for better feature extraction with DRISHTI's small dataset (50 images).

Two separate models are trained:
    1. disc_unet  — segments the optic disc (OD)
    2. cup_unet   — segments the optic cup (OC)

CDR is then computed from the predicted masks:
    CDR = sqrt(cup_area / disc_area)   (diameter ratio approximation)

Why two separate models?
    The disc and cup have different textures, boundaries, and intensities.
    A single model with two output channels tends to underperform on the
    cup (smaller, harder target) — separate models are more reliable for
    a 50-image training set.

Usage
─────
    from models.unet import build_unet, compute_cdr_from_masks

    disc_model = build_unet(encoder='resnet18', pretrained=True)
    cup_model  = build_unet(encoder='resnet18', pretrained=True)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE, SEG_IMAGE_SIZE


# ─────────────────────────────────────────────────────────────────────
# BUILD U-NET
# ─────────────────────────────────────────────────────────────────────
def build_unet(
    encoder: str     = "resnet18",
    pretrained: bool = True,
    in_channels: int = 3,
    classes: int     = 1,
) -> nn.Module:
    """
    Build a U-Net with a ResNet18 encoder using segmentation-models-pytorch.

    Args:
        encoder:     Encoder backbone name
        pretrained:  Use ImageNet pretrained encoder weights
        in_channels: Input image channels (3 = RGB)
        classes:     Output mask channels (1 = binary mask)

    Returns:
        U-Net model on DEVICE
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation-models-pytorch not installed.\n"
            "Run: uv add segmentation-models-pytorch"
        )

    weights  = "imagenet" if pretrained else None
    model    = smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,   # raw logits — we apply sigmoid manually
    )
    model = model.to(DEVICE)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net ({encoder}) | {total:,} params | {trainable:,} trainable | device={DEVICE}")
    return model


# ─────────────────────────────────────────────────────────────────────
# SEGMENTATION LOSS
# ─────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss for binary segmentation.

    Dice loss handles class imbalance (small cup region vs background).
    BCE loss provides stable gradients everywhere.
    Combined loss = 0.5 * Dice + 0.5 * BCE is standard for medical imaging.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE
        bce_loss = self.bce(logits, targets)

        # Dice
        probs   = torch.sigmoid(logits)
        inter   = (probs * targets).sum(dim=(1, 2, 3))
        union   = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice    = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)

        return 0.5 * bce_loss + 0.5 * dice.mean()


# ─────────────────────────────────────────────────────────────────────
# CDR COMPUTATION FROM MASKS
# ─────────────────────────────────────────────────────────────────────
def compute_cdr_from_masks(
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute Cup-to-Disc Ratio from predicted binary masks.

    CDR = sqrt(cup_area / disc_area)
    The square root converts area ratio to diameter ratio,
    consistent with clinical CDR measurement convention.

    Args:
        disc_mask: Predicted disc mask — float (H, W) in [0, 1]
                   OR binary uint8 (H, W)
        cup_mask:  Predicted cup mask  — float (H, W) in [0, 1]
                   OR binary uint8 (H, W)
        threshold: Binarisation threshold (applied if masks are float)

    Returns:
        CDR float in [0, 1], or 0.5 as fallback
    """
    # Binarise if float
    if disc_mask.dtype != np.uint8:
        disc_bin = (disc_mask >= threshold).astype(np.float32)
    else:
        disc_bin = (disc_mask > 0).astype(np.float32)

    if cup_mask.dtype != np.uint8:
        cup_bin = (cup_mask >= threshold).astype(np.float32)
    else:
        cup_bin = (cup_mask > 0).astype(np.float32)

    disc_area = disc_bin.sum()
    cup_area  = cup_bin.sum()

    if disc_area < 1.0:
        return 0.5   # fallback for failed segmentation

    # Cup cannot exceed disc anatomically
    cup_area = min(cup_area, disc_area)
    cdr      = float(np.sqrt(cup_area / disc_area))
    return float(np.clip(cdr, 0.0, 1.0))


def predict_mask(
    model: nn.Module,
    image_tensor: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run inference and return a binary mask as numpy array.

    Args:
        model:        Trained U-Net (on DEVICE)
        image_tensor: (1, 3, H, W) normalised tensor on DEVICE
        threshold:    Sigmoid threshold for binarisation

    Returns:
        Binary mask np.ndarray (H, W) uint8, values 0 or 255
    """
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)          # (1, 1, H, W)
        prob   = torch.sigmoid(logits)[0, 0]  # (H, W)
        mask   = (prob >= threshold).cpu().numpy().astype(np.uint8) * 255
    return mask
