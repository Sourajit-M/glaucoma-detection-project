"""
models/cnn_model.py
====================
ResNet18-based CNN for glaucoma classification.

Design choices
──────────────
- Transfer learning from ImageNet pretrained ResNet18
- Two-stage training:
    Stage 1 (frozen backbone) — only the classifier head is trained
    Stage 2 (full fine-tune)  — all layers unfrozen, lower LR
- Custom classifier head: AdaptiveAvgPool → Dropout → Linear(512→2)
- Mixed precision (torch.cuda.amp) — halves VRAM, ~30% faster on RTX GPUs
- Label smoothing loss — regularises overconfident predictions

Why ResNet18 over larger models?
- 6 GB VRAM limits batch size with ResNet50+
- ResNet18 converges faster, easier to interpret with Grad-CAM
- Literature shows diminishing returns beyond ResNet18 on small fundus datasets

Usage
─────
    from models.cnn_model import build_model, get_optimizer

    model = build_model(num_classes=2, dropout=0.5, pretrained=True)
    optimizer, scheduler = get_optimizer(model, stage=1)  # frozen backbone
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import NUM_CLASSES, DROPOUT_RATE, USE_PRETRAINED, LEARNING_RATE, WEIGHT_DECAY, DEVICE


# ─────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────
class GlaucomaResNet(nn.Module):
    """
    ResNet18 with a custom glaucoma classification head.

    Architecture:
        ResNet18 backbone (pretrained on ImageNet)
            └─ AdaptiveAvgPool2d (already in ResNet18)
            └─ Dropout(p)
            └─ Linear(512 → num_classes)

    The backbone feature extractor outputs a 512-dim vector per image,
    which feeds the lightweight classification head.
    """

    def __init__(
        self,
        num_classes: int  = NUM_CLASSES,
        dropout: float    = DROPOUT_RATE,
        pretrained: bool  = USE_PRETRAINED,
    ):
        super().__init__()

        # Load backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the original FC layer — we'll replace it
        self.feature_dim = backbone.fc.in_features   # 512 for ResNet18
        backbone.fc      = nn.Identity()
        self.backbone    = backbone

        # Custom head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # (B, 512)
        logits   = self.head(features)   # (B, num_classes)
        return logits

    def freeze_backbone(self):
        """Freeze all backbone parameters (Stage 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen — training head only.")

    def unfreeze_backbone(self):
        """Unfreeze all parameters (Stage 2 fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen — full fine-tuning.")

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────
def build_model(
    num_classes: int = NUM_CLASSES,
    dropout: float   = DROPOUT_RATE,
    pretrained: bool = USE_PRETRAINED,
    freeze_backbone: bool = True,
) -> GlaucomaResNet:
    """
    Build and return a GlaucomaResNet moved to the configured device.

    Args:
        num_classes:     Output classes (2 for binary)
        dropout:         Dropout rate in the classification head
        pretrained:      Use ImageNet pretrained weights
        freeze_backbone: If True, start with backbone frozen (Stage 1)

    Returns:
        GlaucomaResNet on DEVICE
    """
    model = GlaucomaResNet(num_classes, dropout, pretrained)

    if freeze_backbone:
        model.freeze_backbone()

    model = model.to(DEVICE)

    total      = sum(p.numel() for p in model.parameters())
    trainable  = model.count_trainable_params()
    print(f"Model     : GlaucomaResNet18")
    print(f"Device    : {DEVICE}")
    print(f"Params    : {total:,} total  |  {trainable:,} trainable")
    return model


# ─────────────────────────────────────────────────────────────────────
# OPTIMISER + SCHEDULER
# ─────────────────────────────────────────────────────────────────────
def get_optimizer(
    model: GlaucomaResNet,
    stage: int = 1,
) -> tuple:
    """
    Returns (optimizer, scheduler) configured for the given training stage.

    Stage 1 — head only, higher LR (backbone frozen):
        AdamW, LR=1e-3, CosineAnnealingLR

    Stage 2 — full fine-tune, lower LR (backbone unfrozen):
        AdamW with layer-wise LR decay:
            backbone LR = 1e-4  (10× lower than head)
            head LR     = 1e-4
        CosineAnnealingLR

    Args:
        model: GlaucomaResNet instance
        stage: 1 (frozen backbone) or 2 (full fine-tune)

    Returns:
        (optimizer, scheduler)
    """
    if stage == 1:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-5
        )

    elif stage == 2:
        # Differential learning rates: backbone gets 10× lower LR
        param_groups = [
            {"params": model.backbone.parameters(), "lr": LEARNING_RATE},
            {"params": model.head.parameters(),     "lr": LEARNING_RATE * 10},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-6
        )

    else:
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    return optimizer, scheduler


# ─────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────
def get_loss_fn(
    y_train: "np.ndarray",
    label_smoothing: float = 0.1,
) -> nn.CrossEntropyLoss:
    """
    Returns a CrossEntropyLoss with:
      - Class weights (inverse frequency) to handle imbalance
      - Label smoothing to prevent overconfident predictions

    Args:
        y_train:         Training labels array (for computing class weights)
        label_smoothing: Smoothing factor (0.1 is a standard value)

    Returns:
        nn.CrossEntropyLoss on DEVICE
    """
    import numpy as np
    counts       = np.bincount(y_train)
    weights      = 1.0 / counts
    weights      = weights / weights.sum() * len(counts)   # normalise
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print(f"Class weights — Normal: {weights[0]:.3f}  Glaucoma: {weights[1]:.3f}")

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


# ─────────────────────────────────────────────────────────────────────
# EFFICIENTNET-B0 MODEL
# ─────────────────────────────────────────────────────────────────────
class GlaucomaEfficientNet(nn.Module):
    """
    EfficientNet-B0 with a custom glaucoma classification head.

    Architecture:
        EfficientNet-B0 backbone (pretrained on ImageNet)
            └─ AdaptiveAvgPool2d (built-in)
            └─ Dropout(p)
            └─ Linear(1280 → num_classes)

    EfficientNet-B0 outputs a 1280-dim feature vector.
    It is more parameter-efficient than ResNet18 (5.3M vs 11.7M params)
    while matching or exceeding accuracy on medical imaging benchmarks.

    Same freeze/unfreeze API as GlaucomaResNet — trainer.py unchanged.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float   = DROPOUT_RATE,
        pretrained: bool = USE_PRETRAINED,
    ):
        super().__init__()

        weights          = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone         = models.efficientnet_b0(weights=weights)

        # EfficientNet classifier head is backbone.classifier[1]
        self.feature_dim = backbone.classifier[1].in_features   # 1280
        backbone.classifier = nn.Identity()
        self.backbone    = backbone

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # (B, 1280)
        logits   = self.head(features) # (B, num_classes)
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("EfficientNet-B0 backbone frozen — training head only.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("EfficientNet-B0 backbone unfrozen — full fine-tuning.")

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────
# UNIFIED FACTORY  (arch-aware)
# ─────────────────────────────────────────────────────────────────────
def build_model_arch(
    arch: str        = "resnet18",
    num_classes: int = NUM_CLASSES,
    dropout: float   = DROPOUT_RATE,
    pretrained: bool = USE_PRETRAINED,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Build any supported glaucoma classification model by name.

    Supported arch values:
        'resnet18'        — GlaucomaResNet      (512-dim,  11.7M params)
        'efficientnet_b0' — GlaucomaEfficientNet (1280-dim,  5.3M params)

    Returns model on DEVICE with the same .backbone / .head structure.
    trainer.py works unchanged with both architectures.
    """
    arch = arch.lower().replace("-", "_")

    if arch == "resnet18":
        model      = GlaucomaResNet(num_classes, dropout, pretrained)
        arch_label = "ResNet18"
    elif arch == "efficientnet_b0":
        model      = GlaucomaEfficientNet(num_classes, dropout, pretrained)
        arch_label = "EfficientNet-B0"
    else:
        raise ValueError(
            f"Unknown arch '{arch}'. Supported: 'resnet18', 'efficientnet_b0'"
        )

    if freeze_backbone:
        model.freeze_backbone()

    model = model.to(DEVICE)

    total     = sum(p.numel() for p in model.parameters())
    trainable = model.count_trainable_params()
    print(f"Architecture : {arch_label}")
    print(f"Device       : {DEVICE}")
    print(f"Params       : {total:,} total  |  {trainable:,} trainable")
    return model


def get_optimizer_for_arch(model: nn.Module, stage: int = 1) -> tuple:
    """
    Optimizer + scheduler for any model with .backbone and .head attributes.
    Identical logic to get_optimizer() — works for both ResNet and EfficientNet.
    """
    if stage == 1:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-5
        )
    elif stage == 2:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": LEARNING_RATE},
            {"params": model.head.parameters(),     "lr": LEARNING_RATE * 10},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-6
        )
    else:
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    return optimizer, scheduler