"""
models/seg_trainer.py
=====================
Dataset class and training loop for U-Net segmentation on DRISHTI-GS1.

DRISHTI has only 50 training images — aggressive augmentation is essential.
We use:
    - Random horizontal + vertical flips
    - Random rotation (±30°)
    - Elastic deformation (simulates retinal image variability)
    - Colour jitter (handles different fundus camera devices)
    - Random crop + resize

Metrics reported per epoch:
    - Dice coefficient (primary segmentation metric)
    - IoU / Jaccard score
    - Training and validation loss

Usage
─────
    from models.seg_trainer import SegDataset, train_unet, SegTrainer

    disc_ds = SegDataset(drishti_df, target='disc')
    trainer = SegTrainer(model, run_name='disc_unet')
    history = trainer.fit(train_loader, val_loader, epochs=30)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from pathlib import Path
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DEVICE, SEG_IMAGE_SIZE, SEG_BATCH_SIZE, SEG_EPOCHS, SEG_LR,
    MODELS_DIR, LOGS_DIR, SEED
)
from models.unet import DiceBCELoss


# ─────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────
class SegDataset(Dataset):
    """
    PyTorch Dataset for DRISHTI-GS1 segmentation.

    Args:
        df:         DataFrame with [image_path, od_mask_path, cup_mask_path]
        target:     'disc' or 'cup' — which mask to return as ground truth
        augment:    Apply augmentation (True for train, False for val/test)
        image_size: Target (H, W)
    """

    # Aggressive augmentation for the tiny DRISHTI training set
    _TRAIN_TRANSFORMS = A.Compose([
        A.Resize(*SEG_IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    _VAL_TRANSFORMS = A.Compose([
        A.Resize(*SEG_IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def __init__(
        self,
        df: pd.DataFrame,
        target: str   = "disc",   # "disc" or "cup"
        augment: bool = True,
        image_size: tuple = SEG_IMAGE_SIZE,
    ):
        assert target in ("disc", "cup"), "target must be 'disc' or 'cup'"
        mask_col = "od_mask_path" if target == "disc" else "cup_mask_path"

        # Keep only rows with valid mask paths
        self.df        = df.dropna(subset=["image_path", mask_col]).reset_index(drop=True)
        self.mask_col  = mask_col
        self.transform = self._TRAIN_TRANSFORMS if augment else self._VAL_TRANSFORMS

        if len(self.df) == 0:
            raise RuntimeError(
                f"No valid samples found for target='{target}'. "
                "Check that mask paths are correct."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_bgr = cv2.imread(str(row["image_path"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load mask — softmap PNG: values 0-255, threshold at 127
        mask_raw = cv2.imread(str(row[self.mask_col]), cv2.IMREAD_GRAYSCALE)
        mask     = (mask_raw > 127).astype(np.float32)   # binary [0, 1]

        # Apply transforms (albumentations handles both image + mask jointly)
        transformed = self.transform(image=img_rgb, mask=mask)
        image  = transformed["image"]                          # (3, H, W) float tensor
        mask_t = transformed["mask"].unsqueeze(0).float()     # (1, H, W) float tensor

        return image, mask_t


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    inter    = (pred_bin * target).sum()
    union    = pred_bin.sum() + target.sum()
    return float((2.0 * inter + smooth) / (union + smooth))


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    inter    = (pred_bin * target).sum()
    union    = pred_bin.sum() + target.sum() - inter
    return float((inter + smooth) / (union + smooth))


# ─────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────
class SegTrainer:
    """
    Training loop for U-Net segmentation with:
    - Mixed precision (AMP)
    - Early stopping on validation Dice
    - Best model checkpointing

    Args:
        model:    U-Net on DEVICE
        run_name: Used for checkpoint filename
    """

    def __init__(self, model: nn.Module, run_name: str):
        self.model     = model
        self.run_name  = run_name
        self.ckpt_path = MODELS_DIR / f"{run_name}_best.pth"
        self.loss_fn   = DiceBCELoss()
        self.scaler    = GradScaler('cuda')

        self.best_dice    = 0.0
        self.best_epoch   = 0
        self.patience_ctr = 0
        self.history      = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "val_iou":    [],
        }
        print(f"SegTrainer: {run_name}  →  checkpoint: {self.ckpt_path}")

    def _epoch(self, loader, train: bool) -> tuple:
        self.model.train() if train else self.model.eval()
        total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, masks in loader:
                imgs  = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast('cuda'):
                    logits = self.model(imgs)
                    loss   = self.loss_fn(logits, masks)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * imgs.size(0)
                total_dice += dice_coefficient(logits.detach(), masks) * imgs.size(0)
                total_iou  += iou_score(logits.detach(), masks) * imgs.size(0)

        n = len(loader.dataset)
        return total_loss / n, total_dice / n, total_iou / n

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = SEG_EPOCHS,
        lr:           float = SEG_LR,
        patience:     int = 10,
    ) -> dict:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        print(f"\n{'═'*55}")
        print(f"  Training {self.run_name}  |  max {epochs} epochs  |  patience {patience}")
        print(f"{'═'*55}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            tr_loss, tr_dice, _       = self._epoch(train_loader, train=True)
            vl_loss, vl_dice, vl_iou  = self._epoch(val_loader,   train=False)
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_dice"].append(tr_dice)
            self.history["val_dice"].append(vl_dice)
            self.history["val_iou"].append(vl_iou)

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:>3}/{epochs} | "
                f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
                f"Dice {tr_dice:.4f}/{vl_dice:.4f} | "
                f"IoU {vl_iou:.4f} | {elapsed:.1f}s"
            )

            if vl_dice > self.best_dice:
                self.best_dice    = vl_dice
                self.best_epoch   = epoch
                self.patience_ctr = 0
                torch.save({
                    "epoch":             epoch,
                    "model_state_dict":  self.model.state_dict(),
                    "val_dice":          vl_dice,
                    "val_iou":           vl_iou,
                }, self.ckpt_path)
                print(f"  ✓ Saved  (val Dice {vl_dice:.4f})")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= patience:
                    print(f"\n  Early stopping at epoch {epoch}  "
                          f"(best Dice {self.best_dice:.4f} at epoch {self.best_epoch})")
                    break

        print(f"\n  Done. Best val Dice: {self.best_dice:.4f}")
        return self.history

    def load_best(self) -> nn.Module:
        ckpt = torch.load(self.ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {self.run_name}: epoch {ckpt['epoch']}, "
              f"Dice {ckpt['val_dice']:.4f}, IoU {ckpt['val_iou']:.4f}")
        return self.model


# ─────────────────────────────────────────────────────────────────────
# BUILD DATALOADERS FOR SEGMENTATION
# ─────────────────────────────────────────────────────────────────────
def build_seg_dataloaders(
    drishti_df: pd.DataFrame,
    target: str  = "disc",
    val_frac: float = 0.2,
    batch_size: int = SEG_BATCH_SIZE,
) -> tuple:
    """
    Split DRISHTI training data into train/val and return DataLoaders.

    Args:
        drishti_df: DataFrame from load_drishti_segmentation()
        target:     'disc' or 'cup'
        val_frac:   Fraction of data to use for validation
        batch_size: Batch size

    Returns:
        (train_loader, val_loader)
    """
    from sklearn.model_selection import train_test_split

    df = drishti_df.dropna(subset=["od_mask_path", "cup_mask_path"]).reset_index(drop=True)

    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=val_frac, random_state=SEED
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = SegDataset(train_df, target=target, augment=True)
    val_ds   = SegDataset(val_df,   target=target, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,   # num_workers=0 for Windows safety
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"[{target.upper()} seg]  Train: {len(train_ds)}  Val: {len(val_ds)}")
    return train_loader, val_loader
