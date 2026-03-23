"""
data/preprocessing.py
=====================
Image preprocessing pipeline for glaucoma fundus images.

Steps applied in order:
  1. Load image (BGR → RGB)
  2. Resize to target size
  3. CLAHE (Contrast Limited Adaptive Histogram Equalisation)
     — enhances optic disc visibility
  4. Green channel extraction (optional; green channel has highest contrast
     in fundus images)
  5. Normalise to [0, 1] then standardise with ImageNet stats

Also provides:
  - FundusPreprocessor  : callable class for use in PyTorch Datasets
  - preprocess_image    : standalone function for single images
  - verify_dataset      : checks all images in a DataFrame are loadable
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


# ─────────────────────────────────────────────────────────────────────
# CLAHE ENHANCEMENT
# ─────────────────────────────────────────────────────────────────────
def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Applies CLAHE on the L channel of LAB colour space.
    Preserves colour while enhancing local contrast.

    Args:
        image_bgr:  OpenCV BGR image (uint8)
        clip_limit: CLAHE clip limit (higher = more contrast)
        tile_size:  Grid size for CLAHE

    Returns: CLAHE-enhanced BGR image (uint8)
    """
    lab   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq  = clahe.apply(l)
    lab   = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────
# CIRCULAR MASK (removes black borders around fundus)
# ─────────────────────────────────────────────────────────────────────
def apply_circular_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Detects the fundus circle and zeroes out non-retinal background.
    Helps models focus on the retinal region only.

    Args:
        image_rgb: RGB image (uint8)

    Returns: RGB image with background masked to black (uint8)
    """
    gray   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find largest contour (the fundus circle)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_rgb

    largest = max(contours, key=cv2.contourArea)
    mask    = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    masked = image_rgb.copy()
    masked[mask == 0] = 0
    return masked


# ─────────────────────────────────────────────────────────────────────
# MAIN PREPROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────
def preprocess_image(
    image_path: str,
    target_size: tuple = IMAGE_SIZE,
    use_clahe: bool = True,
    use_circular_mask: bool = True,
    as_tensor: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single fundus image.

    Args:
        image_path:         Path to the image file
        target_size:        (H, W) to resize to
        use_clahe:          Apply CLAHE enhancement
        use_circular_mask:  Zero out non-fundus background
        as_tensor:          If True, return normalised PyTorch tensor (C, H, W)

    Returns:
        np.ndarray (H, W, 3) uint8  — if as_tensor=False
        torch.Tensor (3, H, W)      — if as_tensor=True, normalised [0,1]
    """
    # ── Load
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    # ── CLAHE (on BGR before converting to RGB)
    if use_clahe:
        img_bgr = apply_clahe(img_bgr)

    # ── BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Circular mask
    if use_circular_mask:
        img_rgb = apply_circular_mask(img_rgb)

    # ── Resize
    img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LANCZOS4)

    if not as_tensor:
        return img_rgb

    # ── Normalise + convert to tensor
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array(NORMALIZE_MEAN, dtype=np.float32)
    std  = np.array(NORMALIZE_STD,  dtype=np.float32)
    img_norm = (img_float - mean) / std
    return torch.from_numpy(img_norm.transpose(2, 0, 1))   # (C, H, W)


# ─────────────────────────────────────────────────────────────────────
# PYTORCH TRANSFORMS
# ─────────────────────────────────────────────────────────────────────
def get_train_transforms(image_size: tuple = IMAGE_SIZE) -> transforms.Compose:
    """
    Augmentation + normalisation pipeline for training.
    Augmentations chosen to reflect realistic clinical variation.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_val_transforms(image_size: tuple = IMAGE_SIZE) -> transforms.Compose:
    """
    Deterministic pipeline for validation and test (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


# ─────────────────────────────────────────────────────────────────────
# PYTORCH DATASET CLASS
# ─────────────────────────────────────────────────────────────────────
import pandas as pd
from torch.utils.data import Dataset

class GlaucomaDataset(Dataset):
    """
    PyTorch Dataset for glaucoma classification.

    Args:
        df:            DataFrame with columns [image_path, label]
        transform:     torchvision transform pipeline
        use_clahe:     Apply CLAHE before passing to transform
        use_mask:      Apply circular mask
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        use_clahe: bool = True,
        use_mask: bool  = True,
    ):
        self.df         = df.reset_index(drop=True)
        self.transform  = transform
        self.use_clahe  = use_clahe
        self.use_mask   = use_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])

        # Load + preprocess (returns RGB np.ndarray)
        img = preprocess_image(
            row["image_path"],
            use_clahe=self.use_clahe,
            use_circular_mask=self.use_mask,
            as_tensor=False,
        )

        img_pil = Image.fromarray(img)

        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """
    Convenience function: build train/val/test DataLoaders.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    from torch.utils.data import DataLoader

    datasets = {
        "train": GlaucomaDataset(train_df, transform=get_train_transforms()),
        "val":   GlaucomaDataset(val_df,   transform=get_val_transforms()),
        "test":  GlaucomaDataset(test_df,  transform=get_val_transforms()),
    }

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"[DataLoader] {split}: {len(ds)} images, {len(loaders[split])} batches")

    return loaders


# ─────────────────────────────────────────────────────────────────────
# DATASET INTEGRITY CHECK
# ─────────────────────────────────────────────────────────────────────
def verify_dataset(df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    """
    Checks every image in the DataFrame is loadable.
    Prints a summary and returns a cleaned DataFrame with broken entries removed.

    Args:
        df:          DataFrame with 'image_path' column
        sample_size: If set, only verify a random sample (useful for AIROGS)
    """
    print(f"Verifying {len(df)} images...")
    if sample_size:
        check_df = df.sample(min(sample_size, len(df)), random_state=42)
    else:
        check_df = df

    bad = []
    for _, row in check_df.iterrows():
        img = cv2.imread(str(row["image_path"]))
        if img is None:
            bad.append(row["image_path"])

    if bad:
        print(f"  Found {len(bad)} unreadable images:")
        for p in bad[:10]:
            print(f"    {p}")
        df = df[~df["image_path"].isin(bad)].copy()
        print(f"  Removed {len(bad)} entries. Remaining: {len(df)}")
    else:
        print(f"  All images verified OK.")

    return df