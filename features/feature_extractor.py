"""
features/feature_extractor.py
==============================
Handcrafted feature extraction for classical ML models.

Three feature groups
────────────────────
1. Colour features  (18-dim)
   RGB and HSV channel statistics: mean, std, skewness per channel.
   Captures colour distribution differences between glaucoma / normal fundus.

2. LBP texture features  (26-dim)
   Local Binary Pattern histogram from the green channel (highest contrast
   in fundus images). Radius=3, n_points=24, method='uniform' → 26 bins.
   Captures micro-structural patterns around the optic disc.

3. CDR proxy feature  (1-dim)
   Cup-to-Disc Ratio estimated from green-channel thresholding.
   For DRISHTI, ground-truth cdr_mean from the .txt files is used directly.
   Clinically the single most important glaucoma indicator.

Total feature vector: 45 dimensions per image.

Usage
─────
    from features.feature_extractor import extract_features, build_feature_matrix

    # Single image
    vec = extract_features("path/to/image.jpg")

    # Full dataset DataFrame → feature matrix ready for sklearn
    X, y, names, valid_df = build_feature_matrix(df, n_jobs=4)
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import stats
from skimage.feature import local_binary_pattern
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGE_SIZE

# ─────────────────────────────────────────────────────────────────────
# FEATURE DIMENSIONS
# ─────────────────────────────────────────────────────────────────────
N_COLOUR = 18    # 3 channels × 3 stats × 2 colour spaces (RGB, HSV)
LBP_BINS = 26    # uniform LBP with n_points=24  →  n_points + 2
N_CDR    = 1
N_TOTAL  = N_COLOUR + LBP_BINS + N_CDR   # 45


# ─────────────────────────────────────────────────────────────────────
# 1. COLOUR FEATURES  (18-dim)
# ─────────────────────────────────────────────────────────────────────
def extract_colour_features(image_rgb: np.ndarray) -> np.ndarray:
    """
    Per-channel statistics (mean, std, skewness) from RGB and HSV spaces.

    Args:
        image_rgb: uint8 RGB image, any spatial size

    Returns:
        np.ndarray (18,)
    """
    img = image_rgb.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 180.0   # H → [0, 1]
    hsv[:, :, 1] /= 255.0   # S → [0, 1]
    hsv[:, :, 2] /= 255.0   # V → [0, 1]

    feats = []
    for space in [img, hsv]:
        for ch in range(3):
            channel = space[:, :, ch].ravel()
            # skew returns NaN on near-constant channels (black-border crops)
            skewness = float(stats.skew(channel))
            if np.isnan(skewness):
                skewness = 0.0
            feats.extend([
                float(channel.mean()),
                float(channel.std()),
                skewness,
            ])

    return np.array(feats, dtype=np.float32)


def colour_feature_names() -> list:
    names = []
    for space, channels in [("RGB", ["R", "G", "B"]), ("HSV", ["H", "S", "V"])]:
        for ch in channels:
            for stat in ["mean", "std", "skew"]:
                names.append(f"{space}_{ch}_{stat}")
    return names   # 18 names


# ─────────────────────────────────────────────────────────────────────
# 2. LBP TEXTURE FEATURES  (26-dim)
# ─────────────────────────────────────────────────────────────────────
def extract_lbp_features(
    image_rgb: np.ndarray,
    radius: int = 3,
    n_points: int = 24,
) -> np.ndarray:
    """
    Normalised LBP histogram from the green channel.

    Green channel is used because it gives the highest contrast between
    optic disc, cup, and surrounding retina in fundus images.
    'uniform' patterns produce (n_points + 2) = 26 bins.

    Args:
        image_rgb: uint8 RGB image
        radius:    LBP neighbourhood radius in pixels
        n_points:  Number of circularly symmetric neighbour points

    Returns:
        np.ndarray (26,)
    """
    green  = image_rgb[:, :, 1]
    lbp    = local_binary_pattern(green, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist    = hist.astype(np.float32)
    total   = hist.sum()
    if total > 0:
        hist /= total
    return hist


def lbp_feature_names(n_points: int = 24) -> list:
    return [f"LBP_bin_{i}" for i in range(n_points + 2)]


# ─────────────────────────────────────────────────────────────────────
# 3. CDR PROXY FEATURE  (1-dim)
# ─────────────────────────────────────────────────────────────────────
def extract_cdr_proxy(image_rgb: np.ndarray) -> float:
    """
    Estimates Cup-to-Disc Ratio using green-channel intensity thresholding.

    Method:
      1. Crop the central 50% of the image (optic disc is roughly centred
         after CLAHE + circular mask + resize preprocessing)
      2. Otsu threshold to detect the disc (bright region)
      3. 75th percentile threshold to detect the cup (brightest core)
      4. CDR = sqrt(cup_area / disc_area)  — sqrt because CDR is a
         diameter ratio, not an area ratio

    NOTE: this is a proxy only. Phase 7 (U-Net segmentation) will replace
    this with a geometrically accurate CDR. For DRISHTI use cdr_gt directly.

    Args:
        image_rgb: Preprocessed uint8 RGB image (224×224 expected)

    Returns:
        float in [0.0, 1.0], fallback 0.5 on failure
    """
    h, w  = image_rgb.shape[:2]
    green = image_rgb[:, :, 1]

    mh, mw = h // 4, w // 4
    roi    = green[mh: h - mh, mw: w - mw]

    if roi.mean() < 10:   # near-black crop → skip
        return 0.5

    _, disc_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cup_thresh   = int(np.percentile(roi, 75))
    _, cup_mask  = cv2.threshold(roi, cup_thresh, 255, cv2.THRESH_BINARY)

    disc_area = float(disc_mask.sum()) / 255.0
    cup_area  = float(cup_mask.sum())  / 255.0

    if disc_area < 1.0:
        return 0.5

    cup_area = min(cup_area, disc_area)   # cup cannot exceed disc
    cdr      = float(np.sqrt(cup_area / disc_area))
    return float(np.clip(cdr, 0.0, 1.0))


def cdr_feature_names() -> list:
    return ["CDR_proxy"]


# ─────────────────────────────────────────────────────────────────────
# COMBINED EXTRACTOR  —  single image  →  45-dim vector
# ─────────────────────────────────────────────────────────────────────
def extract_features(
    image_path: str,
    cdr_gt: float = None,
    target_size: tuple = IMAGE_SIZE,
) -> np.ndarray:
    """
    Full 45-dimensional feature vector for one image.

    Args:
        image_path:  Path to the image file
        cdr_gt:      Ground-truth CDR (e.g. from DRISHTI).
                     If provided, replaces the proxy estimate.
        target_size: (H, W) resize target

    Returns:
        np.ndarray (45,), or None if the image cannot be loaded
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(
        img_rgb,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )

    colour_feats = extract_colour_features(img_rgb)
    lbp_feats    = extract_lbp_features(img_rgb)
    cdr_val      = float(cdr_gt) if (cdr_gt is not None and not np.isnan(cdr_gt)) \
                   else extract_cdr_proxy(img_rgb)
    cdr_feat     = np.array([cdr_val], dtype=np.float32)

    return np.concatenate([colour_feats, lbp_feats, cdr_feat])


def feature_names() -> list:
    """Ordered list of all 45 feature names."""
    return colour_feature_names() + lbp_feature_names() + cdr_feature_names()


# ─────────────────────────────────────────────────────────────────────
# BATCH EXTRACTOR  —  full DataFrame  →  feature matrix
# ─────────────────────────────────────────────────────────────────────
def _extract_row(row: pd.Series) -> np.ndarray:
    """Picklable worker for joblib parallel execution."""
    cdr_gt = row.get("cdr_mean", None)
    if cdr_gt is not None and pd.isna(cdr_gt):
        cdr_gt = None
    return extract_features(row["image_path"], cdr_gt=cdr_gt)


def build_feature_matrix(
    df: pd.DataFrame,
    n_jobs: int = -1,
    desc: str = "Extracting features",
) -> tuple:
    """
    Extract features for every image in a DataFrame in parallel.

    Args:
        df:     DataFrame with [image_path, label].
                If 'cdr_mean' column exists (DRISHTI), it is used as CDR.
        n_jobs: Parallel workers. -1 = all CPU cores.
        desc:   tqdm progress bar description.

    Returns:
        X      — np.ndarray (n_samples, 45)  feature matrix
        y      — np.ndarray (n_samples,)     integer labels
        names  — list[str]                   feature names (length 45)
        valid  — pd.DataFrame                rows that were processed OK
    """
    rows = list(df.itertuples(index=False))
    # Convert namedtuples to dicts so _extract_row can use .get()
    row_dicts = [r._asdict() for r in rows]

    if n_jobs == 1:
        results = [_extract_row(r) for r in tqdm(row_dicts, desc=desc)]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_extract_row)(r) for r in tqdm(row_dicts, desc=desc)
        )

    valid_mask = [r is not None for r in results]
    n_failed   = valid_mask.count(False)
    if n_failed:
        print(f"  Warning: {n_failed} images failed extraction and were skipped.")

    good_results = [r for r, ok in zip(results, valid_mask) if ok]
    valid_df     = df[valid_mask].reset_index(drop=True)

    # Drop any rows whose label is NaN before casting to int
    label_nan_mask = valid_df["label"].isna()
    if label_nan_mask.any():
        n_dropped = label_nan_mask.sum()
        print(f"  Warning: {n_dropped} rows had NaN labels and were dropped.")
        valid_df     = valid_df[~label_nan_mask].reset_index(drop=True)
        good_results = [r for r, drop in zip(good_results, label_nan_mask) if not drop]

    X     = np.stack(good_results).astype(np.float32)
    y     = pd.to_numeric(valid_df["label"], errors="coerce").fillna(0).astype(int)

    print(f"Feature matrix shape : {X.shape}")
    print(f"Class counts         : Normal={int((y==0).sum())}  Glaucoma={int((y==1).sum())}")
    return X, y, feature_names(), valid_df