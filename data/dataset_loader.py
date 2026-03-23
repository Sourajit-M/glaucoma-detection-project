"""
data/dataset_loader.py
======================
Unified dataset loading for all four glaucoma datasets.

Actual folder structures on disk
─────────────────────────────────
ACRIMA/
  database/
    Im001_ACRIMA.jpg          ← normal   (no '_g_' in stem)
    Im686_g_ACRIMA.jpg        ← glaucoma ('_g_' in stem)

DRISHTI-GS1/
  Drishti_GS_trainingData/
    Images/                   ← fundus PNGs
    GT/                       ← mask PNGs live directly here
  Drishti_GS_testingData/
    Images/
    GT/

RIM-ONE_DL/
  Train/
    Glaucoma/
    Normal/
  Test/
    Glaucoma/
    Normal/

EyePACS-AIROGS/
  metadata.csv
  train/
    RG/                       ← Referable Glaucoma
    NRG/                      ← Non-Referable (Normal)
  validation/
    RG/
    NRG/
  test/
    RG/
    NRG/

All classification loaders return a DataFrame with columns:
    image_path | label | dataset | split

label encoding:  1 = glaucoma,  0 = normal
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASETS, SPLIT_RATIOS, SEED


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────
# ACRIMA  –  label encoded in filename  (_g_ = glaucoma)
# ─────────────────────────────────────────────────────────────────────
def load_acrima() -> pd.DataFrame:
    """
    Loads ACRIMA dataset.
    Label rule:  '_g_' in filename stem → glaucoma (1), otherwise normal (0).
    Applies a stratified 70/15/15 train-val-test split (no predefined split).
    """
    cfg  = DATASETS["ACRIMA"]
    root = Path(cfg["root"])

    _check_path(root, "ACRIMA database folder")

    records = []
    for fpath in sorted(root.iterdir()):
        if fpath.suffix.lower() not in VALID_EXT:
            continue
        label = 1 if "_g_" in fpath.stem else 0
        records.append({
            "image_path": str(fpath),
            "label":      label,
            "dataset":    "ACRIMA",
        })

    if not records:
        raise RuntimeError(
            f"No images found in {root}. "
            "Make sure you extracted database.zip into ACRIMA/database/"
        )

    df = pd.DataFrame(records)
    df = _assign_splits(df)
    _print_summary("ACRIMA", df)
    return df


# ─────────────────────────────────────────────────────────────────────
# RIM-ONE DL  –  label from folder name  (Glaucoma / Normal)
# ─────────────────────────────────────────────────────────────────────
def load_rimone() -> pd.DataFrame:
    """
    Loads RIM-ONE DL dataset.
    Predefined Train/Test split from folder structure.
    A validation split is carved out of Train (stratified).
    """
    cfg = DATASETS["RIMONE"]
    records = []

    for split_name, root_key in [("train", "train_root"), ("test", "test_root")]:
        root = Path(cfg[root_key])
        _check_path(root, f"RIM-ONE {split_name} root")

        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            folder = class_dir.name.lower()
            if "glaucoma" in folder:
                label = 1
            elif "normal" in folder:
                label = 0
            else:
                print(f"  [RIMONE] Skipping unrecognised subfolder: '{class_dir.name}'")
                continue

            for fpath in sorted(class_dir.iterdir()):
                if fpath.suffix.lower() not in VALID_EXT:
                    continue
                records.append({
                    "image_path": str(fpath),
                    "label":      label,
                    "dataset":    "RIMONE",
                    "split":      split_name,
                })

    df = pd.DataFrame(records)

    # Carve validation out of train
    train_mask = df["split"] == "train"
    val_frac   = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])
    train_idx, val_idx = train_test_split(
        df[train_mask].index,
        test_size=val_frac,
        stratify=df.loc[train_mask, "label"],
        random_state=SEED,
    )
    df.loc[val_idx, "split"] = "val"

    _print_summary("RIMONE", df)
    return df


# ─────────────────────────────────────────────────────────────────────
# EyePACS-AIROGS  –  label from folder name  (RG / NRG)
# ─────────────────────────────────────────────────────────────────────
def load_airogs(max_samples: int = None) -> pd.DataFrame:
    """
    Loads EyePACS-AIROGS-light-v2.

    Folder layout:
        train/RG/, train/NRG/
        validation/RG/, validation/NRG/
        test/RG/,  test/NRG/

    RG  (Referable Glaucoma)  → label 1
    NRG (Non-Referable)       → label 0

    Args:
        max_samples: Cap total TRAIN images (val and test are never capped).
                     e.g. max_samples=2000 → up to 1000 RG + 1000 NRG from train.
    """
    cfg        = DATASETS["AIROGS"]
    pos_folder = cfg["positive_folder"]   # "RG"

    split_map = {
        "train": Path(cfg["train_root"]),
        "val":   Path(cfg["val_root"]),
        "test":  Path(cfg["test_root"]),
    }

    records = []
    for split_name, root in split_map.items():
        if not root.exists():
            print(f"  [AIROGS] Warning: {split_name} folder not found → {root}, skipping.")
            continue

        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = 1 if class_dir.name.upper() == pos_folder.upper() else 0

            for fpath in sorted(class_dir.iterdir()):
                if fpath.suffix.lower() not in VALID_EXT:
                    continue
                records.append({
                    "image_path": str(fpath),
                    "label":      label,
                    "dataset":    "AIROGS",
                    "split":      split_name,
                })

    df = pd.DataFrame(records)

    # Optional cap — TRAIN only
    if max_samples is not None:
        train_df = df[df["split"] == "train"]
        other_df = df[df["split"] != "train"]
        n_per_class = max_samples // 2
        train_capped = (
            train_df
            .groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), n_per_class), random_state=SEED))
            .reset_index(drop=True)
        )
        df = pd.concat([train_capped, other_df], ignore_index=True)

    _print_summary("AIROGS", df)
    return df


# ─────────────────────────────────────────────────────────────────────
# DRISHTI-GS1  –  segmentation dataset (OD + cup masks + CDR)
#
# Actual GT layout (confirmed from your download):
#   GT/
#     <stem>/
#       SoftMap/
#         <stem>_ODsegSoftmap.png    ← optic disc soft mask
#         <stem>_cupsegSoftmap.png   ← cup soft mask
#       <stem>_cdrValues.txt         ← pre-computed CDR values (4 raters)
#
# NOTE: Test GT does not exist in this dataset release — DRISHTI test
# is a blind evaluation set. Only the training split has ground truth.
# ─────────────────────────────────────────────────────────────────────
def load_drishti_segmentation() -> pd.DataFrame:
    """
    Loads DRISHTI-GS1 training set for segmentation and CDR tasks.

    Returns DataFrame:
        image_path | od_mask_path | cup_mask_path | cdr_path | cdr_mean | split | dataset

    cdr_mean is the mean CDR across the 4 rater annotations stored in
    <stem>_cdrValues.txt. Use this directly instead of computing CDR
    from masks — it is more reliable as it averages expert opinions.
    """
    cfg     = DATASETS["DRISHTI"]
    img_dir = Path(cfg["train_images"])
    gt_dir  = Path(cfg["train_gt"])

    _check_path(img_dir, "DRISHTI training Images folder")
    _check_path(gt_dir,  "DRISHTI training GT folder")

    records = []
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem   # e.g. "drishtiGS_002"

        softmap_dir = gt_dir / stem / "SoftMap"

        od_mask  = softmap_dir / f"{stem}_ODsegSoftmap.png"
        cup_mask = softmap_dir / f"{stem}_cupsegSoftmap.png"
        cdr_file = gt_dir / stem / f"{stem}_cdrValues.txt"

        # Parse mean CDR from the txt file (4 rater values, one per line)
        cdr_mean = _parse_cdr_txt(cdr_file) if cdr_file.exists() else None

        records.append({
            "image_path":    str(img_path),
            "od_mask_path":  str(od_mask)  if od_mask.exists()  else None,
            "cup_mask_path": str(cup_mask) if cup_mask.exists() else None,
            "cdr_path":      str(cdr_file) if cdr_file.exists() else None,
            "cdr_mean":      cdr_mean,
            "split":         "train",
            "dataset":       "DRISHTI",
        })

    df = pd.DataFrame(records)

    # Report
    missing_od  = df["od_mask_path"].isna().sum()
    missing_cup = df["cup_mask_path"].isna().sum()
    missing_cdr = df["cdr_mean"].isna().sum()
    n_valid     = (df["od_mask_path"].notna() & df["cup_mask_path"].notna()).sum()

    print(f"[DRISHTI] Train images : {len(df)}")
    print(f"          With OD mask : {len(df) - missing_od}")
    print(f"          With cup mask: {len(df) - missing_cup}")
    print(f"          With CDR     : {len(df) - missing_cdr}  "
          f"(mean CDR = {df['cdr_mean'].mean():.3f})")
    print(f"          Fully valid  : {n_valid}")
    print(f"  Note: test split has no public GT — train-only for segmentation.")
    return df


def _parse_cdr_txt(cdr_path: Path) -> float:
    """
    Parse a DRISHTI CDR values file.
    File contains one float per line (one value per rater, typically 4 raters).
    Returns the mean across all rater values.
    Returns None if the file cannot be parsed.
    """
    try:
        with open(cdr_path) as f:
            values = [float(v) for v in f.read().split()]
        return float(np.mean(values)) if values else None
    except Exception:
        return None


def debug_drishti_gt():
    """
    Diagnostic helper — prints the first 20 files found in your GT folders.
    Call this if load_drishti_segmentation() reports missing masks.
    """
    cfg = DATASETS["DRISHTI"]
    for label, gt_key in [("Train GT", "train_gt"), ("Test GT", "test_gt")]:
        gt_dir = Path(cfg[gt_key])
        if not gt_dir.exists():
            print(f"  {label}: NOT FOUND at {gt_dir}")
            continue
        all_files = sorted(gt_dir.rglob("*.*"))
        print(f"\n{label} ({gt_dir})  — {len(all_files)} files, showing first 20:")
        for f in all_files[:20]:
            print(f"    {f.relative_to(gt_dir)}")


# ─────────────────────────────────────────────────────────────────────
# COMBINED LOADER
# ─────────────────────────────────────────────────────────────────────
def load_all_datasets(
    include: list = None,
    airogs_max_samples: int = 2000,
) -> pd.DataFrame:
    """
    Load and merge classification datasets into one DataFrame.

    Args:
        include:            Dataset names to include.
                            Default: ["ACRIMA", "RIMONE", "AIROGS"]
        airogs_max_samples: Cap on AIROGS train images (val + test unaffected).

    Returns:
        Combined DataFrame  [image_path, label, dataset, split]
    """
    if include is None:
        include = ["ACRIMA", "RIMONE", "AIROGS"]

    loaders = {
        "ACRIMA": load_acrima,
        "RIMONE": load_rimone,
        "AIROGS": lambda: load_airogs(max_samples=airogs_max_samples),
    }

    frames = []
    for name in include:
        if name not in loaders:
            print(f"  Warning: '{name}' is not a recognised dataset name.")
            continue
        try:
            frames.append(loaders[name]())
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [SKIP] {name}: {e}")

    if not frames:
        raise RuntimeError(
            "No datasets were loaded. "
            "Check ROOT_DATA_DIR in config.py and your folder structure."
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n{'─'*52}")
    print(f"  COMBINED  Total   : {len(combined)}")
    print(f"            Glaucoma: {int(combined['label'].sum())}")
    print(f"            Normal  : {int((combined['label']==0).sum())}")
    print(f"{'─'*52}")
    return combined


# ─────────────────────────────────────────────────────────────────────
# SPLIT HELPER
# ─────────────────────────────────────────────────────────────────────
def get_dataset_splits(
    df: pd.DataFrame,
    dataset: str = None,
) -> tuple:
    """
    Returns (train_df, val_df, test_df).

    Args:
        df:      Full or filtered DataFrame.
        dataset: Optional — filter to one dataset before splitting.
    """
    if dataset:
        df = df[df["dataset"] == dataset].copy()
    return (
        df[df["split"] == "train"].reset_index(drop=True),
        df[df["split"] == "val"].reset_index(drop=True),
        df[df["split"] == "test"].reset_index(drop=True),
    )


def print_dataset_summary(df: pd.DataFrame):
    """Formatted per-dataset × per-split breakdown table."""
    W = 12
    print(f"\n{'Dataset':<{W}} {'Split':<8} {'Glaucoma':>10} {'Normal':>8} {'Total':>8}")
    print("─" * 52)
    for dataset in sorted(df["dataset"].unique()):
        for split in ["train", "val", "test"]:
            sub = df[(df["dataset"] == dataset) & (df["split"] == split)]
            if len(sub) == 0:
                continue
            g = int(sub["label"].sum())
            n = int((sub["label"] == 0).sum())
            print(f"{dataset:<{W}} {split:<8} {g:>10} {n:>8} {len(sub):>8}")
    print("─" * 52)
    g = int(df["label"].sum())
    n = int((df["label"] == 0).sum())
    print(f"{'ALL':<{W}} {'all':<8} {g:>10} {n:>8} {len(df):>8}\n")


# ─────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────
def _assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/15/15 split for datasets without a predefined split."""
    train_df, temp_df = train_test_split(
        df,
        train_size=SPLIT_RATIOS["train"],
        stratify=df["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]),
        stratify=temp_df["label"],
        random_state=SEED,
    )
    df = df.copy()
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index,   "split"] = "val"
    df.loc[test_df.index,  "split"] = "test"
    return df


def _check_path(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n  {path}\n"
            "Check ROOT_DATA_DIR in config.py and verify your folder structure."
        )


def _print_summary(name: str, df: pd.DataFrame):
    g = int(df["label"].sum())
    n = int((df["label"] == 0).sum())
    print(f"[{name:<6}]  Total: {len(df):>5}  |  "
          f"Glaucoma: {g:>4}  |  Normal: {n:>4}")