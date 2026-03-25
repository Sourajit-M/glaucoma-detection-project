"""
models/ensemble.py
==================
Hybrid late-fusion ensemble for glaucoma detection.

Architecture
────────────
Three base predictors feed into a meta-learner:

    CNN (ResNet18)  → P(glaucoma | image)   [deep visual features]
    SVM-RBF         → P(glaucoma | features) [handcrafted colour + LBP]
    U-Net CDR       → CDR value              [structural clinical feature]
         ↓                   ↓                        ↓
    ─────────────────────────────────────────────────────
                     Meta-learner (LR)
                   → Final P(glaucoma)

Why this is novel
─────────────────
Existing glaucoma detection papers use EITHER deep learning OR
handcrafted features OR CDR — rarely all three in a principled fusion.
This ensemble combines:
  - Global image context   (CNN)
  - Interpretable features (SVM + colour/LBP)
  - Clinical biomarker     (CDR from U-Net segmentation)

The meta-learner is a Logistic Regression trained on the stacked
predictions — lightweight, interpretable, and avoids overfitting.

Training protocol
─────────────────
To avoid data leakage, the meta-learner is trained on the VALIDATION
split (which the base models never trained on), and evaluated on the
held-out TEST split (which neither the base models nor meta-learner
have seen during training).

Usage
─────
    from models.ensemble import HybridEnsemble

    ensemble = HybridEnsemble(cnn_model, svm_model, cdr_series)
    ensemble.fit_meta(X_val_meta, y_val)
    metrics = ensemble.evaluate(X_test_meta, y_test)
"""

import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score,
    accuracy_score, roc_curve,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DEVICE, MODELS_DIR, RESULTS_DIR, SEED


# ─────────────────────────────────────────────────────────────────────
# CNN INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────
def get_cnn_probabilities(
    model: torch.nn.Module,
    loader: "DataLoader",
) -> np.ndarray:
    """
    Run CNN inference on a DataLoader and return glaucoma probabilities.

    Args:
        model:  Trained GlaucomaResNet (best checkpoint loaded, eval mode)
        loader: DataLoader for the target split

    Returns:
        np.ndarray (n_samples,) — P(glaucoma) for each image
    """
    from torch.cuda.amp import autocast

    model.eval()
    all_probs = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# FEATURE MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────
def build_meta_features(
    cnn_probs:  np.ndarray,
    svm_probs:  np.ndarray,
    cdr_values: np.ndarray,
) -> np.ndarray:
    """
    Stack CNN probability, SVM probability, and CDR into a 3-column
    meta-feature matrix for the meta-learner.

    Args:
        cnn_probs:  (n,) CNN glaucoma probabilities
        svm_probs:  (n,) SVM glaucoma probabilities
        cdr_values: (n,) U-Net CDR values

    Returns:
        X_meta: np.ndarray (n, 3)
    """
    assert len(cnn_probs) == len(svm_probs) == len(cdr_values), (
        f"Length mismatch: CNN={len(cnn_probs)}, "
        f"SVM={len(svm_probs)}, CDR={len(cdr_values)}"
    )

    # Clamp CDR to [0, 1] as a safety check
    cdr_clamped = np.clip(cdr_values, 0.0, 1.0).astype(np.float32)

    return np.column_stack([
        cnn_probs.astype(np.float32),
        svm_probs.astype(np.float32),
        cdr_clamped,
    ])


# ─────────────────────────────────────────────────────────────────────
# ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────────────
class HybridEnsemble:
    """
    Late-fusion ensemble: CNN + SVM + CDR → Logistic Regression meta-learner.

    Args:
        cnn_model:   Trained GlaucomaResNet (loaded checkpoint)
        svm_model:   Trained SVM pipeline (from classical_ml.py)
        cdr_df:      DataFrame with [image_path, cdr_unet] for all splits
                     (from dataset_with_unet_cdr.csv)
    """

    FEATURE_NAMES = ["CNN_prob", "SVM_prob", "CDR_unet"]

    def __init__(
        self,
        cnn_model,
        svm_model,
        cdr_df: pd.DataFrame,
    ):
        self.cnn_model  = cnn_model
        self.svm_model  = svm_model
        self.cdr_df     = cdr_df.set_index("image_path")

        self.meta_clf   = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )
        self.scaler     = StandardScaler()
        self._fitted    = False

    def _get_cdr(self, image_paths: list) -> np.ndarray:
        """Look up U-Net CDR for each image path. Fallback 0.5 if missing."""
        cdrs = []
        for p in image_paths:
            try:
                cdr = float(self.cdr_df.loc[p, "cdr_unet"])
            except KeyError:
                cdr = 0.5   # fallback for images not in CDR table
            cdrs.append(cdr if not np.isnan(cdr) else 0.5)
        return np.array(cdrs, dtype=np.float32)

    def build_split_features(
        self,
        df_split: pd.DataFrame,
        loader: "DataLoader",
        feature_X: np.ndarray,
    ) -> tuple:
        """
        Build meta-features for one split (val or test).

        Args:
            df_split:  DataFrame for the split with [image_path, label]
            loader:    DataLoader for CNN inference
            feature_X: Classical feature matrix (n, 45) for SVM

        Returns:
            (X_meta, y) — (n, 3) feature matrix and labels
        """
        # CNN probabilities
        cnn_probs = get_cnn_probabilities(self.cnn_model, loader)

        # SVM probabilities
        svm_probs = self.svm_model.predict_proba(feature_X)[:, 1]

        # U-Net CDR values
        cdr_vals  = self._get_cdr(df_split["image_path"].tolist())

        # Align lengths (CNN loader may drop incomplete last batch)
        n = min(len(cnn_probs), len(svm_probs), len(cdr_vals))
        if n < len(df_split):
            print(f"  Note: aligned to {n} samples "
                  f"(dropped {len(df_split)-n} from DataLoader tail).")

        X_meta = build_meta_features(
            cnn_probs[:n], svm_probs[:n], cdr_vals[:n]
        )
        y = df_split["label"].values[:n].astype(int)
        return X_meta, y

    def fit_meta(self, X_meta_val: np.ndarray, y_val: np.ndarray):
        """
        Train the meta-learner on the validation split.

        IMPORTANT: Use ONLY the validation split here — the test split
        must remain completely unseen until final evaluation.

        Args:
            X_meta_val: (n_val, 3) meta-features from validation split
            y_val:      (n_val,) validation labels
        """
        X_scaled = self.scaler.fit_transform(X_meta_val)
        self.meta_clf.fit(X_scaled, y_val)
        self._fitted = True

        coefs = dict(zip(self.FEATURE_NAMES, self.meta_clf.coef_[0]))
        print("Meta-learner trained on validation split.")
        print("  Logistic Regression coefficients:")
        for name, coef in coefs.items():
            print(f"    {name:<12}: {coef:+.4f}")

        # Validation AUC (training performance, not test)
        val_probs = self.scaler.transform(X_meta_val)
        val_probs = self.meta_clf.predict_proba(val_probs)[:, 1]
        val_auc   = roc_auc_score(y_val, val_probs)
        print(f"  Validation AUC (meta-learner): {val_auc:.4f}")

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        """Return P(glaucoma) for the meta-feature matrix."""
        assert self._fitted, "Call fit_meta() before predict_proba()"
        X_scaled = self.scaler.transform(X_meta)
        return self.meta_clf.predict_proba(X_scaled)[:, 1]

    def evaluate(
        self,
        X_meta_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Full evaluation on the test split.

        Returns dict with all metrics + arrays for plotting.
        """
        assert self._fitted, "Call fit_meta() before evaluate()"

        y_prob = self.predict_proba(X_meta_test)
        y_pred = (y_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr, tpr, _    = roc_curve(y_test, y_prob)

        metrics = {
            "auc":         roc_auc_score(y_test, y_prob),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1":          f1_score(y_test, y_pred),
            "accuracy":    accuracy_score(y_test, y_pred),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "fpr":    fpr,
            "tpr":    tpr,
        }
        return metrics

    def save(self, path: Path = None):
        """Save the fitted meta-learner and scaler."""
        if path is None:
            path = MODELS_DIR / "hybrid_ensemble_meta.pkl"
        joblib.dump({"meta_clf": self.meta_clf, "scaler": self.scaler}, path)
        print(f"Ensemble meta-learner saved → {path}")

    def load(self, path: Path = None):
        """Load a previously saved meta-learner and scaler."""
        if path is None:
            path = MODELS_DIR / "hybrid_ensemble_meta.pkl"
        saved = joblib.load(path)
        self.meta_clf = saved["meta_clf"]
        self.scaler   = saved["scaler"]
        self._fitted  = True
        print(f"Loaded ensemble meta-learner ← {path}")
