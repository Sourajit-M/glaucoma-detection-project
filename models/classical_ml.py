"""
models/classical_ml.py
=======================
Classical ML pipeline for glaucoma detection.

Models
──────
- Logistic Regression  (linear baseline)
- SVM with RBF kernel  (non-linear, strong on tabular features)
- Random Forest        (ensemble, provides feature importances)

Each model is wrapped in a sklearn Pipeline:
    StandardScaler → (optional PCA) → Classifier

Design decisions for the paper
───────────────────────────────
- Stratified 5-fold CV on the TRAINING split only (no test leakage)
- GridSearchCV for hyperparameter tuning within CV folds
- Metrics: ROC-AUC (primary), Sensitivity, Specificity, F1, Accuracy
- Class-weight balancing for imbalanced datasets
- All results saved to outputs/results/ as CSV + pkl

Usage
─────
    from models.classical_ml import train_all_models, evaluate_on_test

    results = train_all_models(X_train, y_train)
    test_metrics = evaluate_on_test(results, X_test, y_test)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    f1_score,
    accuracy_score,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, RESULTS_DIR, SEED

# ─────────────────────────────────────────────────────────────────────
# PIPELINE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────
def _make_pipelines() -> dict:
    """
    Returns a dict of {model_name: (pipeline, param_grid)}.

    Pipelines always start with imputation + scaling so they are
    safe to use even if a few NaN values slip through extraction.
    """
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]

    pipelines = {

        # ── Logistic Regression ───────────────────────────────────────
        "LogisticRegression": (
            Pipeline(base_steps + [
                ("clf", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="lbfgs",
                )),
            ]),
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
            },
        ),

        # ── SVM (RBF kernel) ─────────────────────────────────────────
        "SVM_RBF": (
            Pipeline(base_steps + [
                ("clf", SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,      # needed for ROC-AUC
                    random_state=SEED,
                )),
            ]),
            {
                "clf__C":     [0.1, 1.0, 10.0, 100.0],
                "clf__gamma": ["scale", "auto"],
            },
        ),

        # ── Random Forest ─────────────────────────────────────────────
        "RandomForest": (
            Pipeline(base_steps + [
                ("clf", RandomForestClassifier(
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                )),
            ]),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth":    [None, 10, 20],
                "clf__min_samples_split": [2, 5],
            },
        ),
    }

    return pipelines


# ─────────────────────────────────────────────────────────────────────
# METRICS HELPER
# ─────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """
    Computes all clinical and ML metrics for one model evaluation.

    Returns dict with:
        auc, sensitivity, specificity, f1, accuracy,
        tp, fp, tn, fn, fpr (for ROC curve)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall for glaucoma
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall for normal

    return {
        "auc":         roc_auc_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1":          f1_score(y_true, y_pred),
        "accuracy":    accuracy_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────
# TRAIN ALL MODELS  (cross-validated tuning on training split)
# ─────────────────────────────────────────────────────────────────────
def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> dict:
    """
    Tune and train all three models using stratified k-fold CV + GridSearch.

    Args:
        X_train:  Feature matrix for the training split (n_samples, 45)
        y_train:  Labels (0=normal, 1=glaucoma)
        cv_folds: Number of CV folds (default 5)

    Returns:
        dict keyed by model name, each value is a dict with:
            'best_estimator'  — fitted Pipeline (best hyperparams)
            'best_params'     — winning hyperparameter dict
            'cv_auc_mean'     — mean AUC across CV folds
            'cv_auc_std'      — std AUC across CV folds
            'cv_results'      — full GridSearchCV cv_results_ DataFrame
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    results = {}

    for name, (pipeline, param_grid) in _make_pipelines().items():
        print(f"\n{'─'*52}")
        print(f"  Training: {name}")
        print(f"{'─'*52}")

        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            refit=True,        # refit best model on full X_train
            return_train_score=False,
        )
        gs.fit(X_train, y_train)

        cv_results_df = pd.DataFrame(gs.cv_results_)

        print(f"  Best params : {gs.best_params_}")
        print(f"  CV AUC      : {gs.best_score_:.4f}")

        results[name] = {
            "best_estimator": gs.best_estimator_,
            "best_params":    gs.best_params_,
            "cv_auc_mean":    gs.best_score_,
            "cv_auc_std":     cv_results_df.loc[gs.best_index_, "std_test_score"],
            "cv_results":     cv_results_df,
        }

        # Save fitted model
        model_path = MODELS_DIR / f"{name}_best.pkl"
        joblib.dump(gs.best_estimator_, model_path)
        print(f"  Saved → {model_path}")

    return results


# ─────────────────────────────────────────────────────────────────────
# EVALUATE ON TEST SPLIT
# ─────────────────────────────────────────────────────────────────────
def evaluate_on_test(
    trained_results: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate all trained models on the held-out test split.

    Args:
        trained_results: Output of train_all_models()
        X_test:          Test feature matrix
        y_test:          Test labels

    Returns:
        DataFrame with one row per model, all metrics as columns.
        Also saved to outputs/results/classical_ml_results.csv
    """
    rows = []
    roc_data = {}   # stored for plotting

    for name, res in trained_results.items():
        model  = res["best_estimator"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": metrics["auc"]}

        row = {
            "model":       name,
            "cv_auc":      f"{res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}",
            **{k: round(v, 4) for k, v in metrics.items()},
            "best_params": str(res["best_params"]),
        }
        rows.append(row)

        print(f"\n[{name}]")
        print(f"  AUC         : {metrics['auc']:.4f}")
        print(f"  Sensitivity : {metrics['sensitivity']:.4f}")
        print(f"  Specificity : {metrics['specificity']:.4f}")
        print(f"  F1          : {metrics['f1']:.4f}")
        print(f"  Accuracy    : {metrics['accuracy']:.4f}")

    results_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = RESULTS_DIR / "classical_ml_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # Save ROC curve data for plotting
    joblib.dump(roc_data, RESULTS_DIR / "classical_ml_roc_data.pkl")

    return results_df, roc_data


# ─────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE  (Random Forest only)
# ─────────────────────────────────────────────────────────────────────
def get_feature_importances(
    trained_results: dict,
    feature_names: list,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extracts and returns feature importances from the Random Forest model.

    Args:
        trained_results: Output of train_all_models()
        feature_names:   List of feature names (length 45)
        top_n:           How many top features to return

    Returns:
        DataFrame with columns [feature, importance] sorted descending
    """
    if "RandomForest" not in trained_results:
        raise ValueError("RandomForest model not found in trained_results.")

    rf_pipeline = trained_results["RandomForest"]["best_estimator"]
    rf_clf      = rf_pipeline.named_steps["clf"]

    importances = rf_clf.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    # Save
    fi_df.to_csv(RESULTS_DIR / "rf_feature_importances.csv", index=False)
    return fi_df


# ─────────────────────────────────────────────────────────────────────
# LOAD SAVED MODELS
# ─────────────────────────────────────────────────────────────────────
def load_trained_models() -> dict:
    """
    Loads all saved .pkl model files from outputs/models/.
    Useful for re-running evaluation without retraining.

    Returns:
        dict keyed by model name → fitted Pipeline
    """
    model_files = {
        "LogisticRegression": MODELS_DIR / "LogisticRegression_best.pkl",
        "SVM_RBF":            MODELS_DIR / "SVM_RBF_best.pkl",
        "RandomForest":       MODELS_DIR / "RandomForest_best.pkl",
    }
    loaded = {}
    for name, path in model_files.items():
        if path.exists():
            loaded[name] = joblib.load(path)
            print(f"Loaded: {name} ← {path}")
        else:
            print(f"Not found (not yet trained): {path}")
    return loaded
