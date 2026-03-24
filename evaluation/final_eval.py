"""
evaluation/final_eval.py
=========================
Final evaluation utilities for the glaucoma detection paper.

Covers:
- Bootstrap 95% confidence intervals for all metrics
- DeLong's test for AUC comparison between models
- McNemar's test for pairwise classifier comparison
- Cross-dataset performance breakdown
- Publication-ready results table generation
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    f1_score, accuracy_score,
)
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RESULTS_DIR, SEED


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = SEED,
) -> dict:
    """
    Bootstrap 95% CIs for AUC, sensitivity, specificity, F1, accuracy.

    Returns dict keyed by metric → {mean, lower, upper, std}
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    point = {
        "auc":         roc_auc_score(y_true, y_prob),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "accuracy":    accuracy_score(y_true, y_pred),
    }

    boot = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx       = rng.integers(0, n, size=n)
        yt        = y_true[idx]
        yprob     = y_prob[idx]
        ypred     = (yprob >= 0.5).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        cm = confusion_matrix(yt, ypred, labels=[0,1]).ravel()
        if len(cm) < 4:
            continue
        tn_b, fp_b, fn_b, tp_b = cm
        boot["auc"].append(roc_auc_score(yt, yprob))
        boot["sensitivity"].append(tp_b/(tp_b+fn_b) if (tp_b+fn_b)>0 else 0.0)
        boot["specificity"].append(tn_b/(tn_b+fp_b) if (tn_b+fp_b)>0 else 0.0)
        boot["f1"].append(f1_score(yt, ypred, zero_division=0))
        boot["accuracy"].append(accuracy_score(yt, ypred))

    alpha   = 1.0 - ci
    results = {}
    for k in point:
        v = np.array(boot[k])
        results[k] = {
            "mean":  point[k],
            "lower": float(np.percentile(v, 100 * alpha / 2)),
            "upper": float(np.percentile(v, 100 * (1 - alpha / 2))),
            "std":   float(v.std()),
        }
    return results


def delong_auc_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> tuple:
    """
    DeLong's test comparing two AUCs on the same test set.
    Returns (z_statistic, p_value).
    """
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    def _auc_var(yt, yp):
        pos = yp[yt == 1]
        neg = yp[yt == 0]
        n1, n0 = len(pos), len(neg)
        v10 = np.array([np.mean(p > neg) + 0.5*np.mean(p == neg) for p in pos])
        v01 = np.array([np.mean(p < pos) + 0.5*np.mean(p == pos) for p in neg])
        auc   = v10.mean()
        s_var = np.var(v10, ddof=1)/n1 + np.var(v01, ddof=1)/n0
        return auc, s_var

    auc_a, var_a = _auc_var(y_true, y_prob_a)
    auc_b, var_b = _auc_var(y_true, y_prob_b)
    z = (auc_a - auc_b) / np.sqrt(var_a + var_b + 1e-12)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> tuple:
    """
    McNemar's test with Yates continuity correction.
    Returns (chi2_statistic, p_value).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    b = int(np.sum( (y_pred_a == y_true) & (y_pred_b != y_true) ))
    c = int(np.sum( (y_pred_a != y_true) & (y_pred_b == y_true) ))
    chi2 = (abs(b - c) - 1)**2 / (b + c + 1e-12)
    p    = stats.chi2.sf(chi2, df=1)
    return float(chi2), float(p)


def cross_dataset_metrics(
    test_df: pd.DataFrame,
    y_pred_col:  str = "y_pred",
    y_prob_col:  str = "y_prob",
    label_col:   str = "label",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    """
    AUC, sensitivity, specificity, F1, accuracy broken down by dataset.
    Includes an 'Overall' row at the bottom.
    """
    rows = []
    for ds in sorted(test_df[dataset_col].unique()):
        sub = test_df[test_df[dataset_col] == ds]
        if sub[label_col].nunique() < 2:
            continue
        yt    = sub[label_col].values
        yp    = sub[y_pred_col].values.astype(int)
        yprob = sub[y_prob_col].values
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        rows.append({
            "Dataset":     ds,
            "N":           len(sub),
            "Glaucoma":    int(yt.sum()),
            "Normal":      int((yt==0).sum()),
            "AUC":         round(roc_auc_score(yt, yprob), 4),
            "Sensitivity": round(tp/(tp+fn) if (tp+fn)>0 else 0, 4),
            "Specificity": round(tn/(tn+fp) if (tn+fp)>0 else 0, 4),
            "F1":          round(f1_score(yt, yp, zero_division=0), 4),
            "Accuracy":    round(accuracy_score(yt, yp), 4),
        })

    # Overall
    yt    = test_df[label_col].values
    yp    = test_df[y_pred_col].values.astype(int)
    yprob = test_df[y_prob_col].values
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    rows.append({
        "Dataset":     "Overall",
        "N":           len(test_df),
        "Glaucoma":    int(yt.sum()),
        "Normal":      int((yt==0).sum()),
        "AUC":         round(roc_auc_score(yt, yprob), 4),
        "Sensitivity": round(tp/(tp+fn) if (tp+fn)>0 else 0, 4),
        "Specificity": round(tn/(tn+fp) if (tn+fp)>0 else 0, 4),
        "F1":          round(f1_score(yt, yp, zero_division=0), 4),
        "Accuracy":    round(accuracy_score(yt, yp), 4),
    })
    return pd.DataFrame(rows)


def make_results_table(
    model_results: dict,
    ci_results:    dict,
) -> pd.DataFrame:
    """
    Build publication-ready table with 95% CI in each cell.

    model_results: {model_name: {auc, sensitivity, specificity, f1, accuracy}}
    ci_results:    {model_name: output of bootstrap_metrics()}
    """
    rows = []
    for name in model_results:
        m  = model_results[name]
        ci = ci_results.get(name, {})

        def fmt(key):
            val = m.get(key, 0)
            if key in ci:
                lo, hi = ci[key]["lower"], ci[key]["upper"]
                return f"{val:.4f} ({lo:.4f}–{hi:.4f})"
            return f"{val:.4f}"

        rows.append({
            "Model":                name,
            "AUC (95% CI)":         fmt("auc"),
            "Sensitivity (95% CI)": fmt("sensitivity"),
            "Specificity (95% CI)": fmt("specificity"),
            "F1 (95% CI)":          fmt("f1"),
            "Accuracy (95% CI)":    fmt("accuracy"),
        })
    return pd.DataFrame(rows)
