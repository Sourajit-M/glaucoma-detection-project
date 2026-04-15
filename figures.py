import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

RESULTS_DIR = Path("outputs/results")


def print_metric(label: str, results: dict) -> None:
    """Print accuracy when available, otherwise fall back to other saved metrics."""
    if "accuracy" in results:
        print(f"{label:<24}: {results['accuracy']:.4f}")
        return

    if "auc" in results:
        print(f"{label:<24}: accuracy not saved (AUC: {results['auc']:.4f})")
        return

    available = ", ".join(sorted(results.keys()))
    print(f"{label:<24}: accuracy not saved (available keys: {available})")

# Classical ML
classical_df = pd.read_csv(RESULTS_DIR / "classical_ml_results.csv")
print("Classical ML accuracy:")
print(classical_df[["model", "accuracy"]].to_string(index=False))

# CNN ResNet18
cnn = joblib.load(RESULTS_DIR / "cnn_test_results.pkl")
print()
print_metric("ResNet18 accuracy", cnn)

# EfficientNet
eff = joblib.load(RESULTS_DIR / "efficientnet_test_results.pkl")
print_metric("EfficientNet accuracy", eff)

# Ensemble / proposed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("outputs/results")

# Load saved meta features from notebook 08
cache    = joblib.load(RESULTS_DIR / "features_cache.pkl")
cnn_res  = joblib.load(RESULTS_DIR / "cnn_test_results.pkl")
eff_res  = joblib.load(RESULTS_DIR / "efficientnet_test_results.pkl")

# Rebuild the CNN+CDR meta features (cols 0 and 2 from the ablation)
# Load the full meta feature matrix
import pandas as pd
df_cdr = pd.read_csv(RESULTS_DIR / "dataset_with_unet_cdr.csv")

valid_df  = cache["valid_df"]
y_all     = cache["y"]

val_mask  = valid_df["split"] == "val"
test_mask = valid_df["split"] == "test"

# CDR values aligned to test split
test_paths = valid_df[test_mask]["image_path"].values
val_paths  = valid_df[val_mask]["image_path"].values

cdr_lookup = df_cdr.set_index("image_path")["cdr_unet"]

def get_cdr(paths):
    return np.array([
        float(cdr_lookup.loc[p]) if p in cdr_lookup.index else 0.5
        for p in paths
    ], dtype=np.float32)

# CNN probs from saved results (already aligned to test split)
cnn_prob_test = np.array(cnn_res["y_prob"])
cnn_prob_val  = np.array(cnn_res.get("y_prob_val",
                 # fallback: use first len(val) entries if stored together
                 cnn_res["y_prob"]))

cdr_test = get_cdr(test_paths)
cdr_val  = get_cdr(val_paths)

y_val  = y_all[val_mask]
y_test = y_all[test_mask]

# Build meta features: CNN prob + CDR (cols 0 and 2)
X_val_meta  = np.column_stack([cnn_prob_val[:len(y_val)],  cdr_val[:len(y_val)]])
X_test_meta = np.column_stack([cnn_prob_test[:len(y_test)], cdr_test[:len(y_test)]])

scaler = StandardScaler()
clf    = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=1000, random_state=42)
clf.fit(scaler.fit_transform(X_val_meta), y_val)

y_pred = clf.predict(scaler.transform(X_test_meta))
acc    = accuracy_score(y_test[:len(y_pred)], y_pred)
print(f"CNN+CDR accuracy: {acc:.4f}")
