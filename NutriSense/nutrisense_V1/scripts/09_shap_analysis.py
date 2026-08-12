# scripts/09_shap_analysis.py
"""
SHAP explainability — run only after best_model.pkl achieves AUC >= 0.65.
Produces:
  outputs/figures/shap_global_summary.png
  outputs/figures/shap_local_waterfall_example.png
  outputs/figures/shap_beeswarm.png
"""
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

MODEL_DIR  = os.path.join("outputs", "models")
FIGURE_DIR = os.path.join("outputs", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SHAP explainability for the best saved model."
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=float(os.getenv("SHAP_MIN_AUC", "0.65")),
        help="Minimum test AUC required before SHAP runs (default: 0.65 or SHAP_MIN_AUC env var)."
    )
    parser.add_argument(
        "--allow-low-auc",
        action="store_true",
        help="Proceed even if model test AUC is below --min-auc."
    )
    return parser.parse_args()


args = parse_args()

# ── Load artefacts ────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
    meta = json.load(f)

if meta["test_auc"] < args.min_auc:
    if not args.allow_low_auc:
        raise ValueError(
            f"Test AUC is {meta['test_auc']:.4f} — below {args.min_auc:.2f} threshold. "
            "Fix the model first (scripts/08_tune_model.py), or re-run this script with "
            "--allow-low-auc if this is exploratory analysis only."
        )
    print(
        f"WARNING: Model AUC {meta['test_auc']:.4f} is below threshold {args.min_auc:.2f}. "
        "Proceeding due to --allow-low-auc."
    )

print(f"Model AUC: {meta['test_auc']:.4f} — proceeding with SHAP analysis.")

best_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
X_test     = pd.read_csv(os.path.join(MODEL_DIR, "X_test.csv"))
y_test     = pd.read_csv(os.path.join(MODEL_DIR, "y_test.csv")).squeeze()

with open(os.path.join(MODEL_DIR, "feature_names.txt")) as f:
    feature_names = [line.strip() for line in f.readlines()]

X_test.columns = feature_names[:len(X_test.columns)]

# ── Get the actual XGBoost model if wrapped in a pipeline ────────────────────
# ImbPipeline wraps the model; SHAP's TreeExplainer needs the raw estimator.
if hasattr(best_model, "named_steps"):
    clf = best_model.named_steps.get("clf", best_model)
else:
    clf = best_model

# For StackingClassifier, use the RF base estimator for SHAP
if hasattr(clf, "estimators_"):
    print("Stacking detected — using RF base estimator for SHAP TreeExplainer.")
    shap_model = clf.estimators_[0]  # RF
else:
    shap_model = clf  # XGBoost directly

# ── Compute SHAP values ───────────────────────────────────────────────────────
print("Computing SHAP values (this takes 1-3 minutes)...")
explainer   = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_test)

# For binary XGBoost, shap_values shape is (n, features); for RF it's list[2]
if isinstance(shap_values, list):
    sv = shap_values[1]  # class 1 (stunted)
else:
    sv = shap_values

# ── Plot 1: Global Summary (bar) ──────────────────────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_test, plot_type="bar", max_display=15, show=False)
plt.title("SHAP Feature Importance — Stunting Risk (Global)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "shap_global_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ shap_global_summary.png saved")

# ── Plot 2: Beeswarm (direction of effects) ───────────────────────────────────
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_test, max_display=15, show=False)
plt.title("SHAP Beeswarm — Feature Directionality", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ shap_beeswarm.png saved")

# ── Plot 3: Local Waterfall — pick a True Positive (TP) ──────────────────────
# Find a child the model correctly flags as stunted
if hasattr(best_model, "named_steps"):
    y_pred = best_model.predict(X_test)
else:
    y_pred = best_model.predict(X_test)

tp_indices = np.where((y_test.values == 1) & (y_pred == 1))[0]
if len(tp_indices) == 0:
    print("⚠ No true positives found — using first stunted child instead.")
    tp_indices = np.where(y_test.values == 1)[0]

tp_idx = tp_indices[0]
print(f"\nTP example — child index {tp_idx} (stunted and correctly flagged)")

exp_val = explainer.expected_value
if isinstance(exp_val, list):
    exp_val = exp_val[1]

explanation = shap.Explanation(
    values      = sv[tp_idx],
    base_values = exp_val,
    data        = X_test.iloc[tp_idx].values,
    feature_names = X_test.columns.tolist()
)
plt.figure()
shap.waterfall_plot(explanation, max_display=15, show=False)
plt.title(f"SHAP Waterfall — Child #{tp_idx} (True Positive)", fontsize=11)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURE_DIR, "shap_local_waterfall_example.png"),
    dpi=150, bbox_inches="tight"
)
plt.close()
print("✅ shap_local_waterfall_example.png saved")

# ── Print top 10 features (for decisions.md) ─────────────────────────────────
mean_abs_shap = pd.Series(
    np.abs(sv).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

print("\n--- Top 15 Features by Mean |SHAP| ---")
print(mean_abs_shap.head(15).to_string())
print("\n✅ All SHAP figures saved to outputs/figures/")
print("→ Add to project_log/figures_index.md")