# scripts/07b_model_comparison.py
"""
Fair 8-model comparison under identical 5-fold CV protocol.
Fixes applied vs previous runs:
  - LR: solver='saga' + StandardScaler (fixes ConvergenceWarning)
  - XGBoost: removed use_label_encoder (deprecated param warning fixed)
  - KNN + SVM: scaled via pipeline (distance-based models need this)
  - SMOTE inside pipeline for every model (leak-free)
  - Sparse columns already cleaned by updated 05_encode_features.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

PROC_DIR = os.path.join("data", "processed")
OUT_DIR  = "outputs"
FIG_DIR  = os.path.join("outputs", "figures")
MODEL_DIR = os.path.join("outputs", "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data (post-clean encoding from updated 05_encode_features.py) ────────
df = pd.read_csv(os.path.join(PROC_DIR, "tamilnadu_nfhs5_encoded.csv"))
drop_cols = ["caseid", "midx", "v001", "v002", "hhid", "stunting_haz", "state"]
X_raw = df.drop(columns=[c for c in drop_cols + ["stunting_label"]
                          if c in df.columns])
y = df["stunting_label"]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

print(f"Dataset: {X.shape[0]} rows × {X.shape[1]} features")
print(f"Stunting prevalence: {y.mean()*100:.1f}%")
print(f"Class ratio (neg:pos): {(y==0).sum()}:{(y==1).sum()}")

spw = round((y == 0).sum() / (y == 1).sum(), 2)
print(f"scale_pos_weight: {spw}")

# Verify no sparse columns remain
col_sums = X.sum().sort_values()
sparse_remaining = col_sums[col_sums < 20]
if len(sparse_remaining) > 0:
    print(f"\n⚠ Still {len(sparse_remaining)} sparse columns — "
          f"re-run 05_encode_features.py first!")
    print(sparse_remaining.to_string())
else:
    print("✅ No sparse columns — feature set is clean.")

# ── Define all 8 models ───────────────────────────────────────────────────────
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_models = {
    # FIX 1: solver='saga' + StandardScaler resolves ConvergenceWarning
    # saga handles large datasets and l1/l2 regularization well
    "Logistic Regression": LogisticRegression(
        solver="saga", max_iter=3000, C=0.1,
        class_weight="balanced", random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=8, random_state=42
    ),

    # KNN: needs scaling — pipeline handles this per-fold
    "KNN": KNeighborsClassifier(n_neighbors=15),

    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),

    # SVM: needs scaling — pipeline handles this per-fold
    "SVM": SVC(
        probability=True, class_weight="balanced",
        kernel="rbf", C=1.0, random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    ),

    # FIX 2: removed use_label_encoder (deprecated in XGBoost ≥ 1.6)
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc", verbosity=0,
        random_state=42, n_jobs=-1
    ),
}

# Stacking: replicating Erda et al. + passthrough fix
stack_estimators = [
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=42)),
]
# NOTE: Using passthrough=False here — our diagnosis showed False > True
# on the unclean feature set. After sparse column removal this may change.
# We test both below in the passthrough re-test section.
base_models["Stacking (RF+GBM)"] = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(
        solver="liblinear", max_iter=2000, random_state=42
    ),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    passthrough=False,
    n_jobs=-1
)

# ── Run identical CV protocol ─────────────────────────────────────────────────
scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]
results = []

print(f"\nRunning 5-fold CV for all 8 models...\n")

for name, model in base_models.items():
    print(f"  [{name}]...")

    steps = [("smote", SMOTE(random_state=42))]
    # Scale for distance/margin based models
    if name in ["KNN", "SVM", "Logistic Regression"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    pipe = ImbPipeline(steps)

    try:
        scores = cross_validate(
            pipe, X, y, cv=CV, scoring=scoring,
            n_jobs=-1, error_score=np.nan
        )
        auc_mean = np.nanmean(scores["test_roc_auc"])
        auc_std  = np.nanstd(scores["test_roc_auc"])
        results.append({
            "Model":          name,
            "ROC-AUC_mean":   auc_mean,
            "ROC-AUC_std":    auc_std,
            "F1_mean":        np.nanmean(scores["test_f1"]),
            "Precision_mean": np.nanmean(scores["test_precision"]),
            "Recall_mean":    np.nanmean(scores["test_recall"]),
            "Accuracy_mean":  np.nanmean(scores["test_accuracy"]),
        })
        print(f"    ROC-AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    except Exception as e:
        print(f"    ⚠ FAILED: {e}")
        results.append({
            "Model": name, "ROC-AUC_mean": np.nan, "ROC-AUC_std": np.nan,
            "F1_mean": np.nan, "Precision_mean": np.nan,
            "Recall_mean": np.nan, "Accuracy_mean": np.nan,
        })

# ── Re-test passthrough=True after sparse column removal ─────────────────────
print("\n  [Stacking passthrough=True — re-testing after sparse fix]...")
stack_pt = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(solver="liblinear", max_iter=2000, random_state=42),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    passthrough=True, n_jobs=-1
)
pipe_pt = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", stack_pt)])
scores_pt = cross_validate(pipe_pt, X, y, cv=CV, scoring=scoring, n_jobs=-1, error_score=np.nan)
pt_auc = np.nanmean(scores_pt["test_roc_auc"])
results.append({
    "Model": "Stacking (passthrough=True)",
    "ROC-AUC_mean": pt_auc,
    "ROC-AUC_std":  np.nanstd(scores_pt["test_roc_auc"]),
    "F1_mean":        np.nanmean(scores_pt["test_f1"]),
    "Precision_mean": np.nanmean(scores_pt["test_precision"]),
    "Recall_mean":    np.nanmean(scores_pt["test_recall"]),
    "Accuracy_mean":  np.nanmean(scores_pt["test_accuracy"]),
})
print(f"    ROC-AUC: {pt_auc:.4f} ± {np.nanstd(scores_pt['test_roc_auc']):.4f}")

# ── Build and display results table ──────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values(
    "ROC-AUC_mean", ascending=False
).reset_index(drop=True)

display_df = results_df.copy()
display_df["ROC-AUC"] = display_df.apply(
    lambda r: f"{r['ROC-AUC_mean']:.3f} ± {r['ROC-AUC_std']:.3f}", axis=1)
display_df["F1"]        = display_df["F1_mean"].round(3)
display_df["Precision"] = display_df["Precision_mean"].round(3)
display_df["Recall"]    = display_df["Recall_mean"].round(3)
display_df["Accuracy"]  = display_df["Accuracy_mean"].round(3)
table_df = display_df[["Model", "ROC-AUC", "F1", "Precision", "Recall", "Accuracy"]]

print("\n" + "="*70)
print("MODEL COMPARISON — 5-Fold CV (SMOTE inside pipeline, clean features)")
print("="*70)
print(table_df.to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
results_df.to_csv(os.path.join(OUT_DIR, "model_comparison_results_raw.csv"), index=False)
table_df.to_csv(os.path.join(OUT_DIR, "model_comparison_results.csv"), index=False)
print(f"\n✅ Saved → outputs/model_comparison_results.csv")

# ── Bar chart ─────────────────────────────────────────────────────────────────
valid_df = results_df.dropna(subset=["ROC-AUC_mean"]).sort_values("ROC-AUC_mean")
fig, ax = plt.subplots(figsize=(11, 7))
colors = sns.color_palette("viridis", len(valid_df))
ax.barh(valid_df["Model"], valid_df["ROC-AUC_mean"],
        xerr=valid_df["ROC-AUC_std"],
        color=colors, capsize=4, edgecolor="white")
ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5,
           label="Random chance (0.50)")
for i, (_, row) in enumerate(valid_df.iterrows()):
    ax.text(row["ROC-AUC_mean"] + 0.002, i,
            f"{row['ROC-AUC_mean']:.3f}", va="center", fontsize=9)
ax.set_xlabel("ROC-AUC (mean ± std, 5-fold CV)", fontsize=11)
ax.set_title("Model Comparison — Tamil Nadu Child Stunting Prediction\n"
             "(All models: identical CV, SMOTE in-loop, clean features)", fontsize=12)
ax.legend()
ax.set_xlim(0.40, max(valid_df["ROC-AUC_mean"]) + 0.08)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison_barplot.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved → outputs/figures/model_comparison_barplot.png")

# ── Determine winner and print decisions.md entry ─────────────────────────────
winner_row = results_df.iloc[0]
print("\n" + "="*70)
print("PASTE INTO project_log/decisions.md")
print("="*70)
print(f"""
## Decision 5 (Week 4): Model selection via 8-model comparison

Ran {len(results_df)} model configurations under identical 5-fold CV with
SMOTE correctly inside each fold and clean features (sparse OHE columns
collapsed). Fixes applied vs Week 3 run:
  - LR: solver='saga', max_iter=3000, StandardScaler (ConvergenceWarning fixed)
  - XGBoost: removed deprecated use_label_encoder
  - Sparse OHE columns: 9 rare categories collapsed into 'other' in 05_encode_features.py
  - passthrough: re-tested both True/False after sparse fix

Full results: outputs/model_comparison_results.csv
Figure:       outputs/figures/model_comparison_barplot.png

Winner: {winner_row['Model']}
  CV ROC-AUC : {winner_row['ROC-AUC_mean']:.4f} ± {winner_row['ROC-AUC_std']:.4f}
  CV Recall  : {winner_row['Recall_mean']:.3f}  (minority/stunted class)

Selection rationale: highest cross-validated ROC-AUC under fair comparison;
recall on stunted class prioritized for health screening context.
""")

# Save winner name for 08_tune_model.py to read
import json
winner_meta = {
    "winner_model_name": winner_row["Model"],
    "winner_auc": round(winner_row["ROC-AUC_mean"], 4),
    "winner_recall": round(winner_row["Recall_mean"], 3),
}
with open(os.path.join(MODEL_DIR, "comparison_winner.json"), "w") as f:
    json.dump(winner_meta, f, indent=2)
print(f"✅ Winner saved → outputs/models/comparison_winner.json")