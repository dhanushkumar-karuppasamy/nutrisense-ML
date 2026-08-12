# scripts/07_diagnose_baseline.py
"""
Diagnose why Week 3 stacking ensemble AUC was 0.59.
Three hypotheses tested in order:
  (a) SMOTE-before-CV leakage
  (b) passthrough=False starving the meta-learner
  (c) Sparse one-hot columns adding noise to tree splits
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

PROC_DIR = os.path.join("data", "processed")
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def load_data():
    df = pd.read_csv(os.path.join(PROC_DIR, "tamilnadu_nfhs5_encoded.csv"))
    drop_cols = ["caseid", "midx", "v001", "v002", "hhid", "stunting_haz", "state"]
    X = df.drop(columns=[c for c in drop_cols + ["stunting_label"] if c in df.columns])
    y = df["stunting_label"]
    # Impute here for diagnosis purposes (imputer fitted on full X — acceptable
    # for diagnosis scripts; in production pipelines, impute inside CV loop)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imp, y

def make_stacking(passthrough=False):
    estimators = [
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            solver="liblinear", max_iter=2000, random_state=42
        ),
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        passthrough=passthrough,
        n_jobs=-1
    )

def diagnose_a_leakage(X, y):
    """
    HYPOTHESIS A: SMOTE applied once before StackingClassifier.fit() leaks
    synthetic samples across its internal CV folds. Fix: put SMOTE INSIDE
    the cross-validation loop using ImbPipeline.
    """
    print("\n" + "="*60)
    print("DIAGNOSIS A - SMOTE+CV Leakage")
    print("="*60)

    # Leak-free: SMOTE inside the CV loop
    stacking = make_stacking(passthrough=False)
    pipeline_fixed = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   stacking)
    ])
    scores_fixed = cross_val_score(
        pipeline_fixed, X, y, cv=CV, scoring="roc_auc", n_jobs=-1
    )
    print(f"  Leak-free CV ROC-AUC  : {scores_fixed.mean():.4f} +/- {scores_fixed.std():.4f}")
    print(f"  Week 3 reported AUC   : 0.5900  (single train/test split, SMOTE outside CV)")

    if scores_fixed.mean() < 0.57:
        print("  -> CONFIRMED: Leak-free AUC is near chance. Leakage was masking a")
        print("    genuinely weak model. BOTH leakage AND model quality need fixing.")
    elif scores_fixed.mean() < 0.63:
        print("  -> PARTIAL: Some leakage inflation, but the model is still weak.")
        print("    Proceed to Diagnosis B and XGBoost benchmark.")
    else:
        print("  -> Model is actually decent; leakage was the primary culprit.")

    return scores_fixed.mean()

def diagnose_b_passthrough(X, y):
    """
    HYPOTHESIS B: passthrough=False gives meta-learner only 2 features
    (RF_prob, GBM_prob). With correlated tree models, there's almost no
    signal to combine. passthrough=True adds all 42 original features.
    """
    print("\n" + "="*60)
    print("DIAGNOSIS B - passthrough=False vs passthrough=True")
    print("="*60)

    for pt in [False, True]:
        stacking = make_stacking(passthrough=pt)
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf",   stacking)
        ])
        scores = cross_val_score(pipeline, X, y, cv=CV, scoring="roc_auc", n_jobs=-1)
        label = "passthrough=True " if pt else "passthrough=False"
        print(f"  {label} -> {scores.mean():.4f} +/- {scores.std():.4f}")

def diagnose_c_sparse_columns(X, y):
    """
    HYPOTHESIS C: Rare one-hot categories create near-empty columns.
    Columns with <20 non-zero entries in ~4700 rows are essentially noise
    for tree splits — they get tried as split candidates but never split
    on real signal.
    """
    print("\n" + "="*60)
    print("DIAGNOSIS C - Sparse One-Hot Columns")
    print("="*60)

    col_sums = X.sum().sort_values()
    sparse = col_sums[col_sums < 20]
    print(f"  Columns with <20 non-zero entries ({len(sparse)} total):")
    if len(sparse) > 0:
        print(sparse.to_string())
        print(f"\n  Recommendation: collapse these rare categories into 'other'")
        print(f"  in script 05_encode_features.py before re-running models.")
    else:
        print("  No severely sparse columns found. Encoding looks clean.")

    # Also show the bottom 10 regardless
    print(f"\n  Bottom 10 columns by non-zero count:")
    print(col_sums.head(10).to_string())

    return sparse.index.tolist()

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data()
    print(f"  Shape: {X.shape}  |  Stunting rate: {y.mean()*100:.1f}%")

    auc_a = diagnose_a_leakage(X, y)
    diagnose_b_passthrough(X, y)
    sparse_cols = diagnose_c_sparse_columns(X, y)

    print("\n" + "="*60)
    print("SUMMARY - Record these in project_log/decisions.md")
    print("="*60)
    print(f"  Leak-free stacking AUC  : {auc_a:.4f}")
    print(f"  Sparse columns to drop  : {len(sparse_cols)}")
    print("  -> Run scripts/08_tune_model.py next (XGBoost benchmark)")