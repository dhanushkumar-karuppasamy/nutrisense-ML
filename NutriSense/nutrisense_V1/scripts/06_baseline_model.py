# scripts/06_baseline_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

def run_baseline():
    proc_dir  = os.path.join("data", "processed")
    model_dir = os.path.join("outputs", "models")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(proc_dir, "tamilnadu_nfhs5_encoded.csv"))

    # ── Drop non-feature columns ─────────────────────────────────────────────
    drop_cols = ["caseid", "midx", "v001", "v002", "hhid",
                 "stunting_haz", "state"]   # keep stunting_label as target
    X = df.drop(columns=[c for c in drop_cols + ["stunting_label"] if c in df.columns])
    y = df["stunting_label"]

    print(f"Feature matrix: {X.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True).mul(100).round(1)}")

    # ── Stratified train/test split ──────────────────────────────────────────
    # WHY stratified: With 20% stunting, a random split could give test set
    # with 10% or 30% stunting by chance — making metrics unreliable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Train stunting rate: {y_train.mean()*100:.1f}%")
    print(f"Test  stunting rate: {y_test.mean()*100:.1f}%")

    # ── Impute missing feature values before SMOTE / model fitting ─────────
    # SMOTE and the tree models below require fully numeric, non-missing inputs.
    # Fit on training data only to avoid leaking information from the test set.
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)
    print(f"\nRemaining NaNs after imputation: train={int(X_train.isna().sum().sum())}, "
          f"test={int(X_test.isna().sum().sum())}")

    # ── SMOTE on training set only (NEVER on test set) ───────────────────────
    # WHY only on train: Applying SMOTE to test set would inflate your metrics.
    # Synthetic samples should only help the model learn, not pollute evaluation.
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE — Train: {X_train_res.shape[0]} "
          f"(stunted: {y_train_res.sum()}, not: {(y_train_res==0).sum()})")

    # ── Stacking Ensemble (replicating Erda et al. structure) ────────────────
    # Level-1 estimators: Random Forest + Gradient Boosting
    # WHY these two: RF is low-bias, GBM is low-variance; they make errors on 
    # different samples, so stacking them reduces both.
    # Level-2 meta-learner: Logistic Regression
    # WHY LR as meta-learner: Simple, interpretable, avoids overfitting the
    # blended predictions. Erda et al. used LR as their meta-learner.
    estimators = [
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=False,   # meta-learner sees only L1 predictions, not raw features
        n_jobs=-1
    )

    print("\nFitting stacking ensemble...")
    stacking_clf.fit(X_train_res, y_train_res)

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred  = stacking_clf.predict(X_test)
    y_proba = stacking_clf.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["Not Stunted", "Stunted"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, "baseline_stacking.pkl")
    joblib.dump(stacking_clf, model_path)
    print(f"\nModel saved → {model_path}")

    # ── Save feature names for SHAP in Week 4 ───────────────────────────────
    feature_path = os.path.join(model_dir, "feature_names.txt")
    with open(feature_path, "w") as f:
        f.write("\n".join(X.columns.tolist()))
    print(f"Feature names saved → {feature_path}")

if __name__ == "__main__":
    run_baseline()