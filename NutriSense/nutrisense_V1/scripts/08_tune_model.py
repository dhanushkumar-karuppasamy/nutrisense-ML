# scripts/08_tune_model.py
"""
Tunes the model that won 07b_model_comparison.py.
Reads winner from outputs/models/comparison_winner.json.
Saves best_model.pkl for all downstream scripts (09, 10, app/).
"""
import os, json, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV,
                                      train_test_split)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                              f1_score, precision_score, recall_score,
                              confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")

PROC_DIR  = os.path.join("data", "processed")
MODEL_DIR = os.path.join("outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
CV_REPEATED = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42) 

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(PROC_DIR, "tamilnadu_nfhs5_encoded.csv"))
drop_cols = ["caseid", "midx", "v001", "v002", "hhid", "stunting_haz", "state"]
X_raw = df.drop(columns=[c for c in drop_cols + ["stunting_label"] if c in df.columns])
y = df["stunting_label"]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

spw = round((y_train == 0).sum() / (y_train == 1).sum(), 2)
print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"scale_pos_weight: {spw}")

# ── Read winner from comparison run ───────────────────────────────────────────
winner_path = os.path.join(MODEL_DIR, "comparison_winner.json")
if os.path.exists(winner_path):
    with open(winner_path) as f:
        cw = json.load(f)
    winner = cw["winner_model_name"]
    print(f"\nWinner from comparison: {winner} (AUC={cw['winner_auc']})")
else:
    print("⚠ comparison_winner.json not found — defaulting to XGBoost")
    print("  Run scripts/07b_model_comparison.py first.")
    winner = "XGBoost"

# ── Build the winner model for tuning ─────────────────────────────────────────
print(f"\n--- Tuning: {winner} ---")

if "XGBoost" in winner:
    param_dist = {
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
        "n_estimators":     [200, 300, 400, 500],
        "min_child_weight": [1, 3, 5, 7],
        "subsample":        [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "gamma":            [0, 0.1, 0.2, 0.3],
        "reg_alpha":        [0, 0.01, 0.1],
        "reg_lambda":       [1, 1.5, 2],
    }
    base_model = XGBClassifier(
        scale_pos_weight=spw, eval_metric="auc",
        verbosity=0, random_state=42, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist,
        n_iter=40, cv=CV, scoring="roc_auc",
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)

elif "Random Forest" in winner:
    param_dist = {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth":    [6, 8, 10, 12, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", 0.5],
    }
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", rf)])
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist,
        n_iter=40, cv=CV, scoring="roc_auc",
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)

elif "Gradient Boosting" in winner:
    param_dist = {
        "clf__n_estimators":   [100, 200, 300, 400],
        "clf__learning_rate":  [0.01, 0.03, 0.05, 0.08],
        "clf__max_depth":      [3, 4, 5, 6],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__subsample":      [0.7, 0.8, 0.9],
    }
    gb = GradientBoostingClassifier(random_state=42)
    pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", gb)])

    search = RandomizedSearchCV(
    pipe, param_distributions=param_dist,
    n_iter=30, cv=CV_REPEATED, scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)

elif "Logistic" in winner:
    from sklearn.pipeline import Pipeline as SkPipeline
    param_dist = {
        "clf__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
    }
    lr = LogisticRegression(solver="saga", max_iter=3000, class_weight="balanced",
                             random_state=42)
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", lr)
    ])
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist,
        n_iter=20, cv=CV, scoring="roc_auc",
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)

else:
    # Fallback: stacking
    print(f"Winner '{winner}' — using Stacking with passthrough based on comparison result")
    use_pt = "passthrough=True" in winner
    stack_estimators = [
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("gbm", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]
    stacking = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(solver="liblinear", max_iter=2000,
                                            random_state=42),
        cv=CV, passthrough=use_pt, n_jobs=-1
    )
    pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", stacking)])
    param_dist = {
        "clf__rf__n_estimators":   [100, 200, 300],
        "clf__gbm__n_estimators":  [100, 200],
        "clf__gbm__learning_rate": [0.05, 0.08, 0.1],
    }
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist,
        n_iter=20, cv=CV, scoring="roc_auc",
        n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
best_model = search.best_estimator_
print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
print(f"Best params: {json.dumps({k: str(v) for k, v in search.best_params_.items()}, indent=2)}")

y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred  = best_model.predict(X_test)
test_auc = roc_auc_score(y_test, y_proba)

print(f"\nTest ROC-AUC  : {test_auc:.4f}")
print(f"Test F1       : {f1_score(y_test, y_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Test Recall   : {recall_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Stunted", "Stunted"],
                             zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Gate check ────────────────────────────────────────────────────────────────
if test_auc < 0.60:
    print(f"\n⚠ WARNING: Test AUC {test_auc:.4f} is still below 0.60.")
    print("  Possible causes:")
    print("  1. Sparse columns not yet cleaned — re-run 05_encode_features.py")
    print("  2. Interaction features not created — check output of 05_encode_features.py")
    print("  3. Data signal is genuinely limited — acceptable, document honestly")
    print("  Do NOT proceed to SHAP until AUC >= 0.62.")
elif test_auc < 0.65:
    print(f"\n⚠ AUC {test_auc:.4f} is between 0.60-0.65.")
    print("  Acceptable for this dataset size (~5890). Proceed to SHAP cautiously.")
    print("  Document limitation: small TN-specific dataset limits discriminative power.")
else:
    print(f"\n✅ AUC {test_auc:.4f} clears 0.65 threshold. Proceed to SHAP (script 09).")

# ── Save all artefacts ────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(imputer,    os.path.join(MODEL_DIR, "imputer.pkl"))
X_test.to_csv(os.path.join(MODEL_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(MODEL_DIR, "y_test.csv"), index=False)
X_train.to_csv(os.path.join(MODEL_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(MODEL_DIR, "y_train.csv"), index=False)
with open(os.path.join(MODEL_DIR, "feature_names.txt"), "w") as f:
    f.write("\n".join(X.columns.tolist()))

meta = {
    "winner": winner,
    "best_cv_auc":   round(search.best_score_, 4),
    "test_auc":      round(test_auc, 4),
    "best_params":   {k: str(v) for k, v in search.best_params_.items()},
    "scale_pos_weight": spw,
    "feature_count": X.shape[1],
    "train_size":    int(X_train.shape[0]),
    "test_size":     int(X_test.shape[0]),
}
with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ best_model.pkl saved → {MODEL_DIR}")
print(f"✅ model_meta.json saved → {MODEL_DIR}")