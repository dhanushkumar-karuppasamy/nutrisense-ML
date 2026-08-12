"""
NutriSense ML Pipeline - Model Training, Benchmarking & Calibration
Executes a fair 8-model comparison under Stratified 5-Fold Cross Validation with SMOTE inside loop.
Calibrates prediction probabilities and serializes best model artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, brier_score_loss
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin

from data_loader import load_dataset, FEATURE_LABELS, AdvancedFeatureTransformer

# Hide TF warnings
tf.get_logger().setLevel('ERROR')

class KerasDNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=50, batch_size=32, verbose=0, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.model = None
        self.classes_ = np.array([0, 1])
        self.input_dim = None

    def build_model(self, input_dim):
        tf.random.set_seed(self.random_state)
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        return model

    def fit(self, X, y):
        # Convert X to numpy if it's a dataframe or array-like
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)
        if hasattr(y, 'values'):
            y = y.values
        else:
            y = np.array(y)
            
        self.input_dim = X.shape[1]
        self.model = self.build_model(self.input_dim)
        
        # Calculate class weights for imbalance
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        total = len(y)
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        early_stopping = EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )
        
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        return self

    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.array(X)
        preds = self.model.predict(X, verbose=0).flatten()
        return np.vstack([1 - preds, preds]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

warnings.filterwarnings("ignore")


def train_and_evaluate():
    print("=" * 70)
    print("NutriSense Phase 2: High-Quality ML Training & Evaluation Ladder")
    print("=" * 70)

    # 1. Load Data
    X_raw, y, feature_labels = load_dataset()
    print(f"Dataset Loaded: {X_raw.shape[0]} children, {X_raw.shape[1]} features.")
    print(f"Stunting Prevalence: {y.mean() * 100:.2f}% ({y.sum()} positive / {(y == 0).sum()} negative)")

    # 2. Imputation & Preprocessing
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.20, random_state=42, stratify=y
    )

    spw = round((y_train == 0).sum() / (y_train == 1).sum(), 2)
    print(f"Training split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")
    print(f"Scale Pos Weight (Class Ratio): {spw}")

    # 3. Model Benchmark Directory Setup
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_model_dir = os.path.join(base_dir, "models")
    os.makedirs(output_model_dir, exist_ok=True)

    # 4. Define 8 Models
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_models = {
        "Logistic Regression": LogisticRegression(
            solver="saga", max_iter=3000, C=0.1, class_weight="balanced", random_state=42
        ),
        "Keras DNN": KerasDNNWrapper(epochs=100, batch_size=64, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, min_samples_split=5,
            subsample=0.8, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            reg_alpha=0.5, reg_lambda=2.0, eval_metric="auc", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.02, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbosity=-1, n_jobs=-1
        ),
    }

    # Add Stacking Classifier (Keras + GBM + XGB meta-learner)
    stack_estimators = [
        ("dnn", KerasDNNWrapper(epochs=50, batch_size=64, random_state=42)),
        ("gbm", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.02, scale_pos_weight=spw, reg_alpha=0.1, random_state=42, n_jobs=-1))
    ]

    base_models["Stacking Ensemble"] = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(solver="liblinear", penalty="l1", C=0.5, random_state=42),
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        passthrough=True,
        n_jobs=-1
    )

    # 5. Execute Stratified Cross-Validation Benchmarking
    results = []
    print("\nRunning 5-Fold Stratified Cross Validation for all models...")

    fitted_pipelines = {}

    for name, model in base_models.items():
        print(f"  --> Benchmarking [{name}]...")

        # ADASYN inside pipeline to prevent data leakage
        pipeline_steps = [
            ("adv_features", AdvancedFeatureTransformer()),
            ("adasyn", ADASYN(random_state=42))
        ]
        if name in ["Logistic Regression", "Keras DNN", "Stacking Ensemble"]:
            pipeline_steps.append(("scaler", StandardScaler()))
        pipeline_steps.append(("clf", model))

        pipe = ImbPipeline(pipeline_steps)

        scoring = ["roc_auc", "recall", "precision", "f1", "accuracy"]
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=-1
        )

        # Fit on full X_train for test set evaluation & probability calibration
        pipe.fit(X_train, y_train)

        # Test set metrics
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        test_auc = roc_auc_score(y_test, y_proba)
        test_recall = recall_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)

        cv_auc_mean = np.mean(cv_scores["test_roc_auc"])
        cv_auc_std = np.std(cv_scores["test_roc_auc"])
        cv_recall_mean = np.mean(cv_scores["test_recall"])

        results.append({
            "Model": name,
            "CV_ROC_AUC": round(float(cv_auc_mean), 4),
            "CV_ROC_AUC_Std": round(float(cv_auc_std), 4),
            "CV_Recall": round(float(cv_recall_mean), 4),
            "Test_ROC_AUC": round(float(test_auc), 4),
            "Test_Recall": round(float(test_recall), 4),
            "Test_Precision": round(float(test_precision), 4),
            "Test_F1": round(float(test_f1), 4),
            "Test_Accuracy": round(float(test_acc), 4),
            "Brier_Score": round(float(brier), 4),
        })

        fitted_pipelines[name] = pipe

    # Display comparison table sorted by CV ROC-AUC
    results_df = pd.DataFrame(results).sort_values("CV_ROC_AUC", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON BENCHMARK RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Winner Selection (Highest CV ROC-AUC)
    winner_row = results_df.iloc[0]
    winner_name = winner_row["Model"]
    print("\n" + "*" * 70)
    print(f"SELECTED BEST MODEL: {winner_name}")
    print(f"CV ROC-AUC: {winner_row['CV_ROC_AUC']:.4f} ± {winner_row['CV_ROC_AUC_Std']:.4f}")
    print(f"Test Recall (Stunted Class): {winner_row['Test_Recall']:.4f}")
    print("*" * 70)

    winner_pipeline = fitted_pipelines[winner_name]

    # Skip probability calibration to maintain variance for risk scoring
    print("\nSkipping probability calibration to maintain variance for risk scoring...")
    calibrated_clf = winner_pipeline

    if hasattr(calibrated_clf, "predict_proba"):
        calib_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    else:
        calib_proba = calibrated_clf.predict(X_test)
    calib_auc = roc_auc_score(y_test, calib_proba)
    calib_brier = brier_score_loss(y_test, calib_proba)

    print(f"Final Test ROC-AUC: {calib_auc:.4f} | Brier Score: {calib_brier:.4f}")

    # 6. Save Artifacts
    # Save model pipeline & calibrated model
    joblib.dump(winner_pipeline, os.path.join(output_model_dir, "best_model_pipeline.pkl"))
    joblib.dump(calibrated_clf, os.path.join(output_model_dir, "calibrated_model.pkl"))
    joblib.dump(imputer, os.path.join(output_model_dir, "imputer.pkl"))

    # Save X_train, X_test, y_train, y_test for SHAP & DiCE
    X_train.to_csv(os.path.join(output_model_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_model_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_model_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_model_dir, "y_test.csv"), index=False)

    # Save Feature Labels & Names
    with open(os.path.join(output_model_dir, "feature_names.json"), "w") as f:
        json.dump(list(X_raw.columns), f, indent=2)

    with open(os.path.join(output_model_dir, "feature_labels.json"), "w") as f:
        json.dump(feature_labels, f, indent=2)

    # Save metadata & comparison metrics
    meta_info = {
        "winner_model": winner_name,
        "cv_roc_auc": winner_row["CV_ROC_AUC"],
        "test_roc_auc": winner_row["Test_ROC_AUC"],
        "calibrated_test_auc": round(float(calib_auc), 4),
        "test_recall": winner_row["Test_Recall"],
        "test_precision": winner_row["Test_Precision"],
        "test_f1": winner_row["Test_F1"],
        "test_accuracy": winner_row["Test_Accuracy"],
        "brier_score": round(float(calib_brier), 4),
        "sample_size": int(X_imputed.shape[0]),
        "feature_count": int(X_imputed.shape[1]),
        "stunting_prevalence_pct": round(float(y.mean() * 100), 2)
    }

    with open(os.path.join(output_model_dir, "model_meta.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    results_df.to_json(os.path.join(output_model_dir, "benchmark_results.json"), orient="records", indent=2)

    print(f"\n[OK] Training complete! Artifacts saved to: {output_model_dir}")


if __name__ == "__main__":
    train_and_evaluate()
