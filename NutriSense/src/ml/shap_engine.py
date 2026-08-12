"""
NutriSense ML Pipeline - SHAP Explainability Engine
Decomposes individual child stunting risk scores into feature-level positive and negative contributions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap


class NutriSenseSHAPExplainer:
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            models_dir = os.path.join(base_dir, "models")

        self.models_dir = models_dir

        # Load pipeline & dataset
        self.model_pipeline = joblib.load(os.path.join(models_dir, "best_model_pipeline.pkl"))
        self.calibrated_model = joblib.load(os.path.join(models_dir, "calibrated_model.pkl"))
        self.X_train = pd.read_csv(os.path.join(models_dir, "X_train.csv"))

        with open(os.path.join(models_dir, "feature_names.json")) as f:
            self.feature_names = json.load(f)

        with open(os.path.join(models_dir, "feature_labels.json")) as f:
            self.feature_labels = json.load(f)

        # Use KernelExplainer on the FULL pipeline so we explain the original features
        # (Handling the dynamically injected K-Means/PCA features seamlessly)
        background = self.X_train.sample(min(50, len(self.X_train)), random_state=42)
        
        # We need a prediction function that outputs 1D array or 2D array of probabilities
        def predict_fn(X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.X_train.columns)
            return self.model_pipeline.predict_proba(X)
            
        self.explainer = shap.KernelExplainer(predict_fn, background)

        # Base value (expected value)
        exp_val = self.explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            self.expected_value = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
        else:
            self.expected_value = float(exp_val)

    def explain_sample(self, sample_df):
        """
        Calculates local SHAP values for a single child's feature vector.
        Returns a structured dictionary of feature contributions, sorted by absolute impact.
        """
        # Ensure correct column order
        sample_ordered = sample_df[self.X_train.columns].copy()

        shap_vals = self.explainer.shap_values(sample_ordered)

        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]  # Class 1 (Stunted)
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        feature_contributions = []
        for i, col in enumerate(self.X_train.columns):
            contrib = float(sv[i])
            val = float(sample_ordered.iloc[0][col])
            label = self.feature_labels.get(col, col)

            # Human readable explanation direction
            if contrib > 0.01:
                direction = "increases_risk"
            elif contrib < -0.01:
                direction = "reduces_risk"
            else:
                direction = "neutral"

            feature_contributions.append({
                "feature_code": col,
                "feature_name": label,
                "value": round(val, 2),
                "shap_value": round(contrib, 4),
                "impact_pct": round(contrib * 100, 2),
                "direction": direction
            })

        # Sort by absolute SHAP impact
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Risk Factors (increase risk) vs Protective Factors (reduce risk)
        risk_factors = [f for f in feature_contributions if f["direction"] == "increases_risk"]
        protective_factors = [f for f in feature_contributions if f["direction"] == "reduces_risk"]

        return {
            "expected_base_value": round(self.expected_value, 4),
            "expected_base_risk_pct": round(self.expected_value * 100, 2),
            "top_features": feature_contributions[:10],
            "risk_factors": risk_factors[:5],
            "protective_factors": protective_factors[:5],
            "all_features": feature_contributions
        }

    def get_global_importance(self, top_n=15):
        """
        Computes global mean |SHAP| feature importance across training sample.
        """
        # Reduce sample size for KernelExplainer to keep API fast
        sample_batch = self.X_train.sample(min(50, len(self.X_train)), random_state=42)
        shap_vals = self.explainer.shap_values(sample_batch)

        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals
            
        if len(sv.shape) == 3:
            sv = sv[:, :, 1]

        mean_abs = np.abs(sv).mean(axis=0)

        global_rankings = []
        for i, col in enumerate(self.X_train.columns):
            label = self.feature_labels.get(col, col)
            global_rankings.append({
                "feature_code": col,
                "feature_name": label,
                "mean_abs_shap": round(float(mean_abs[i]), 4),
                "importance_score": round(float(mean_abs[i] * 100), 2)
            })

        global_rankings.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        return global_rankings[:top_n]
