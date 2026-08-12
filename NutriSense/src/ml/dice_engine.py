"""
NutriSense ML Pipeline - DiCE Counterfactual & Intervention Engine
Generates minimal actionable counterfactual packages to reduce childhood stunting risk.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import dice_ml


class NutriSenseDiCEEngine:
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            models_dir = os.path.join(base_dir, "models")

        self.models_dir = models_dir

        self.calibrated_model = joblib.load(os.path.join(models_dir, "calibrated_model.pkl"))
        self.pipeline = joblib.load(os.path.join(models_dir, "best_model_pipeline.pkl"))
        self.X_train = pd.read_csv(os.path.join(models_dir, "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(models_dir, "y_train.csv")).squeeze()

        with open(os.path.join(models_dir, "feature_labels.json")) as f:
            self.feature_labels = json.load(f)

        # Immutable features (cannot be modified by interventions)
        self.features_to_vary = [
            c for c in self.X_train.columns
            if c not in ["child_sex", "child_age_months", "birth_order", "birth_interval", "caesarean", "mother_age", "mother_age_group"]
        ]

        # Setup DiCE data and model wrapper
        dataset_dice = pd.concat([self.X_train, self.y_train], axis=1)
        self.d = dice_ml.Data(
            dataframe=dataset_dice,
            continuous_features=[c for c in self.X_train.columns if c in ["child_age_months", "mother_age", "mother_bmi", "anc_visits", "birth_interval"]],
            outcome_name="stunting_label"
        )
        self.m = dice_ml.Model(model=self.calibrated_model, backend="sklearn")
        try:
            self.exp = dice_ml.Dice(self.d, self.m, method="random")
        except Exception:
            self.exp = None

    def _calculate_sri(self, prob: float) -> float:
        """Scales raw probability to a 0-100 Stunting Risk Index (SRI)"""
        base_rate = 0.258
        max_prob = 0.60
        if prob < base_rate:
            sri = (prob / base_rate) * 40.0
        else:
            sri = 40.0 + ((prob - base_rate) / (max_prob - base_rate)) * 60.0
        return max(0.0, min(100.0, sri))

    def generate_intervention_packages(self, sample_df, current_risk_prob):
        """
        Generates 3 actionable intervention packages for a child:
        1. WASH Package (Sanitation & Safe Water)
        2. Maternal & Clinical Care Package (ANC visits, infection control, anemia support)
        3. Comprehensive Integrated Package (WASH + Maternal Care + Hygiene)
        """
        base_features = sample_df.copy()
        current_risk_pct = round(self._calculate_sri(float(current_risk_prob)), 2)

        packages = []

        # --- 1. WASH Intervention Package ---
        wash_df = base_features.copy()
        # Upgrade toilet to flush/septic (toilet_type_11 or toilet_type_12 = 1, open defecation = 0)
        for col in wash_df.columns:
            if col.startswith("toilet_type_"):
                wash_df[col] = 1.0 if col == "toilet_type_11" else 0.0
            if col == "share_toilet":
                wash_df[col] = 0.0
            if col.startswith("water_source_"):
                wash_df[col] = 1.0 if col == "water_source_11" else 0.0
            if col == "wash_vulnerability_score":
                wash_df[col] = 0.0

        wash_prob = float(self.calibrated_model.predict_proba(wash_df)[:, 1][0])
        wash_risk_pct = round(self._calculate_sri(wash_prob), 2)
        wash_delta_pct = round(current_risk_pct - wash_risk_pct, 2)

        packages.append({
            "package_id": "wash_package",
            "package_name": "Sanitation & Clean Water (WASH)",
            "icon": "faucet",
            "simulated_risk_pct": wash_risk_pct,
            "risk_reduction_pct": wash_delta_pct,
            "key_actions": [
                "Provide access to private flush/septic toilet facility",
                "Ensure clean piped/borehole drinking water source",
                "Eliminate open defecation & shared latrine exposure"
            ]
        })

        # --- 2. Maternal & Clinical Care Package ---
        care_df = base_features.copy()
        if "anc_visits" in care_df.columns:
            care_df["anc_visits"] = max(4.0, float(care_df["anc_visits"].iloc[0]) + 3.0)
        if "anc_adequacy_flag" in care_df.columns:
            care_df["anc_adequacy_flag"] = 1.0
        if "diarrhea_recent" in care_df.columns:
            care_df["diarrhea_recent"] = 0.0
        if "fever_recent" in care_df.columns:
            care_df["fever_recent"] = 0.0
        if "mother_anemia" in care_df.columns:
            care_df["mother_anemia"] = max(0.0, float(care_df["mother_anemia"].iloc[0]) - 1.0)
        if "interact_unsafe_water_x_diarrhea" in care_df.columns:
            care_df["interact_unsafe_water_x_diarrhea"] = 0.0

        care_prob = float(self.calibrated_model.predict_proba(care_df)[:, 1][0])
        care_risk_pct = round(self._calculate_sri(care_prob), 2)
        care_delta_pct = round(current_risk_pct - care_risk_pct, 2)

        packages.append({
            "package_id": "maternal_care_package",
            "package_name": "Maternal Healthcare & Clinical Support",
            "icon": "user-nurse",
            "simulated_risk_pct": care_risk_pct,
            "risk_reduction_pct": care_delta_pct,
            "key_actions": [
                "Increase Antenatal Care (ANC) visits to ≥4 WHO recommended checkups",
                "Prompt clinical treatment for childhood diarrhea and fever",
                "Maternal iron supplementation for anemia management"
            ]
        })

        # --- 3. Comprehensive Integrated Package ---
        comp_df = wash_df.copy()
        if "anc_visits" in comp_df.columns:
            comp_df["anc_visits"] = max(5.0, float(comp_df["anc_visits"].iloc[0]) + 3.0)
        if "anc_adequacy_flag" in comp_df.columns:
            comp_df["anc_adequacy_flag"] = 1.0
        if "diarrhea_recent" in comp_df.columns:
            comp_df["diarrhea_recent"] = 0.0
        if "fever_recent" in comp_df.columns:
            comp_df["fever_recent"] = 0.0
        if "mother_anemia" in comp_df.columns:
            comp_df["mother_anemia"] = 0.0
        if "mother_education" in comp_df.columns:
            comp_df["mother_education"] = max(2.0, float(comp_df["mother_education"].iloc[0]) + 1.0)

        comp_prob = float(self.calibrated_model.predict_proba(comp_df)[:, 1][0])
        comp_risk_pct = round(self._calculate_sri(comp_prob), 2)
        comp_delta_pct = round(current_risk_pct - comp_risk_pct, 2)

        packages.append({
            "package_id": "comprehensive_package",
            "package_name": "Integrated WASH + Maternal & Social Package",
            "icon": "shield-halved",
            "simulated_risk_pct": comp_risk_pct,
            "risk_reduction_pct": comp_delta_pct,
            "key_actions": [
                "Full WASH upgrade + Regular ANC checkups",
                "Community health worker monthly monitoring",
                "Maternal health literacy & nutritional support"
            ]
        })

        # Try generating raw DiCE counterfactuals if requested
        dice_counterfactuals = []
        if self.exp is not None and current_risk_prob > 0.35:
            try:
                cf = self.exp.generate_counterfactuals(
                    sample_df, total_CFs=2, desired_class=0, features_to_vary=self.features_to_vary
                )
                cf_df = cf.cf_examples_list[0].final_cfs_df
                if cf_df is not None:
                    for idx, row in cf_df.iterrows():
                        diffs = []
                        for col in self.features_to_vary:
                            orig_v = sample_df.iloc[0][col]
                            new_v = row[col]
                            if orig_v != new_v:
                                diffs.append({
                                    "feature": self.feature_labels.get(col, col),
                                    "from": orig_v,
                                    "to": new_v
                                })
                        dice_counterfactuals.append({"changes": diffs})
            except Exception:
                pass

        return {
            "baseline_risk_pct": current_risk_pct,
            "packages": packages,
            "dice_counterfactuals": dice_counterfactuals
        }
