"""
NutriSense - Flask Web Application & REST API Server
Provides endpoints for childhood stunting risk prediction, SHAP explainability,
DiCE counterfactual intervention simulation, model benchmarking, and field worker tools.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add src/ml to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "ml"))

from data_loader import FEATURE_LABELS
from shap_engine import NutriSenseSHAPExplainer
from dice_engine import NutriSenseDiCEEngine

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Lazy-loaded Globals
model_pipeline = None
calibrated_model = None
imputer = None
shap_explainer = None
dice_engine = None
feature_names = []
model_meta = {}
benchmark_results = []


def initialize_ml_components():
    global model_pipeline, calibrated_model, imputer, shap_explainer, dice_engine, feature_names, model_meta, benchmark_results
    if calibrated_model is None:
        try:
            model_pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model_pipeline.pkl"))
            calibrated_model = joblib.load(os.path.join(MODELS_DIR, "calibrated_model.pkl"))
            imputer = joblib.load(os.path.join(MODELS_DIR, "imputer.pkl"))

            with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
                feature_names = json.load(f)

            with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
                model_meta = json.load(f)

            if os.path.exists(os.path.join(MODELS_DIR, "benchmark_results.json")):
                with open(os.path.join(MODELS_DIR, "benchmark_results.json")) as f:
                    benchmark_results = json.load(f)

            shap_explainer = NutriSenseSHAPExplainer(MODELS_DIR)
            dice_engine = NutriSenseDiCEEngine(MODELS_DIR)
            print("[OK] All NutriSense ML & Explainability engines loaded successfully.")
        except Exception as e:
            print(f"[WARN] Warning initializing ML engines: {e}")


def construct_feature_vector(input_dict):
    """
    Constructs a complete single-row pandas DataFrame aligned with training feature columns.
    Sets defaults and builds OHE & interaction features dynamically.
    """
    row = {col: 0.0 for col in feature_names}

    # Direct Mappings
    if "child_age_months" in input_dict:
        row["child_age_months"] = float(input_dict.get("child_age_months", 24))
    if "child_sex" in input_dict:
        row["child_sex"] = float(input_dict.get("child_sex", 0))  # 0=Male, 1=Female
    if "birth_order" in input_dict:
        row["birth_order"] = float(input_dict.get("birth_order", 1))
    if "birth_interval" in input_dict:
        row["birth_interval"] = float(input_dict.get("birth_interval", 36))
    if "anc_visits" in input_dict:
        row["anc_visits"] = float(input_dict.get("anc_visits", 4))
    if "caesarean" in input_dict:
        row["caesarean"] = float(input_dict.get("caesarean", 0))
    if "diarrhea_recent" in input_dict:
        row["diarrhea_recent"] = float(input_dict.get("diarrhea_recent", 0))
    if "fever_recent" in input_dict:
        row["fever_recent"] = float(input_dict.get("fever_recent", 0))
    if "mother_age" in input_dict:
        row["mother_age"] = float(input_dict.get("mother_age", 26))
        # Derive age group 1-7
        m_age = row["mother_age"]
        row["mother_age_group"] = min(7, max(1, int((m_age - 15) // 5) + 1))
    if "mother_education" in input_dict:
        row["mother_education"] = float(input_dict.get("mother_education", 2))
    if "wealth_index" in input_dict:
        row["wealth_index"] = float(input_dict.get("wealth_index", 3))
    if "mother_anemia" in input_dict:
        row["mother_anemia"] = float(input_dict.get("mother_anemia", 0))
    if "electricity" in input_dict:
        row["electricity"] = float(input_dict.get("electricity", 1))
    if "share_toilet" in input_dict:
        row["share_toilet"] = float(input_dict.get("share_toilet", 0))
    if "mother_bmi" in input_dict:
        row["mother_bmi"] = float(input_dict.get("mother_bmi", 21.5))

    # OHE Toilet Type
    toilet_val = input_dict.get("toilet_type", "toilet_type_11")
    if toilet_val in row:
        row[toilet_val] = 1.0

    # OHE Water Source
    water_val = input_dict.get("water_source", "water_source_11")
    if water_val in row:
        row[water_val] = 1.0

    # OHE Delivery Place
    delivery_val = input_dict.get("delivery_place", "delivery_place_21.0")
    if delivery_val in row:
        row[delivery_val] = 1.0

    # Interactions & Derived Features
    row["interact_wealth_x_education"] = row["wealth_index"] * row["mother_education"]
    is_unsafe_water = 1.0 if water_val in ["water_source_31", "water_source_61", "water_source_92", "water_source_other"] else 0.0
    row["interact_unsafe_water_x_diarrhea"] = is_unsafe_water * row["diarrhea_recent"]

    open_defecation = 1.0 if toilet_val == "toilet_type_31" else 0.0
    row["wash_vulnerability_score"] = (open_defecation * 2.0) + (row["share_toilet"] * 1.0)
    row["anc_adequacy_flag"] = 1.0 if row["anc_visits"] >= 4.0 else 0.0

    df_row = pd.DataFrame([row])[feature_names]
    return df_row


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "system": "NutriSense",
        "model_loaded": calibrated_model is not None,
        "metadata": model_meta
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    initialize_ml_components()
    data = request.json or {}
    try:
        sample_df = construct_feature_vector(data)
        prob = float(calibrated_model.predict_proba(sample_df)[:, 1][0])
        
        # Scale probability to 0-100 Stunting Risk Index (SRI)
        base_rate = 0.258
        max_prob = 0.60
        if prob < base_rate:
            sri = (prob / base_rate) * 40.0
        else:
            sri = 40.0 + ((prob - base_rate) / (max_prob - base_rate)) * 60.0
        sri = max(0.0, min(100.0, sri))
        
        risk_pct = round(sri, 2)

        # Categorize Risk Level
        if risk_pct < 15.0:
            tier = "Low Risk"
            badge_class = "success"
            action_summary = "Child exhibits healthy nutritional trajectory. Recommend standard monitoring."
        elif risk_pct < 35.0:
            tier = "Moderate Risk"
            badge_class = "warning"
            action_summary = "Child shows mild vulnerability factors. Routine health checkups recommended."
        elif risk_pct < 60.0:
            tier = "High Risk"
            badge_class = "danger"
            action_summary = "Significant stunting risk detected. Priority intervention recommended."
        else:
            tier = "Severe Risk"
            badge_class = "critical"
            action_summary = "Urgent clinical and nutritional intervention required!"

        return jsonify({
            "success": True,
            "stunting_probability": prob,
            "stunting_risk_pct": risk_pct,
            "risk_tier": tier,
            "badge_class": badge_class,
            "action_summary": action_summary,
            "feature_vector": sample_df.to_dict(orient="records")[0]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/explain", methods=["POST"])
def explain():
    initialize_ml_components()
    data = request.json or {}
    try:
        sample_df = construct_feature_vector(data)
        explanation = shap_explainer.explain_sample(sample_df)
        return jsonify({
            "success": True,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/counterfactual", methods=["POST"])
def counterfactual():
    initialize_ml_components()
    data = request.json or {}
    try:
        sample_df = construct_feature_vector(data)
        prob = float(calibrated_model.predict_proba(sample_df)[:, 1][0])
        interventions = dice_engine.generate_intervention_packages(sample_df, prob)
        return jsonify({
            "success": True,
            "interventions": interventions
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


benchmark_cache = None

@app.route("/api/benchmark", methods=["GET"])
def benchmark():
    global benchmark_cache
    initialize_ml_components()
    
    if benchmark_cache is None:
        global_shap = []
        if shap_explainer:
            global_shap = shap_explainer.get_global_importance(top_n=15)
        benchmark_cache = {
            "success": True,
            "model_meta": model_meta,
            "benchmark_results": benchmark_results,
            "global_shap_importance": global_shap
        }
    return jsonify(benchmark_cache)

@app.route("/api/benchmark/retest", methods=["POST"])
def benchmark_retest():
    global benchmark_cache
    benchmark_cache = None
    return benchmark()


@app.route("/api/sample_children", methods=["GET"])
def sample_children():
    samples = [
        {
            "id": "high_risk_rural",
            "name": "Anitha (Rural Vulnerable Household)",
            "description": "24-month girl in rural household with open defecation, unsafe water, low ANC visits & mother anemia.",
            "data": {
                "child_age_months": 24, "child_sex": 1, "birth_order": 3, "birth_interval": 20,
                "anc_visits": 1, "diarrhea_recent": 1, "fever_recent": 1, "mother_age": 22,
                "mother_education": 0, "wealth_index": 1, "mother_anemia": 2, "electricity": 0,
                "share_toilet": 1, "mother_bmi": 17.5, "toilet_type": "toilet_type_31",
                "water_source": "water_source_31", "delivery_place": "delivery_place_31.0"
            }
        },
        {
            "id": "moderate_risk_semiurban",
            "name": "Karthik (Semi-Urban Moderate Risk)",
            "description": "18-month boy with moderate wealth, pit latrine, public tap water, and 3 ANC visits.",
            "data": {
                "child_age_months": 18, "child_sex": 0, "birth_order": 2, "birth_interval": 30,
                "anc_visits": 3, "diarrhea_recent": 0, "fever_recent": 1, "mother_age": 25,
                "mother_education": 1, "wealth_index": 2, "mother_anemia": 1, "electricity": 1,
                "share_toilet": 0, "mother_bmi": 20.2, "toilet_type": "toilet_type_22",
                "water_source": "water_source_13", "delivery_place": "delivery_place_21.0"
            }
        },
        {
            "id": "low_risk_urban",
            "name": "Priya (Urban Protection Profile)",
            "description": "36-month girl in urban household with flush toilet, dwelling tap water, higher maternal education & 8 ANC visits.",
            "data": {
                "child_age_months": 36, "child_sex": 1, "birth_order": 1, "birth_interval": 48,
                "anc_visits": 8, "diarrhea_recent": 0, "fever_recent": 0, "mother_age": 28,
                "mother_education": 3, "wealth_index": 5, "mother_anemia": 0, "electricity": 1,
                "share_toilet": 0, "mother_bmi": 23.5, "toilet_type": "toilet_type_11",
                "water_source": "water_source_11", "delivery_place": "delivery_place_23.0"
            }
        }
    ]
    return jsonify({"success": True, "samples": samples})


if __name__ == "__main__":
    initialize_ml_components()
    app.run(host="0.0.0.0", port=5000, debug=True)
