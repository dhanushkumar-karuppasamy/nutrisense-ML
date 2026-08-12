"""
NutriSense ML Pipeline - Data Loader & Feature Engineering Module
Ingests NFHS-5 Tamil Nadu dataset, performs domain-specific feature engineering,
imputation, and metadata mapping for explainable AI.
"""

import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Feature Labels Map for UI & SHAP Explanations
FEATURE_LABELS = {
    "child_age_months": "Child Age (Months)",
    "child_sex": "Child Sex (Female=1, Male=0)",
    "birth_order": "Birth Order Number",
    "birth_interval": "Preceding Birth Interval (Months)",
    "anc_visits": "Antenatal Care (ANC) Visits",
    "caesarean": "Caesarean Delivery",
    "diarrhea_recent": "Recent Diarrhea (Past 2 Weeks)",
    "fever_recent": "Recent Fever (Past 2 Weeks)",
    "mother_age": "Mother's Age (Years)",
    "mother_age_group": "Mother's Age Group",
    "mother_education": "Mother's Education Level (0-3)",
    "wealth_index": "Household Wealth Quintile (1-5)",
    "mother_anemia": "Mother's Anemia Level (0=None, 3=Severe)",
    "electricity": "Electricity Connection",
    "share_toilet": "Shared Sanitation Facility",
    "mother_bmi": "Mother's BMI (kg/m²)",
    "interact_wealth_x_education": "Wealth × Maternal Education Synergy",
    "interact_unsafe_water_x_diarrhea": "Unsafe Water × Diarrhea Risk Pathway",
    "toilet_type_11": "Flush Toilet (Piped Sewer)",
    "toilet_type_12": "Flush Toilet (Septic Tank)",
    "toilet_type_13": "Flush Toilet (Pit Latrine)",
    "toilet_type_22": "Pit Latrine (Slab)",
    "toilet_type_31": "No Facility (Open Defecation)",
    "toilet_type_41": "Composting Toilet",
    "toilet_type_other": "Other Toilet Facility",
    "water_source_11": "Piped Water into Dwelling",
    "water_source_12": "Piped Water into Yard/Plot",
    "water_source_13": "Public Tap / Standpipe",
    "water_source_14": "Tube Well / Borehole",
    "water_source_21": "Protected Well",
    "water_source_31": "Unprotected Well",
    "water_source_61": "Tanker Truck Water",
    "water_source_71": "Bottled / Cart Water",
    "water_source_92": "Surface Water (River/Stream)",
    "water_source_other": "Other Water Source",
    "delivery_place_21.0": "Public Health Facility Delivery",
    "delivery_place_23.0": "Private Health Facility Delivery",
    "delivery_place_24.0": "NGO / Trust Facility Delivery",
    "delivery_place_25.0": "Home Delivery (Qualified Attendant)",
    "delivery_place_31.0": "Home Delivery (Unassisted)",
    "delivery_place_other": "Other Delivery Location",
    "wash_vulnerability_score": "WASH Vulnerability Score",
    "anc_adequacy_flag": "Adequate ANC Visits (≥4)",
    "disease_burden_index": "Acute Disease Burden (Diarrhea/Fever)",
    "maternal_vulnerability_score": "Maternal Vulnerability Score",
    "child_vulnerability_score": "Child Vulnerability Score",
    "wealth_education_synergy": "Wealth-Education Synergy",
    "unsafe_water_diarrhea_pathway": "Unsafe Water x Diarrhea Risk",
    "bmi_anemia_synergy": "BMI-Anemia Synergy Risk",
    "child_age_x_wealth": "Child Age x Wealth Index",
    "education_x_anc": "Maternal Education x ANC Visits",
    "child_age_log": "Child Age (Log Scale)",
    "is_critical_age": "Critical Age Window (6-24m)",
    "mother_bmi_sq": "Maternal BMI (Non-Linear)",
    "wash_age_synergy": "WASH Vulnerability x Critical Age",
    "wealth_edu_anc_triple": "Wealth x Edu x ANC Risk",
    "pca_socio_wash_1": "Latent Socio-WASH Factor 1",
    "pca_socio_wash_2": "Latent Socio-WASH Factor 2",
    "kmeans_dist_0": "Cluster Distance 0",
    "kmeans_dist_1": "Cluster Distance 1",
    "kmeans_dist_2": "Cluster Distance 2",
    "kmeans_dist_3": "Cluster Distance 3",
    "kmeans_dist_4": "Cluster Distance 4"
}


def load_dataset(dataset_path=None):
    """
    Load the encoded NFHS-5 dataset and extract features & target.
    """
    if dataset_path is None:
        # Default to processed dataset from nutrisense_V1
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        dataset_path = os.path.join(
            base_dir, "nutrisense_V1", "data", "processed", "tamilnadu_nfhs5_encoded.csv"
        )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Identifiers and targets to exclude from features
    drop_cols = [
        "caseid", "midx", "v001", "v002", "hhid",
        "stunting_haz", "state", "stunting_label"
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Clean & Impute Missing Data in DataFrame
    X = df[feature_cols].copy()
    y = df["stunting_label"].astype(int)

    # Domain Feature Enhancements
    # 1. WASH Vulnerability Score: (shared toilet or open defecation or unsafe water)
    open_defecation = X.get("toilet_type_31", pd.Series(0, index=X.index))
    shared_toilet = X.get("share_toilet", pd.Series(0, index=X.index)).fillna(0)
    X["wash_vulnerability_score"] = (open_defecation * 2.0) + (shared_toilet * 1.0)

    # 2. Adequate ANC Visits flag (≥4 visits according to WHO guidelines)
    if "anc_visits" in X.columns:
        X["anc_adequacy_flag"] = (X["anc_visits"] >= 4.0).astype(float)

    # 3. Disease Burden Index
    diarrhea = X.get("diarrhea_recent", pd.Series(0, index=X.index)).fillna(0)
    fever = X.get("fever_recent", pd.Series(0, index=X.index)).fillna(0)
    X["disease_burden_index"] = diarrhea + fever

    # 4. Maternal Vulnerability Score (Age <20 or >35, Low Edu, High Anemia, Low BMI)
    mom_age = X.get("mother_age", pd.Series(25, index=X.index))
    mom_age_risk = ((mom_age < 20) | (mom_age > 35)).astype(float)
    mom_edu = X.get("mother_education", pd.Series(1, index=X.index))
    mom_edu_risk = (mom_edu == 0).astype(float) * 1.5 + (mom_edu == 1).astype(float) * 0.5
    mom_anemia = X.get("mother_anemia", pd.Series(0, index=X.index)).fillna(0)
    mom_bmi = X.get("mother_bmi", pd.Series(20.0, index=X.index)).fillna(20.0)
    mom_bmi_risk = (mom_bmi < 18.5).astype(float) * 2.0
    X["maternal_vulnerability_score"] = mom_age_risk + mom_edu_risk + mom_anemia + mom_bmi_risk

    # 5. Child Vulnerability Score (High Birth Order + Short Interval)
    b_order = X.get("birth_order", pd.Series(1, index=X.index))
    b_interval = X.get("birth_interval", pd.Series(30, index=X.index))
    X["child_vulnerability_score"] = (b_order > 3).astype(float) * 1.5 + (b_interval < 24).astype(float) * 1.5

    # 6. Socio-Economic Synergy (Wealth x Education)
    wealth = X.get("wealth_index", pd.Series(3, index=X.index))
    X["wealth_education_synergy"] = wealth * mom_edu

    # 7. Unsafe Water x Diarrhea Pathway
    unsafe_water = X.get("water_source_31", pd.Series(0, index=X.index)) + X.get("water_source_92", pd.Series(0, index=X.index))
    X["unsafe_water_diarrhea_pathway"] = (unsafe_water > 0).astype(float) * diarrhea

    # 8. Extra High-Impact Interactions
    X["bmi_anemia_synergy"] = (mom_bmi < 18.5).astype(float) * mom_anemia
    child_age = X.get("child_age_months", pd.Series(24, index=X.index))
    X["child_age_x_wealth"] = child_age * wealth
    anc = X.get("anc_visits", pd.Series(2, index=X.index))
    X["education_x_anc"] = mom_edu * anc

    # 9. Non-Linear Transformations
    X["child_age_log"] = np.log(child_age + 1)
    is_critical_age = ((child_age >= 6) & (child_age <= 24)).astype(float)
    X["is_critical_age"] = is_critical_age
    X["mother_bmi_sq"] = mom_bmi ** 2
    X["wash_age_synergy"] = X["wash_vulnerability_score"] * is_critical_age
    X["wealth_edu_anc_triple"] = wealth * mom_edu * anc

    return X, y, FEATURE_LABELS

class AdvancedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn Transformer to apply PCA and KMeans clustering.
    Fit purely on training data to prevent data leakage.
    """
    def __init__(self, n_clusters=5, n_components=2, random_state=42):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        
        # Columns to use for PCA and KMeans
        self.pca_cols = [
            "wealth_index", "mother_education", "wash_vulnerability_score", 
            "toilet_type_31", "water_source_31", "electricity"
        ]
        self.cluster_cols = [
            "mother_age", "mother_bmi", "wealth_index", "mother_education", "child_age_months"
        ]
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            # Fallback if not a dataframe (though we expect it to be)
            return self
            
        # Ensure columns exist
        pca_valid = [c for c in self.pca_cols if c in X.columns]
        cluster_valid = [c for c in self.cluster_cols if c in X.columns]
        
        if pca_valid:
            self.pca.fit(X[pca_valid])
        if cluster_valid:
            self.kmeans.fit(X[cluster_valid])
            
        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            # Since pipeline steps might convert to numpy, try converting back or returning X
            # But the pipeline steps after SMOTE will receive numpy array.
            # So this transformer MUST be placed BEFORE SMOTE in the pipeline!
            return X
            
        X_out = X.copy()
        
        pca_valid = [c for c in self.pca_cols if c in X.columns]
        cluster_valid = [c for c in self.cluster_cols if c in X.columns]
        
        if pca_valid and hasattr(self.pca, "components_"):
            pca_features = self.pca.transform(X[pca_valid])
            for i in range(self.n_components):
                X_out[f"pca_socio_wash_{i+1}"] = pca_features[:, i]
        else:
            for i in range(self.n_components):
                X_out[f"pca_socio_wash_{i+1}"] = 0.0
                
        if cluster_valid and hasattr(self.kmeans, "cluster_centers_"):
            cluster_dists = self.kmeans.transform(X[cluster_valid])
            for i in range(self.n_clusters):
                X_out[f"kmeans_dist_{i}"] = cluster_dists[:, i]
        else:
            for i in range(self.n_clusters):
                X_out[f"kmeans_dist_{i}"] = 0.0
                
        return X_out
