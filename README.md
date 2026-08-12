# NutriSense: Explainable AI Framework for Childhood Stunting Prevention

NutriSense is an advanced Explainable AI (XAI) dashboard and Machine Learning pipeline designed to predict childhood stunting risk indicators (under the NFHS-5 dataset standard) and generate actionable, patient-specific healthcare interventions.

The tool is aimed at policy makers, clinical experts, and community health workers (such as ASHA/ANM workers) in India to target malnutrition proactive interventions.

---

## 🌟 Key Features

1. **Stunting Risk Screening**: Enter demographic, clinical, and WASH (Water, Sanitation, and Hygiene) data to calculate a 0–100 Stunting Risk Index (SRI).
2. **SHAP Interpretability Hub**: Uses Game-Theory-based KernelSHAP values to decompose individual child risk predictions into positive risk accelerators and protective buffers.
3. **DiCE "What-If" Simulation**: Simulates counterfactual intervention packages (WASH upgrades, maternal healthcare, and integrated support packages) showing target risk reduction values.
4. **Model Trust & CV Benchmarking**: Audits 8 classifiers (Logistic Regression, RandomForest, Gradient Boosting, XGBoost, LightGBM, and Stacking Ensembles) trained under Stratified 5-Fold Cross-Validation with ADASYN oversampling.
5. **ASHA Quick-Check**: A simplified, mobile-friendly village screening interface for community workers in remote areas.

---

## 📂 Repository Structure

The codebase is organized into clean, modular layers:

```
nutrisense-ML/
├── .gitignore                  # Ignores large datasets, virtual envs, and compiled binaries
├── README.md                   # Main documentation
├── NutriSense/                 # Core Web Application Directory
│   ├── app.py                  # Flask web backend & REST API endpoints
│   ├── models/                 # Model metadata, performance stats, and schemas
│   │   ├── model_meta.json     # Accuracy, precision, recall, and selected best model details
│   │   ├── feature_names.json  # Feature ordering schemas
│   │   └── feature_labels.json # Human-readable label strings for dashboards
│   ├── static/                 # Web assets
│   │   ├── css/styles.css      # Premium dark-accented CSS stylesheet
│   │   └── js/app.js           # Client-side Chart.js visualizations & routing
│   ├── templates/
│   │   └── index.html          # HTML dashboard grid and tabs
│   └── src/
│       └── ml/
│           ├── data_loader.py  # Advanced feature engineering & preprocessing
│           ├── train_model.py  # Model comparison, stratified cross-validation, and training
│           ├── shap_engine.py  # SHAP explanation vector builder
│           └── dice_engine.py  # DiCE counterfactual packages generator
└── NutriSense/nutrisense_V1/   # Historical data pre-processing scripts (NFHS-5 data merge & QC)
```

> [!NOTE]
> Training dataset CSV files (`X_train.csv`, `X_test.csv`) and heavy pre-trained model pickles (`best_model_pipeline.pkl`, `calibrated_model.pkl`) are excluded from this repository via `.gitignore` to keep git clean and lightweight. You can generate them at any time by running the training pipeline.

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/dhanushkumar-karuppasamy/nutrisense-ML.git
cd nutrisense-ML
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install flask joblib scikit-learn shap dice-ml pandas numpy xgboost lightgbm tensorflow imbalanced-learn
```

---

## 🚀 Running the Project

### Start the Dashboard Web App
Run the Flask server from the root of the project:
```bash
python NutriSense/app.py
```
Open **`http://127.0.0.1:5000`** in your browser to view the interactive dashboard.

---

## 📈 Training & Model Calibration

If you need to retrain the classifiers, compute benchmarks, or regenerate the serialized model pickles, run:
```bash
# Execute model comparison pipeline
python NutriSense/src/ml/train_model.py
```
This script will:
1. Load dataset configurations from `data_loader.py`.
2. Compare 8 classifiers using Stratified 5-Fold CV.
3. Automatically select the model with the highest CV ROC-AUC.
4. Serialize and save the model pipelines and metadata back into the `models/` directory.
