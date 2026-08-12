# NutriSense — Project Status Report
**Generated:** 2026-08-06  
**Repository:** [dhanushkumar-karuppasamy/nutrisense](https://github.com/dhanushkumar-karuppasamy/nutrisense)  
**Scope:** Tamil Nadu, NFHS5 | **Target:** Child stunting prediction (HAZ < −2 SD)

---

## Repository Structure (Verified 2026-08-06)

```
nutrisense/
├── scripts/
│   ├── 01_extract_metadata.py          ✅ Done
│   ├── 02_build_feature_dictionary.py  ✅ Done
│   ├── 03_merge_datasets.py            ✅ Done (Week 3 expanded)
│   ├── 04_quality_checks.py            ✅ Done (imputation strategy)
│   ├── 05_encode_features.py           ✅ Done
│   └── 06_baseline_model.py            ✅ Done (stacking ensemble)
├── notebooks/
│   ├── 01_metadata_inspection.ipynb    ✅ Done
│   ├── 02_feature_mapping.ipynb        ✅ Done
│   ├── 03_eda_expanded.ipynb           ✅ Done
│   ├── 03_merge_and_qc.ipynb           ✅ Done
│   ├── 04_eda_plots.ipynb              ✅ Done
│   ├── 05_feature_engineering.ipynb    ✅ Done
│   └── 06_model_baseline.ipynb         ✅ Done
├── outputs/
│   ├── missingness_report.csv          ✅ Present
│   ├── figures/                        ✅ 12 PNGs (see figures index below)
│   └── models/
│       ├── baseline_stacking.pkl       ✅ Present (23.4 MB)
│       └── feature_names.txt           ✅ Present
├── data/
│   ├── raw/         (DTA files — gitignored, local only)
│   └── processed/   (CSVs — gitignored, local only)
├── paper/
│   └── abstract_skeleton.md            ✅ Present
├── project_log/
│   ├── decisions.md                    ✅ Present (4 decisions logged)
│   ├── week02_notes.md                 ✅ Present
│   └── project_status_report.md        ✅ THIS FILE
├── requirements.txt                    ✅ Updated
└── README.md                           ⚠️  Needs update (still reflects early scope)
```

---

## Figures Index (`outputs/figures/`)

| File | What it shows | Used in paper? |
|---|---|---|
| `stunting_distribution.png` | HAZ score distribution across TN children | Yes — Section III |
| `stunting_prevalence.png` | Class split: 74.2% not stunted / 25.8% stunted | Yes — Abstract |
| `stunting_by_category.png` | Stunting rate by wealth, education, toilet type | Yes — EDA section |
| `correlation_heatmap.png` | Full feature-vs-feature heatmap | Optional |
| `correlation_matrix.png` | Numeric correlation matrix | Optional |
| `corr_haz.png` | Per-feature correlation bar chart vs HAZ | Yes — Feature selection justification |
| `missing_data_matrix.png` | Missingness pattern across variables | Yes — Data preprocessing |
| `anc_by_stunting.png` | ANC visits boxplot by stunting label | Yes — Novelty argument |
| `mother_bmi_vs_stunting.png` | BMI boxplot by stunting label | Yes — Clinical feature value |
| `prev_by_education.png` | Stunting prevalence by mother education level | Yes — EDA |
| `roc_curve_lr.png` | ROC curve for baseline model | Yes — Results |
| `confusion_matrix_lr.png` | Confusion matrix for baseline model | Yes — Results |

---

## Week-by-Week Completed Work

### Week 0 — Project Setup & Literature Review
**Status: ✅ Complete**

- Identified research gap: existing ML stunting models (Arya et al., Erda et al.) lack explainability and use only socioeconomic features
- Chose NFHS5 (DHS) as data source — Tamil Nadu scope selected for state-level actionability
- Defined novelty: (1) clinical feature enrichment, (2) SHAP interpretability, (3) DiCE counterfactuals for intervention design
- Selected base comparison paper: Erda et al. (stacking ensemble, socioeconomic features only)
- Set up repository structure: `scripts/`, `notebooks/`, `data/`, `outputs/`, `paper/`
- Created `requirements.txt`

**Key outputs:** Repo initialized, literature review completed mentally, research gap defined.

---

### Week 1 — Data Exploration & Metadata
**Status: ✅ Complete**

- `01_extract_metadata.py`: Extracted DHS variable labels and code books from raw `.DTA` files using `pyreadstat`
- `02_build_feature_dictionary.py`: Built a human-readable mapping of DHS variable codes → feature names
- `notebooks/01_metadata_inspection.ipynb`: Visual inspection of DHS column structure across KR, IR, HR files
- `notebooks/02_feature_mapping.ipynb`: Confirmed Tamil Nadu state code = 33 in `v024`/`hv024`
- Identified target variable: `hw70` (HAZ × 100), confirmed WHO threshold HAZ < −2 SD = stunted

**Key outputs:** Feature dictionary, confirmed TN state code, verified variable availability across files.

---

### Week 2 — 3-Way Merge Pipeline
**Status: ✅ Complete**

- `03_merge_datasets.py` (original): Chunked reading via `pyreadstat` to avoid OOM on ~500k-row raw files
- Filtered KR, IR, HR to Tamil Nadu (state code = 33) before merging
- KR × IR: inner join on `caseid` (child's mother link)
- KR+IR × HR: inner join on `v001`+`v002` (cluster + household number)
- **Output:** `tamilnadu_nfhs5_merged.csv` — 13 columns (6 socioeconomic + IDs + target)
- `notebooks/03_merge_and_qc.ipynb`: Verified merge integrity, row counts, no duplicates
- `04_quality_checks.py` (original): Sentinel codes → NaN replacement only (no imputation yet)

**Gap identified at end of Week 2:** Feature set was purely socioeconomic — no clinical variables. Novelty claim vs. Erda et al. was not substantiated yet.

**Key outputs:** `tamilnadu_nfhs5_merged.csv`, chunked-loading pipeline validated.

---

### Week 3 — Feature Expansion, Imputation, Encoding, EDA, Baseline Model
**Status: ✅ Complete**

#### Task 1 — Feature Set Expansion ✅
- `03_merge_datasets.py` **updated**: Now pulls 22+ columns across KR, IR, HR
- **New clinical variables added:**
  - KR: `b4` (child sex), `hw1` (child age months), `b11` (birth interval), `bord` (birth order), `h11` (diarrhea), `h22` (fever), `m14_1` (ANC visits), `m15_1` (delivery place), `m17_1` (caesarean)
  - IR: `v012` (mother age), `v013` (mother age group), `v457` (mother anemia)
  - HR: `hv201` (water source)
- Feature matrix grew from 13 → **42 features** (including one-hot expanded nominals)
- Stunting prevalence confirmed at **25.8%** (5,890 total children in TN)

#### Task 2 — Missing Value Strategy ✅
- `04_quality_checks.py` **fully rewritten**: Documented imputation strategy per variable type
  - Continuous (BMI, ANC visits, birth interval): median imputation if ≤15% missing, MICE if >15%
  - Ordinal (education, wealth, anemia): mode OR explicit "missing" category (-1) if >10% missing
  - Nominal (toilet type, water source): mode imputation
  - Binary (electricity, sex, caesarean): mode imputation
- `outputs/missingness_report.csv` generated: per-column missing counts and percentages
- `outputs/figures/missing_data_matrix.png` generated

#### Task 3 — Feature Encoding ✅
- `05_encode_features.py` created:
  - Ordinal (education, wealth, anemia, age group): kept as ordered integers — preserves rank signal for XGBoost
  - Nominal (toilet type, water source, delivery place): one-hot encoded
  - Binary (electricity, share_toilet, caesarean, child sex): 0/1, left as-is or recoded
  - Target created: `stunting_label = (stunting_haz < -2.0).astype(int)`
- Output: `tamilnadu_nfhs5_encoded.csv`

#### Task 4 — EDA Notebook ✅
- `notebooks/03_eda_expanded.ipynb`: Full EDA on expanded 42-feature dataset
- `notebooks/04_eda_plots.ipynb`: Extended plots
- Figures generated (12 PNGs in `outputs/figures/`):
  - Correlation heatmap, HAZ distribution, stunting by category
  - ANC visits by stunting status (key novelty visualization)
  - BMI vs stunting, prevalence by education

#### Task 5 — Class Imbalance ✅
- Confirmed: **25.8% stunted / 74.2% not stunted** → 2.9:1 ratio (moderate imbalance)
- Decision: **SMOTE** applied (documented in `decisions.md`)
- After SMOTE: 3,498 stunted + 3,498 not stunted in training set (6,996 total)
- SMOTE applied on training set only — test set untouched

#### Task 6 — Baseline Model ✅
- `06_baseline_model.py`: Stacking ensemble replicating Erda et al. structure
  - Level 1: RandomForest (n=100) + GradientBoosting (n=100)
  - Level 2 meta-learner: Logistic Regression
  - CV: StratifiedKFold(n_splits=5)
- Stratified 80/20 train-test split confirmed (train 25.8% / test 25.7% stunting rate)
- `outputs/models/baseline_stacking.pkl` saved (23.4 MB)
- `outputs/models/feature_names.txt` saved (42 features listed)

#### Task 7 — Project Log Setup ✅
- `project_log/decisions.md`: 4 decisions logged (TN scope, HAZ target, SMOTE, stacking baseline)
- `project_log/week02_notes.md`: Retroactive Week 2 notes written
- `project_log/week03_notes.md`: ⚠️ **MISSING** — needs to be created

#### Task 8 — Paper Drafting (Parallel) ✅
- `paper/abstract_skeleton.md`: Abstract with placeholder numbers, structured correctly
- ⚠️ Related Work section not yet written — due this week

---

## Baseline Model Results Summary

| Metric | Not Stunted | Stunted | Overall |
|---|---|---|---|
| Precision | 0.76 | 0.42 | — |
| Recall | 0.94 | 0.13 | — |
| F1-Score | 0.84 | 0.19 | — |
| Accuracy | — | — | 0.73 |
| **ROC-AUC** | — | — | **0.5919** |

**Confusion Matrix:**
```
              Predicted Not  Predicted Stunted
Actual Not         823              52
Actual Stunted     265              38
```

### ⚠️ Critical Issue: Baseline ROC-AUC = 0.59

A ROC-AUC of 0.59 is only marginally above random (0.50). This is the **most important issue to address in Week 4**. The model is correctly classifying non-stunted children (recall 0.94) but almost completely failing on stunted children (recall 0.13 — only 38 of 303 stunted children caught).

**Root cause is almost certainly one of:**
1. The stacking ensemble's internal `StratifiedKFold` is running on SMOTE-augmented data, which may cause cross-validation leakage
2. The `GradientBoostingClassifier` meta-features are weak with default hyperparameters on 42 one-hot features
3. The `passthrough=False` means the LR meta-learner only sees 2 probabilities — may be information-starved

**Week 4 must investigate this before SHAP/DiCE — a model that can't find stunted children has nothing useful to explain.**

---

## Pending Tasks & Issues

### 🔴 Critical (Must Fix Before Moving Forward)

| # | Issue | Action |
|---|---|---|
| C1 | ROC-AUC = 0.59, stunted recall = 0.13 | Diagnose and fix in Week 4 (see Week 4 plan) |
| C2 | `week03_notes.md` missing from `project_log/` | Create immediately |
| C3 | `figures_index.md` missing from `project_log/` | Create (template in Week 3 plan) |

### 🟡 Important (Needed for Paper Quality)

| # | Issue | Action |
|---|---|---|
| I1 | README.md not updated since Week 0 | Update to reflect current 6-script pipeline |
| I2 | Related Work section not drafted | Write 3 paragraphs (Poshan Tracker → Arya et al. → Erda et al.) |
| I3 | `myenv/` folder committed to repo (23k+ files) | Add `myenv/` to `.gitignore`, remove from tracking |
| I4 | `baseline_stacking.pkl` is 23.4 MB in git | Add `outputs/models/*.pkl` to `.gitignore`, use Git LFS or exclude |
| I5 | `decisions.md` has no Week 3 model decisions | Log: why stacking L1 = RF+GBM, why passthrough=False, SMOTE strategy |

### 🟢 Next Phase Work (Week 4–6)

| # | Task | Week |
|---|---|---|
| N1 | Diagnose + fix baseline model (tune, XGBoost swap test) | Week 4 |
| N2 | SHAP analysis on best model | Week 4 |
| N3 | DiCE counterfactual generation | Week 5 |
| N4 | Cross-validation metrics (not just one split) | Week 4 |
| N5 | Paper: Related Work section (3 paragraphs) | Week 4 (parallel) |
| N6 | Paper: Methodology section | Week 5 |
| N7 | Paper: Results & Discussion | Week 6 |
| N8 | Final model selection + hyperparameter tuning | Week 5 |

---

## Week 4 Plan

### Primary Goal: Fix the model, then explain it

#### Task 4.1 — Diagnose Baseline Failure
Run these diagnostics on the existing `baseline_stacking.pkl` before any changes:

```python
# Check if SMOTE is causing CV leakage in StackingClassifier
# StackingClassifier's internal CV sees SMOTE-augmented data, which is correct
# But verify by checking cross_val_score separately
from sklearn.model_selection import cross_val_score, StratifiedKFold
scores = cross_val_score(stacking_clf, X_train, y_train, 
                          cv=StratifiedKFold(5), scoring='roc_auc')
print(f"CV ROC-AUC on ORIGINAL train (no SMOTE): {scores.mean():.4f} ± {scores.std():.4f}")
```

#### Task 4.2 — Try XGBoost as Single Model Benchmark
Before concluding the stacking is weak, test a tuned XGBoost directly:

```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                     scale_pos_weight=2.9,   # handles imbalance natively
                     eval_metric='auc', random_state=42)
xgb.fit(X_train, y_train)   # NO SMOTE — use scale_pos_weight instead
```

If XGBoost alone beats the stacking ensemble ROC-AUC, use XGBoost as your primary model (still keep stacking as the baseline comparison point in the paper).

#### Task 4.3 — SHAP Analysis (once model ≥ 0.70 AUC)
```python
# scripts/07_shap_analysis.py
import shap
import joblib

model = joblib.load("outputs/models/best_model.pkl")
explainer = shap.TreeExplainer(model)   # works for RF, GBM, XGBoost
shap_values = explainer.shap_values(X_test)

# Global: top 15 features
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.savefig("outputs/figures/shap_global_summary.png", dpi=150, bbox_inches="tight")

# Local: explain one stunted child
shap.waterfall_plot(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[idx],
    feature_names=X_test.columns.tolist()
))
```

#### Task 4.4 — Paper: Related Work Section
Three-paragraph structure (write in `paper/related_work.md`):

1. **Para 1 — Government/Production Systems:** Poshan Tracker (GoI) — monitoring/reporting tool, not predictive ML, no explainability
2. **Para 2 — ML-for-stunting literature:** Arya et al. — socioeconomic features only, feature importance (not SHAP), no counterfactuals
3. **Para 3 — Direct comparison:** Erda et al. — stacking ensemble, best prior work, no explainability layer, no intervention recommendations → leads directly to your contribution statement

---

## Contribution Statement (for paper)

> We extend Erda et al.'s stacking ensemble by:
> 1. Incorporating maternal and child **clinical features** (ANC visits, birth interval, birth order, maternal anemia) beyond the socioeconomic proxies used in prior work
> 2. Applying **SHAP** for per-child feature-level interpretability, enabling practitioners to understand individual risk factors
> 3. Generating **DiCE counterfactuals** to produce actionable intervention recommendations for modifiable risk factors

---

## Immediate Action Items (Do Before Week 4 Starts)

```
[ ] Create project_log/week03_notes.md
[ ] Create project_log/figures_index.md
[ ] Add myenv/ to .gitignore and run: git rm -r --cached myenv/
[ ] Add outputs/models/*.pkl to .gitignore
[ ] Update README.md to reflect 6-script pipeline
[ ] Add 3 more entries to decisions.md (model architecture choices)
[ ] Write paper/related_work.md (3 paragraphs)
```

---

*Report auto-generated from live repo audit on 2026-08-06. Update after each week.*
