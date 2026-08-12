# NutriSense — Running Decisions Log

## [2026-07-31] Decision: TN-only scope
**Context:** NFHS5 covers all of India; project scope needed narrowing.
**Options:** All-India vs. state-level vs. multi-state
**Choice:** Tamil Nadu only
**Reasoning:** Reduces confounders (agro-ecological, policy variation), makes 
clinical recommendations actionable at state level, consistent with DHS sub-national analysis literature.
**Reversible?:** Yes — TN_STATE_CODE=33 is one constant in 03_merge_datasets.py.

## [2026-07-31] Decision: HAZ as target (not WAZ or WHZ)
**Context:** WHO defines three z-scores for child nutrition status.
**Options:** HAZ (height-for-age), WAZ (weight-for-age), WHZ (weight-for-height)
**Choice:** HAZ → stunting (HAZ < -2 SD)
**Reasoning:** Stunting measures chronic malnutrition (long-term), not acute wasting.
Our interventions (ANC visits, birth spacing) operate on chronic pathways.
HAZ is the primary outcome in Erda et al. — needed for baseline comparison.
**Reversible?:** Yes — change threshold in 05_encode_features.py.

## [2026-07-31] Decision: SMOTE over class weights
**Context:** Stunting prevalence ~19–23% (moderate imbalance).
**Options:** Class weights in model | SMOTE | leave as-is
**Choice:** SMOTE
**Reasoning:** Class weights only change the loss function; SMOTE actually 
provides the model with more minority-class training examples. 
At 4:1 ratio, SMOTE with k=5 is standard practice (Chawla et al. 2002).
"Proper class-imbalance handling" is stated novelty vs. Erda et al.
**Reversible?:** Yes — comment out SMOTE block in 06_baseline_model.py and pass class_weight='balanced' to estimators instead.

## [2026-07-31] Decision: Stacking ensemble as baseline
**Context:** Need a benchmark before adding SHAP/DiCE.
**Options:** XGBoost alone | Random Forest alone | Stacking (Erda et al. structure)
**Choice:** Stacking
**Reasoning:** Replicating Erda et al.'s approach makes the comparison in the 
paper fair (apples-to-apples). Stating "our SHAP+DiCE addition improves 
ROC-AUC from X to Y over the same ensemble" is a clean contribution.
**Reversible?:** N/A — this is the baseline, not the final model.

## Decision 6 (Week 4): Final AUC ceiling — documented as data limitation

After fixing three root causes sequentially:
  1. Sparse OHE columns (9 rare categories) — collapsed into 'other'
  2. Missing high-value predictors (anc_visits, delivery_place, caesarean) —
     root cause: 03_merge_datasets.py requested m14_1/m15_1/m17_1, but the
     actual DTA variable names in this DHS round are m14/m15/m17 (no suffix).
     pyreadstat silently drops unmatched usecols instead of erroring.
  3. Verified: no SMOTE leakage, no BMI sentinel corruption, mild household
     overlap only (23% of households have >1 child)

Result: CV ROC-AUC improved from 0.590 -> 0.602 (tuned Stacking RF+GBM).
Test ROC-AUC remained flat: 0.5804 -> 0.5818.

Conclusion: The Tamil Nadu NFHS-5 subsample (5,890 records, 25.8% stunted)
has a genuine AUC ceiling around 0.58-0.60 for this feature set. This is
consistent with known limitations of cross-sectional household survey data
for individual-level clinical prediction — stunting is a cumulative,
multi-year biological outcome, while most available predictors capture a
single point in time. This ceiling is documented as a limitation in the
Results/Discussion section rather than pursued further via additional
model tuning, which showed diminishing and non-generalizing returns.

Final model: Stacking (RF+GBM), Test AUC 0.582, Test Recall (stunted) 0.125.
Proceeding to SHAP with this AUC labeled as exploratory/limited-signal
analysis, per faculty guidance on transparent limitation reporting.