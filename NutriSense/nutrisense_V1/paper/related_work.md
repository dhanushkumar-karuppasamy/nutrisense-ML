# Related Work

## Government Nutrition Monitoring Systems

The Government of India's Poshan Tracker is the primary operational tool for
monitoring child nutrition at the national level. It functions as a program
delivery dashboard, tracking beneficiary coverage of ICDS (Integrated Child
Development Services) interventions such as supplementary nutrition, growth
monitoring sessions, and health check-ups [cite NITI Aayog / Poshan Tracker
documentation]. While Poshan Tracker enables real-time tracking of scheme
delivery, it is not a predictive tool: it records whether a child received an
intervention but cannot identify which currently healthy children are at
elevated risk of future stunting. It further lacks any per-child explainability
or actionable counterfactual guidance. NutriSense addresses precisely this gap
by shifting from retrospective monitoring to prospective, explainable risk
prediction.

## Regression-Based Feature Importance Approaches

Arya et al. (2022) conducted a multi-state regression analysis of stunting
determinants using NFHS-4 data, identifying maternal education, wealth index,
and sanitation access as significant predictors at the district level. Their
approach established statistically rigorous associations between socioeconomic
variables and stunting outcomes, providing macro-level policy evidence [cite
Arya et al.]. However, regression-based feature importance (standardized
coefficients or odds ratios) operates at a population level: it reports that
"children from lower wealth quintiles are 2.3× more likely to be stunted on
average" but cannot generate a risk score or an actionable recommendation for
a specific child. NutriSense transitions this population-level insight into
individual-level prediction by treating stunting as a supervised classification
problem, enabling per-child risk scoring that district-level regression cannot
produce.

## Ensemble Machine Learning Without Explainability

Erda et al. (2023) introduced a stacking ensemble approach — combining Random
Forest and Gradient Boosting as level-1 classifiers with Logistic Regression
as a meta-learner — for childhood stunting prediction on NFHS data, reporting
an accuracy of 0.79 [cite Erda et al.]. However, their evaluation relied
exclusively on accuracy, a metric that is uninformative under class imbalance
(approximately 35–38% stunting prevalence nationally): a classifier predicting
"not stunted" for all children would achieve ~62–65% accuracy without any
predictive value. Their reported ROC-AUC was not provided, a significant
omission for a binary health risk classifier. Additionally, their model
provides no post-hoc explainability layer (no SHAP attributions, no
counterfactual guidance) and no Tamil Nadu-specific analysis. NutriSense
directly addresses these limitations: we replicate their stacking architecture
as our baseline (correcting the evaluation methodology to report ROC-AUC),
add SHAP-based individual explainability, integrate DiCE counterfactual
recommendations restricted to modifiable clinical variables, and tailor the
analysis to Tamil Nadu's socioeconomic and nutritional profile using NFHS-5.