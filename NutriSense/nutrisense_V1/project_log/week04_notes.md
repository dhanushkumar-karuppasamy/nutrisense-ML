# Week 4 Notes
**Date:** [fill in your actual date]

## What we did
- Diagnosed Week 3 stacking AUC (0.59) via scripts/07_diagnose_baseline.py
  - Root cause: SMOTE applied before StackingClassifier.fit() caused synthetic
    sample leakage across internal CV folds
  - Leak-free CV AUC: [fill from your run]
- Benchmarked XGBoost (scale_pos_weight) vs fixed stacking — winner: [XGBoost/Stacking]
- RandomizedSearchCV (30 iterations) — best CV AUC: [fill]
- SHAP analysis run — top 3 features: [fill after running 09_shap_analysis.py]
- Related work section drafted: paper/related_work.md

## Key decisions
- See project_log/decisions.md → Decision 5

## Blockers / open questions
- [fill in anything that didn't work]