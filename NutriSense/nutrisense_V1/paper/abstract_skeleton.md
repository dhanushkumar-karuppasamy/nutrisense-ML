# Abstract (Skeleton — Week 3, numbers TBD)

Child stunting remains a critical public health challenge in Tamil Nadu, India, 
with prevalence rates of approximately [X]% in NFHS5 data. Existing ML-based 
prediction systems, including Erda et al.'s stacking ensemble, lack explainability 
layers and actionable intervention recommendations. We present NutriSense, a 
multi-feature stacking ensemble augmented with SHAP-based per-child interpretability 
and DiCE counterfactual generation for intervention design. Our model incorporates 
maternal clinical variables (anemia, ANC visits, delivery place) and child-level 
features (birth interval, birth order) that go beyond the socioeconomic proxies used 
in prior work. On the TN NFHS5 dataset, our baseline stacking model achieves 
ROC-AUC of [X.XX], with SHAP analysis identifying [top feature] as the strongest 
predictor. DiCE counterfactuals demonstrate actionable intervention paths for [X]% 
of stunted children through modifiable risk factors.

**Keywords:** child stunting, SHAP, DiCE, counterfactuals, NFHS5, Tamil Nadu