# SHAP Explainability and Reliability Analysis

## 1. Objective

The objective of this experiment is to investigate how SHAP (SHapley Additive exPlanations) explains the predictions made by the Random Forest model in the FinGraph-Sentinel project.

The analysis focuses on identifying influential features, understanding their contribution to model predictions, and evaluating the stability of global feature importance rankings under different background dataset selections.

## 2. Experimental Setup

- Dataset: Elliptic Bitcoin Transaction Dataset
- Model: Temporal Random Forest Classifier
- Explanation method: SHAP TreeExplainer
- SHAP stability analysis: Interventional explanations using two  background samples of 50 training transactions each
- Global explanation dataset: 200 transactions
- Feature count: 171
- Training period: Time steps 1–41
- Testing period: Time steps 42–49
## 3. SHAP Explanation Stability

To evaluate the sensitivity of SHAP explanations to background dataset selection, two different background samples were selected from the training period.

- Background A: 50 transactions (random_state=42)
- Background B: 50 transactions (random_state=100)
- Common explanation dataset: 100 test transactions

### Results

| Evaluation Metric | Result |
|---|---:|
| Spearman rank correlation | 0.990479 |
| P-value | 1.78 × 10⁻¹⁴⁷ |
| Top-10 feature overlap | 8/10 |
| Mean absolute importance difference | 0.000266 |

### Interpretation

The Spearman correlation of approximately 0.99 indicates very high consistency between the global SHAP feature rankings obtained using the two background samples.

Eight out of the top ten features were common to both experiments.

**Finding:** Global SHAP feature rankings showed strong consistency under the tested background sampling conditions.

**Limitation:** This experiment evaluates only background-sampling sensitivity. It does not establish stability across different time periods, models, or individual transaction explanations.
## 4. Most Influential Features

The following features appeared among the top 15 global SHAP features in the stability experiment.

| Feature | Background A Importance | Background B Importance |
|---|---:|---:|
| Feature 53 | 0.021843 | 0.019115 |
| Feature 55 | 0.014371 | 0.011295 |
| Feature 65 | 0.011882 | 0.012588 |
| Feature 59 | 0.010019 | 0.010332 |
| Feature 66 | 0.008492 | 0.009253 |
| Feature 5 | 0.007806 | 0.005139 |
| Feature 90 | 0.007472 | 0.005930 |
| Feature 16 | 0.007204 | 0.007665 |
| Feature 52 | 0.006920 | 0.005655 |
| Feature 23 | 0.006670 | 0.006817 |

Feature 53 had the highest mean absolute SHAP importance in both background experiments.

**Observation:** The leading features maintained relatively consistent importance rankings, although their individual importance magnitudes changed slightly between background samples.
## 5. SHAP Visualizations

The SHAP analysis generated two visualizations to interpret the Random Forest model.

### 5.1 SHAP Summary Plot

File: `reports/figures/shap_summary.png`

This plot displays the distribution of SHAP values across transactions and features. It helps identify which features have the greatest influence on model predictions and how their values relate to prediction contributions.

### 5.2 SHAP Feature Importance Plot

File: `reports/figures/shap_importance.png`

This plot presents global feature importance using mean absolute SHAP values. It provides an overall ranking of features based on their contribution magnitude.

Both visualizations were generated using 200 test transactions and 171 input features.
## 6. Overall Conclusion

The SHAP experiments demonstrate that the Random Forest model can be interpreted using global feature attribution.

The main findings are:

1. SHAP successfully generated explanations for 200 test transactions across 171 features.
2. Feature 53 showed the highest global mean absolute SHAP importance in the stability experiment.
3. The global feature rankings demonstrated high consistency across two background samples.
4. The top-10 feature overlap was 8 out of 10.
5. Background selection caused small changes in individual feature importance magnitudes.

### Limitations

- The stability experiment used only two background samples.
- The analysis focused on global feature rankings rather than individual prediction explanations.
- Stability across different temporal periods has not yet been established.
- SHAP importance indicates model attribution, not causal influence.
- The experiments do not establish that the model's predictions are correct or fair.

### Future Work

Further experiments should investigate temporal explanation stability, local explanation consistency, and agreement between SHAP and other explainability methods such as LIME.