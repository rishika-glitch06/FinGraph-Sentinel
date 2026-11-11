# Feature 2 Investigation

## 1. Objective

The objective of this investigation is to understand the behaviour of Feature 2 in the Elliptic Bitcoin Transaction Dataset and examine its relationship with temporal distribution shifts, model predictions, classification errors, and interactions with other features.

The investigation focuses on understanding whether Feature 2 contributes meaningful predictive information to the Random Forest model and whether its behaviour changes across different time periods.

## 2. Dataset and Experimental Scope

- Dataset: Elliptic Bitcoin Transaction Dataset
- Model: Random Forest Classifier
- Target timestep: 42
- Total labelled transactions at timestep 42: 2,154
- Licit transactions: 239
- Illicit transactions: 1,915
- Original input features: 165

The investigation uses the trained Random Forest model and focuses on Feature 2 and its relationship with selected features.
## 3. Temporal Drift Analysis

Feature 2 showed substantial distribution changes between the training and testing periods.

| Class | Training Mean | Testing Mean | KS Statistic |
|---|---:|---:|---:|
| Licit | -0.10247 | 0.29136 | 0.851335 |
| Illicit | -0.05522 | 0.67450 | 0.590648 |

Both classes exhibited considerable distribution shifts.

The KS statistics indicate that Feature 2 behaves differently across the training and testing periods, even when transactions are examined separately by class.

**Observation:** Feature 2 demonstrates substantial temporal drift. However, this alone does not establish that it causes model performance deterioration.
## 4. Feature 2 and Classification Errors

The analysis compared correctly classified licit transactions with licit transactions incorrectly predicted as illicit at timestep 42.

| Metric | Correctly Classified | False Positives |
|---|---:|---:|
| Sample count | 190 | 49 |
| Mean Feature 2 | 0.086437 | 0.103501 |

The difference in mean Feature 2 values was relatively small.

Statistical tests produced the following results:

- Kolmogorov–Smirnov statistic: 0.428571
- KS p-value: 5.56 × 10⁻⁷
- Mann–Whitney U p-value: 0.319545

The tests provide different perspectives on the distributions. The Mann–Whitney test does not show statistically significant evidence of a difference in central tendency.

**Observation:** Feature 2 alone does not provide sufficient evidence to explain the classification errors at timestep 42.
## 5. Outlier Analysis

An exploratory outlier analysis was conducted using the Interquartile Range (IQR) method.

The combined IQR boundaries for Feature 2 were:

- Lower bound: -0.10684
- Upper bound: 0.27990

Among the 49 false-positive transactions, 11 were identified as outliers.

After excluding these outliers from the false-positive group:

- Remaining false positives: 38
- Mean Feature 2: 0.051979
- Median Feature 2: 0.012390
- Mann–Whitney U p-value: 0.00319

The original Mann–Whitney p-value was 0.319545.

**Observation:** The statistical result changes after outlier exclusion. This suggests that extreme observations influence the comparison. However, this is an exploratory sensitivity analysis, not evidence that outliers cause classification errors.
## 6. Relationship Between Feature 2 and Model Predictions

The relationship between Feature 2 and the Random Forest model's predicted illicit probabilities was examined at timestep 42.

### Spearman Correlation

| Transaction Group | Spearman Correlation | P-value |
|---|---:|---:|
| Overall | -0.311407 | 1.18 × 10⁻⁴⁹ |
| Licit | -0.025549 | 0.69434 |
| Illicit | -0.256999 | 2.90 × 10⁻³⁰ |

### Pearson Correlation

- Overall: 0.015679
- Licit: 0.0780
- Illicit: -0.12424

The overall Spearman correlation is negative, while the overall Pearson correlation is close to zero.

This indicates that the relationship is not well described by a simple linear pattern. The relationship also differs between licit and illicit transactions.

**Observation:** Feature 2 has a non-linear association with predicted probabilities, particularly among illicit transactions. Correlation does not establish causation or prove that Feature 2 drives model predictions.
## 7. Permutation Importance Analysis

A permutation experiment was performed to evaluate the marginal predictive contribution of Feature 2.

Feature 2 values were randomly shuffled 20 times while keeping the remaining features unchanged.

### Baseline Performance

| Metric | Baseline |
|---|---:|
| ROC-AUC | 0.965998 |
| Balanced Accuracy | 0.896706 |
| Macro F1 | 0.933107 |

### Average Performance Change After Permutation

| Metric | Mean Performance Drop |
|---|---:|
| ROC-AUC | -0.000353 |
| Balanced Accuracy | -0.000065 |
| Macro F1 | 0.000101 |

Additional observations:

- Mean predicted probability change: 0.003639
- Mean rank correlation: 0.975877

**Observation:** Permuting Feature 2 produces almost no deterioration in model performance at timestep 42. This suggests that its marginal predictive contribution is negligible in this experiment. Correlated features may contain overlapping information.
## 8. Conditional Permutation Analysis

To investigate whether Feature 2 provides additional information beyond the other features, an approximate conditional permutation experiment was performed.

K-means clustering was applied to the remaining 164 features using 10 clusters. Feature 2 was then shuffled within these clusters over 20 repetitions.

### Results

| Metric | Baseline | Mean Performance Drop |
|---|---:|---:|
| ROC-AUC | 0.965998 | -0.000808 |
| Balanced Accuracy | 0.896706 | 0.000000 |
| Macro F1 | 0.933107 | 0.000000 |

Additional observations:

- Mean predicted probability change: 0.002500
- Mean rank correlation: 0.986585

**Observation:** Feature 2 continues to show negligible predictive contribution under this approximate conditional permutation approach.

**Limitation:** K-means clustering does not guarantee that the conditional distribution is preserved. Therefore, these results should be interpreted as exploratory rather than a formal conditional importance estimate.
## 9. Feature Interaction Analysis

Feature 2 was examined alongside five other important features using two-dimensional Partial Dependence Plots (PDPs).

The interaction strength was estimated using the Root Mean Square (RMS) of the non-additive interaction surface.

### Results

| Feature Pair | Interaction RMS | Maximum Absolute Interaction |
|---|---:|---:|
| Feature 2 × Feature 55 | 0.000130 | 0.001669 |
| Feature 2 × Feature 53 | 0.000432 | 0.007982 |
| Feature 2 × Feature 54 | 0.000040 | 0.000422 |
| Feature 2 × Feature 142 | 0.000034 | 0.000547 |
| Feature 2 × Feature 144 | 0.000077 | 0.001147 |

### Key Findings

- Feature 2 × Feature 53 showed the strongest interaction among the tested pairs.
- Feature 2 × Feature 55 showed the second strongest interaction.
- The remaining interaction magnitudes were comparatively smaller.
- The maximum observed interaction deviation was approximately 0.008.

**Observation:** The tested feature pairs exhibit small non-additive PDP interaction patterns. These values do not establish statistical significance or causality.

**Limitation:** PDP evaluates combinations of feature values that may not naturally occur together, particularly when features are correlated.
## 10. Overall Conclusion

The investigation examined Feature 2 from multiple perspectives, including temporal drift, classification errors, outliers, prediction probabilities, permutation importance, conditional permutation, and feature interactions.

The main findings are:

1. **Temporal drift:** Feature 2 exhibits substantial distribution shifts between training and testing periods, even within individual classes.
2. **Classification errors:** Feature 2 alone does not sufficiently explain false-positive predictions at timestep 42.
3. **Outliers:** Extreme observations influence the statistical comparison between correctly classified and incorrectly classified licit transactions.
4. **Prediction probabilities:** Feature 2 has a non-linear association with predicted illicit probabilities, particularly for illicit transactions.
5. **Permutation importance:** Both marginal and approximate conditional permutation experiments indicate negligible predictive contribution at timestep 42.
6. **Feature interactions:** Feature 2 × Feature 53 exhibits the strongest interaction pattern among the five tested pairs, although the measured magnitude remains small.

### Final Interpretation

Feature 2 is an interesting feature from a temporal drift and distribution-analysis perspective. However, the experiments do not establish that it is an important independent predictor or a direct cause of Random Forest classification errors.

Its behaviour highlights the importance of investigating temporal distribution shifts, feature dependencies, and model reliability rather than relying solely on feature importance rankings.

### Limitations

- Permutation experiments were restricted to timestep 42.
- Conditional permutation used approximate K-means clusters.
- PDP interaction analysis may be affected by correlated features.
- Statistical associations do not establish causality.
- Results should not be generalized to all timesteps without further experiments.