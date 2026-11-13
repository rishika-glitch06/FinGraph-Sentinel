
# SHAP vs GNNExplainer Comparison

## Objective

This experiment compares feature-level explanations from two
different machine learning architectures:

- Random Forest explained using SHAP.
- Graph Convolutional Network explained using GNNExplainer.

## Experimental Setup

- Dataset: Elliptic Bitcoin transaction dataset.
- Temporal evaluation window: timesteps 42–49.
- SHAP aggregation: mean absolute SHAP values over 200 or fewer
  test samples.
- GNN explanation: one local node explanation.
- Common features compared: 165.

## Rank Correlation

- Spearman correlation: 0.122264
- Spearman p-value: 1.177137e-01
- Kendall correlation: 0.084405
- Kendall p-value: 1.075677e-01

## Top-K Feature Overlap

 top_k  overlap_count  overlap_percentage                                            common_features
     5              1                20.0                                                 feature_53
    10              2                20.0                                     feature_53, feature_90
    20              5                25.0 feature_52, feature_53, feature_55, feature_80, feature_90

## Top 15 Features

    feature  shap_importance  gnn_importance  shap_rank  gnn_rank  rank_difference
 feature_53         0.035741        0.428931        1.0       5.0              4.0
 feature_76         0.022709        0.364136        2.0     157.0            155.0
 feature_55         0.021705        0.417358        3.0      11.0              8.0
  feature_5         0.021056        0.380750        4.0      94.0             90.0
 feature_90         0.019070        0.426894        5.0       7.0              2.0
feature_132         0.019047        0.395307        6.0      35.0             29.0
 feature_14         0.018392        0.367047        7.0     149.0            142.0
 feature_18         0.018107        0.386618        8.0      66.0             58.0
feature_138         0.015654        0.375778        9.0     117.0            108.0
 feature_81         0.015462        0.367134       10.0     148.0            138.0
 feature_41         0.012833        0.367238       11.0     147.0            136.0
 feature_80         0.012446        0.415575       12.0      13.0              1.0
feature_156         0.012414        0.369365       13.0     142.0            129.0
feature_163         0.011843        0.383201       14.0      79.0             65.0
feature_144         0.010915        0.380253       15.0      97.0             82.0

## Interpretation

The rank correlation measures the degree of agreement between
the feature rankings produced by the two explanation methods.

Top-K overlap measures how many highly ranked features are shared.

A low agreement may indicate that the models rely on different
feature patterns, but it does not establish why the models differ.

## Limitations

1. SHAP explains a Random Forest, while GNNExplainer explains a GCN.
2. SHAP aggregates explanations across multiple test transactions.
3. The current GNNExplainer result represents one local node.
4. Raw importance magnitudes are not directly comparable.
5. Feature importance does not establish causality.
6. A single local GNN explanation cannot represent the entire graph.
7. These results do not establish which explanation is more correct.

## Conclusion

This experiment provides an initial comparison of tree-based
and graph-based explanations for illicit Bitcoin transaction
classification.

The results should be interpreted as model-specific explanation
agreement rather than evidence of explanation correctness.
