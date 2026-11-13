# 🔷 FinGraph Sentinel
### Graph-Based Illicit Bitcoin Transaction Detection & Explainable AI

FinGraph Sentinel is a research-oriented machine learning project focused on identifying illicit Bitcoin transactions using the Elliptic Bitcoin Transaction Dataset.

The project combines traditional machine learning, graph neural networks, temporal analysis, and Explainable AI to investigate not only predictive performance but also model transparency and explanation reliability.

---

## 🚀 Project Overview

The project investigates four major areas:

- **Transaction Classification:** Detecting licit and illicit Bitcoin transactions.
- **Graph Learning:** Modeling transaction relationships using Graph Convolutional Networks.
- **Temporal Drift:** Investigating changes in transaction feature distributions over time.
- **Explainable AI:** Comparing SHAP and GNNExplainer explanations.

## 🧠 Models Implemented

| Model | Approach |
|---|---|
| Random Forest | Traditional supervised learning |
| Temporal GCN | Graph-based classification |
| SHAP | Feature attribution |
| GNNExplainer | Local graph explanation |

## 📊 Dataset

**Dataset:** Elliptic Bitcoin Transaction Dataset

| Property | Value |
|---|---:|
| Transactions | 203,769 |
| Graph Edges | 234,355 |
| Time Steps | 49 |
| Original Features | 165 |

The dataset contains licit, illicit, and unknown transaction labels.

## 📈 Model Evaluation

### Random Forest

| Metric | Result |
|---|---:|
| Accuracy | 97.34% |
| Balanced Accuracy | 73.53% |
| Macro F1 | 80.39% |
| ROC-AUC | 87.67% |
| Illicit PR-AUC | 99.27% |

### Temporal GCN

| Metric | Result |
|---|---:|
| Accuracy | 59.50% |
| Balanced Accuracy | 70.28% |
| Macro F1 | 44.58% |
| ROC-AUC | 72.99% |
| Illicit PR-AUC | 98.25% |

The results highlight the importance of evaluating class imbalance rather than relying exclusively on accuracy.

## 🔬 Explainability Research

### SHAP Analysis
- Global feature importance.
- Feature attribution analysis.
- Stability analysis across background samples.

### GNNExplainer
- Local graph explanation.
- Node feature importance.
- Edge importance analysis.

### SHAP vs GNNExplainer

The comparison investigates the agreement between feature-level and graph-level explanations.

The experiment produced:
- Spearman rank correlation: 0.1223
- Kendall rank correlation: 0.0844
- Top-10 feature overlap: 20%

These results indicate weak agreement in the particular experiment. The methods have different explanation scopes, so this comparison should be interpreted as exploratory rather than a direct correctness evaluation.

## ⏳ Temporal Drift Analysis

The project investigates:
- Feature distribution changes.
- Class-conditional drift.
- Feature importance and drift relationships.
- Temporal model performance.
- Drift and prediction error associations.

Observed correlations are not interpreted as causal relationships.

## 🖥️ Interactive Dashboard

FinGraph Sentinel includes a Streamlit dashboard with the following sections:

1. Overview
2. Model Evaluation
3. Temporal Drift
4. Explainability
5. Transaction Explorer
6. Research Reports

The dashboard provides interactive access to generated experiment outputs and processed transaction data.

## 🏗️ Project Structure

```text
FinGraph-Sentinel/
│
├── app/
│   ├── dashboard.py
│   └── dashboard_backup.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── explainability/
│   ├── models/
│   ├── preprocessing/
│   └── features/
│
├── reports/
│   ├── explainability_comparison/
│   ├── gnn_explainability/
│   └── figures/
│
├── requirements.txt
└── README.md
```

*Directory names may vary slightly depending on the current implementation.*

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rishika-glitch06/FinGraph-Sentinel.git
cd FinGraph-Sentinel
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser.

## ⚠️ Research Limitations

- The dataset contains a substantial class imbalance.
- High accuracy does not necessarily indicate strong minority-class performance.
- Temporal drift does not establish causal relationships.
- The current GNNExplainer experiment represents a local explanation, not a dataset-wide explanation.
- SHAP and GNNExplainer operate at different explanation scopes.
- This project is a research prototype and is not a production financial surveillance system.

## 🎯 Research Direction

The broader research direction is to investigate the reliability, stability, and interpretability of machine learning explanations in graph-based financial transaction analysis.

---

**FinGraph Sentinel**  
Graph Machine Learning | Temporal Analysis | Explainable AI