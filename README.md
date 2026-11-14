# 🔷 FinGraph Sentinel
### Graph-Based Illicit Bitcoin Transaction Detection & Explainable AI

FinGraph Sentinel is a research-oriented machine learning project focused on detecting illicit Bitcoin transactions using the Elliptic Bitcoin Transaction Dataset.

The project combines traditional machine learning, graph neural networks, temporal analysis, and Explainable AI (XAI) to investigate predictive performance, feature drift, and explanation reliability.

---

## 🚀 Project Overview

The project investigates four major areas:

- **Transaction Classification:** Identifying licit and illicit Bitcoin transactions.
- **Graph Learning:** Modeling transaction relationships using Graph Convolutional Networks.
- **Temporal Drift:** Studying changes in transaction feature distributions across time.
- **Explainable AI:** Comparing SHAP and GNNExplainer explanations.

## 🧠 Models and Methods

| Model / Method | Purpose |
|---|---|
| Random Forest | Supervised transaction classification |
| Temporal GCN | Graph-based transaction classification |
| SHAP | Feature attribution and explanation stability |
| GNNExplainer | Local graph and node explanation |
| Temporal Drift Analysis | Feature distribution and prediction error investigation |

## 📊 Dataset

**Dataset:** Elliptic Bitcoin Transaction Dataset

| Property | Value |
|---|---:|
| Transactions | 203,769 |
| Graph Edges | 234,355 |
| Time Steps | 49 |
| Original Features | 165 |
| Illicit Transactions | 4,545 |
| Licit Transactions | 42,019 |
| Unknown Transactions | 157,205 |

The dataset contains transaction features, transaction labels, and directed relationships between transactions.

Raw and processed datasets are excluded from Git version control because of their size.

## 📈 Model Evaluation

### Random Forest

Temporal evaluation uses earlier time steps for training, intermediate time steps for validation, and later time steps for testing.

| Metric | Result |
|---|---:|
| Accuracy | 97.34% |
| Balanced Accuracy | 73.53% |
| Macro F1 | 80.39% |
| ROC-AUC | 87.67% |
| Illicit PR-AUC | 99.27% |
| Licit Recall | 47.30% |
| Illicit Recall | 99.76% |

### Temporal GCN

| Metric | Result |
|---|---:|
| Accuracy | 59.50% |
| Balanced Accuracy | 70.28% |
| Macro F1 | 44.58% |
| ROC-AUC | 72.99% |
| Illicit PR-AUC | 98.25% |
| Licit Recall | 82.11% |
| Illicit Recall | 58.45% |

The results demonstrate why accuracy alone is insufficient for evaluating models on highly imbalanced transaction datasets.

## 🔬 Explainability Research

### SHAP Analysis

- Global feature importance.
- Feature attribution analysis.
- Explanation stability across background samples.
- Feature ranking consistency.

### GNNExplainer

- Local graph explanations.
- Node feature importance.
- Edge importance analysis.

### SHAP vs GNNExplainer

The experiment investigates agreement between feature-level and graph-level explanations.

| Metric | Result |
|---|---:|
| Spearman Correlation | 0.1223 |
| Kendall Correlation | 0.0844 |
| Top-5 Feature Overlap | 20% |
| Top-10 Feature Overlap | 20% |
| Top-20 Feature Overlap | 25% |

The observed agreement was weak and statistically nonsignificant in this experiment.

The comparison is exploratory: SHAP uses aggregated feature importance, whereas GNNExplainer represents a local explanation for an individual graph prediction. The results should not be interpreted as a direct correctness comparison.

## ⏳ Temporal Drift Analysis

The project investigates:

- Feature distribution changes over time.
- Class-conditional feature drift.
- Feature importance and drift relationships.
- Temporal model performance.
- Prediction errors associated with feature drift.
- Feature-level investigations and ablation experiments.

Observed statistical associations are not interpreted as causal relationships.

## 🖥️ Interactive Dashboard

The project includes a Streamlit dashboard with six sections:

1. Overview
2. Model Evaluation
3. Temporal Drift
4. Explainability
5. Transaction Explorer
6. Research Reports

The dashboard reads generated reports and processed transaction data.

## 📁 Project Structure

```text
FinGraph-Sentinel/
│
├── app/
│   └── dashboard.py
│
├── configs/
│
├── data/
│   ├── raw/                  # Downloaded source datasets
│   └── processed/            # Generated datasets
│
├── experiments/
│
├── models/                   # Locally saved trained models
│
├── notebooks/
│
├── reports/
│   ├── explainability_comparison/
│   ├── gnn_explainability/
│   └── figures/
│
├── src/
│   ├── evaluation/
│   ├── explainability/
│   ├── features/
│   ├── models/
│   └── preprocessing/
│
├── tests/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

**Note:** Large datasets, locally saved models, and the Python virtual environment are excluded from Git.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/rishika-glitch06/FinGraph-Sentinel.git
cd FinGraph-Sentinel
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 📦 Dataset Setup

Download the three original Elliptic dataset CSV files.

Required filenames:

```text
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

Place them inside:

```text
data/
└── raw/
    ├── elliptic_txs_features.csv
    ├── elliptic_txs_classes.csv
    └── elliptic_txs_edgelist.csv
```

The raw dataset files are not included in this repository.

## 🔄 Data Preprocessing and Feature Engineering

Run the scripts from the project root directory, in the following order.

### Step 1: Clean and merge the raw dataset

```bash
python src/preprocessing/clean_data.py
```

Generates:

- `data/processed/transactions.csv`
- `data/processed/edges.csv`

### Step 2: Generate graph features

```bash
python src/features/graph_features.py
```

Generates:

- `data/processed/transactions_with_graph_features.csv`

### Step 3: Generate advanced graph features

```bash
python src/features/advanced_graph_features.py
```

Generates:

- `data/processed/transactions_advanced_features.csv`

This dataset includes connected-component size, clustering coefficient, and core number.

### Optional: Generate temporal graph features

```bash
python src/features/temporal_graph_features.py
```

Generates:

- `data/processed/transactions_temporal_features.csv`

This is a separate temporal feature-generation pipeline.

**Note:** Feature generation on the complete dataset can require substantial memory and processing time.

## ▶️ Run the Dashboard

After generating the advanced graph feature dataset:

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser.

## ⚠️ Research Limitations

- The dataset contains substantial class imbalance.
- High accuracy does not necessarily indicate strong minority-class performance.
- Temporal feature drift does not establish causal relationships.
- The current GNNExplainer experiment represents a local explanation, not a dataset-wide explanation.
- SHAP and GNNExplainer operate at different explanation scopes.
- The project is a research prototype and is not intended for production financial surveillance.

## 🎯 Research Direction

The broader research direction is to investigate the reliability, stability, and interpretability of machine learning explanations in graph-based financial transaction analysis.

## 📚 Dataset Reference

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., and Leiserson, C. E.

*Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.*

KDD 2019 Workshop on Anomaly Detection in Finance.

---

**FinGraph Sentinel**

Graph Machine Learning | Temporal Analysis | Explainable AI