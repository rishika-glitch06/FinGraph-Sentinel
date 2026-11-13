from pathlib import Path
import streamlit as st
import pandas as pd

# ============================================================
# FIN GRAPH SENTINEL — RESEARCH DASHBOARD
# Redesigned: compact top navigation, editorial hero, KPI cards,
# focused analytical workspaces, and responsive report panels.
# ============================================================

st.set_page_config(
    page_title="FinGraph Sentinel",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed"

# ---------- Visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
  --ink: #14243a;
  --muted: #68788d;
  --paper: #f4f7fb;
  --white: #ffffff;
  --line: #e2e9f1;
  --navy: #10243d;
  --blue: #2563eb;
  --teal: #0f9d91;
  --amber: #e6a23c;
}
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  background: var(--paper);
  color: var(--ink);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container {
  max-width: 1450px;
  padding-top: 1.25rem;
  padding-bottom: 3rem;
}
#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

.brand-row {
  display:flex; align-items:center; justify-content:space-between;
  padding: 4px 0 14px 0; border-bottom:1px solid var(--line);
  margin-bottom: 18px;
}
.brand {
  display:flex; gap:11px; align-items:center;
  color:var(--navy); font-family:Manrope,sans-serif;
  font-weight:800; font-size:19px; letter-spacing:-.5px;
}
.brand-mark {
  width:37px; height:37px; border-radius:12px;
  background:linear-gradient(135deg,#2563eb,#14b8a6);
  display:flex; align-items:center; justify-content:center;
  color:white; font-size:19px; font-weight:800;
  box-shadow:0 6px 16px #2563eb30;
}
.brand-meta {
  color:var(--muted); font-size:11px; letter-spacing:1.3px;
  font-weight:700; text-transform:uppercase;
}
.hero {
  background:linear-gradient(115deg,#10243d 0%,#173e63 62%,#126b70 100%);
  border-radius:22px; padding:30px 34px; color:white;
  position:relative; overflow:hidden; margin:8px 0 22px 0;
  box-shadow:0 14px 34px #173b5d20;
}
.hero:after {
  content:""; position:absolute; width:310px; height:310px;
  border:1px solid #ffffff20; border-radius:50%;
  right:-65px; top:-150px; box-shadow:0 0 0 35px #ffffff08,0 0 0 75px #ffffff06;
}
.hero-kicker {
  color:#9ddbd7; text-transform:uppercase; letter-spacing:2px;
  font-size:10px; font-weight:800; margin-bottom:10px;
}
.hero h1 {
  color:white; font-family:Manrope,sans-serif; font-size:32px;
  line-height:1.15; margin:0 0 9px 0; font-weight:800;
  letter-spacing:-1px;
}
.hero p { color:#d4e4f1; font-size:14px; margin:0; max-width:720px; }
.hero-tag {
  display:inline-block; margin-top:18px; padding:6px 11px;
  border:1px solid #ffffff35; border-radius:20px;
  color:#e4f4f5; font-size:11px; font-weight:600;
}
.section-head {
  display:flex; align-items:end; justify-content:space-between;
  margin:22px 0 12px 0;
}
.section-head h2 {
  font-family:Manrope,sans-serif; color:var(--navy);
  font-size:19px; font-weight:800; margin:0;
}
.section-head span { color:var(--muted); font-size:11px; }
.kpi {
  background:white; border:1px solid var(--line); border-radius:16px;
  padding:17px 18px; min-height:108px;
  box-shadow:0 3px 12px #19324b08;
}
.kpi-label {
  color:var(--muted); font-size:11px; font-weight:700;
  letter-spacing:.5px; text-transform:uppercase;
}
.kpi-value {
  color:var(--navy); font-family:Manrope,sans-serif;
  font-size:27px; font-weight:800; margin-top:7px;
  letter-spacing:-.7px;
}
.kpi-note { color:#8492a4; font-size:10px; margin-top:2px; }
.panel {
  background:white; border:1px solid var(--line); border-radius:17px;
  padding:19px 20px; box-shadow:0 3px 12px #19324b08;
  margin-bottom:14px;
}
.panel-title {
  font-family:Manrope,sans-serif; color:var(--navy);
  font-size:14px; font-weight:800; margin-bottom:4px;
}
.panel-sub { color:var(--muted); font-size:11px; margin-bottom:12px; }
.pill {
  display:inline-block; border-radius:20px; padding:5px 9px;
  background:#eaf2ff; color:#2457a6; font-size:10px; font-weight:700;
}
div[data-testid="stMetric"] {
  background:white; border:1px solid var(--line); border-radius:14px;
  padding:13px 15px; box-shadow:0 3px 12px #19324b08;
}
div[data-testid="stMetricLabel"] { color:var(--muted); }
div[data-testid="stMetricValue"] { color:var(--navy); }
.stTabs [data-baseweb="tab-list"] {
  gap:8px; background:#eaf0f6; padding:5px; border-radius:12px;
}
.stTabs [data-baseweb="tab"] {
  height:38px; border-radius:9px; color:#52647a; font-weight:700;
}
.stTabs [aria-selected="true"] {
  background:white !important; color:#1d4ed8 !important;
}
div[data-testid="stDataFrame"] {
  border:1px solid var(--line); border-radius:12px; overflow:hidden;
}
.stButton button, .stDownloadButton button {
  border-radius:10px; font-weight:700;
}
div[role="radiogroup"] {
  gap:6px; flex-wrap:wrap; padding:5px 0 12px 0;
}
div[role="radiogroup"] label {
  background:white; border:1px solid var(--line);
  border-radius:20px; padding:7px 13px !important;
  color:#53647a; font-size:12px; font-weight:700;
}
div[role="radiogroup"] label:has(input:checked) {
  background:#e8f0ff; border-color:#9dbaf5; color:#1d4ed8;
}
div[data-testid="stAlert"] { border-radius:12px; }
.footer {
  border-top:1px solid var(--line); margin-top:32px; padding-top:15px;
  color:#8a98a9; font-size:10px; display:flex;
  justify-content:space-between;
}
.small-note { color:var(--muted); font-size:11px; }
</style>
""", unsafe_allow_html=True)


# ---------- Data and artifact helpers ----------
@st.cache_data(show_spinner=False)
def read_csv(path_string, nrows=None):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return None


def find_artifact(filename):
    direct = REPORTS / filename
    if direct.exists():
        return direct
    matches = list(REPORTS.rglob(filename)) if REPORTS.exists() else []
    return matches[0] if matches else None


def image_panel(filename, caption=None):
    path = find_artifact(filename)
    if path:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Visual not generated yet: {filename}")


def csv_panel(filename, height=300):
    path = find_artifact(filename)
    if not path:
        st.info(f"Report not found: {filename}")
        return
    df = read_csv(str(path))
    if df is None:
        st.warning(f"Could not read {filename}")
        return
    st.dataframe(df, use_container_width=True, height=height)


def markdown_panel(filename):
    path = find_artifact(filename)
    if not path:
        st.info(f"Report not found: {filename}")
        return
    try:
        st.markdown(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not open report: {exc}")


def text_panel(filename):
    path = find_artifact(filename)
    if path:
        st.text(path.read_text(encoding="utf-8"))
    else:
        st.info(f"Report not found: {filename}")


def heading(title, subtitle=""):
    st.markdown(
        f'<div class="section-head"><div><h2>{title}</h2>'
        f'<span>{subtitle}</span></div></div>',
        unsafe_allow_html=True
    )


def kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True
    )


# ---------- Brand header ----------
st.markdown("""
<div class="brand-row">
  <div class="brand"><div class="brand-mark">F</div>
    <div>FINGRAPH <span style="color:#0f9d91">SENTINEL</span></div>
  </div>
  <div class="brand-meta">TRANSACTION INTELLIGENCE&nbsp;&nbsp; / &nbsp;&nbsp; RESEARCH LAB</div>
</div>
""", unsafe_allow_html=True)

# Horizontal app navigation — replaces the oversized sidebar
pages = [
    "Overview", "Model Lab", "Temporal Intelligence",
    "Explainability Studio", "Transaction Explorer", "Reports"
]
page = st.radio("Workspace", pages, horizontal=True, label_visibility="collapsed")

# ---------- Shared hero ----------
hero_text = {
    "Overview": ("RESEARCH DASHBOARD", "Graph-based illicit Bitcoin transaction detection",
                 "A unified view of dataset scale, predictive performance and explainability experiments."),
    "Model Lab": ("MODEL PERFORMANCE", "Evaluate the models beyond accuracy",
                  "Compare Random Forest and Temporal GCN across class-wise and time-aware metrics."),
    "Temporal Intelligence": ("DISTRIBUTION MONITORING", "Understand how transaction patterns shift over time",
                              "Inspect feature drift, temporal performance and the relationship between drift and model errors."),
    "Explainability Studio": ("MODEL TRANSPARENCY", "Inspect what the models rely on",
                              "Explore SHAP feature attributions, GNNExplainer graph explanations and their level of agreement."),
    "Transaction Explorer": ("DATA WORKSPACE", "Explore processed transaction records",
                             "Filter the processed Elliptic dataset by time step and class, then export a sample."),
    "Reports": ("EXPERIMENT ARCHIVE", "Research outputs and reproducibility",
                "Browse generated experiment tables, figures and written summaries.")
}
kicker, title, desc = hero_text[page]
st.markdown(
    f'<div class="hero"><div class="hero-kicker">{kicker}</div>'
    f'<h1>{title}</h1><p>{desc}</p>'
    f'<div class="hero-tag">ELLIPTIC DATASET&nbsp; • &nbsp;RANDOM FOREST&nbsp; • &nbsp;TEMPORAL GCN</div></div>',
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    heading("Dataset at a glance", "The scale of the transaction graph used in the experiments")
    cols = st.columns(4)
    values = [
        ("TRANSACTIONS", "203,769", "Elliptic transaction nodes"),
        ("GRAPH EDGES", "234,355", "Directed transaction links"),
        ("TIME STEPS", "49", "Temporal snapshots"),
        ("ORIGINAL FEATURES", "165", "Transaction attributes"),
    ]
    for col, item in zip(cols, values):
        with col:
            kpi(*item)

    heading("Predictive performance", "Held-out test results — accuracy is shown alongside imbalance-aware metrics")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Random Forest</div>'
                    '<div class="panel-sub">Strong overall classification performance</div>'
                    '<span class="pill">TREE ENSEMBLE</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "97.34%")
        c2.metric("Balanced accuracy", "73.53%")
        c1.metric("Macro F1", "80.39%")
        c2.metric("ROC-AUC", "87.67%")
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Temporal GCN</div>'
                    '<div class="panel-sub">Graph-based classification using transaction relationships</div>'
                    '<span class="pill">GRAPH NEURAL NETWORK</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "59.50%")
        c2.metric("Balanced accuracy", "70.28%")
        c1.metric("Macro F1", "44.58%")
        c2.metric("ROC-AUC", "72.99%")

    heading("Model comparison", "Use class-wise metrics to understand the minority-class trade-off")
    image_panel("overall_model_comparison.png")
    a, b = st.columns(2)
    with a:
        image_panel("classwise_recall_comparison.png")
    with b:
        image_panel("minority_class_distribution.png")

    st.markdown('<div class="panel"><div class="panel-title">Research question</div>'
                '<div class="panel-sub">What can we learn when predictive performance, temporal drift and explanations are examined together?</div>'
                '<p style="color:#42546b;font-size:13px;line-height:1.7">'
                'This project studies illicit Bitcoin transaction classification using tabular and graph-based '
                'learning. It evaluates performance under class imbalance and temporal splits, examines feature '
                'distribution drift, and compares feature-level SHAP explanations with local graph explanations '
                'from GNNExplainer.</p></div>', unsafe_allow_html=True)
    st.warning("Interpretation note: the dataset is highly imbalanced. Accuracy alone can conceal weak performance on the licit class.")


# ============================================================
# MODEL LAB
# ============================================================
elif page == "Model Lab":
    heading("Model scorecard", "Held-out test metrics")
    l, r = st.columns(2)
    with l:
        st.markdown("#### Random Forest")
        st.metric("Accuracy", "97.34%")
        st.metric("Balanced accuracy", "73.53%")
        st.metric("Macro F1", "80.39%")
        st.metric("ROC-AUC", "87.67%")
        st.metric("Illicit PR-AUC", "99.27%")
    with r:
        st.markdown("#### Temporal GCN")
        st.metric("Accuracy", "59.50%")
        st.metric("Balanced accuracy", "70.28%")
        st.metric("Macro F1", "44.58%")
        st.metric("ROC-AUC", "72.99%")
        st.metric("Illicit PR-AUC", "98.25%")

    heading("Evaluation visuals", "Overall and class-wise comparisons")
    image_panel("overall_model_comparison.png")
    image_panel("classwise_recall_comparison.png")
    tabs = st.tabs(["Model comparison", "Random Forest by time", "GCN by time", "Error analysis"])
    with tabs[0]:
        csv_panel("gcn_vs_rf_comparison.csv")
    with tabs[1]:
        image_panel("per_timestep_performance.png")
        csv_panel("rf_per_timestep_metrics.csv")
    with tabs[2]:
        csv_panel("gnn_per_timestep_metrics.csv")
    with tabs[3]:
        csv_panel("rf_error_summary.csv")
        csv_panel("rf_error_feature_analysis.csv")
        image_panel("minority_class_probability.png")
    st.info("The Random Forest has stronger overall macro-F1 and ROC-AUC in these experiments; the GCN shows a different class-recall trade-off.")


# ============================================================
# TEMPORAL INTELLIGENCE
# ============================================================
elif page == "Temporal Intelligence":
    heading("Drift observatory", "How feature distributions change across time")
    c1, c2 = st.columns(2)
    with c1:
        image_panel("feature_drift_analysis.png")
    with c2:
        image_panel("class_conditional_drift.png")
    image_panel("temporal_probability_drift.png")
    tabs = st.tabs(["Feature-level drift", "Feature groups", "Drift and errors"])
    with tabs[0]:
        csv_panel("feature_drift_results.csv", 420)
    with tabs[1]:
        csv_panel("feature_group_drift_summary.csv", 420)
        csv_panel("feature_group_drift_details.csv", 420)
    with tabs[2]:
        csv_panel("drift_error_correlation.csv")
        st.caption("Correlations are observational associations; they do not establish that drift causes model errors.")


# ============================================================
# EXPLAINABILITY STUDIO
# ============================================================
elif page == "Explainability Studio":
    heading("Explanation workbench", "Three complementary views of model behaviour")
    tabs = st.tabs(["SHAP", "GNNExplainer", "Method agreement"])
    with tabs[0]:
        st.markdown("#### Global feature attribution")
        c1, c2 = st.columns(2)
        with c1:
            image_panel("shap_summary.png")
        with c2:
            image_panel("shap_importance.png")
        csv_panel("shap_stability_comparison.csv")
        markdown_panel("shap_explainability_summary.md")
    with tabs[1]:
        st.markdown("#### Local graph explanation")
        c1, c2 = st.columns(2)
        with c1:
            image_panel("gnn_feature_importance.png")
        with c2:
            image_panel("gnn_edge_importance.png")
        csv_panel("gnn_feature_importance.csv")
        csv_panel("gnn_edge_importance.csv")
        text_panel("gnn_explanation_summary.txt")
        st.caption("The current GNNExplainer result is a local explanation for one selected node and should not be generalized to all transactions.")
    with tabs[2]:
        st.markdown("#### SHAP vs GNNExplainer")
        image_panel("shap_vs_gnn_importance.png")
        image_panel("shap_vs_gnn_rank.png")
        image_panel("shap_gnn_topk_overlap.png")
        csv_panel("shap_gnn_feature_comparison.csv")
        csv_panel("shap_gnn_topk_overlap.csv")
        markdown_panel("shap_gnn_comparison_summary.md")
        st.warning("The compared explanations have different scopes: aggregated global SHAP importance versus a single local GNNExplainer case. Their agreement is therefore exploratory, not a direct correctness test.")


# ============================================================
# TRANSACTION EXPLORER
# ============================================================
elif page == "Transaction Explorer":
    heading("Transaction data browser", "Explore a bounded preview of the processed feature table")
    dataset = DATA / "transactions_advanced_features.csv"
    df = read_csv(str(dataset), nrows=20000)
    if df is None:
        st.error(f"Dataset not found: {dataset}")
    else:
        st.markdown(f'<span class="pill">PREVIEW LOADED · {len(df):,} ROWS</span>', unsafe_allow_html=True)
        filtered = df.copy()
        f1, f2 = st.columns(2)
        if "time_step" in filtered.columns:
            available_steps = sorted(filtered["time_step"].dropna().unique().tolist())
            with f1:
                chosen_steps = st.multiselect(
                    "Time steps", available_steps,
                    default=available_steps[:min(5, len(available_steps))]
                )
            if chosen_steps:
                filtered = filtered[filtered["time_step"].isin(chosen_steps)]
        if "class" in filtered.columns:
            available_classes = sorted(filtered["class"].dropna().astype(str).unique().tolist())
            with f2:
                chosen_classes = st.multiselect("Class labels", available_classes, default=available_classes)
            if chosen_classes:
                filtered = filtered[filtered["class"].astype(str).isin(chosen_classes)]
        st.metric("Rows matching filters", f"{len(filtered):,}")
        st.dataframe(filtered.head(500), use_container_width=True, height=430)
        st.download_button(
            "Export filtered preview as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="fingraph_sentinel_filtered_preview.csv",
            mime="text/csv",
        )
        st.caption("Explorer loads at most 20,000 rows to keep the browser responsive; this is a preview, not the complete dataset.")


# ============================================================
# REPORTS
# ============================================================
elif page == "Reports":
    heading("Research report library", "Generated summaries and experiment artifacts")
    report_options = {
        "SHAP explainability summary": "shap_explainability_summary.md",
        "SHAP vs GNNExplainer comparison": "shap_gnn_comparison_summary.md",
    }
    selected = st.selectbox("Choose a written report", list(report_options.keys()))
    markdown_panel(report_options[selected])
    heading("Experiment tables", "Open the generated CSV outputs")
    report_names = [
        "gcn_vs_rf_comparison.csv",
        "rf_per_timestep_metrics.csv",
        "gnn_per_timestep_metrics.csv",
        "rf_error_summary.csv",
        "rf_error_feature_analysis.csv",
        "feature_drift_results.csv",
        "feature_group_drift_summary.csv",
        "drift_error_correlation.csv",
        "shap_stability_comparison.csv",
        "shap_gnn_feature_comparison.csv",
        "shap_gnn_topk_overlap.csv",
    ]
    chosen = st.selectbox("Select a table", report_names)
    csv_panel(chosen, 420)
    heading("Artifact availability", "Files found in the reports directory")
    for name in report_names:
        status = "Available" if find_artifact(name) else "Not generated"
        st.write(("🟢 " if status == "Available" else "⚪ ") + f"`{name}` — {status}")


# ---------- Footer ----------
st.markdown("""
<div class="footer">
  <span>FINGRAPH SENTINEL · RESEARCH PROTOTYPE</span>
  <span>ELLIPTIC TRANSACTION GRAPH · EXPLAINABLE ML</span>
</div>
""", unsafe_allow_html=True)
