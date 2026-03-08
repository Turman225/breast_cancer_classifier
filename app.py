import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load model + meta ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_meta():
    with open("model_meta.json") as f:
        return json.load(f)

model = load_model()
meta  = load_meta()

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0b0f1a;
    --surface:  #131929;
    --border:   #1e2d45;
    --accent1:  #3dd9c5;
    --accent2:  #e85d9b;
    --warn:     #f5a623;
    --text:     #dce8f5;
    --muted:    #6a849e;
    --safe:     #3dd9c5;
    --danger:   #e85d9b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text) !important;
}

.stSlider label, .stSelectbox label, .stNumberInput label {
    color: var(--muted) !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace !important;
}

.stSlider > div > div > div > div {
    background: var(--accent1) !important;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    line-height: 1;
    margin: 0.3rem 0;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
}

.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    border: 2px solid;
}

.result-benign {
    background: rgba(61,217,197,0.08);
    border-color: var(--safe);
}

.result-malignant {
    background: rgba(232,93,155,0.08);
    border-color: var(--danger);
}

.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    margin: 0;
}

.result-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.4rem;
}

.prob-bar-bg {
    background: var(--border);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin: 0.4rem 0 1rem;
}

.feature-tag {
    display: inline-block;
    background: rgba(61,217,197,0.12);
    color: var(--accent1);
    border: 1px solid rgba(61,217,197,0.3);
    border-radius: 99px;
    padding: 0.15rem 0.65rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    margin: 0.15rem;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent1), #2bb8a7) !important;
    color: #0b0f1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100%;
    transition: all .2s ease !important;
    letter-spacing: 0.03em;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(61,217,197,0.35) !important;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.05rem !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:2rem 0 1rem;'>
  <span style='font-family:DM Mono,monospace;font-size:.75rem;letter-spacing:.15em;
               text-transform:uppercase;color:#3dd9c5;'>
    Gradient Boosting · Kaggle Dataset · 96.5% CV Accuracy
  </span>
  <h1 style='margin:.4rem 0 .2rem;font-size:2.6rem;'>Breast Cancer Classifier</h1>
  <p style='color:#6a849e;font-size:.95rem;max-width:560px;'>
    Predict whether a tumour is <b style='color:#3dd9c5'>benign</b> or 
    <b style='color:#e85d9b'>malignant</b> from 30 cell-nucleus features 
    extracted via Fine Needle Aspirate (FNA) biopsy imagery.
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Top metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, label, color in [
    (c1, f"{meta['cv_mean']*100:.1f}%",   "CV Accuracy",       "#3dd9c5"),
    (c2, f"±{meta['cv_std']*100:.1f}%",   "Std Deviation",     "#dce8f5"),
    (c3, f"{meta['n_samples']}",           "Total Samples",     "#dce8f5"),
    (c4, f"{meta['n_features']}",          "Input Features",    "#dce8f5"),
]:
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{color}'>{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─── Sidebar – feature inputs ─────────────────────────────────────────────────
feature_names = meta["feature_names"]
fmin  = meta["feature_min"]
fmax  = meta["feature_max"]
fmean = meta["feature_mean"]

GROUPS = {
    "🔵  Mean Features":  list(range(0, 10)),
    "🟡  SE Features":    list(range(10, 20)),
    "🔴  Worst Features": list(range(20, 30)),
}

with st.sidebar:
    st.markdown("## INPUT FEATURES")
    vals = {}
    for group_label, indices in GROUPS.items():
        with st.expander(group_label, expanded=(group_label == "🔵  Mean Features")):
            for i in indices:
                name = feature_names[i]
                lo, hi, def_ = float(fmin[i]), float(fmax[i]), float(fmean[i])
                step = max((hi - lo) / 200, 1e-6)
                vals[i] = st.slider(
                    name, min_value=lo, max_value=hi,
                    value=def_, step=step,
                    format="%.4f",
                )
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔬 Run Prediction")

# ─── Main panel ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # ── Prediction result ──────────────────────────────────────────────────
    input_vec = np.array([[vals[i] for i in range(30)]])
    proba     = model.predict_proba(input_vec)[0]       # [malignant, benign]
    pred_idx  = int(proba.argmax())
    pred_name = meta["target_names"][pred_idx]          # "malignant" or "benign"
    confidence = float(proba[pred_idx]) * 100

    is_benign = (pred_name == "benign")
    card_cls  = "result-benign" if is_benign else "result-malignant"
    icon      = "✦" if is_benign else "⚠"
    color     = "#3dd9c5" if is_benign else "#e85d9b"
    label_txt = pred_name.upper()

    st.markdown(f"""
    <div class='result-card {card_cls}'>
      <div style='font-size:2.2rem;'>{icon}</div>
      <p class='result-label' style='color:{color}'>{label_txt}</p>
      <p class='result-sub'>Model prediction · {confidence:.1f}% confidence</p>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("**Class probabilities**")
    for idx, (name, p) in enumerate(zip(meta["target_names"], proba)):
        bar_color = "#3dd9c5" if name == "benign" else "#e85d9b"
        pct = p * 100
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;margin-bottom:.2rem;'>
          <span style='font-family:DM Mono,monospace;font-size:.82rem;color:#dce8f5'>{name}</span>
          <span style='font-family:DM Mono,monospace;font-size:.82rem;color:{bar_color}'>{pct:.1f}%</span>
        </div>
        <div class='prob-bar-bg'>
          <div style='background:{bar_color};height:100%;width:{pct:.1f}%;
                      border-radius:99px;transition:width .4s ease;'></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Radar chart of mean features ──────────────────────────────────────
    st.markdown("**Feature profile vs. dataset mean (mean block)**")
    mean_idx    = list(range(10))
    input_norm  = [(vals[i] - fmin[i]) / max(fmax[i] - fmin[i], 1e-9) for i in mean_idx]
    dataset_norm = [(fmean[i] - fmin[i]) / max(fmax[i] - fmin[i], 1e-9) for i in mean_idx]
    labels = [feature_names[i].replace(" mean", "") for i in mean_idx]

    fig = go.Figure()
    for trace_vals, trace_name, clr in [
        (input_norm,   "Your Input",    "#3dd9c5"),
        (dataset_norm, "Dataset Mean",  "#6a849e"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=trace_vals + [trace_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=trace_name,
            line=dict(color=clr, width=2),
            fillcolor=clr.replace("#", "rgba(") + ",0.08)" if clr != "#6a849e" else "rgba(106,132,158,0.06)",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#131929",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="#1e2d45", linecolor="#1e2d45"),
            angularaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45",
                             tickfont=dict(family="DM Mono", size=10, color="#6a849e")),
        ),
        paper_bgcolor="#0b0f1a",
        plot_bgcolor="#0b0f1a",
        font=dict(family="DM Sans"),
        legend=dict(font=dict(color="#dce8f5", size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=30, t=20, b=20),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    # ── Feature importance bar ─────────────────────────────────────────────
    st.markdown("**Top-10 Feature Importances**")
    fi      = np.array(meta["feature_importances"])
    top_idx = np.argsort(fi)[::-1][:10]
    top_fi  = fi[top_idx]
    top_nm  = [feature_names[i] for i in top_idx]

    fig2 = go.Figure(go.Bar(
        x=top_fi,
        y=top_nm,
        orientation="h",
        marker=dict(
            color=top_fi,
            colorscale=[[0, "#1e2d45"], [0.5, "#3dd9c5"], [1, "#e85d9b"]],
            showscale=False,
        ),
    ))
    fig2.update_layout(
        paper_bgcolor="#0b0f1a",
        plot_bgcolor="#131929",
        xaxis=dict(showgrid=True, gridcolor="#1e2d45", color="#6a849e",
                   tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(autorange="reversed", color="#dce8f5",
                   tickfont=dict(family="DM Mono", size=10)),
        margin=dict(l=10, r=10, t=10, b=30),
        height=340,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Input summary table ────────────────────────────────────────────────
    st.markdown("**Input vs. Dataset Range**")
    rows = []
    for i in range(30):
        v = vals[i]
        lo, hi, mu = fmin[i], fmax[i], fmean[i]
        pct = (v - lo) / max(hi - lo, 1e-9) * 100
        rows.append({
            "Feature":  feature_names[i],
            "Value":    f"{v:.4f}",
            "Mean":     f"{mu:.4f}",
            "Pct-rank": f"{pct:.0f}%",
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        height=360,
        hide_index=True,
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr class='divider'>
<div style='display:flex;justify-content:space-between;align-items:center;
            font-family:DM Mono,monospace;font-size:.72rem;color:#3a526b;padding:.5rem 0 1rem;'>
  <span>Breast Cancer Wisconsin (Diagnostic) · UCI / Kaggle</span>
  <span>GradientBoostingClassifier · sklearn · joblib</span>
  <span>Streamlit Cloud deployment</span>
</div>
""", unsafe_allow_html=True)
