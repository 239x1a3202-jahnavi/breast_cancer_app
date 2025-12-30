import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG & THEME
# =====================================================
st.set_page_config(page_title="OncoPredict AI", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; }
    .result-container {
        display: flex; flex-direction: column; align-items: center; text-align: center;
        padding: 50px; background: white; border-radius: 30px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border: 1px solid #EDF2F7; margin: 30px auto; max-width: 800px;
    }
    .status-badge {
        padding: 8px 24px; border-radius: 50px; font-weight: bold;
        text-transform: uppercase; font-size: 13px; margin-bottom: 20px;
    }
    .main-title { font-size: 3.2rem; font-weight: 800; margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA & MODEL
# =====================================================
@st.cache_resource
def load_and_train():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X, data.feature_names

model, scaler, X_raw, feature_names = load_and_train()

# =====================================================
# HEADER & INPUTS
# =====================================================
st.markdown("<h1 style='text-align: center; color: #0F172A;'>🧬 OncoPredict <span style='color: #3B82F6;'>AI</span></h1>", unsafe_allow_html=True)

important_features = [f for f in feature_names if "mean" in f]
user_input = {}

with st.expander("📋 Patient Data Entry", expanded=True):
    cols = st.columns(3)
    for i, feature in enumerate(important_features):
        with cols[i % 3]:
            user_input[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                min_value=0.0, value=float(X_raw[feature].mean()), format="%.3f"
            )

analyze = st.button("EXECUTE DIAGNOSTIC SCAN", use_container_width=True, type="primary")

# =====================================================
# RESULTS
# =====================================================
if analyze:
    input_df = pd.DataFrame([user_input]).reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    
    is_benign = (prediction == 1)
    risk_score = probs[0] * 100 # Malignant probability as a percentage
    
    res_color = "#10B981" if is_benign else "#EF4444"
    res_bg = "#ECFDF5" if is_benign else "#FEF2F2"
    res_label = "BENIGN (NO RISK)" if is_benign else "MALIGNANT (HIGH RISK)"

    st.markdown(f"""
        <div class="result-container">
            <div style="background-color: {res_bg}; color: {res_color};" class="status-badge">Clinical Assessment</div>
            <h1 class="main-title" style="color: {res_color};">{"✔️" if is_benign else "⚠️"} {res_label}</h1>
            <p style="color: #64748B; margin-top: 15px;">Model Confidence: {max(probs)*100:.1%}</p>
        </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🌡️ Risk Magnitude")
        # GAUGE CHART
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Malignancy Risk %", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': res_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 30], 'color': '#ECFDF5'},
                    {'range': [30, 70], 'color': '#FFFBEB'},
                    {'range': [70, 100], 'color': '#FEF2F2'}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score}}))
        fig_gauge.update_layout(height=350, margin=dict(t=50, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        st.markdown("#### 🕸️ Feature Morphology (Radar)")
        # RADAR CHART
        categories = [f.replace(" mean", "") for f in important_features[:6]]
        values = [user_input[f] for f in important_features[:6]]
        # Normalize values for display
        max_vals = [X_raw[f].max() for f in important_features[:6]]
        norm_values = [v/m for v, m in zip(values, max_vals)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_values,
            theta=categories,
            fill='toself',
            fillcolor=res_color,
            opacity=0.3,
            line=dict(color=res_color)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False, height=350, margin=dict(t=50, b=50)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.warning("🚨 This tool is for educational purposes and provides AI-based risk estimates only.")
