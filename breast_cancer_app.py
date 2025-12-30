import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="OncoPredict AI | Clinical Dashboard",
    page_icon="🧬",
    layout="wide"
)

# =====================================================
# CUSTOM CSS – HOSPITAL & CLINICAL THEME
# =====================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Modern Card Styling */
    .metric-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Centered Hero Result */
    .result-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 60px;
        background: white;
        border-radius: 30px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border: 1px solid #EDF2F7;
        margin: 20px auto;
        max-width: 800px;
    }

    .status-badge {
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 14px;
        margin-bottom: 15px;
    }

    .malignant-text { color: #DC2626; }
    .benign-text { color: #059669; }
    
    /* Input Label Styling */
    .stNumberInput label {
        font-weight: 600 !important;
        color: #1E293B !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA & MODEL CACHING
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
# SIDEBAR / HEADER
# =====================================================
st.markdown("<h1 style='text-align: center; color: #0F172A;'>🧬 OncoPredict <span style='color: #3B82F6;'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; font-size: 1.1rem;'>Clinical Grade Diagnostic Decision Support System</p>", unsafe_allow_html=True)
st.divider()

# =====================================================
# MANUAL INPUT SECTION
# =====================================================
st.markdown("### 🩺 Patient Biometry Input")
st.info("Please enter the measurements from the pathology report below.")

# Filtering for 'mean' features as requested
important_features = [f for f in feature_names if "mean" in f]

user_input = {}
# Create a 4-column grid for the manual inputs
cols = st.columns(4)

for i, feature in enumerate(important_features):
    with cols[i % 4]:
        # Using number_input instead of slider for manual entry
        user_input[feature] = st.number_input(
            label=feature.replace("_", " ").title(),
            min_value=0.0,
            max_value=float(X_raw[feature].max() * 2), # Allow for outliers
            value=float(X_raw[feature].mean()),
            format="%.3f",
            key=feature
        )

st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    analyze = st.button("RUN DIAGNOSTIC ANALYSIS", use_container_width=True, type="primary")

# =====================================================
# ANALYSIS & RESULTS
# =====================================================
if analyze:
    # Prepare Data
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)

    # Predictions
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    
    # 0 = Malignant, 1 = Benign in Breast Cancer Dataset
    is_benign = prediction == 1
    conf_score = probs[1] if is_benign else probs[0]
    
    st.divider()

    # CENTERED RESULT CARD
    res_color = "#059669" if is_benign else "#DC2626"
    res_bg = "#ECFDF5" if is_benign else "#FEF2F2"
    res_text = "BENIGN (NO CANCER)" if is_benign else "MALIGNANT (RISK DETECTED)"
    res_icon = "✅" if is_benign else "⚠️"

    st.markdown(f"""
        <div class="result-container">
            <div style="background-color: {res_bg}; color: {res_color};" class="status-badge">
                Analysis Result
            </div>
            <h1 style="color: {res_color}; font-size: 3.5rem; margin-top: 0;">{res_icon} {res_text}</h1>
            <p style="font-size: 1.2rem; color: #475569; max-width: 600px;">
                The AI model has analyzed the submitted pathology metrics with 
                <b>{conf_score:.1%} confidence</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # CLASSY BAR CHART
    st.markdown("### 📊 Probability Distribution")
    
    # Custom Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Malignant', 'Benign'],
        y=[probs[0], probs[1]],
        marker_color=['#EF4444', '#10B981'],
        width=0.4
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="Confidence Level", tickformat='.0%', range=[0, 1]),
        xaxis=dict(font=dict(size=14, color="#1E293B"))
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # WARNING
    st.warning("🚨 **Disclaimer:** This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-size: 0.8rem; border-top: 1px solid #E2E8F0; padding-top: 20px;">
        Designed by Jahnavi | Oncology ML Research v2.0 | © 2024
    </div>
""", unsafe_allow_html=True)
