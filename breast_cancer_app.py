import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG & CLINICAL THEME
# =====================================================
st.set_page_config(page_title="OncoPredict AI | Clinical", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F4F7F9; }
    
    /* Central Diagnostic Card */
    .diagnostic-card {
        background: white;
        padding: 60px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin: 40px auto;
        max-width: 900px;
    }

    .result-heading { font-size: 3.5rem; font-weight: 800; margin-bottom: 5px; }
    .conf-text { font-size: 1.2rem; color: #64748B; margin-bottom: 25px; }
    
    /* Badge styling */
    .status-label {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 50px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }
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
# HEADER
# =====================================================
st.markdown("<h1 style='text-align: center; color: #1E293B;'>🩺 OncoPredict <span style='color: #3B82F6;'>System</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>Clinical Decision Support Tool</p>", unsafe_allow_html=True)

# =====================================================
# INPUT SECTION
# =====================================================
important_features = [f for f in feature_names if "mean" in f]
user_input = {}

with st.container():
    st.markdown("### 📋 Enter Patient Pathology Values")
    cols = st.columns(4)
    for i, feature in enumerate(important_features):
        with cols[i % 4]:
            user_input[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                min_value=0.0,
                value=float(X_raw[feature].mean()),
                format="%.3f"
            )

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("GENERATE DIAGNOSTIC REPORT", use_container_width=True, type="primary")

# =====================================================
# DIAGNOSTIC OUTPUT
# =====================================================
if analyze:
    # Model Logic
    input_df = pd.DataFrame([user_input]).reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    
    is_benign = (prediction == 1)
    confidence = max(probs) * 100
    
    # UI Logic
    res_color = "#10B981" if is_benign else "#EF4444"
    res_bg = "#ECFDF5" if is_benign else "#FEF2F2"
    res_text = "BENIGN" if is_benign else "MALIGNANT"
    res_icon = "✅" if is_benign else "⚠️"

    # MAIN DIAGNOSTIC CARD
    st.markdown(f"""
        <div class="diagnostic-card">
            <div class="status-label" style="background: {res_bg}; color: {res_color};">
                PATHOLOGY ANALYSIS COMPLETED
            </div>
            <h1 class="result-heading" style="color: {res_color};">{res_icon} {res_text}</h1>
            <p class="conf-text">Diagnostic Confidence: {confidence:.2%}</p>
            <div style="border-top: 2px solid #F1F5F9; padding-top: 25px; color: #475569; line-height: 1.6;">
                The AI model has processed the submitted biometry. Based on the feature patterns, 
                the assessment indicates a <b>{"low-risk benign" if is_benign else "high-risk malignant"}</b> profile. 
                This report should be reviewed by a radiologist or oncologist.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # DATA SUMMARY TABLE
    st.markdown("### 🧬 Submitted Metrics Summary")
    summary_df = pd.DataFrame({
        "Measurement": [f.replace("_", " ").title() for f in user_input.keys()],
        "Value": list(user_input.values())
    })
    st.table(summary_df)

    st.markdown("---")
    st.caption("🚨 Disclaimer: This system is for educational/support use only and is not a replacement for professional medical diagnosis.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<p style='text-align: center; color: #94A3B8; margin-top: 50px;'>Jahnavi | Medical ML Research v3.0</p>", unsafe_allow_html=True)
