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
        margin: 40px auto;
        max-width: 850px;
    }

    .status-badge {
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 14px;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }

    /* Input Label Styling */
    .stNumberInput label {
        font-weight: 600 !important;
        color: #334155 !important;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
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
# HEADER
# =====================================================
st.markdown("<h1 style='text-align: center; color: #0F172A;'>🧬 OncoPredict <span style='color: #3B82F6;'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; font-size: 1.1rem;'>Medical Grade Diagnostic Decision Support System</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# MANUAL INPUT SECTION
# =====================================================
with st.container():
    st.markdown("### 🩺 Patient Pathology Entry")
    st.info("Manual Entry Mode: Please input the precise mean values from the lab report.")

    # Filter for 'mean' features
    important_features = [f for f in feature_names if "mean" in f]
    user_input = {}
    
    # 3-column grid for inputs
    cols = st.columns(3)
    for i, feature in enumerate(important_features):
        with cols[i % 3]:
            user_input[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                min_value=0.0,
                max_value=float(X_raw[feature].max() * 2),
                value=float(X_raw[feature].mean()),
                format="%.3f"
            )

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("RUN DIAGNOSTIC ANALYSIS", use_container_width=True, type="primary")

# =====================================================
# RESULTS SECTION
# =====================================================
if analyze:
    # Processing
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    
    # Logic (0=Malignant, 1=Benign)
    is_benign = (prediction == 1)
    conf_score = probs[1] if is_benign else probs[0]
    
    res_color = "#10B981" if is_benign else "#EF4444"
    res_bg = "#ECFDF5" if is_benign else "#FEF2F2"
    res_label = "BENIGN (LOW RISK)" if is_benign else "MALIGNANT (HIGH RISK)"
    res_icon = "🟢" if is_benign else "🔴"

    # THE CENTERED MAIN VIBE CARD
    st.markdown(f"""
        <div class="result-container">
            <div style="background-color: {res_bg}; color: {res_color};" class="status-badge">
                Final Assessment
            </div>
            <h1 class="main-title" style="color: {res_color};">{res_icon} {res_label}</h1>
            <p style="font-size: 1.4rem; color: #1E293B; margin-top: 10px;">
                AI Confidence Level: <b>{conf_score:.1%}</b>
            </p>
            <hr style="width: 50%; border: 0.5px solid #E2E8F0; margin: 25px 0;">
            <p style="color: #64748B; max-width: 600px;">
                The diagnostic model has identified characteristics associated with 
                {"non-cancerous" if is_benign else "malignant"} cellular patterns based on the 
                provided mean measurements.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # CLASSY PLOTLY BAR CHART
    st.markdown("### 📊 Probability Distribution")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Malignant Risk', 'Benign Probability'],
        y=[probs[0], probs[1]],
        marker_color=['#EF4444', '#10B981'],
        width=0.4,
        text=[f"{probs[0]:.1%}", f"{probs[1]:.1%}"],
        textposition='outside'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis=dict(
            title="Model Confidence", 
            tickformat='.0%', 
            range=[0, 1.1],
            gridcolor='#F1F5F9'
        ),
        xaxis=dict(
            tickfont=dict(size=14, color="#334155", family="Arial")
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.warning("🚨 **Clinical Note:** This software is a decision-support tool and does not replace a tissue biopsy or a doctor's final diagnosis.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; border-top: 1px solid #E2E8F0; padding-top: 25px;">
        Designed & Developed by Jahnavi • Oncology ML System v2.1 • Built with Streamlit & Scikit-Learn
    </div>
""", unsafe_allow_html=True)
