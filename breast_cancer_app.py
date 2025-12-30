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
        padding: 50px;
        background: white;
        border-radius: 30px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border: 1px solid #EDF2F7;
        margin: 30px auto;
        max-width: 800px;
    }

    .status-badge {
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 13px;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }

    .stNumberInput label {
        font-weight: 600 !important;
        color: #334155 !important;
    }
    
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0;
        line-height: 1.2;
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
    
    # Train model on scaled data
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X, data.feature_names

model, scaler, X_raw, feature_names = load_and_train()

# =====================================================
# HEADER
# =====================================================
st.markdown("<h1 style='text-align: center; color: #0F172A; margin-bottom:0;'>🧬 OncoPredict <span style='color: #3B82F6;'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; font-size: 1.2rem;'>Advanced Clinical Diagnostic Decision Support</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# MANUAL INPUT SECTION
# =====================================================
with st.container():
    st.markdown("### 🩺 Patient Pathology Metrics")
    st.caption("Please enter the 'Mean' values from the diagnostic report below.")

    # Filter for 'mean' features only
    important_features = [f for f in feature_names if "mean" in f]
    user_input = {}
    
    # 3-column grid for manual entry
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
analyze = st.button("EXECUTE DIAGNOSTIC SCAN", use_container_width=True, type="primary")

# =====================================================
# RESULTS SECTION
# =====================================================
if analyze:
    # 1. Prepare Data
    input_df = pd.DataFrame([user_input])
    # Align with all 30 features the model expects (fill missing with 0)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_df)

    # 2. Prediction
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    
    # Logic: 0 = Malignant (Red), 1 = Benign (Green)
    is_benign = (prediction == 1)
    conf_score = probs[1] if is_benign else probs[0]
    
    res_color = "#10B981" if is_benign else "#EF4444"
    res_bg = "#ECFDF5" if is_benign else "#FEF2F2"
    res_label = "BENIGN (NO RISK)" if is_benign else "MALIGNANT (HIGH RISK)"
    res_icon = "✔️" if is_benign else "⚠️"

    # THE CENTERED "MAIN VIBE" RESULT CARD
    st.markdown(f"""
        <div class="result-container">
            <div style="background-color: {res_bg}; color: {res_color};" class="status-badge">
                Diagnostic Assessment Result
            </div>
            <h1 class="main-title" style="color: {res_color};">{res_icon} {res_label}</h1>
            <p style="font-size: 1.5rem; color: #1E293B; margin-top: 15px; font-weight: 500;">
                Confidence Level: {conf_score:.1%}
            </p>
            <div style="width: 100px; height: 4px; background: {res_color}; margin: 20px 0; border-radius: 10px;"></div>
            <p style="color: #64748B; max-width: 550px; line-height: 1.6;">
                The analysis indicates that the provided cellular measurements align with 
                <b>{"non-malignant" if is_benign else "malignant"}</b> patterns. 
                Further clinical correlation is required.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 3. CLASSY CHARTS SECTION
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 📊 Probability Distribution")
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            x=['Malignant', 'Benign'],
            y=[probs[0], probs[1]],
            marker_color=['#EF4444', '#10B981'],
            width=0.5,
            text=[f"{probs[0]:.1%}", f"{probs[1]:.1%}"],
            textposition='outside'
        ))
        fig_prob.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            font=dict(family="Arial", size=13, color="#334155"),
            yaxis=dict(range=[0, 1.2], showticklabels=False, showgrid=False),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with chart_col2:
        st.markdown("#### 🔍 Key Diagnostic Drivers")
        importances = model.feature_importances_
        feat_imp = pd.Series(importances[:len(important_features)], index=important_features)
        top_5 = feat_imp.nlargest(5)

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=top_5.values,
            y=[f.replace("_", " ").title() for f in top_5.index],
            orientation='h',
            marker_color='#3B82F6',
            width=0.6
        ))
        fig_imp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            font=dict(family="Arial", size=13, color="#334155"),
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.warning("🚨 **Notice:** This AI model is designed for decision support. It should be used in conjunction with official pathology reports and biopsy results.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; border-top: 1px solid #E2E8F0; padding-top: 25px;">
        Designed & Developed by Jahnavi • Clinical ML Systems v2.2 • © 2024
    </div>
""", unsafe_allow_html=True)
