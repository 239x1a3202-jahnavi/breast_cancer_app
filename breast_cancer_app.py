import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Breast Cancer Diagnostic System",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM CSS – HOSPITAL LIGHT BLUE THEME
# =====================================================
st.markdown("""
<style>
body {
    background-color: #f2f8fc;
}

.section-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0, 123, 255, 0.08);
    border-left: 6px solid #5dade2;
}

.main-card {
    background: linear-gradient(135deg, #eaf4fb, #ffffff);
    padding: 50px;
    border-radius: 22px;
    box-shadow: 0 12px 35px rgba(0, 123, 255, 0.15);
    text-align: center;
    border-top: 10px solid #5dade2;
}

.result-green {
    color: #1abc9c;
}

.result-red {
    color: #e74c3c;
}

.subtle-text {
    color: #555;
    font-size: 18px;
}

.ml-caption {
    margin-top: 20px;
    font-size: 14px;
    color: #2c3e50;
    background: #e8f4fd;
    padding: 8px 14px;
    border-radius: 10px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1 style="text-align:center;">🩺 Breast Cancer Diagnostic System</h1>
<p style="text-align:center; color:#555;">
AI-assisted medical risk assessment tool
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# LOAD DATA & MODEL
# =====================================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# =====================================================
# INPUT SECTION – ONLY IMPORTANT FEATURES
# =====================================================
important_features = [f for f in X.columns if "mean" in f]

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📋 Key Medical Measurements")

cols = st.columns(3)
user_input = {}

for i, feature in enumerate(important_features):
    with cols[i % 3]:
        user_input[feature] = st.slider(
            feature.replace("_", " ").title(),
            float(X[feature].min()),
            float(X[feature].max()),
            float(X[feature].mean())
        )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("🔍 Analyze Report", use_container_width=True)

# =====================================================
# RESULT SECTION – IMPRESSIVE & CLEAR
# =====================================================
if analyze:
    input_df = pd.DataFrame([user_input])

    # Fill missing features with 0
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        title = "NO CANCER DETECTED"
        subtitle = "Benign Tumor"
        color_class = "result-green"
        explanation = "The model did not detect malignant patterns in the provided data."
        confidence = probability[1]
        icon = "✅"
    else:
        title = "CANCER RISK DETECTED"
        subtitle = "Malignant Tumor"
        color_class = "result-red"
        explanation = "The model identified strong malignant characteristics."
        confidence = probability[0]
        icon = "⚠️"

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-card">
        <h1 class="{color_class}" style="font-size:54px;">{icon} {title}</h1>
        <h3 style="color:#2c3e50;">{subtitle}</h3>
        <p class="subtle-text">{explanation}</p>
        <h2>{confidence:.1%} Confidence Level</h2>

        <div class="ml-caption">
            🤖 This result is generated using a Machine Learning prediction model
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ How was this prediction made?"):
        st.write(
            "The system analyzes key tumor measurements such as radius, texture, "
            "smoothness, and concavity patterns using a trained Random Forest "
            "machine learning model."
        )

    st.warning(
        "⚠️ This is an AI-based prediction for educational and decision-support "
        "purposes only. Always consult a qualified medical professional."
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Designed & Developed by Jahnavi • Medical ML System")
