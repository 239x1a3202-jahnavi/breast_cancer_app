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
# CUSTOM CSS (MODERN LOOK)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #f6f8fb;
}
.main-card {
    background: white;
    padding: 45px;
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    text-align: center;
}
.result-green {
    color: #1e8449;
}
.result-red {
    color: #c0392b;
}
.subtle-text {
    color: #555;
    font-size: 18px;
}
.section-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1 style="text-align:center;">🩺 Breast Cancer Diagnostic System</h1>
<p style="text-align:center; color:gray;">
Enter values from the medical report to assess cancer risk
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
# INPUT SECTION
# =====================================================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("📋 Medical Report Entry")

user_input = {}
cols = st.columns(3)

for i, feature in enumerate(X.columns):
    with cols[i % 3]:
        user_input[feature] = float(
            st.text_input(
                feature.replace("_", " ").title(),
                value=f"{X[feature].mean():.5f}"
            )
        )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("🔍 Analyze Report", use_container_width=True)

# =====================================================
# RESULT SECTION (HERO)
# =====================================================
if analyze:
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        title = "NO CANCER DETECTED"
        subtitle = "Benign Tumor"
        color_class = "result-green"
        explanation = "No strong indicators of cancer were found."
        confidence = probability[1]
    else:
        title = "CANCER DETECTED"
        subtitle = "Malignant Tumor"
        color_class = "result-red"
        explanation = "Strong indicators of cancerous cells were detected."
        confidence = probability[0]

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-card">
        <h1 class="{color_class}" style="font-size:52px;">{title}</h1>
        <h3 style="color:#666;">({subtitle})</h3>
        <p class="subtle-text">{explanation}</p>
        <h2>{confidence:.2%} Confidence</h2>
    </div>
    """, unsafe_allow_html=True)

    # OPTIONAL EXPLANATION
    with st.expander("ℹ️ Why this result? (optional)"):
        st.write(
            "The model analyzes multiple measurements from the medical report "
            "such as radius, texture, symmetry, and concavity patterns to "
            "identify cancerous characteristics."
        )

    st.warning(
        "⚠️ This system is for educational and decision-support purposes only. "
        "Always consult a certified medical professional."
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Designed & Developed by Jahnavi • Modern Medical ML UI")


