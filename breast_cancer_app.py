import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Breast Cancer Diagnostic System",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1 style="text-align:center;">🩺 Breast Cancer Diagnostic System</h1>
<p style="text-align:center; color:gray;">
Simple & clear cancer prediction using medical report values
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# =====================================================
# MODEL TRAINING
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# =====================================================
# FEATURE GROUPS
# =====================================================
mean_features = [f for f in X.columns if "mean" in f]
error_features = [f for f in X.columns if "error" in f]
worst_features = [f for f in X.columns if "worst" in f]

# =====================================================
# INPUT SECTION
# =====================================================
st.subheader("📋 Enter Medical Report Values")
st.caption("Type values exactly as mentioned in the pathology report")

user_input = {}

def render_inputs(features):
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            user_input[feature] = float(
                st.text_input(
                    f"{feature.replace('_', ' ').title()}",
                    value=f"{X[feature].mean():.5f}"
                )
            )

with st.expander("🟦 Mean Features", expanded=True):
    render_inputs(mean_features)

with st.expander("🟨 Error Features"):
    render_inputs(error_features)

with st.expander("🟥 Worst Features"):
    render_inputs(worst_features)

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Analyze Report", use_container_width=True)

# =====================================================
# RESULT SECTION
# =====================================================
if predict:
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    benign_prob = probability[1]
    malignant_prob = probability[0]

    if prediction == 1:
        main_result = "NO CANCER DETECTED"
        medical_term = "Benign Tumor"
        color = "#2ecc71"
        confidence = benign_prob
        explanation = "The model did not find signs of cancerous cells."
    else:
        main_result = "CANCER DETECTED"
        medical_term = "Malignant Tumor"
        color = "#e74c3c"
        confidence = malignant_prob
        explanation = "The model found strong signs of cancerous cells."

    st.markdown("---")

    # ================= MAIN RESULT CARD =================
    st.markdown(
        f"""
        <div style="
            background-color:#ffffff;
            padding:45px;
            border-radius:20px;
            text-align:center;
            box-shadow:0 8px 25px rgba(0,0,0,0.18);
            border-top:12px solid {color};
        ">
            <h1 style="color:{color}; font-size:56px;">{main_result}</h1>
            <h3 style="color:#555;">({medical_term})</h3>
            <p style="font-size:20px; color:#444;">{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>")

    # ================= CONFIDENCE =================
    st.subheader("🔬 Prediction Confidence")
    st.progress(confidence)
    st.markdown(
        f"<h3 style='text-align:center;'>{confidence:.2%}</h3>",
        unsafe_allow_html=True
    )

    # ================= PROBABILITY METRICS =================
    col1, col2 = st.columns(2)
    with col1:
        st.metric("No Cancer Probability", f"{benign_prob:.2%}")
    with col2:
        st.metric("Cancer Probability", f"{malignant_prob:.2%}")

    # ================= PREMIUM FEATURE IMPORTANCE =================
    st.markdown("---")
    st.subheader("📊 Key Factors Influencing the Result")

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importances.index, importances.values)
    ax.set_xlabel("Influence Level")
    ax.set_title("Top Diagnostic Features")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig)

    # ================= DISCLAIMER =================
    st.warning(
        "⚠️ This tool is for educational and decision-support purposes only. "
        "Always consult a certified medical professional for diagnosis."
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Developed by Jahnavi • Machine Learning • Streamlit")

