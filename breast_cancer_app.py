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
Clinical decision-support tool for manual pathology report entry
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
# INPUT FORM (MAIN PAGE)
# =====================================================
st.subheader("📋 Patient Diagnostic Feature Entry")
st.caption("Type values exactly as mentioned in the pathology report")

user_input = {}

def render_inputs(features):
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            value = st.text_input(
                f"Enter {feature.replace('_', ' ').title()}",
                value=f"{X[feature].mean():.5f}"
            )
            user_input[feature] = float(value)

with st.expander("🟦 Mean Features", expanded=True):
    render_inputs(mean_features)

with st.expander("🟨 Error Features"):
    render_inputs(error_features)

with st.expander("🟥 Worst Features"):
    render_inputs(worst_features)

# =====================================================
# PREDICT BUTTON
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Cancer", use_container_width=True)

# =====================================================
# RESULT SECTION (MAIN FOCUS)
# =====================================================
if predict:
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    benign_prob = probability[1]
    malignant_prob = probability[0]

    if prediction == 1:
        diagnosis = "BENIGN"
        color = "#2ecc71"
        confidence = benign_prob
        message = "Low likelihood of malignancy detected"
    else:
        diagnosis = "MALIGNANT"
        color = "#e74c3c"
        confidence = malignant_prob
        message = "High likelihood of malignancy detected"

    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------- MAIN RESULT CARD -----------------
    st.markdown(
        f"""
        <div style="
            background-color:#ffffff;
            padding:40px;
            border-radius:18px;
            text-align:center;
            box-shadow:0 6px 18px rgba(0,0,0,0.15);
            border-top:10px solid {color};
        ">
            <h1 style="color:{color}; font-size:52px;">{diagnosis}</h1>
            <p style="font-size:22px; color:#555;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------- CONFIDENCE -----------------
    st.subheader("🔬 Model Confidence")
    st.progress(confidence)
    st.markdown(
        f"<h3 style='text-align:center;'>{confidence:.2%}</h3>",
        unsafe_allow_html=True
    )

    # ----------------- PROBABILITIES -----------------
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Benign Probability", f"{benign_prob:.2%}")
    with col2:
        st.metric("Malignant Probability", f"{malignant_prob:.2%}")

    # ----------------- FEATURE IMPORTANCE -----------------
    st.markdown("---")
    st.subheader("📊 Top Influential Features")

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    importances[:10].plot(kind="bar", ax=ax)
    ax.set_ylabel("Importance")
    ax.set_title("Most Influential Diagnostic Features")
    st.pyplot(fig)

    # ----------------- DISCLAIMER -----------------
    st.warning(
        "⚠️ **Medical Disclaimer:** This prediction is generated by a machine learning model "
        "and is intended only for educational and decision-support purposes. "
        "It must not be used as a substitute for professional medical diagnosis."
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Developed by Jahnavi • Streamlit • Machine Learning")
