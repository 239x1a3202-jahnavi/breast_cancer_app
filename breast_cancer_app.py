import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostic System",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# Header
# -------------------------
st.markdown("""
<h1 style='text-align:center;'>🩺 Breast Cancer Diagnostic System</h1>
<p style='text-align:center;color:gray;'>
Clinical decision-support tool for manual pathology report entry
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Load Dataset
# -------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -------------------------
# Train Model
# -------------------------
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

# -------------------------
# Feature Groups
# -------------------------
mean_features = [f for f in X.columns if "mean" in f]
error_features = [f for f in X.columns if "error" in f]
worst_features = [f for f in X.columns if "worst" in f]

user_input = {}

# -------------------------
# INPUT FORM
# -------------------------
st.subheader("📋 Patient Diagnostic Feature Entry")
st.caption("Enter values exactly as shown in the pathology report")

with st.expander("🟦 Mean Features", expanded=True):
    cols = st.columns(3)
    for i, feature in enumerate(mean_features):
        with cols[i % 3]:
            user_input[feature] = float(
                st.text_input(
                    feature.replace("_", " ").title(),
                    value=f"{X[feature].mean():.5f}"
                )
            )

with st.expander("🟨 Error Features"):
    cols = st.columns(3)
    for i, feature in enumerate(error_features):
        with cols[i % 3]:
            user_input[feature] = float(
                st.text_input(
                    feature.replace("_", " ").title(),
                    value=f"{X[feature].mean():.5f}"
                )
            )

with st.expander("🟥 Worst Features"):
    cols = st.columns(3)
    for i, feature in enumerate(worst_features):
        with cols[i % 3]:
            user_input[feature] = float(
                st.text_input(
                    feature.replace("_", " ").title(),
                    value=f"{X[feature].mean():.5f}"
                )
            )

# -------------------------
# Prediction Button
# -------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Cancer", use_container_width=True)

# -------------------------
# Prediction Output
# -------------------------
if predict:
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("🧪 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("🟢 Benign (Not Cancer)")
        else:
            st.error("🔴 Malignant (Cancer)")

    with col2:
        st.metric("Benign Probability", f"{probability[1]:.2f}")
        st.metric("Malignant Probability", f"{probability[0]:.2f}")

    # -------------------------
    # Feature Importance
    # -------------------------
    st.subheader("📊 Top Feature Importance")
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    importances[:10].plot(kind="bar", ax=ax)
    ax.set_ylabel("Importance")
    ax.set_title("Most Influential Diagnostic Features")
    st.pyplot(fig)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Developed by Jahnavi • Streamlit • Machine Learning")

