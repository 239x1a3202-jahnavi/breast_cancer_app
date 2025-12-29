import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction System")
st.write(
    "Please **manually type each feature value** exactly as shown in the medical report."
)

# -------------------------
# Load Dataset
# -------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Cancer")

# -------------------------
# Train Model
# -------------------------
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

# -------------------------
# Manual Input Form (TEXT ONLY)
# -------------------------
st.sidebar.header("Patient Feature Entry")

user_input = {}
input_errors = False

for feature in X.columns:
    value = st.sidebar.text_input(
        label=f"Enter value of {feature.replace('_', ' ').title()}",
        value=str(round(X[feature].mean(), 5))
    )

    try:
        user_input[feature] = float(value)
    except ValueError:
        st.sidebar.error(f"Invalid numeric value for {feature}")
        input_errors = True

# -------------------------
# Prediction
# -------------------------
if st.sidebar.button("🔍 Predict Cancer"):
    if input_errors:
        st.error("❌ Please correct invalid inputs before prediction.")
    else:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("🟢 Benign (Not Cancer)")
        else:
            st.error("🔴 Malignant (Cancer)")

        st.subheader("Prediction Probability")
        st.write(f"🟢 Benign Probability: **{probability[1]:.2f}**")
        st.write(f"🔴 Malignant Probability: **{probability[0]:.2f}**")

        # -------------------------
        # Feature Importance
        # -------------------------
        st.subheader("Top 10 Important Features")

        importances = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        importances[:10].plot(kind="bar", ax=ax)
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Developed by Jahnavi | Clinical Data Entry ML Application")

