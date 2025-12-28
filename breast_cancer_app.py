#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------
# Page config and style
# -------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    body { background-color: #f3e5ab; }  /* soft coffee background */
    h1, h2, h3 { color: #6f4e37; font-family: 'Arial'; }
    .stButton>button { background-color: #6f4e37; color: white; font-weight:bold; }
    .stSlider>div>div>div>div>div { color: #6f4e37; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Breast Cancer Predictor Dashboard")
st.write("Enter patient features in the sidebar to predict cancer type.")

# -------------------------
# Load dataset and train model
# -------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Cancer")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# -------------------------
# Sidebar input for top features
# -------------------------
st.sidebar.header("Patient Features Input")
top_features = feature_importances.index[:8]  # top 8 features

input_data = []
for feature in top_features:
    val = st.sidebar.slider(
        feature.replace("_", " ").title(),
        float(X[feature].min()),
        float(X[feature].max()),
        float(X[feature].mean())
    )
    input_data.append(val)

# Fill remaining features with zeros
full_input = np.array(input_data + [0]*(X.shape[1]-len(top_features))).reshape(1, -1)
full_input_scaled = scaler.transform(full_input)

# -------------------------
# Prediction
# -------------------------
if st.sidebar.button("Predict"):
    prediction = rf_model.predict(full_input_scaled)[0]
    prediction_prob = rf_model.predict_proba(full_input_scaled)[0]

    cancer_class = "Benign (Not Cancer)" if prediction == 1 else "Malignant (Cancer)"
    prob_malignant = prediction_prob[0]
    prob_benign = prediction_prob[1]

    # Display result in main area
    st.subheader("Prediction Result")
    st.markdown(f"<h2 style='color:#6f4e37; text-align:center;'>{cancer_class}</h2>", unsafe_allow_html=True)

    # Show probability bars
    st.subheader("Prediction Probability")
    st.progress(int(prob_benign*100))
    st.write(f"Benign probability: {prob_benign:.2f}")
    st.write(f"Malignant probability: {prob_malignant:.2f}")

    # Feature importance chart
    st.subheader("Top Feature Importance")
    fig, ax = plt.subplots(figsize=(8,4))
    feature_importances[:10].plot(kind='bar', color="#6f4e37", ax=ax)
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    ax.set_title("Top 10 Features Influencing Prediction")
    st.pyplot(fig)

    # Optional: allow download of input and prediction
    result_df = pd.DataFrame([full_input[0]], columns=X.columns)
    result_df['Prediction'] = cancer_class
    st.download_button("Download Result as CSV", result_df.to_csv(index=False), "prediction.csv", "text/csv")
