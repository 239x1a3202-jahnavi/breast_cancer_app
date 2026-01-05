import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="OncoPredict AI",
    layout="wide"
)

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

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_scaled, y)

    return model, scaler, X, data.feature_names

model, scaler, X_raw, feature_names = load_and_train()

# =====================================================
# HEADER
# =====================================================
st.title("OncoPredict AI")
st.subheader("Clinical Decision Support Tool")

# =====================================================
# INPUT SECTION
# =====================================================
st.markdown("### Enter Patient Pathology Values")

important_features = [f for f in feature_names if "mean" in f]
user_input = {}

cols = st.columns(4)
for i, feature in enumerate(important_features):
    with cols[i % 4]:
        user_input[feature] = st.number_input(
            label=feature.replace("_", " ").title(),
            min_value=0.0,
            value=float(X_raw[feature].mean()),
            format="%.3f"
        )

analyze = st.button("Generate Diagnostic Report")

# =====================================================
# OUTPUT
# =====================================================
if analyze:
    input_df = pd.DataFrame([user_input]).reindex(
        columns=feature_names,
        fill_value=0
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]

    is_benign = prediction == 1
    confidence = max(probs) * 100

    st.markdown("## Diagnostic Result")

    if is_benign:
        st.success("Result: BENIGN (Low Risk)")
    else:
        st.error("Result: MALIGNANT (High Risk)")

    st.write(f"Confidence: **{confidence:.2f}%**")

    st.markdown(
        "This prediction is based on statistical patterns learned from the "
        "breast cancer dataset. The result should be reviewed by a qualified "
        "medical professional."
    )

    st.markdown("### Submitted Metrics")
    st.dataframe(
        pd.DataFrame({
            "Measurement": user_input.keys(),
            "Value": user_input.values()
        }),
        use_container_width=True
    )

    st.warning(
        "Disclaimer: This tool is for educational purposes only and "
        "must not be used as a standalone medical diagnosis."
    )

# =====================================================
# FOOTER
# =====================================================
st.caption("Jahnavi | Medical ML Research v3.0")
