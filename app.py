# app.py
import streamlit as st
import numpy as np
import joblib
import os

# ----------------------------
# Paths
# ----------------------------
MODEL_DIR = "Model"

# Load saved artifacts
svm_model = joblib.load(os.path.join(MODEL_DIR, "breast_cancer_svm_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "breast_cancer_scaler.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "breast_cancer_pca.pkl"))

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Breast Cancer Prediction 🩺", layout="wide", page_icon="💖")
st.title("Breast Cancer Prediction 🩺")
st.markdown("""
Predict whether a tumor is **Malignant (M)** or **Benign (B)** using SVM.
Use the sliders to input tumor features.
""")

# Features grouped by type
FEATURE_GROUPS = {
    "Mean Features": [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ],
    "SE Features": [
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se'
    ],
    "Worst Features": [
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
}

st.subheader("Tumor Features Input")

# Store input values
input_data = []

for group, features in FEATURE_GROUPS.items():
    with st.expander(group, expanded=True):
        for feature in features:
            val = st.slider(feature, 0.0, 50.0, 0.0, 0.1)
            input_data.append(val)

# Convert input to array
input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)
input_pca = pca.transform(input_scaled)

# Predict button
if st.button("Predict"):
    prediction = svm_model.predict(input_pca)[0]
    decision_score = svm_model.decision_function(input_pca)[0]

    if prediction == 0:
        st.markdown(f"<h2 style='color:red;'>Prediction: Malignant (M) ❌</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4>Decision Score: {decision_score:.4f}</h4>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:green;'>Prediction: Benign (B) ✅</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4>Decision Score: {decision_score:.4f}</h4>", unsafe_allow_html=True)
