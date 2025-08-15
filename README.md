# Breast Cancer Prediction

## Overview
Predicts whether a tumor is **benign or malignant** using machine learning.

## Dataset
**Wisconsin Breast Cancer Dataset (WBCD)**  
- **ID:** Patient identifier  
- **Diagnosis:** Malignant (M) / Benign (B) **[Target]**  
- **Features:** Cell nuclei measurements (Mean, SE, Worst) for:
  - Radius, Texture, Perimeter, Area  
  - Smoothness, Compactness, Concavity, Concave points  
  - Symmetry, Fractal Dimension  

## Tools & Libraries
- Python: `pandas`, `numpy`, `scikit-learn`  
- Visualization: `matplotlib`, `seaborn`, `plotly`  
- Model saving: `joblib`  
- Optional UI: `Streamlit`

## Models
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- AdaBoost  
- **Resampling:** SMOTE (to handle imbalanced data)

## Workflow
1. Preprocess data (missing values, encoding, scaling)  
2. Train/test split  
3. Train & evaluate models (Accuracy, Recall, F1-score)  
4. Save best model for deployment

## Run
```bash
git clone https://github.com/Melissiasamir/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction
pip install -r requirements.txt
streamlit run app.py
