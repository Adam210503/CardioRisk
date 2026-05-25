import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🫀 Multi-Class Heart Disease Risk Assessment")

# Load the pre-trained pipeline exactly ONCE into memory
@st.cache_resource
def load_model():
    return joblib.load('optimized_heart_disease_model.pkl')

model = load_model()

# Setup interactive input sliders for real-time inference
st.sidebar.header("Patient Clinical Metrics")
age = st.sidebar.slider("Age", 20, 100, 50)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0)

# Build the feature array based on user entries
input_data = pd.DataFrame([[age, 1, 3, 120, chol, 0, 1, thalach, 0, oldpeak, 1, 0, 3]], 
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

if st.button("Analyze Cardiac Risk Profile"):
    # Run instant prediction with calibrated probability arrays
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.write(f"### Predicted Diagnosis: Class {prediction}")
    st.write(f"Confidence Matrix: {np.round(probabilities * 100, 2)}")