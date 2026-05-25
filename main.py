from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize the FastAPI application
app = FastAPI(title="CardioRisk ML Scoring Engine", version="1.0")

# 2. Load the trained model brain asset directly into memory once
model = joblib.load('optimized_heart_disease_model.pkl')

# 3. Define the strict data structure expected for an incoming patient
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"status": "online", "message": "CardioRisk API is operational."}

@app.post("/predict")
def predict_cardiac_risk(patient: PatientData):
    # Convert incoming JSON data cleanly into a Pandas DataFrame for the model
    input_df = pd.DataFrame([patient.dict()])
    
    # Run instant inference
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0].tolist()
    
    # Return the clean prediction data structures back to the caller
    return {
        "severity_class": prediction,
        "probabilities": probabilities
    }