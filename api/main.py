# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

# ----------------------------
# Initialize FastAPI
# ----------------------------
app = FastAPI(title="Credit Scoring API")

# ----------------------------
# Load model and metadata
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
THRESH_PATH = os.path.join(BASE_DIR, "model", "best_threshold.json")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.json")

model = joblib.load(MODEL_PATH)

with open(THRESH_PATH, "r") as f:
    best_threshold = json.load(f).get("threshold", 0.5)

with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)["features"]

# ----------------------------
# Define input data model
# ----------------------------
class ClientData(BaseModel):
    client_id: int
    features: list  # Features must follow feature_names order

# ----------------------------
# Root test
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running"}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(client: ClientData):
    if len(client.features) != len(feature_names):
        raise HTTPException(status_code=400, detail="Feature length mismatch")

    X = pd.DataFrame([client.features], columns=feature_names)
    proba = model.predict_proba(X)[0][1]
    prediction = "Loan NOT approved, risk of default!" if proba >= best_threshold else "Loan approved!"

    return {
        "client_id": client.client_id,
        "default_probability": float(proba),
        "threshold": best_threshold,
        "prediction": prediction
    }
