# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import os

# --------------------------------------------------
# App initialization
# --------------------------------------------------
app = FastAPI(title="Credit Scoring API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
THRESH_PATH = os.path.join(BASE_DIR, "model", "best_threshold.json")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "df_clients_sample.pkl")

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
model = joblib.load(MODEL_PATH)

with open(THRESH_PATH, "r") as f:
    best_threshold = json.load(f)["threshold"]

with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

df_clients = joblib.load(DATA_PATH)

# --------------------------------------------------
# Input schema
# --------------------------------------------------
class ClientRequest(BaseModel):
    client_id: int

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "API is running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(request: ClientRequest):

    if request.client_id not in df_clients["SK_ID_CURR"].values:
        raise HTTPException(status_code=404, detail="Client ID not found")

    client_row = df_clients[df_clients["SK_ID_CURR"] == request.client_id]

    X = client_row[feature_names]

    proba = model.predict_proba(X)[0, 1]

    decision = (
        "Loan NOT approved (high default risk) !"
        if proba >= best_threshold
        else "Loan approved."
    )

    return {
        "client_id": request.client_id,
        "default_probability": float(proba),
        "threshold": best_threshold,
        "decision": decision
    }
