# Implementez-un-modele-de-scoring
Implement a credit scoring model with a FastAPI API and a Streamlit dashboard.

# Credit Scoring API & Dashboard

## Project Overview
This project implements:
- A FastAPI inference API returning credit default probability
- A Streamlit dashboard calling the API to visualize results

## API
- Framework: FastAPI
- Endpoint: /predict
- Input: client_id
- Output: probability, threshold, decision

## Dashboard
- Framework: Streamlit
- Calls API via HTTP
- Displays:
  - Credit decision
  - Gauge visualization
  - Client information
  - SHAP explanations

## Deployment
- Code versioned with GitHub
- Ready for cloud deployment (Railway)

## Author
Hope90 for OpenClassrooms – Projet 7
