#!/bin/bash

# Run FastAPI API
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Run Streamlit Dashboard
streamlit run dashboard/dashboard.py --server.port 8080 --server.address 0.0.0.0
