import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}


def test_predict_valid_client():
    response = client.post("/predict", json={"client_id": 101099})
    assert response.status_code == 200

    data = response.json()
    assert "default_probability" in data
    assert "threshold" in data
    assert "decision" in data


def test_predict_invalid_client():
    response = client.post("/predict", json={"client_id": 999999999})
    assert response.status_code == 404
