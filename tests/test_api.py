import requests

# Local Run API
#API_URL = "http://127.0.0.1:8000/predict"
API_URL = "credit-scoring-api.up.railway.app"


client_id = 100005
response = requests.post(API_URL, json={"client_id": client_id})

if response.status_code == 200:
    result = response.json()
    print("Default Probability:", result["default_probability"])
    print("Threshold:", result["threshold"])
    print("Prediction:", result["prediction"])
else:
    print("API call failed with status code", response.status_code)
