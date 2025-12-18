# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import numpy as np
import os

# ----------------------------
# 初始化 FastAPI
# ----------------------------
app = FastAPI(title="Credit Scoring API")

# ----------------------------
# 加载模型和数据
# ----------------------------
MODEL_PATH = r"C:\Users\xwei3\OneDrive - TEN\Perso Info\DS\P7\credit_scoring_api\model\best_model.pkl"
DATA_PATH = r"C:\Users\xwei3\OneDrive - TEN\Perso Info\DS\P7\credit_scoring_api\data\df_clients.pkl"
THRESH_PATH = r"C:\Users\xwei3\OneDrive - TEN\Perso Info\DS\P7\credit_scoring_api\model\best_threshold.json"

# 加载 LightGBM 模型
model = joblib.load(MODEL_PATH)

# 加载所有客户数据
df_clients = joblib.load(DATA_PATH)

# 加载阈值
with open(THRESH_PATH, "r") as f:
    threshold_dict = json.load(f)
best_threshold = threshold_dict.get("threshold", 0.51)  # 默认0.51

# ----------------------------
# 定义输入数据模型
# ----------------------------
class ClientID(BaseModel):
    client_id: int

# ----------------------------
# 根路径测试
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running"}

# ----------------------------
# 预测接口
# ----------------------------
@app.post("/predict")
def predict(client: ClientID):
    # 检查客户ID是否存在
    if client.client_id not in df_clients['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail="Client ID not found")

    # 取出客户特征
    X = df_clients[df_clients['SK_ID_CURR'] == client.client_id].drop(columns=['SK_ID_CURR'])

    # 模型预测概率
    proba = model.predict_proba(X)[0][1]  # 正类概率

    # 根据阈值判断信用结果
    prediction = "Loan NOT approved, risk of default!" if proba >= best_threshold else "Loan approved!"

    return {
        "client_id": client.client_id,
        "default_probability": float(proba),
        "threshold": best_threshold,
        "prediction": prediction
    }



