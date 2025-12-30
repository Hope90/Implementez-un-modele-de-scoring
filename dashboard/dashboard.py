# --------------------------
# dashboard/dashboard.py
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import shap
import requests
import joblib
import json
import os

# --------------------------
# Set page
# --------------------------
st.set_page_config(page_title="Credit Dashboard")
st.title("Prêt à Dépenser - Credit Scoring Dashboard")

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

API_URL = "https://credit-scoring-api.up.railway.app/predict"  # update this to your deployed API URL

# --------------------------
# Load data
# --------------------------
df_clients_fe = joblib.load(os.path.join(DATA_DIR, "df_clients_sample.pkl"))  # FE features
X_train = joblib.load(os.path.join(DATA_DIR, "X_train_sample.pkl"))           # for SHAP & global stats
app_test = pd.read_csv(os.path.join(DATA_DIR, "application_test_sample.csv")) # raw features

# Ensure consistent types
for col in df_clients_fe.select_dtypes(include="object").columns:
    df_clients_fe[col] = df_clients_fe[col].astype(str)
for col in df_clients_fe.select_dtypes(include="bool").columns:
    df_clients_fe[col] = df_clients_fe[col].astype(int)

# --------------------------
# Select client
# --------------------------
client_id = st.selectbox("Select client ID", app_test["SK_ID_CURR"].tolist())

# Get row indices
client_row_raw = app_test[app_test["SK_ID_CURR"] == client_id].index[0]
client_row_fe = df_clients_fe[df_clients_fe["SK_ID_CURR"] == client_id].index[0]

# --------------------------
# Call API for prediction
# --------------------------
with st.spinner("Calling scoring API..."):
    response = requests.post(API_URL, json={"client_id": int(client_id)})

if response.status_code != 200:
    st.error("❌ API call failed")
    st.stop()

result = response.json()
proba = result["default_probability"]
threshold = result["threshold"]
decision = result["decision"]

# --------------------------
# Show credit decision
# --------------------------
st.subheader("Credit Decision")
col1, col2 = st.columns(2)
col1.metric("Default Probability", f"{proba:.2%}")
col2.metric("Decision", decision)

# --------------------------
# Gauge visualization
# --------------------------
def plot_gauge(proba, threshold):
    fig, ax = plt.subplots(figsize=(6,3))
    wedge_green = Wedge((0,0), 1, 0, 180*threshold, facecolor='#00FF00', alpha=0.6)
    wedge_red = Wedge((0,0), 1, 180*threshold, 180, facecolor='#FF0000', alpha=0.6)
    ax.add_patch(wedge_green)
    ax.add_patch(wedge_red)

    angle = 180*proba
    ax.arrow(0, 0, 0.8*np.cos(np.radians(angle)), 0.8*np.sin(np.radians(angle)),
             width=0.02, head_width=0.05, head_length=0.1, fc='k', ec='k')
    ax.add_patch(Circle((0,0), 0.05, color='k'))

    threshold_angle = 180*threshold
    ax.plot([0.85*np.cos(np.radians(threshold_angle)), 1.05*np.cos(np.radians(threshold_angle))],
            [0.85*np.sin(np.radians(threshold_angle)), 1.05*np.sin(np.radians(threshold_angle))],
            color='black', linestyle='--', linewidth=2)
    ax.text(1.1*np.cos(np.radians(threshold_angle)),
            1.1*np.sin(np.radians(threshold_angle)),
            f"Threshold\n{threshold:.2%}",
            ha='center', va='center', fontsize=6, fontweight='bold')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    return fig

st.subheader("Credit Risk Gauge")
st.pyplot(plot_gauge(proba, threshold))

# --------------------------
# Show client raw info
# --------------------------
st.subheader("Client Information")
st.dataframe(app_test.loc[[client_row_raw]])

# --------------------------
# SHAP explanations
# --------------------------
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
explainer = shap.TreeExplainer(model)
x_df = df_clients_fe.loc[[client_row_fe], X_train.columns]
shap_values = explainer(x_df)

# Local SHAP top 10
st.subheader("Local Feature Importance (Top 10)")
shap_arr = shap_values.values[0]
top_idx = np.argsort(np.abs(shap_arr))[-10:]
top_features = X_train.columns[top_idx]
top_values = shap_arr[top_idx]

fig, ax = plt.subplots(figsize=(6,4))
ax.barh(top_features, top_values, color=['#008bfb' if v>0 else '#ED1C24' for v in top_values])
ax.set_xlabel("SHAP value")
ax.set_title("Top 10 Feature Impact")
st.pyplot(fig)

# Local SHAP waterfall
st.subheader("Local SHAP Waterfall")
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(plt.gcf())

# Global SHAP summary
st.subheader("Global Feature Importance")
shap_values_global = explainer(X_train[:20])
shap.summary_plot(shap_values_global, X_train[:20], show=False)
st.pyplot(plt.gcf())

# --------------------------
# Univariate feature distribution
# --------------------------
st.subheader("Feature Distribution")
feature = st.selectbox("Select Feature", X_train.columns, key="feat_dist")
fig, ax = plt.subplots()
ax.hist(df_clients_fe[feature], bins=30, alpha=0.6)
ax.axvline(df_clients_fe.loc[client_row_fe, feature], color="red", linewidth=3)
ax.set_title(f"Distribution of {feature}")
st.pyplot(fig)

# --------------------------
# Bivariate plot
# --------------------------
st.subheader("Bivariate Plot")
feature1 = st.selectbox("Select First Feature", X_train.columns, key="f1")
feature2 = st.selectbox("Select Second Feature", X_train.columns, key="f2")
fig, ax = plt.subplots()
sc = ax.scatter(df_clients_fe[feature1], df_clients_fe[feature2],
                c=df_clients_fe[X_train.columns[0]], alpha=0.3, cmap='viridis')
ax.scatter(df_clients_fe.loc[client_row_fe, feature1], df_clients_fe.loc[client_row_fe, feature2],
           color='red', s=200)
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_title(f"{feature1} vs {feature2}")
st.pyplot(fig)
