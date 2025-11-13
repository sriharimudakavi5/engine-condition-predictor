
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

st.title("⚙️ Predictive Maintenance: Engine Condition Predictor")

st.write("Upload data or input manually to predict engine condition using XGBoost model.")

# Constants for model/scaler paths on Hugging Face Hub
HF_TOKEN = os.getenv("HF_TOKEN") # Get token from environment variables (Colab secrets or GitHub secrets)
MODEL_REPO_ID = "sriharimudakavi/engine-condition-xgboost-tuned"
MODEL_FILENAME = "xgboost_tuned_model.joblib"
SCALER_REPO_ID = "sriharimudakavi/engine-data" # Assuming scaler is in the dataset repo
SCALER_FILENAME = "scaler.joblib"

# Download model and scaler
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type="model", token=HF_TOKEN)
    scaler_path = hf_hub_download(repo_id=SCALER_REPO_ID, filename=SCALER_FILENAME, repo_type="dataset", token=HF_TOKEN)
except Exception as e:
    st.error(f"Error downloading model or scaler: {e}")
    st.stop()

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

option = st.sidebar.selectbox("Input Method", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    rpm = st.number_input("Engine RPM", 0, 3000, 800)
    oil_p = st.number_input("Lube Oil Pressure", 0.0, 10.0, 3.0)
    fuel_p = st.number_input("Fuel Pressure", 0.0, 25.0, 6.0)
    cool_p = st.number_input("Coolant Pressure", 0.0, 10.0, 2.0)
    oil_t = st.number_input("Lube Oil Temp (°C)", 60.0, 120.0, 80.0)
    cool_t = st.number_input("Coolant Temp (°C)", 60.0, 200.0, 90.0)
    input_df = pd.DataFrame([[rpm, oil_p, fuel_p, cool_p, oil_t, cool_t]],
                              columns=["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"])
    st.write(input_df)
    if st.button("🔍 Predict Engine Condition"):
        # Scale the input data
        scaled_input_df = scaler.transform(input_df)
        pred = model.predict(scaled_input_df)[0]
        st.success(f"Predicted Condition: {'Normal (0)' if pred==0 else 'Faulty (1)'}")
else:
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        input_df = pd.read_csv(file)
        st.write("Uploaded Data:")
        st.dataframe(input_df)
        if st.button("🔍 Predict Engine Condition from CSV"):
            # Ensure the columns match the training data
            if not all(col in input_df.columns for col in ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"]):
                st.error("CSV file must contain 'Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp' columns.")
            else:
                # Scale the input data
                scaled_input_df = scaler.transform(input_df[["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"]])
                preds = model.predict(scaled_input_df)
                input_df["Predicted Condition"] = np.where(preds==0, "Normal (0)", "Faulty (1)")
                st.write("Predictions:")
                st.dataframe(input_df)
