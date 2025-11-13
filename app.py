
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    from huggingface_hub import hf_hub_download

    st.title("⚙️ Predictive Maintenance: Engine Condition Predictor")

    st.write("Upload data or input manually to predict engine condition using XGBoost model.")
    from huggingface_hub import HfApi, HfFolder
import os

SPACE_ID = "sriharimudakavi/engine-condition-predictor"
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# Download app.py
api.repo_download(
    repo_id=SPACE_ID,
    repo_type="space",
    local_dir="temp_space"
)

app_path = "temp_space/app.py"

# Read and replace
text = open(app_path).read()
text = text.replace(
    'sriharimudakavi/engine-model',
    'sriharimudakavi/engine-condition-xgboost-tuned'
)

open(app_path, "w").write(text)

# Upload back
api.upload_file(
    path_or_fileobj=app_path,
    repo_id=SPACE_ID,
    repo_type="space",
    path_in_repo="app.py"
)

print("Updated repo_id and uploaded app.py!")
    model = joblib.load(model_path)

    option = st.sidebar.selectbox("Input Method", ["Manual Entry", "Upload CSV"])

    if option == "Manual Entry":
        rpm = st.number_input("Engine RPM", 0, 3000, 800)
        oil_p = st.number_input("Lube Oil Pressure", 0.0, 10.0, 3.0)
        fuel_p = st.number_input("Fuel Pressure", 0.0, 25.0, 6.0)
        cool_p = st.number_input("Coolant Pressure", 0.0, 10.0, 2.0)
        oil_t = st.number_input("Lube Oil Temp (°C)", 60.0, 120.0, 80.0)
        cool_t = st.number_input("Coolant Temp (°C)", 60.0, 200.0, 90.0)
        df = pd.DataFrame([[rpm, oil_p, fuel_p, cool_p, oil_t, cool_t]],
                          columns=["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"])
        st.write(df)
        if st.button("🔍 Predict Engine Condition"):
            pred = model.predict(df)[0]
            st.success(f"Predicted Condition: {'Normal' if pred==1 else 'Abnormal'}")
    else:
        file = st.file_uploader("Upload CSV file", type=["csv"])
        if file:
            df = pd.read_csv(file)
            preds = model.predict(df)
            df["Predicted Condition"] = np.where(preds==1, "Normal", "Abnormal")
            st.write(df)
