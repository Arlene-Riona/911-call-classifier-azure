import json
import os
import joblib
import numpy as np
import pandas as pd

model = None

NON_FEATURE_COLS = [
    "file_name", "label", "label_encoded"
]

def init():
    global model
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model_output", "model.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data["data"])
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        X = df[feature_cols].values
        preds = model.predict(X)
        proba = model.predict_proba(X)
        return {
            "predictions": preds.tolist(),
            "probabilities": proba.tolist()
        }
    except Exception as e:
        return {"error": str(e)}