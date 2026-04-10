import requests
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

ENDPOINT_URL = "https://call911-endpoint.qatarcentral.inference.ml.azure.com/score"
API_KEY = ""

NON_FEATURE_COLS = [
    "file_name", "label", "label_encoded"
]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def main():
    # Load the gold parquet directly — you don't have a deploy split,
    # so just use your test set or the full gold as a proxy
    df = pd.read_parquet("test_filtered.parquet")

    y_true = df["label_encoded"].values
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]

    payload = {"data": X.to_dict(orient="records")}

    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(payload)
    )

    result = response.json()
    print("Raw response:", result)  # add this
    preds = result["predictions"]

    print("Deployment Accuracy:", accuracy_score(y_true, preds))
    print(classification_report(
        y_true, preds,
        target_names=["medical", "fire", "violence"]
    ))

if __name__ == "__main__":
    main()