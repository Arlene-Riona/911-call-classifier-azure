import argparse
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",    type=str)
    parser.add_argument("--test_data",     type=str)
    parser.add_argument("--mi_percentile", type=int, default=50)
    parser.add_argument("--train_filtered", type=str)
    parser.add_argument("--test_filtered",  type=str)
    args = parser.parse_args()

    # Load train and test
    train_df = pd.read_parquet(os.path.join(args.train_data, "train.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.test_data,  "test.parquet"))

    print(f"Train rows : {len(train_df)}")
    print(f"Test rows  : {len(test_df)}")
    print(f"Features before filter: {len(train_df.columns) - 1}")

    # Separate features and labels
    X_train = train_df.drop(columns=["label_encoded"])
    y_train = train_df["label_encoded"]
    X_test  = test_df.drop(columns=["label_encoded"])
    y_test  = test_df["label_encoded"]

    # Mutual information filter - fit on train only
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    threshold = np.percentile(mi_scores, args.mi_percentile)
    selected_features = X_train.columns[mi_scores >= threshold].tolist()

    print(f"Features after MI filter : {len(selected_features)}")
    print(f"Features removed         : {len(X_train.columns) - len(selected_features)}")

    # Apply to both train and test
    X_train_filtered = X_train[selected_features]
    X_test_filtered  = X_test[selected_features]

    # Save filtered train
    os.makedirs(args.train_filtered, exist_ok=True)
    train_out = X_train_filtered.copy()
    train_out["label_encoded"] = y_train.values
    train_out.to_parquet(
        os.path.join(args.train_filtered, "train_filtered.parquet"),
        index=False
    )

    # Save filtered test
    os.makedirs(args.test_filtered, exist_ok=True)
    test_out = X_test_filtered.copy()
    test_out["label_encoded"] = y_test.values
    test_out.to_parquet(
        os.path.join(args.test_filtered, "test_filtered.parquet"),
        index=False
    )

    print("Filter selection complete ✅")

if __name__ == "__main__":
    main()