import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data",   type=str)
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--random_seed",  type=int,   default=42)
    parser.add_argument("--output_train", type=str)
    parser.add_argument("--output_test",  type=str)
    args = parser.parse_args()

    # Load curated features from gold layer
    print(f"Loading from: {args.input_data}")
    files = [
        os.path.join(args.input_data, f)
        for f in os.listdir(args.input_data)
        if f.endswith(".parquet")
    ]
    df = pd.concat([pd.read_parquet(f) for f in files])

    print(f"Total rows    : {len(df)}")
    print(f"Total columns : {len(df.columns)}")

    # Drop non-feature columns — keep label_encoded
    drop_cols = ["file_name", "label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate features and label
    X = df.drop(columns=["label_encoded"])
    y = df["label_encoded"]

    print(f"Features : {X.shape[1]}")
    print(f"Label distribution:\n{y.value_counts()}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y
    )

    print(f"\nTrain rows : {len(X_train)}")
    print(f"Test rows  : {len(X_test)}")

    # Save train
    os.makedirs(args.output_train, exist_ok=True)
    train_df = X_train.copy()
    train_df["label_encoded"] = y_train.values
    train_df.to_parquet(
        os.path.join(args.output_train, "train.parquet"),
        index=False
    )

    # Save test
    os.makedirs(args.output_test, exist_ok=True)
    test_df = X_test.copy()
    test_df["label_encoded"] = y_test.values
    test_df.to_parquet(
        os.path.join(args.output_test, "test.parquet"),
        index=False
    )

    print("Split complete ✅")

if _name_ == "_main_":
    main()