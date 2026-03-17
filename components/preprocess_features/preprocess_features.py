import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   type=str)
    parser.add_argument("--test_size",   type=float, default=0.2)
    parser.add_argument("--random_seed", type=int,   default=42)
    parser.add_argument("--train_data",  type=str)
    parser.add_argument("--test_data",   type=str)
    args = parser.parse_args()

    # Load data from curated/features_gold
    print(f"Loading data from: {args.data_path}")
    files = [
        os.path.join(args.data_path, f) 
        for f in os.listdir(args.data_path) 
        if f.endswith(".parquet")
    ]
    df = pd.concat([pd.read_parquet(f) for f in files])
    print(f"Loaded rows    : {len(df)}")
    print(f"Loaded columns : {len(df.columns)}")

    # Drop non-feature columns except label
    drop_cols = ["file_name", "label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate features and label
    X = df.drop(columns=["label_encoded"])
    y = df["label_encoded"]

    print(f"Features : {X.shape[1]}")
    print(f"Labels   : {y.value_counts().to_dict()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y  # maintain class distribution
    )

    print(f"Train rows : {len(X_train)}")
    print(f"Test rows  : {len(X_test)}")

    # Save train set
    os.makedirs(args.train_data, exist_ok=True)
    train_df = X_train.copy()
    train_df["label_encoded"] = y_train
    train_df.to_parquet(
        os.path.join(args.train_data, "train.parquet"), 
        index=False
    )

    # Save test set
    os.makedirs(args.test_data, exist_ok=True)
    test_df = X_test.copy()
    test_df["label_encoded"] = y_test
    test_df.to_parquet(
        os.path.join(args.test_data, "test.parquet"), 
        index=False
    )

    print("Train/test split complete")

if __name__ == "__main__":
    main()