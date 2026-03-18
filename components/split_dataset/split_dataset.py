import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data",   type=str)
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--random_seed",  type=int, default=42)
    parser.add_argument("--output_train", type=str)
    parser.add_argument("--output_test",  type=str)
    args = parser.parse_args()

    # Load curated features from your Notebook 3 output
    print(f"Loading from: {args.input_data}")
    df = pd.read_parquet(os.path.join(args.input_data, "911_call_features.parquet"))
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Encode labels
    label_map = {"medical": 0, "fire": 1, "violence": 2}
    df["label_encoded"] = df["emergency_category"].map(label_map)

    # Separate features and label
    X = df.drop(columns=["emergency_category", "file_name", "title", "description", "label_encoded"])
    y = df["label_encoded"]

    print(f"Features: {X.shape[1]}")
    print(f"Label distribution:\n{y.value_counts()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y
    )

    print(f"\nTrain: {len(X_train)} rows")
    print(f"Test: {len(X_test)} rows")

    # Save train
    os.makedirs(args.output_train, exist_ok=True)
    train_df = X_train.copy()
    train_df["label_encoded"] = y_train
    train_df.to_parquet(os.path.join(args.output_train, "train.parquet"), index=False)

    # Save test
    os.makedirs(args.output_test, exist_ok=True)
    test_df = X_test.copy()
    test_df["label_encoded"] = y_test
    test_df.to_parquet(os.path.join(args.output_test, "test.parquet"), index=False)

    print("Split complete ✅")

if __name__ == "__main__":
    main()