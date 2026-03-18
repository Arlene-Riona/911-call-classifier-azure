import argparse
import pandas as pd
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",         type=str)
    parser.add_argument("--test_data",          type=str)
    parser.add_argument("--n_estimators",       type=int,   default=100)
    parser.add_argument("--max_depth",          type=int,   default=10)
    parser.add_argument("--random_seed",        type=int,   default=42)
    parser.add_argument("--model_output",       type=str)
    parser.add_argument("--evaluation_metrics", type=str)
    parser.add_argument("--test_predictions",   type=str)
    args = parser.parse_args()

    # Load train and test
    print(f"Loading train from: {args.train_data}")
    train_df = pd.read_parquet(os.path.join(args.train_data, "train.parquet"))

    print(f"Loading test from: {args.test_data}")
    test_df = pd.read_parquet(os.path.join(args.test_data, "test.parquet"))

    print(f"Train rows : {len(train_df)}")
    print(f"Test rows  : {len(test_df)}")

    # Separate features and labels
    X_train = train_df.drop(columns=["label_encoded"])
    y_train = train_df["label_encoded"]
    X_test  = test_df.drop(columns=["label_encoded"])
    y_test  = test_df["label_encoded"]

    print(f"Features : {X_train.shape[1]}")
    print(f"Train label distribution:\n{y_train.value_counts()}")

    # Train Random Forest with class weights to handle imbalance
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_seed,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Training complete ✅")

    # Predict
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    auc      = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    report   = classification_report(
        y_test, y_pred,
        target_names=["medical", "fire", "violence"],
        output_dict=True
    )

    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["medical", "fire", "violence"]
    ))

    # Save model
    os.makedirs(args.model_output, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_output, "model.pkl"))
    print(f"Model saved ✅")

    # Save metrics
    os.makedirs(args.evaluation_metrics, exist_ok=True)
    metrics = {
        "accuracy":                accuracy,
        "f1_weighted":             f1,
        "auc_roc":                 auc,
        "classification_report":   report
    }
    with open(os.path.join(args.evaluation_metrics, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved ✅")

    # Save predictions with probabilities
    os.makedirs(args.test_predictions, exist_ok=True)
    predictions_df = pd.DataFrame({
        "true_label":      y_test.values,
        "predicted_label": y_pred,
        "prob_medical":    y_pred_proba[:, 0],
        "prob_fire":       y_pred_proba[:, 1],
        "prob_violence":   y_pred_proba[:, 2]
    })
    predictions_df.to_parquet(
        os.path.join(args.test_predictions, "test_predictions.parquet"),
        index=False
    )
    print(f"Predictions saved ✅")
    print("\nPipeline complete ✅")

if _name_ == "_main_":
    main()