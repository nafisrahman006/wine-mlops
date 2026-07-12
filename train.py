import argparse
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn


def train(csv_path: str = "data/winequality-red.csv"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    for attempt in range(10):
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("wine-quality-prediction")
            break
        except Exception as e:
            print(f"Waiting for MLflow... ({attempt + 1}/10): {e}")
            time.sleep(5)
    else:
        raise RuntimeError("MLflow tracking server not available")

    df = pd.read_csv(csv_path)
    print(f"Loaded data from: {csv_path}")
    print(f"Dataset shape: {df.shape}")

    df["target"] = (df["quality"] >= 7).astype(int)
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")

    feature_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "alcohol"]
    X = df[feature_cols]
    y = df["target"]

    training_stats = {
        "alcohol_mean": float(X["alcohol"].mean()),
        "alcohol_std": float(X["alcohol"].std()),
        "fixed_acidity_mean": float(X["fixed acidity"].mean()),
        "fixed_acidity_std": float(X["fixed acidity"].std()),
        "volatile_acidity_mean": float(X["volatile acidity"].mean()),
        "volatile_acidity_std": float(X["volatile acidity"].std()),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42, probability=True)
    }

    print("\n" + "="*50)
    print("TRAINING ALL MODELS - NO AUTO-REGISTRATION")
    print("="*50)
    print("\nAfter training, go to MLflow UI and:")
    print("1. Click 'wine-quality-prediction' experiment")
    print("2. Compare runs, pick the best model")
    print("3. Click the best run -> Artifacts -> Register Model")
    print("4. Name: wine-model")
    print("5. Go to Model Registry -> wine-model -> Add alias -> 'production'")
    print("="*50 + "\n")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            for stat_name, stat_value in training_stats.items():
                mlflow.log_param(f"train_stat_{stat_name}", stat_value)

            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"\n{name}:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Train time: {train_time:.2f}s")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("train_time_sec", train_time)

            # Log model artifact
            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
                print(f"  Model artifact logged to MLflow")
            except Exception as e:
                print(f"  Failed to log model: {e}")

    print("\n" + "="*50)
    print("ALL MODELS TRAINED")
    print("="*50)
    print("\nNext steps:")
    print("1. Open MLflow UI: http://localhost:5001")
    print("2. Go to 'wine-quality-prediction' experiment")
    print("3. Compare runs, note the best Run ID")
    print("4. Click best run -> Artifacts -> Register Model")
    print("5. Name it 'wine-model', then add alias 'production'")
    print("6. Restart API: docker-compose restart api")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/winequality-red.csv")
    args = parser.parse_args()

    train(args.data)