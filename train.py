import argparse
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn


def train(csv_path: str = "data/winequality-red.csv", model_output: str = "wine_model.pkl"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    for attempt in range(10):
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("wine-quality-prediction")
            break
        except Exception:
            print(f"Waiting for MLflow... ({attempt + 1}/10)")
            time.sleep(5)
    else:
        raise RuntimeError("MLflow tracking server not available")

    df = pd.read_csv(csv_path)
    print(f"Loaded data from: {csv_path}")
    
    df["target"] = (df["quality"] >= 7).astype(int)
    
    X = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "alcohol"]]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42)
    }
    
    best_model = None
    best_name = None
    best_acc = 0
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"{name} accuracy: {acc:.4f}")
            
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, artifact_path="model")
            
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name
    
    output_dir = os.path.dirname(model_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(best_model, model_output)
    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")
    print(f"Model saved as {model_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/winequality-red.csv")
    parser.add_argument("--output", type=str, default="wine_model.pkl")
    args = parser.parse_args()

    train(args.data, args.output)