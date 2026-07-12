import os
import time
import logging
import json
from collections import deque
from typing import List

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

missing = [k for k, v in {
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "MLFLOW_S3_ENDPOINT_URL": MLFLOW_S3_ENDPOINT_URL,
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
}.items() if not v]

if missing:
    raise ValueError(f"Missing environment variables: {', '.join(missing)}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

print("All environment variables loaded from .env")

# 1. MONITORING & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 2. MODEL PERFORMANCE METRICS
metrics_store = {
    "total_predictions": 0,
    "good_quality_count": 0,
    "bad_quality_count": 0,
    "prediction_times_ms": deque(maxlen=10000),
    "recent_inputs": deque(maxlen=100),
}

# 3. INPUT VALIDATION
class WineInput(BaseModel):
    fixed_acidity: float = Field(..., ge=0, le=20)
    volatile_acidity: float = Field(..., ge=0, le=2)
    citric_acid: float = Field(..., ge=0, le=1)
    residual_sugar: float = Field(..., ge=0, le=20)
    alcohol: float = Field(..., ge=8, le=15)

    @validator('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'alcohol')
    def check_non_negative(cls, v):
        if v < 0:
            raise ValueError('Value cannot be negative')
        return v

# 4. BATCH PREDICTION SCHEMA
class BatchWineInput(BaseModel):
    wines: List[WineInput]

# LOAD MODEL FROM REGISTRY
client = MlflowClient()
MODEL_ALIAS = "production"

def load_model_from_registry(max_retries: int = 30, delay: int = 5):
    for attempt in range(max_retries):
        try:
            model_version = client.get_model_version_by_alias("wine-model", MODEL_ALIAS)
            model_uri = f"models:/{model_version.name}@{MODEL_ALIAS}"
            model = mlflow.sklearn.load_model(model_uri)

            logger.info(f"Loaded model: wine-model@{MODEL_ALIAS}")
            logger.info(f"Version: {model_version.version}")
            logger.info(f"Run ID: {model_version.run_id}")
            return model

        except Exception as e:
            if attempt == 0:
                print("\n" + "="*60)
                print("NO PRODUCTION MODEL FOUND IN REGISTRY")
                print("="*60)
                print("\nTraining is running. After it finishes:")
                print("1. Open MLflow UI: http://localhost:5001")
                print("2. Go to 'wine-quality-prediction' experiment")
                print("3. Pick best model, click it -> Artifacts -> Register Model")
                print("4. Name: wine-model")
                print("5. Go to Model Registry -> wine-model -> Add alias -> 'production'")
                print("6. Then this API will auto-load it")
                print("="*60 + "\n")

            logger.warning(f"Waiting for model in registry... ({attempt + 1}/{max_retries}): {e}")
            time.sleep(delay)

    raise RuntimeError(f"Failed to load model 'wine-model@{MODEL_ALIAS}' from MLflow. Please register a model manually.")

model = load_model_from_registry()

# 5. DRIFT DETECTION
TRAINING_STATS = {
    "alcohol_mean": 10.42,
    "alcohol_std": 1.07,
    "fixed_acidity_mean": 8.32,
    "fixed_acidity_std": 1.74,
    "volatile_acidity_mean": 0.53,
    "volatile_acidity_std": 0.18,
}

DRIFT_THRESHOLD = 2.0

def detect_drift(recent_inputs: deque) -> dict:
    if len(recent_inputs) < 10:
        return {"drift_detected": False, "reason": "Not enough data"}

    recent = list(recent_inputs)
    recent_alcohol = [w["alcohol"] for w in recent]
    recent_alcohol_mean = np.mean(recent_alcohol)
    alcohol_drift = abs(recent_alcohol_mean - TRAINING_STATS["alcohol_mean"]) / TRAINING_STATS["alcohol_std"]

    recent_fa = [w["fixed_acidity"] for w in recent]
    recent_fa_mean = np.mean(recent_fa)
    fa_drift = abs(recent_fa_mean - TRAINING_STATS["fixed_acidity_mean"]) / TRAINING_STATS["fixed_acidity_std"]

    drift_detected = alcohol_drift > DRIFT_THRESHOLD or fa_drift > DRIFT_THRESHOLD

    result = {
        "drift_detected": drift_detected,
        "alcohol_drift_score": round(alcohol_drift, 2),
        "fixed_acidity_drift_score": round(fa_drift, 2),
        "recent_alcohol_mean": round(recent_alcohol_mean, 2),
        "recent_fixed_acidity_mean": round(recent_fa_mean, 2),
    }

    if drift_detected:
        logger.warning(f"DATA DRIFT DETECTED! {json.dumps(result)}")

    return result

app = FastAPI(title="Wine Quality Prediction API", version="2.0.0")

@app.get("/")
def health_check():
    return {
        "message": "Wine Quality Prediction API is live",
        "model": "wine-model@production",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "features": [
            "monitoring_logging",
            "performance_metrics",
            "input_validation",
            "batch_prediction",
            "drift_detection"
        ]
    }

@app.get("/metrics")
def get_metrics():
    total = metrics_store["total_predictions"]
    good = metrics_store["good_quality_count"]
    bad = metrics_store["bad_quality_count"]

    times = list(metrics_store["prediction_times_ms"])
    avg_time = round(sum(times) / len(times), 2) if times else 0

    return {
        "total_predictions": total,
        "good_quality_count": good,
        "bad_quality_count": bad,
        "good_percentage": round((good / total * 100), 2) if total > 0 else 0,
        "avg_prediction_time_ms": avg_time,
        "drift_status": detect_drift(metrics_store["recent_inputs"])
    }

@app.post("/predict")
def predict(data: WineInput):
    start_time = time.time()

    try:
        input_dict = {
            "fixed_acidity": data.fixed_acidity,
            "volatile_acidity": data.volatile_acidity,
            "citric_acid": data.citric_acid,
            "residual_sugar": data.residual_sugar,
            "alcohol": data.alcohol
        }
        logger.info(f"Prediction request: {json.dumps(input_dict)}")

        metrics_store["recent_inputs"].append(input_dict)

        input_data = np.array([[
            data.fixed_acidity,
            data.volatile_acidity,
            data.citric_acid,
            data.residual_sugar,
            data.alcohol
        ]])
        prediction = model.predict(input_data)[0]
        pred_time_ms = round((time.time() - start_time) * 1000, 3)

        metrics_store["total_predictions"] += 1
        if prediction:
            metrics_store["good_quality_count"] += 1
        else:
            metrics_store["bad_quality_count"] += 1
        metrics_store["prediction_times_ms"].append(pred_time_ms)

        logger.info(f"Result: good_quality={bool(prediction)}, Time: {pred_time_ms}ms")

        drift = detect_drift(metrics_store["recent_inputs"])

        return {
            "good_quality": bool(prediction),
            "prediction_time_ms": pred_time_ms,
            "drift_warning": drift["drift_detected"],
            "drift_details": drift if drift["drift_detected"] else None
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
def predict_batch(data: BatchWineInput):
    start_time = time.time()

    try:
        logger.info(f"Batch prediction request: {len(data.wines)} wines")

        predictions = []
        for wine in data.wines:
            input_dict = {
                "fixed_acidity": wine.fixed_acidity,
                "volatile_acidity": wine.volatile_acidity,
                "citric_acid": wine.citric_acid,
                "residual_sugar": wine.residual_sugar,
                "alcohol": wine.alcohol
            }
            metrics_store["recent_inputs"].append(input_dict)

            input_data = np.array([[
                wine.fixed_acidity,
                wine.volatile_acidity,
                wine.citric_acid,
                wine.residual_sugar,
                wine.alcohol
            ]])
            pred = model.predict(input_data)[0]
            predictions.append({"good_quality": bool(pred)})

            metrics_store["total_predictions"] += 1
            if pred:
                metrics_store["good_quality_count"] += 1
            else:
                metrics_store["bad_quality_count"] += 1

        total_time_ms = round((time.time() - start_time) * 1000, 3)
        logger.info(f"Batch result: {len(predictions)} predictions, Time: {total_time_ms}ms")

        return {
            "predictions": predictions,
            "total": len(predictions),
            "time_ms": total_time_ms
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")