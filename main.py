import os
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import numpy as np

# Load .env file
load_dotenv()

app = FastAPI()

# ========== Environment Variables (no defaults) ==========
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Validate all required vars
missing = []
if not MLFLOW_TRACKING_URI:
    missing.append("MLFLOW_TRACKING_URI")
if not MLFLOW_S3_ENDPOINT_URL:
    missing.append("MLFLOW_S3_ENDPOINT_URL")
if not AWS_ACCESS_KEY_ID:
    missing.append("AWS_ACCESS_KEY_ID")
if not AWS_SECRET_ACCESS_KEY:
    missing.append("AWS_SECRET_ACCESS_KEY")

if missing:
    raise ValueError(f"Missing environment variables: {', '.join(missing)}. Check .env file")

# Set MLflow config
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

print(" All environment variables loaded from .env")
# =======================================================

# ========== Load Model from Registry ==========
client = MlflowClient()
model_version = client.get_model_version_by_alias("wine-model", "production")
model_uri = f"models:/{model_version.name}@{model_version.aliases[0]}"
model = mlflow.sklearn.load_model(model_uri)

print(f" Loaded model: wine-model@production")
print(f" Model ID: {model_version.run_id}")
print(f" Version: {model_version.version}")
print(f" Source: {model_version.source}")
# ==============================================

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    alcohol: float

@app.get("/")
def read_root():
    return {"message": "Wine Quality Prediction API is live"}

@app.post("/predict")
def predict(data: WineInput):
    input_data = np.array([[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.alcohol
    ]])
    prediction = model.predict(input_data)[0]
    return {"good_quality": bool(prediction)}