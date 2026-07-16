# 🍷 Wine Quality MLOps Pipeline

An end-to-end MLOps project for wine quality prediction using **MLflow**, **MinIO**, **PostgreSQL**, **FastAPI**, and **Docker**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![MinIO](https://img.shields.io/badge/MinIO-S3_Compatable-red)


---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Services](#-services)
- [MLflow Tracking](#-mlflow-tracking)
- [MinIO Object Storage](#-minio-object-storage)
- [Model Registry](#-model-registry)
- [API Usage](#-api-usage)
- [Screenshots](#-screenshots)


---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Client    │────▶│  FastAPI     │────▶│  MLflow Registry │
│  (curl/web) │     │   (Port 8000)│     │  (Port 5001)    │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                  │
                       ┌──────────────┐          │
                       │   PostgreSQL │◀─────────┘
                       │  (Metadata)  │
                       └──────────────┘
                              │
                       ┌──────────────┐
                       │    MinIO     │
                       │  (Artifacts) │
                       └──────────────┘
```

**Data Flow:**
```
winequality-red.csv
        │
        ▼
   train.py (4 Models)
        │
   ├────┴────┬────────┬──────────┐
   │         │        │          │
   ▼         ▼        ▼          ▼
PostgreSQL  MinIO   MLflow     Model
(Params)   (Files)  (Metrics)  Registry
   │         │        │          │
   └─────────┴────────┴──────────┘
              │
              ▼
        main.py (API)
              │
              ▼
      /predict endpoint
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multi-Model Training** | RandomForest, GradientBoosting, LogisticRegression, SVM |
| 📊 **MLflow Tracking** | Parameters, metrics, artifacts logging |
| 🗂️ **Model Registry** | Version control with aliases (production, staging) |
| 🪣 **MinIO Storage** | S3-compatible artifact storage |
| 🐘 **PostgreSQL** | MLflow backend |
| 🚀 **FastAPI** | High-performance prediction API |
| 🔒 **Secret Management** | Environment variables via `.env` |
| 🐳 **Docker Compose** | One-command full stack deployment |
| 📈 **Data Drift Detection** | Z-score monitoring on alcohol & fixed acidity vs. training stats |
| 🧪 **Batch Prediction** | `POST /predict/batch` for multiple wines in one request |
| 📉 **Performance Metrics** | `GET /metrics` for prediction counts, accuracy, latency, drift status |
| ✅ **Input Validation** | Pydantic models enforce realistic chemical ranges |
| 🔄 **Registry Retry Logic** | API polls MLflow for 2.5 min waiting for the production alias |
| 📝 **Structured Logging** | Dual file + stdout logging with timestamps |
| ❤️ **Docker Healthcheck** | Built-in container health check via curl |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### 1. Clone Repository

```bash
git clone https://github.com/nafisrahman006/wine-mlops.git
cd wine-quality-mlops
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your secrets (optional for local dev)
nano .env
```

### 3. Start All Services

```bash
docker-compose up --build
```

This will:
1. Start PostgreSQL database
2. Start MinIO object storage
3. Create MLflow bucket in MinIO
4. Start MLflow tracking server
5. Train 4 models (RandomForest, GradientBoosting, LogisticRegression, SVM)
6. Start FastAPI prediction service

### 4. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **FastAPI Docs** | http://localhost:8000/docs | - |
| **MLflow UI** | http://localhost:5001 | - |
| **MinIO Console** | http://localhost:9001 | 

---

## 📁 Project Structure

```
wine-quality-mlops/
│
├── 📂 data/
│   └── winequality-red.csv          # Dataset (Git-tracked)
│
├── 📄 main.py                        # FastAPI application
├── 📄 train.py                       # Multi-model training script
├── 📄 requirements.txt               # Python dependencies
├── 📄 Dockerfile                     # Container image
│
├── 📄 docker-compose.yml             # Full stack orchestration
├── 📄 .env.example                   # Environment template
├── 📄 .env                           # Secrets (Git-ignored)
│
├── 📄 .gitignore                     # Git ignore rules
├── 📄 .dockerignore                  # Docker ignore rules
│
└── 📄 README.md                      # This file
```

---

## 🔧 Services

### PostgreSQL (Database)

- **Image:** `postgres:15`
- **Port:** `5432` (internal)
- **Purpose:** MLflow metadata storage (runs, params, metrics)
- **Volume:** `pgdata` (persistent)

### MinIO (Object Storage)

- **Image:** `minio/minio:latest`
- **Ports:** `9000` (API), `9001` (Console)
- **Purpose:** Model artifacts, dataset backups
- **Volume:** `minio_data` (persistent)
- **Buckets:** `mlflow` (artifacts)

### MLflow (Tracking Server)

- **Port:** `5001`
- **Backend:** PostgreSQL
- **Artifacts:** MinIO S3
- **Features:** Experiment tracking, Model Registry

### Train (Model Training)

- **Trains 4 models:** RandomForest, GradientBoosting, LogisticRegression, SVM
- **Logs to:** MLflow + MinIO
- **Outputs:** Training statistics (logged as MLflow params for drift reference)

### API (FastAPI)

- **Port:** `8000`
- **Loads model from:** MLflow Model Registry (`wine-model@production`)
- **Endpoints:** `GET /`, `GET /metrics`, `POST /predict`, `POST /predict/batch`
- **Features:** Input validation, drift detection, performance metrics, structured logging

---

## 📊 MLflow Tracking

### View Experiments

1. Open http://localhost:5001
2. Click **"wine-quality-prediction"** experiment
3. See all 4 model runs with metrics

### Compare Models

| Run Name | Accuracy | Duration |
|----------|----------|----------|
| RandomForest | ~0.906 | ~13s |
| GradientBoosting | ~0.903 | ~9s |
| SVM | ~0.872 | ~8s |
| LogisticRegression | ~0.866 | ~12s |

### Register Model

1. Click best run (e.g., RandomForest)
2. Go to **Artifacts** tab
3. Click **"Register Model"**
4. Name: `wine-model`
5. Go to **Model Registry** → `wine-model`
6. Click **"Add alias"** → `production`

---

## 🪣 MinIO Object Storage

### Console Access

- **URL:** http://localhost:9001
- **Login:** minioadmin / minioadmin123

### Bucket Structure

```
mlflow/
└── 1/                                    # Experiment ID
    └── models/
        └── m-xxxxxxxxxxxxxxxx/          # Model ID
            └── artifacts/
                ├── MLmodel               # Metadata
                ├── model.skops           # Model file
                ├── conda.yaml            # Environment
                └── requirements.txt      # Dependencies
```

---

## 🎯 Model Registry

### How It Works

```
Training
    │
    ▼
MLflow Log Model ──▶ Registry ──▶ Alias: "production"
    │                              │
    ▼                              ▼
MinIO Storage              API Loads from Alias
```

### API Loading

```python
# main.py
model = mlflow.sklearn.load_model("models:/wine-model@production")
```

**Benefits:**
- No hardcoded Run IDs
- Easy model promotion
- Rollback support
- Auto-retry on startup (waits up to 2.5 min for the alias)

---

## 🌐 API Usage

### Health Check

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "Wine Quality Prediction API is live",
  "model": "wine-model@production",
  "features": [
    "monitoring_logging",
    "performance_metrics",
    "input_validation",
    "batch_prediction",
    "drift_detection"
  ]
}
```

### Get Metrics

```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "total_predictions": 42,
  "good_quality_count": 30,
  "bad_quality_count": 12,
  "good_percentage": 71.43,
  "avg_prediction_time_ms": 2.15,
  "drift_status": {
    "drift_detected": false,
    "alcohol_drift_score": 0.85,
    "fixed_acidity_drift_score": 0.32,
    "recent_alcohol_mean": 10.6,
    "recent_fixed_acidity_mean": 8.1
  }
}
```

### Predict Single Wine

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "alcohol": 9.4
  }'
```

**Response:**
```json
{
  "good_quality": false,
  "prediction_time_ms": 1.234,
  "drift_warning": false,
  "drift_details": null
}
```

### Predict Good Quality Wine

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.3,
    "citric_acid": 0.3,
    "residual_sugar": 2.0,
    "alcohol": 12.0
  }'
```

**Response:**
```json
{
  "good_quality": true,
  "prediction_time_ms": 1.145,
  "drift_warning": false,
  "drift_details": null
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "wines": [
      {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "alcohol": 9.4
      },
      {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.3,
        "citric_acid": 0.3,
        "residual_sugar": 2.0,
        "alcohol": 12.0
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {"good_quality": false},
    {"good_quality": true}
  ],
  "total": 2,
  "time_ms": 2.341
}
```

---

## 📸 Screenshots

### MLflow Experiments

![MLflow Experiments](screenshots/mlflow.png)

*MLflow UI showing all 4 model runs with accuracy metrics*

### MLflow Run Details

![MLflow Run](screenshots/mlflow2.png)

*Detailed view of a single run showing parameters and metrics*

### MLflow Model Registry

![Model Registry](screenshots/mlflow3.png)

*Registered models with version and alias management*

### MinIO Console

![MinIO Console](screenshots/minlo.png)

*MinIO web console showing MLflow artifacts bucket*

### MinIO Artifacts

![MinIO Artifacts](screenshots/minilo3.png)

*Model artifacts stored in MinIO (MLmodel, model.skops, etc.)*

### FastAPI Prediction

![FastAPI Docs](screenshots/fastapi.png)

---

## 🙏 Acknowledgments

- [MLflow](https://mlflow.org/) - Machine Learning Lifecycle Platform
- [MinIO](https://min.io/) - High Performance Object Storage
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Web Framework
- [Scikit-learn](https://scikit-learn.org/) - Machine Learning Library

---

## 📧 Contact

For questions or contributions, open an issue on GitHub.

**Happy MLOps! 🚀**
