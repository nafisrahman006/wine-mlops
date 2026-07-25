"""
Shared pytest fixtures.

main.py talks to a live MLflow registry at import time (it calls
load_model_from_registry() as a module-level statement). CI has no
MLflow/Postgres/MinIO running, so we mock the MLflow client and the
sklearn model loader before importing main.py, and stub the required
env vars main.py checks for.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# main.py lives at the project root, one level above tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Required env vars main.py validates on import
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-key")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret")


class _DummyModel:
    """Predicts good_quality=True when alcohol >= 11, else False."""

    def predict(self, X):
        import numpy as np
        alcohol_col = 4  # matches feature order in main.py
        return np.array([1 if row[alcohol_col] >= 11 else 0 for row in X])


@pytest.fixture(scope="session")
def app_module():
    """Import main.py with MLflow calls mocked out."""
    fake_model_version = MagicMock()
    fake_model_version.name = "wine-model"
    fake_model_version.version = "1"
    fake_model_version.run_id = "fake-run-id"

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.tracking.MlflowClient") as mock_client_cls, \
         patch("mlflow.sklearn.load_model", return_value=_DummyModel()):

        mock_client_cls.return_value.get_model_version_by_alias.return_value = fake_model_version

        # Ensure a clean import even if a prior test imported main already
        sys.modules.pop("main", None)
        import main as main_module

        yield main_module

        sys.modules.pop("main", None)


@pytest.fixture()
def client(app_module):
    return TestClient(app_module.app)
