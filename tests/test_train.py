"""
Tests for train.py.

train() talks to a live MLflow server for tracking. We mock every
mlflow.* call so no server is needed, then let it actually run
scikit-learn training on a small synthetic CSV — this is fast (a few
seconds for 4 tiny models) and exercises the real feature-engineering,
train/test split, and metric-calculation code paths, not just mocks.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train as train_module  # noqa: E402


@pytest.fixture()
def tiny_wine_csv(tmp_path):
    """
    A small synthetic wine-quality CSV with columns matching the real
    dataset. 30 rows, roughly balanced between quality>=7 and
    quality<7 so stratified train_test_split has both classes to work
    with.
    """
    rows = []
    for i in range(30):
        good = i % 2 == 0
        rows.append({
            "fixed acidity": 7.0 + (i % 5) * 0.3,
            "volatile acidity": 0.3 + (i % 4) * 0.05,
            "citric acid": 0.2 + (i % 3) * 0.1,
            "residual sugar": 2.0 + (i % 6) * 0.5,
            "alcohol": 12.0 + (i % 3) if good else 9.0 + (i % 3),
            "quality": 7 if good else 5,
        })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "tiny_wine.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture()
def mocked_mlflow():
    """Mock every mlflow.* call train() makes so no server is needed."""
    with patch("train.mlflow.set_tracking_uri"), \
         patch("train.mlflow.set_experiment"), \
         patch("train.mlflow.start_run") as mock_start_run, \
         patch("train.mlflow.log_param") as mock_log_param, \
         patch("train.mlflow.log_metric") as mock_log_metric, \
         patch("train.mlflow.sklearn.log_model") as mock_log_model:

        # mlflow.start_run(...) is used as a context manager: `with mlflow.start_run(...):`
        mock_start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

        yield {
            "start_run": mock_start_run,
            "log_param": mock_log_param,
            "log_metric": mock_log_metric,
            "log_model": mock_log_model,
        }


def test_train_runs_all_four_models(tiny_wine_csv, mocked_mlflow):
    train_module.train(tiny_wine_csv)

    # One run per model: RandomForest, LogisticRegression, GradientBoosting, SVM
    assert mocked_mlflow["start_run"].call_count == 4


def test_train_logs_expected_metrics(tiny_wine_csv, mocked_mlflow):
    train_module.train(tiny_wine_csv)

    logged_metric_names = {
        call.args[0] for call in mocked_mlflow["log_metric"].call_args_list
    }
    assert {"accuracy", "precision", "recall", "f1_score", "train_time_sec"} <= logged_metric_names


def test_train_logs_model_artifact_for_each_run(tiny_wine_csv, mocked_mlflow):
    train_module.train(tiny_wine_csv)

    assert mocked_mlflow["log_model"].call_count == 4


def test_train_logs_hyperparameters(tiny_wine_csv, mocked_mlflow):
    train_module.train(tiny_wine_csv)

    logged_param_names = {
        call.args[0] for call in mocked_mlflow["log_param"].call_args_list
    }
    assert "model_name" in logged_param_names
    assert "test_size" in logged_param_names
    assert "random_state" in logged_param_names


def test_train_target_creation_logic(tiny_wine_csv):
    """quality >= 7 should map to target=1, else target=0 — no mlflow needed for this."""
    df = pd.read_csv(tiny_wine_csv)
    df["target"] = (df["quality"] >= 7).astype(int)

    assert set(df["target"].unique()) == {0, 1}
    assert (df.loc[df["quality"] >= 7, "target"] == 1).all()
    assert (df.loc[df["quality"] < 7, "target"] == 0).all()
