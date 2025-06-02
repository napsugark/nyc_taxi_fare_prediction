import os
import time
from urllib.parse import urlparse
import pytest
import mlflow
import pandas as pd
import uuid
from pathlib import Path
import shutil
from src.constants import DATASET_DOWNLOAD_PATH, EXPERIMENT_NAME, VERSION, MODEL_TYPE, ALPHA
from src.utils.logging_config import logger
from src.train_model import train_and_log_model
from src.preprocessing import load_data, preprocess_data, prepare_features, split_and_preprocess



@pytest.fixture(scope='session', autouse=True)
def prepare_sample():
    sample_data_path = f"{DATASET_DOWNLOAD_PATH}/dataset_sample.csv"
    if not os.path.exists(sample_data_path):
        print("Generating sample dataset...")
        df_large = pd.read_csv('data/01_raw/train.csv')
        sampled_df = df_large[df_large['pickup_datetime'] >= '2014-01-01'].sample(n=10000, random_state=42)
        sampled_df.to_csv(sample_data_path, index=False)
    else:
        print("Sample dataset already exists.")


@pytest.fixture(scope="module")
def unique_experiment_name():
    return f"{EXPERIMENT_NAME}_{uuid.uuid4()}"

@pytest.fixture(scope="module", autouse=True)
def cleanup_before_and_after(unique_experiment_name):
    """Fixture to clean up MLflow experiment and feature importance directory before and after tests.
    """
    # Cleanup before
    experiment = mlflow.get_experiment_by_name(unique_experiment_name)
    if experiment:
        mlflow.delete_experiment(experiment.experiment_id)

    yield


def test_full_training_pipeline(unique_experiment_name):
    logger.info("Starting full integration test of training pipeline")

    # Load & preprocess
    data_path = Path(DATASET_DOWNLOAD_PATH) / "dataset_sample.csv"
    assert data_path.exists(), f"{data_path} does not exist"

    df = load_data(data_path)
    assert not df.empty, "Loaded dataset is empty"

    df = preprocess_data(df)
    X, y, numeric, boolean, cyclic, categorical = prepare_features(df)
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(
        X, y, numeric, boolean, cyclic, categorical
    )

    train_and_log_model(
        X_train, X_test, y_train, y_test,
        preprocessor=preprocessor,
        version=VERSION,
        experiment_name=unique_experiment_name,
        model_type=MODEL_TYPE,
        alpha=ALPHA
    )

    # Check MLflow run exists
    experiment = mlflow.get_experiment_by_name(unique_experiment_name)
    assert experiment is not None, f"Experiment '{unique_experiment_name}' not found"
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert not runs.empty, "No MLflow runs found"

    # time.sleep(10)  # Wait for MLflow to log everything
   
    # runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    last_run = runs.iloc[0]
    logger.info(f"Metrics columns found: {[col for col in last_run.index if col.startswith('metrics.')]}")
    # logger.info(f"MLflow run columns:\n{list(last_run.index)}")
    # logger.info(f"MLflow run content:\n{last_run.to_dict()}")


    for metric in ["mse", "rmse", "mae", "r2"]:
        assert f"metrics.{metric}" in last_run, f"metrics.{metric} not found in MLflow run"

    artifact_uri = last_run["artifact_uri"]
    parsed_uri = urlparse(artifact_uri)
    artifact_path = Path(parsed_uri.path.lstrip("/"))  # remove leading slash if present
    model_artifact = artifact_path / "model" / "MLmodel"
    assert model_artifact.exists(), f"MLmodel artifact not found at {model_artifact}"

    # Check coefficient CSV exists
    coefs_dir = Path("data/feature_importance")
    coef_files = list(coefs_dir.glob(f"feature_importance_{VERSION}_f*.csv"))
    assert coef_files, "No feature importance CSV file found"

    assert "params.features" in last_run.to_dict(), "Feature names were not logged as a parameter"
