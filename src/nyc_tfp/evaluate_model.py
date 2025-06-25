import joblib
import pandas as pd
import mlflow
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from src.utils.logging_config import logger
from src.nyc_tfp.config import TRAIN_TEST_SPLIT_DIR

# Initialize DagsHub for MLflow tracking
import dagshub
dagshub.init(repo_owner='napsugar.kelemen',
             repo_name='nyc_taxi_fare_prediction',
             mlflow=True)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred)
    }

    logger.info("Evaluation metrics:\n" + json.dumps(metrics, indent=2))
    return metrics

def main():
    logger.info("Starting evaluation...")

    model = joblib.load("models/model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    X_test = pd.read_csv(f"{TRAIN_TEST_SPLIT_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{TRAIN_TEST_SPLIT_DIR}/y_test.csv")

    X_test = pd.DataFrame(
    preprocessor.transform(X_test),
    columns=preprocessor.get_feature_names_out()
)

    run_id_path = Path("evaluation/run_id.txt")
    if not run_id_path.exists():
        raise FileNotFoundError("Run ID file not found at 'evaluation/run_id.txt'.")

    run_id = run_id_path.read_text().strip()

    with mlflow.start_run(run_id=run_id):
        metrics = evaluate(model, X_test, y_test)

        for key, value in metrics.items():
            mlflow.log_metric(f"test_{key}", value)

        output_file = Path("evaluation/metrics.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation complete. Metrics saved to: {output_file}")

if __name__ == "__main__":
    main()
