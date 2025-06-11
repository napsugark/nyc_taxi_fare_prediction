import os
import subprocess
import pandas as pd
from pathlib import Path

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.data
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from src import preprocess
from src.utils.helpers import get_dvc_md5, save_data
from src.visualize_results import plot_and_save_grouped_bar_mlruns_metrics, plot_and_save_best_models_summary
from src.utils.logging_config import logger
from datetime import datetime


from src.config import DATASET_DOWNLOAD_PATH, FINAL_DATASET_PATH, PREPROCESSED_DATASET_PATH, VERSION, EXPERIMENT_NAME, MODEL_TYPE, ALPHA
from src.preprocess import load_data, prepare_features, preprocess_columns,  split_data

def train_and_log_model(X_train, X_test, y_train, y_test, preprocessor, version=VERSION, experiment_name=EXPERIMENT_NAME, model_type=MODEL_TYPE, alpha=ALPHA):
    if model_type == 'Lasso':
        logger.info(f"Training Lasso model with alpha={alpha}...")
        model = Lasso(alpha=alpha)
    elif model_type == 'LinearRegression':
        logger.info("Training Linear Regression model...")
        model = LinearRegression()
    else:
        raise ValueError(f"model_type must be 'linear' or 'lasso', not {model_type}")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_type}_{version}"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Training results: MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        logger.info(f"Logging model parameters and metrics to MLflow...")
        feature_names = preprocessor.get_feature_names_out()
        num_features = len(feature_names)

        mlflow.log_param("features", ", ".join(feature_names))
        mlflow.log_param("num_features", num_features)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.log_param("Final_dataset_version_md5", get_dvc_md5("data/03_final/final_df.csv.dvc"))

        if model_type == 'Lasso':
            mlflow.log_param("alpha", alpha)

        logger.info("Saving model coefficients and feature importance...")
        coefs = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        coefs_file = Path(f"data/feature_importance/feature_importance_{version}_f{num_features}.csv")
        coefs.to_csv(coefs_file, index=False)
        mlflow.log_artifact(str(coefs_file))

        logger.info(f"Model coefficients saved to {coefs_file}")
    
        logger.info("Logging model to MLflow...")
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        logger.info("Model logged successfully.")

def main():
    logger.info("Starting the training process...")
    df = load_data(PREPROCESSED_DATASET_PATH)
    X, y, numeric, boolean, cyclic, categorical, final_df = prepare_features(df)

    save_data(final_df, Path(FINAL_DATASET_PATH))
    subprocess.run(["dvc", "add", str(FINAL_DATASET_PATH)], check=True)

    X_train, X_test, y_train, y_test = split_data(X, y)
   
    X_train_prep, X_test_prep, preprocessor = preprocess_columns(
        X_train, X_test, numeric, boolean, cyclic, categorical
    )

    train_and_log_model(X_train_prep, X_test_prep, y_train, y_test, preprocessor, version=VERSION, experiment_name=EXPERIMENT_NAME, model_type=MODEL_TYPE,alpha=ALPHA)

    # plot_and_save_grouped_bar_mlruns_metrics(experiment_name=EXPERIMENT_NAME)
    # plot_and_save_best_models_summary(experiment_name=EXPERIMENT_NAME)
if __name__ == "__main__":
    main()