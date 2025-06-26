import os
import joblib
import pandas as pd
from pathlib import Path

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso
from src.utils.helpers import get_dvc_md5, save_data, save_train_test_split
from src.utils.logging_config import logger


from src.nyc_tfp.config import FINAL_DATASET_PATH, MODEL_PATH, PREPROCESSED_DATASET_PATH, PREPROCESSOR_PATH, REGISTERED_BY, RUN_TYPE, TRAIN_TEST_SPLIT_DIR, VERSION, EXPERIMENT_NAME, MODEL_TYPE, ALPHA
from src.nyc_tfp.preprocess import load_data, prepare_features, preprocess_columns, split_data

# Initialize DagsHub for MLflow tracking
import dagshub
dagshub.init(repo_owner='napsugar.kelemen',
             repo_name='nyc_taxi_fare_prediction',
             mlflow=True)

def split_and_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Splits the dataset into training and testing sets, applies preprocessing, and returns the processed data.
    
    This function is a wrapper around the split_and_preprocess function to handle the entire preprocessing pipeline.
    
    Returns:
        tuple: A tuple containing:
            - X_train_prep: Preprocessed training features.
            - X_test_prep: Preprocessed testing features.
            - y_train: Training target variable.
            - y_test: Testing target variable.
            - preprocessor: The fitted ColumnTransformer used for preprocessing.
    """
    try:
        logger.info("Preprocessing data for splitting and transformation...")
        X, y, numeric, boolean, cyclic, categorical, final_df = prepare_features(df)
        save_data(final_df, Path(FINAL_DATASET_PATH))
        logger.info(f"Final dataset saved to {FINAL_DATASET_PATH}")
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        save_train_test_split(X_train, X_test, y_train, y_test, output_dir=TRAIN_TEST_SPLIT_DIR)
        logger.info(f"Train-test split saved to {TRAIN_TEST_SPLIT_DIR}")
        logger.info("Preprocessing training and test features...")
        X_train_prep, X_test_prep, preprocessor = preprocess_columns(
            X_train, X_test, numeric, boolean, cyclic, categorical
        )
        return X_train_prep, X_test_prep, y_train, y_test, preprocessor
    except Exception as e:
        logger.error(f"Failed to split and transform data: {e}")
        raise RuntimeError(f"Failed to split and transform data: {e}")


def train_and_log_model(X_train, X_test,
 y_train, y_test, preprocessor, version=VERSION, experiment_name=EXPERIMENT_NAME,
 model_type=MODEL_TYPE, alpha=ALPHA
):
    if model_type == 'Lasso':
        logger.info(f"Training Lasso model with alpha={alpha}...")
        model = Lasso(alpha=alpha)
    elif model_type == 'LinearRegression':
        logger.info("Training Linear Regression model...")
        model = LinearRegression()
    else:
        raise ValueError(f"model_type must be 'LinearRegression' or 'Lasso', not {model_type}")
    
    mlflow.sklearn.autolog(log_input_examples=True, silent=True)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:

        logger.info("Fitting the model...")
        model.fit(X_train, y_train)

        run_id = run.info.run_id
        Path("evaluation").mkdir(parents=True, exist_ok=True) 
        with open("evaluation/run_id.txt", "w") as f:
            f.write(run_id) 

        mlflow.set_tag("run_type", RUN_TYPE)

        logger.info(f"Logging {RUN_TYPE=} with {run_id=} model parameters and metrics to MLflow... ")
        feature_names = preprocessor.get_feature_names_out()
        num_features = len(feature_names)

        mlflow.log_param("features", ", ".join(feature_names))
        mlflow.log_param("num_features", num_features)
        mlflow.log_param("Final_dataset_version_md5", get_dvc_md5("data/03_final/final_df.csv.dvc"))

        if model_type == 'Lasso':
            mlflow.log_param("alpha", alpha)

        mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(X_train), name="X_train"), context="training")
        mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_train, columns=["target"]), name="y_train"), context="training")
        mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(X_test), name="X_test"), context="validation")
        mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_test, columns=["target"]), name="y_test"), context="validation")

        logger.info("Saving model coefficients and feature importance...")
        coefs = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        coefs_file = Path(f"data/feature_importance/feature_importance_{version}_f{num_features}.csv")
        coefs_file.parent.mkdir(parents=True, exist_ok=True)
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
        model_path = Path(MODEL_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        preprocessor_path = Path(PREPROCESSOR_PATH)
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(str(preprocessor_path))

        logger.info(f"Model and preprocessor saved and logged to MLflow.")

        # Register the model manually
        model_uri = f"runs:/{run_id}/model"
        model_name = f"{model_type}_Model"
        tags = {
            "model_type": model_type,
            "registered_by": REGISTERED_BY,
            "run_type": RUN_TYPE,
        }
        logger.info(f"Registering model under name: {model_name}")
        result = mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias=RUN_TYPE,  # prod or exp or repro
            version=result.version
        )

def main():
    logger.info("Starting the training process...")
    df = load_data(PREPROCESSED_DATASET_PATH)
    X_train_prep, X_test_prep, y_train, y_test, preprocessor = split_and_transform(df)
    
    train_and_log_model(X_train_prep, X_test_prep, y_train, y_test, preprocessor, version=VERSION, experiment_name=EXPERIMENT_NAME, model_type=MODEL_TYPE,alpha=ALPHA)

if __name__ == "__main__":
    main()