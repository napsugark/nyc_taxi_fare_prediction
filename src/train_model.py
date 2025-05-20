import pandas as pd
from pathlib import Path

import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from mlflow.models.signature import infer_signature


from constants import VERSION, EXPERIMENT_NAME, DATA_PATH
from preprocessing import load_data, prepare_features, split_and_preprocess

def train_and_log_model(X_train, X_test, y_train, y_test, preprocessor, version=VERSION, experiment_name=EXPERIMENT_NAME):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"LinearRegression_{version}"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        feature_names = preprocessor.get_feature_names_out()
        len_features = len(feature_names)
        mlflow.log_param("features", ", ".join(feature_names))
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        coefs = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        coefs_file = Path(f"data/feature_importance_{version}_f{len_features}.csv")
        coefs.to_csv(coefs_file, index=False)
        mlflow.log_artifact(str(coefs_file))

    
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

def main():
    data_path = Path(DATA_PATH)
    df = load_data(data_path)
    X, y, numeric, boolean, cyclic, categorical = prepare_features(df)
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(X, y, numeric, boolean, cyclic, categorical)
    train_and_log_model(X_train, X_test, y_train, y_test, preprocessor)

if __name__ == "__main__":
    main()