import pandas as pd
import os
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature

from preprocessing import add_cyclic_features, get_feature_lists, build_preprocessor

def main():
    df = pd.read_csv('data/02_processed/df_cleaned.csv')

    df = add_cyclic_features(df)
    df.drop(columns=['id', 'pickup_datetime'], inplace=True)

    numeric, boolean, cyclic = get_feature_lists()
    all_features = numeric + boolean + cyclic

    preprocessor = build_preprocessor(numeric, boolean, cyclic)

    X = df[all_features]
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    mlflow.set_experiment("LinearRegression_NYC_Taxi")

    with mlflow.start_run(run_name="LinearRegression_v4"):
        model = LinearRegression()
        model.fit(X_train_preprocessed, y_train)
        y_pred = model.predict(X_test_preprocessed)

        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        mlflow.log_param("features", ", ".join(X_train.columns))
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        coefs = pd.DataFrame({"feature": X_train.columns, "coefficient": model.coef_})
        coefs_file = "data/feature_importance.csv"
        coefs.to_csv(coefs_file, index=False)
        mlflow.log_artifact(coefs_file)

        signature = infer_signature(X_train_preprocessed, model.predict(X_train_preprocessed))
        input_example = X_train_preprocessed[:5]

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

if __name__ == "__main__":
    main()
