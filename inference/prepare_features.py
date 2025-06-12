import pandas as pd
from src.config import PREDICTION_COLUMNS
from src.preprocess import (
    add_cyclic_features,
    add_haversine,
    select_columns,
    split_date_cols
)

def prepare_features_for_prediction(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the features for the model by applying necessary transformations:
    - Splits datetime columns.
    - Computes haversine distance between pickup and dropoff.
    - Adds cyclic time features.
    - Selects only the columns needed for prediction.

    Args:
        input_df (pd.DataFrame): Raw input data containing at least the required fields.

    Returns:
        pd.DataFrame: Transformed DataFrame ready for model prediction.

    Raises:
        ValueError: If required transformations fail or input data is invalid.
    """
    try:
        df = input_df.copy()
        df = split_date_cols(df, date_col='pickup_datetime')
        df = add_haversine(df)
        df = add_cyclic_features(df)
        df = select_columns(df=df, columns=PREDICTION_COLUMNS)
        return df

    except Exception as e:
        raise ValueError(f"Error during feature preparation: {e}")

