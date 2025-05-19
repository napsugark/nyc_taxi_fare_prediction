# preprocessing.py

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cyclic (sine and cosine) transformations for time-based features.

    Args:
        df (pd.DataFrame): Input DataFrame containing the following columns:
            - 'pickup_datetime_hour'
            - 'pickup_datetime_dayofweek'
            - 'pickup_datetime_month'
            - 'pickup_datetime_dayofyear'

    Returns:
        pd.DataFrame: DataFrame with added cyclic features and original time-based columns removed.
    
    Raises:
        KeyError: If any required column is missing.
        RuntimeError: For any other processing error.
    """
    try:
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df['pickup_datetime_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['pickup_datetime_hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['pickup_datetime_dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['pickup_datetime_dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['pickup_datetime_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['pickup_datetime_month'] / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['pickup_datetime_dayofyear'] / 366)
        df['doy_cos'] = np.cos(2 * np.pi * df['pickup_datetime_dayofyear'] / 366)

        return df.drop(['pickup_datetime_hour', 'pickup_datetime_dayofweek',
                        'pickup_datetime_month', 'pickup_datetime_dayofyear'], axis=1)
    except KeyError as e:
        raise KeyError(f"Missing required column in DataFrame: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to add cyclic features: {e}")

def get_feature_lists() -> tuple[list[str], list[str], list[str]]:
    """
    Returns lists of feature names for preprocessing.

    Returns:
        tuple[list[str], list[str], list[str]]: A tuple containing:
            - numeric: List of numeric feature names.
            - boolean: List of boolean feature names.
            - cyclic: List of cyclic (transformed time) feature names.
    """
    numeric = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
               'dropoff_latitude', 'trip_distance', 'passenger_count',
               'pickup_datetime_year']
    boolean = ['pickup_datetime_is_weekend', 'is_late_night', 'is_night',
               'is_early_morning', 'is_rush_hour']
    cyclic = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
              'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
    return numeric, boolean, cyclic

def build_preprocessor(numeric: list[str], boolean: list[str], cyclic: list[str]) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for preprocessing features.

    Args:
        numeric (list[str]): List of numeric feature names to be standardized.
        boolean (list[str]): List of boolean feature names to be passed through.
        cyclic (list[str]): List of cyclic feature names to be passed through.

    Returns:
        ColumnTransformer: A ColumnTransformer that applies preprocessing steps.

    Raises:
        RuntimeError: If an error occurs while creating the transformer.
    """
    try:
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric),
            ('bool', 'passthrough', boolean),
            ('cyclic', 'passthrough', cyclic),
        ])
        return preprocessor
    except Exception as e:
        raise RuntimeError(f"Failed to build preprocessor: {e}")