import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from constants import SELECTED_COLUMNS, NUMERIC_FEATURES, BOOLEAN_FEATURES, CYCLIC_FEATURES, CATEGORICAL_FEATURES, TARGET_FEATURE


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

def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Returns lists of feature names for preprocessing, after checking they exist in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check against.

    Returns:
        tuple[list[str], list[str], list[str], list[str]]: A tuple containing:
            - numeric: List of numeric feature names.
            - boolean: List of boolean feature names.
            - cyclic: List of cyclic (transformed time) feature names.
            - categorical: List of categorical feature names (currently empty).
    """
    try:
        numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
        boolean = [col for col in BOOLEAN_FEATURES if col in df.columns]
        cyclic = [col for col in CYCLIC_FEATURES if col in df.columns]
        categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
        return numeric, boolean, cyclic, categorical
    except Exception as e:
        raise RuntimeError(f"Failed to get feature lists: {e}")

def build_preprocessor(numeric: list[str], boolean: list[str], cyclic: list[str], categorical: list[str]) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for preprocessing features.

    Args:
        numeric (list[str]): List of numeric feature names to be standardized.
        boolean (list[str]): List of boolean feature names to be passed through.
        cyclic (list[str]): List of cyclic feature names to be passed through.
        categorical (list[str]): List of categorical feature names to be one-hot encoded.

    Returns:
        ColumnTransformer: A ColumnTransformer that applies preprocessing steps.

    Raises:
        RuntimeError: If an error occurs while creating the transformer.
    """
    try:
        transformers = []

        if numeric:
            transformers.append(('num', StandardScaler(), numeric))
        if boolean:
            transformers.append(('bool', 'passthrough', boolean))
        if cyclic:
            transformers.append(('cyclic', 'passthrough', cyclic))
        if categorical:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical))

        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor

    except Exception as e:
        raise RuntimeError(f"Failed to build preprocessor: {e}")
    
def load_data(file_path: Path) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file and applies preprocessing.
    Args:
        file_path (Path): Path to the CSV file.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    try:
        df = pd.read_csv(file_path)
        df = add_cyclic_features(df)
        df = select_columns(df=df, columns=SELECTED_COLUMNS)
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        raise RuntimeError(f"Empty data error: {e}")

def prepare_features(df: pd.DataFrame):
    """
    Prepares features for modeling by selecting relevant columns and splitting into X and y.
    """
    try:
        numeric, boolean, cyclic, categorical = get_feature_lists(df)
        all_features = numeric + boolean + cyclic + categorical
        X = df[all_features]
        y = df[TARGET_FEATURE]
        return X, y, numeric, boolean, cyclic, categorical
    except Exception as e:
        raise RuntimeError(f"Failed to prepare features: {e}")

def split_and_preprocess(X, y, numeric, boolean, cyclic, categorical, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets, applies preprocessing, and returns the processed data.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable.
        numeric (list[str]): List of numeric feature names to be standardized.
        boolean (list[str]): List of boolean feature names to be passed through.
        cyclic (list[str]): List of cyclic feature names to be passed through.
        categorical (list[str]): List of categorical feature names to be one-hot encoded.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
       
    Returns:
        tuple: A tuple containing:
            - X_train_prep: Preprocessed training features.
            - X_test_prep: Preprocessed testing features.
            - y_train: Training target variable.
            - y_test: Testing target variable.
            - preprocessor: The fitted ColumnTransformer used for preprocessing.
    
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        preprocessor = build_preprocessor(numeric, boolean, cyclic, categorical)
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)
        return X_train_prep, X_test_prep, y_train, y_test, preprocessor
    except Exception as e:
        raise RuntimeError(f"Failed to split and preprocess data: {e}")
    
def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Selects specific columns from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str]): List of column names to select.

    Returns:
        pd.DataFrame: DataFrame with only the selected columns.
    
    Raises:
        KeyError: If any of the specified columns are not found in the DataFrame.
        RuntimeError: For any other processing error.
    """
    try:
        return df[columns]
    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to select columns: {e}")