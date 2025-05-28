import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import SELECTED_COLUMNS, NUMERIC_FEATURES, BOOLEAN_FEATURES, CYCLIC_FEATURES, CATEGORICAL_FEATURES, TARGET_FEATURE, DATASET_DOWNLOAD_PATH

import pandas as pd
from pandas import DataFrame
from src.utils.load_dataset import download_kaggle_competition_data
from src.utils.logging_config import logger


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    Args:
        file_path (Path): Path to the CSV file.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    try:
        download_kaggle_competition_data(competition_name="new-york-city-taxi-fare-prediction", dataset_download_path=DATASET_DOWNLOAD_PATH)
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, parse_dates=['pickup_datetime'])
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        raise RuntimeError(f"Empty data error: {e}")

def split_date_cols(df: DataFrame, date_col: str) -> DataFrame:
    """Adds date-related features to a DataFrame from a datetime column.

    Converts the specified column to datetime format and extracts various
    time-based features such as day of year, month, hour, and flags for
    specific time intervals (e.g. night, rush hour).

    Args:
        df (pd.DataFrame): The input DataFrame containing the date column.
        date_col (str): The name of the column with datetime values.

    Returns:
        pd.DataFrame: The input DataFrame with additional columns for:
           
    Raises:
        Exception: If any error occurs during processing, it is caught and
        an error message is printed. The original DataFrame is returned.
    """
    try:
        logger.info(f"Splitting date column: {date_col}")
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        df[f"{date_col}_dayofyear"] = df[date_col].dt.dayofyear
        df[f"{date_col}_month"] = df[date_col].dt.month
        df[f"{date_col}_year"] = df[date_col].dt.year
        df[f"{date_col}_hour"] = df[date_col].dt.hour
        df[f"{date_col}_dayofweek"] = df[date_col].dt.dayofweek
        df[f"{date_col}_is_weekend"] = df[date_col].dt.dayofweek >= 5

        df[f"{date_col}_is_late_night"] = df[f"{date_col}_hour"].between(22, 23)
        df[f"{date_col}_is_night"] = df[f"{date_col}_hour"].between(0, 3)
        df[f"{date_col}_is_early_morning"] = df[f"{date_col}_hour"].between(4, 6)
        df[f"{date_col}_is_rush_hour"] = df[f"{date_col}_hour"].between(7, 8) | df[f"{date_col}_hour"].between(16, 18)

        return df

    except Exception as e:
        raise RuntimeError(f"Error while processing date column '{date_col}': {e}")
    
def calculate_haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two points on the Earth specified by latitude and longitude.
    Uses the Haversine formula to compute the distance in kilometers.
    Args:
        lat1 (float or pd.Series): Latitude of the first point.
        lon1 (float or pd.Series): Longitude of the first point.
        lat2 (float or pd.Series): Latitude of the second point.
        lon2 (float or pd.Series): Longitude of the second point.
    Returns:
        float or pd.Series: Haversine distance in kilometers.
    Raises:
        ValueError: If the input coordinates are not valid.
    """
    try:
        R = 6371  # Radius of the Earth in kilometers
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c  # Distance in kilometers
    except Exception as e:
        raise ValueError(f"Invalid coordinates for Haversine calculation: {e}")


def add_haversine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'trip_distance' column to the DataFrame using the Haversine formula.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'pickup_latitude', 'pickup_longitude','dropoff_latitude', and 'dropoff_longitude' columns.

    Returns:
        pd.DataFrame: DataFrame with an additional 'trip_distance' column.
    
    Raises:
        KeyError: If any required column is missing.
        RuntimeError: For any other processing error.
    """
    try:
        logger.info("Calculating trip distance using Haversine formula")
        df = df.copy()
        df['trip_distance'] = calculate_haversine(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        # Further cleaning based on trip distance
        # Trip distance should be between 0.5 and 50 km
        # 0.5 km is the minimum distance for a trip
        df = df[(df['trip_distance'] < 50) & (df['trip_distance'] > 0.5)]
        return df
    except KeyError as e:
        raise KeyError(f"Missing required column in DataFrame: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to calculate trip distance: {e}")

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
        logger.info("Adding cyclic features to DataFrame")
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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by removing rows with NaN values and incorrect data.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with NaN values and duplicates removed.
        
    Raises:
        RuntimeError: If any error occurs during cleaning.
    """
    try:
        logger.info("Cleaning data...")

        ### The amount should be positive
        df = df[df['fare_amount'] > 0]
        df = df[df['fare_amount'] < 1000]

        # Coordinates - shouldn't be null, zero and should be a NYC location - 
        df = df.dropna(subset=['dropoff_longitude', 'dropoff_latitude'], how='any')
        df = df.dropna(subset=['pickup_longitude', 'pickup_latitude'], how='any')
       


        df = df[(df['pickup_longitude'] != 0) & (df['dropoff_longitude'] != 0) & (df['pickup_latitude'] != 0) & (df['dropoff_latitude'] != 0)]

        # The longitude should be between -180 and 180
        # The latitude should be between -90 and 90
        df = df[df['pickup_longitude'].between(-180, 180) & df['dropoff_longitude'].between(-180, 180) & df['pickup_latitude'].between(-90,90) & df['dropoff_latitude'].between(-90, 90)]
        # The values of the longitude and latitude are not in the range of New York City

        df = df[df['pickup_longitude'].between(-75, -70) & df['dropoff_longitude'].between(-75, -70) & df['pickup_latitude'].between(35,45) & df['dropoff_latitude'].between(35, 45)]

        # The passenger count should be between 1 and 6
        df = df[(df['passenger_count'] <= 6) & (df['passenger_count'] > 0)]
        logger.info(f"Data cleaned successfully, shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to clean data: {e}")
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by applying date splitting, cyclic feature addition, and column selection.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with selected columns.
        
    Raises:
        RuntimeError: If any error occurs during preprocessing.
    """
    try:
        logger.info("Preprocessing data...")
        df = clean_data(df)  
        df = split_date_cols(df, date_col='pickup_datetime')
        df = set_date_range(df, start_year=2013, end_year=2015)
        df = add_haversine(df)
        df = add_cyclic_features(df)
        df = select_columns(df=df, columns=SELECTED_COLUMNS)
        logger.info(f"Data preprocessed successfully, shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess data: {e}")
    
def set_date_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows where 'pickup_datetime_year'
    is between start_year and end_year (inclusive).

    Args:
        df (pd.DataFrame): Input DataFrame that includes a 'pickup_datetime_year' column.
        start_year (int): Start of the year range (inclusive).
        end_year (int): End of the year range (inclusive).

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows within the specified year range.

    Raises:
        RuntimeError: If filtering fails.
    """
    try:
        logger.info(f"Reducing data to years between {start_year} and {end_year}...")
        df = df[df['pickup_datetime_year'].between(start_year, end_year)]
        logger.info(f"Date ranged set successfully, shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to reduce data: {e}")
  

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
        logger.info("Splitting and preprocessing data...")
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