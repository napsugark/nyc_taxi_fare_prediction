import pandas as pd
import pytest
from src.nyc_tfp.preprocess import split_date_cols, calculate_haversine, add_cyclic_features
from math import isclose

def test_split_date_cols_basic(df_timestamps):
    """Test basic functionality of split_date_cols."""
    result = split_date_cols(df_timestamps.copy(), date_col="timestamp")

    assert all(col in result.columns for col in [
        "timestamp_dayofyear", "timestamp_month", "timestamp_year",
        "timestamp_hour", "timestamp_dayofweek", "timestamp_is_weekend",
        "timestamp_is_late_night", "timestamp_is_night",
        "timestamp_is_early_morning", "timestamp_is_rush_hour"
    ])

    assert result.loc[0, "timestamp_is_late_night"] == True
    assert result.loc[1, "timestamp_is_night"] == True
    assert result.loc[2, "timestamp_is_early_morning"] == True
    assert result.loc[3, "timestamp_is_rush_hour"] == True
    assert result.loc[4, "timestamp_is_rush_hour"] == False

def test_split_date_cols_invalid_date(df_invalid):
    """Test handling of invalid date formats in the timestamp column."""
    result = split_date_cols(df_invalid.copy(), "timestamp")

    # First row should be NaT
    assert pd.isna(result.loc[0, "timestamp"])
    # The second should be parsed correctly
    assert result.loc[1, "timestamp_year"] == 2024


def test_split_date_cols_empty(df_empty):
    """Test handling of empty DataFrame."""
    result = split_date_cols(df_empty.copy(), "timestamp")

    assert result.empty
    assert all(col in result.columns for col in [
        "timestamp_dayofyear", "timestamp_month", "timestamp_year",
        "timestamp_hour", "timestamp_dayofweek", "timestamp_is_weekend",
        "timestamp_is_late_night", "timestamp_is_night",
        "timestamp_is_early_morning", "timestamp_is_rush_hour"
    ])

def test_split_date_cols_missing_column(df_timestamps):
    """Test handling when the date column is missing."""
    with pytest.raises(KeyError, match="Column 'missing' not found in DataFrame."):
        split_date_cols(df_timestamps.copy(), "missing")

def test_haversine_known_distance():
    # Distance between New York City and London (~5567 km)
    nyc_lat, nyc_lon = 40.7128, -74.0060
    london_lat, london_lon = 51.5074, -0.1278
    distance = calculate_haversine(nyc_lat, nyc_lon, london_lat, london_lon)
    assert isclose(distance, 5567, rel_tol=0.01)

def test_haversine_zero_distance():
    # Distance should be 0
    lat, lon = 47.0722, 21.9211  # Oradea, Romania
    distance = calculate_haversine(lat, lon, lat, lon)
    assert distance == 0

def test_haversine_series_input():
    # Series input (vectorized)
    lats1 = pd.Series([0.0, 10.0])
    lons1 = pd.Series([0.0, 10.0])
    lats2 = pd.Series([0.0, 20.0])
    lons2 = pd.Series([0.0, 30.0])

    distances = calculate_haversine(lats1, lons1, lats2, lons2)
    assert isinstance(distances, pd.Series)
    assert len(distances) == 2
    assert distances[0] == 0  # Same point
    assert distances[1] > 0   # Some distance

def test_haversine_invalid_input():
    with pytest.raises(ValueError):
        calculate_haversine("invalid", 0, 0, 0)

def test_add_cyclic_features(df_cyclic):
    """Test adding cyclic features to a DataFrame."""
    result = add_cyclic_features(df_cyclic.copy())

    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "dow_sin" in result.columns
    assert "dow_cos" in result.columns
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns
    assert "doy_sin" in result.columns
    assert "doy_cos" in result.columns

    # Check some values
    assert isclose(result.loc[0, "hour_sin"], 0)
    assert isclose(result.loc[0, "hour_cos"], 1)