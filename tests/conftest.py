import pytest
import pandas as pd

@pytest.fixture
def df_timestamps():
    return pd.DataFrame({
        "timestamp": [
            "2023-05-15 22:30:00",  # late night
            "2023-05-16 02:15:00",  # night
            "2023-05-17 05:45:00",  # early morning
            "2023-05-17 07:10:00",  # rush hour
            "2023-05-18 15:00:00",  # normal
        ]
    })

@pytest.fixture
def df_invalid():
    return pd.DataFrame({
        "timestamp": ["invalid-date", "2024-01-01 12:00:00"]
    })

@pytest.fixture
def df_empty():
    return pd.DataFrame({
        "timestamp": []
    })


@pytest.fixture
def df_cyclic():
    return pd.DataFrame({
        "pickup_datetime_hour": [0, 6, 12, 18, 23],  # hours of the day
        "pickup_datetime_dayofweek": [0, 1, 2, 3, 4],  # days of the week
        "pickup_datetime_month": [1, 2, 3, 4, 5],  # months of the year
        "pickup_datetime_dayofyear": [1, 32, 60, 90, 120]  # days of the year
    })
