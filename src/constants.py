VERSION = "v4"
MODEL_TYPE = "Lasso" # 'LinearRegression' or 'Lasso'
ALPHA = 0.01 # for lasso regression

#ORIGINAL COLUMNS
# 'key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count'

NUMERIC_FEATURES = [
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude',
    'dropoff_latitude', 
    'trip_distance', #feature enginered column
    'passenger_count',
    # 'pickup_datetime_year', #feature enginered column from pickup_datetime
    # 'pickup_datetime_month', #feature enginered column from pickup_datetime
    # 'pickup_datetime_dayofyear', #feature enginered column from pickup_datetime
    # 'pickup_datetime_dayofweek', #feature enginered column from pickup_datetime
    # 'pickup_datetime_hour', #feature enginered column from pickup_datetime
]

BOOLEAN_FEATURES = [
    'pickup_datetime_is_weekend', #feature enginered column
    'is_late_night', #feature enginered column
    'is_night',#feature enginered column
    'is_early_morning', #feature enginered column
    'is_rush_hour'#feature enginered column
]

CYCLIC_FEATURES = [
    'hour_sin', #feature enginered column
    'hour_cos', #feature enginered column
    'dow_sin', #feature enginered column
    'dow_cos', #feature enginered column
    'month_sin', #feature enginered column
    'month_cos', #feature enginered column
    'doy_sin', #feature enginered column
    'doy_cos'#feature enginered column
]

CATEGORICAL_FEATURES = []

TARGET_FEATURE = 'fare_amount'

# #v1
# SELECTED_COLUMNS = NUMERIC_FEATURES + [TARGET_FEATURE]

# #v2 with feature trip_distance
# SELECTED_COLUMNS = NUMERIC_FEATURES + [TARGET_FEATURE]

# #v3 with feature trip_distance
# SELECTED_COLUMNS = NUMERIC_FEATURES + CYCLIC_FEATURES + [TARGET_FEATURE]

#v4
SELECTED_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES + CYCLIC_FEATURES + [TARGET_FEATURE]

DATA_PATH = "data/02_processed/df_cleaned_00.csv"
EXPERIMENT_NAME = "NYC_Taxi_Fare_Pred"