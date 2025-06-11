import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

VERSION = params["version"]
MODEL_TYPE = params["model"]["type"]
ALPHA = params["model"]["alpha"]

# Features
NUMERIC_FEATURES = params["features"]["numeric"]
BOOLEAN_FEATURES = params["features"]["boolean"]
CYCLIC_FEATURES = params["features"]["cyclic"]
CATEGORICAL_FEATURES = params["features"]["categorical"]

TARGET_FEATURE = params["target_feature"]

SELECTED_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CYCLIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_FEATURE]

# Paths
DATASET_DOWNLOAD_PATH = params["paths"]["dataset_download_path"]
RAW_DATASET_PATH = params["paths"]["raw_dataset_path"]
PREPROCESSED_DATASET_PATH = params["paths"]["preprocessed_dataset_path"]
FINAL_DATASET_PATH = params["paths"]["final_dataset_path"]

# Experiment
EXPERIMENT_NAME = params["experiment"]["name"]