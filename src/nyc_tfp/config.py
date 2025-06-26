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

PREDICTION_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CYCLIC_FEATURES + CATEGORICAL_FEATURES

# Paths
DATASET_DOWNLOAD_PATH = params["paths"]["dataset_download_path"]
RAW_DATASET_PATH = params["paths"]["raw_dataset_path"]
PREPROCESSED_DATASET_PATH = params["paths"]["preprocessed_dataset_path"]
FINAL_DATASET_PATH = params["paths"]["final_dataset_path"]
MODEL_PATH = params["paths"]["model_path"]
PREPROCESSOR_PATH = params["paths"]["preprocessor_path"]
TRAIN_TEST_SPLIT_DIR = params["paths"]["train_test_split_dir"]

# Experiment
EXPERIMENT_NAME = params["experiment"]["name"]
RAND_SEED = params["experiment"]["random_seed"]
TEST_SIZE = params["experiment"]["test_size"]
RUN_TYPE = params["experiment"]["run_type"]
REGISTERED_BY = params["experiment"]["registered_by"]