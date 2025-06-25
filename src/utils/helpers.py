import yaml
from pathlib import Path
import pandas as pd
from src.utils.logging_config import logger


def save_data(df: pd.DataFrame, file_path: str | Path) -> None:
    """
    Saves the dataset to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (Path): Path where the CSV file will be saved.
    
    Raises:
        IOError: If saving the file fails.
    """
    try:
        logger.info(f"Ensuring directory {file_path.parent} exists...")
        file_path = Path(file_path)  # Ensure file_path is a Path object
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving data to {file_path}...")
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully at {file_path}")
    except IOError as e:
        raise IOError(f"Failed to save file: {e}")
    
def get_dvc_md5(dvc_file_path):
    with open(dvc_file_path, "r") as f:
        dvc_data = yaml.safe_load(f)
        return dvc_data.get("outs", [{}])[0].get("md5")

def save_train_test_split(X_train, X_test, y_train, y_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data(X_train, output_dir / "X_train.csv")
    save_data(X_test, output_dir / "X_test.csv")
    save_data(y_train.reset_index(drop=True), output_dir / "y_train.csv")
    save_data(y_test.reset_index(drop=True), output_dir / "y_test.csv")