import yaml
from pathlib import Path
import pandas as pd
from src.utils.logging_config import logger


def save_data(df: pd.DataFrame, file_path: Path) -> None:
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