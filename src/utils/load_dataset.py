import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle import api
from dotenv import load_dotenv
from src.utils.logging_config import logger


def download_kaggle_competition_data(competition_name: str, dataset_download_path: str = "data/01_raw", expected_file: str = "train.csv") -> None:
    """
    Downloads and extracts Kaggle competition data using environment variables for credentials.
    
    Environment variables required:
        - KAGGLE_USERNAME
        - KAGGLE_KEY

    Args:
        competition_name (str): Kaggle competition slug, e.g., "new-york-city-taxi-fare-prediction"
        download_path (str): Directory to store the downloaded dataset
    """
    try:
        # Load env vars from .env if it exists
        load_dotenv()

        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if not username or not key:
            raise EnvironmentError("KAGGLE_USERNAME and KAGGLE_KEY must be set as environment variables.")

        os.makedirs(dataset_download_path, exist_ok=True)
        zip_file_path = Path(dataset_download_path) / f"{competition_name}.zip"
        extracted_file_path = Path(dataset_download_path) / expected_file

        if extracted_file_path.exists():
            logger.info(f"File '{extracted_file_path}' already exists. Skipping downloading from kaggle.")
            return


        logger.info(f"Connecting to Kaggle API, downloading dataset to {dataset_download_path}...")
        api = KaggleApi()
        api.authenticate()

        try: 
            logger.debug(f"Downloading data for competition: {competition_name}")
            api.competition_download_files(competition_name, path=dataset_download_path)
        except Exception as e:
            logger.error(f"Failed to download competition data: {e}")
            raise

        if zip_file_path.exists():
            logger.info(f"Extracting {zip_file_path}...")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(dataset_download_path)
            zip_file_path.unlink()
            logger.info("Download and extraction complete.")
        else:
            logger.info("No zip file found. Nothing extracted.")
    except Exception as e:
        logger.error(f"An error occurred while downloading or extracting the dataset: {e}")
        raise
