import os
from datetime import datetime
import logging
import logging.handlers


def configure_logger():
    logs_directory_path = 'data/logs'

    if not os.path.exists(logs_directory_path):
        os.makedirs(logs_directory_path)

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'{logs_directory_path}/logfile_{current_date}.log'

    logger = logging.getLogger('NYC_Taxi_Fare_Pred_Logger')
    logger.setLevel(logging.DEBUG)

    # Check if handlers are already added to avoid duplicate handlers
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename, maxBytes=10**6, backupCount=3)  # Rotate logs if they exceed 1MB
        
        # Set logging levels
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        # Define the formatter 
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)


        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = configure_logger()