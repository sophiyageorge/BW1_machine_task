# logging_config/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Common logger (system-wide metrics)
common_logger = logging.getLogger("common")
common_logger.setLevel(logging.INFO)

common_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "common.log"),
    maxBytes=5*1024*1024,  # 5MB per file
    backupCount=5
)
common_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
common_handler.setFormatter(common_formatter)
common_logger.addHandler(common_handler)


# Individual logger (per request/user)
individual_logger = logging.getLogger("individual")
individual_logger.setLevel(logging.INFO)

individual_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "individual.log"),
    maxBytes=5*1024*1024,
    backupCount=5
)
individual_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
individual_handler.setFormatter(individual_formatter)
individual_logger.addHandler(individual_handler)