# src/utils.py

import yaml
import logging
import os

def load_config(config_path="configs/config.yaml"):
    """
    Load YAML config into a Python dict.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def init_logger(log_file="logs/prefect.log"):
    """
    Initialize a root logger that writes to the specified log file.
    Returns: logging.Logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger()
