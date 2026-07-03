import yaml
import os
from pathlib import Path

def get_project_base_path():
    """Return the base path of the project."""
    # This assumes the script is in src/utils
    current_path = Path(__file__).resolve()

    return current_path.parents[2]

def read_config(path_to_config):
    """
    Load the configuration file from the specified path.

    Parameters:
    - path_to_config (str): The file path to the configuration file.

    Returns:
    - dict: The configuration settings as a dictionary.
    """
    with open(path_to_config, "r") as ymlfile:
        return yaml.safe_load(ymlfile)