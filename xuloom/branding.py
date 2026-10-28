"""Public name and configuration location for Xuloom."""
from pathlib import Path

APP_NAME = "Xuloom"
CONFIG_DIRECTORY = ".xuloom"


def configuration_file(root, relative):
    return Path(root) / CONFIG_DIRECTORY / relative
