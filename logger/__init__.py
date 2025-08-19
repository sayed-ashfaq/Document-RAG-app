# LOGGER/__init__.py

from .custom_logger import CustomLogger

# Create single shared logger instance

GLOBAL_LOGGER = CustomLogger().get_logger(__name__)