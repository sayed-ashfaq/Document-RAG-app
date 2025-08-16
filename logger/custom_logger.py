import os
import logging
import structlog
from datetime import datetime


class CustomLogger:
    """
    A custom JSON-based logger using Python's logging + structlog.

    This logger writes logs both to a timestamped file (persistent) and to the console,
    while structuring them in JSON for machine readability and easy integration with
    log aggregators (e.g., ELK stack, Datadog, Splunk).

    Attributes:
        logs_dir (str): Directory where log files will be stored.
        log_file_path (str): Full path of the log file for this session.

    """

    def __init__(self, log_dir="logs"):
        """
        Initialize the logger directory and create a timestamped log file.

        Args:
            log_dir (str, optional): Directory where logs will be stored. Defaults to "logs".
        """
        # Ensure logs directory exists
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)  # Create if missing

        # Create a timestamped log file name
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name=__file__):
        """
        Configure and return a structured JSON logger.

        Args:
            name (str, optional): Name of the logger, usually the module/file name. Defaults to __file__.

        Returns:
            structlog.BoundLogger: A JSON-structured logger instance.
        """
        logger_name = os.path.basename(name)

        # File handler → logs saved to disk
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))  # raw JSON (structlog formats it)

        # Console handler → logs printed to stdout
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # Configure Python's logging module
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",  # structlog will format actual JSON
            handlers=[file_handler, console_handler]
        )

        # Configure structlog for structured JSON logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt='iso', utc=True, key="timestamp"),  # ISO UTC timestamp
                structlog.processors.add_log_level,  # Adds log level
                structlog.processors.EventRenamer(to="event"),  # Standardizes "msg" → "event"
                structlog.processors.JSONRenderer()  # Converts dict → JSON string
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(logger_name)


# --- Usage Example ---
if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)
    logger.info("user uploaded a file", user_id=123, filename="report.pdf")
    logger.error("failed to process PDF", error="File not found", user_id=456)
