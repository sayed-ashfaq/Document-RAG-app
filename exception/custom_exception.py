import sys
import traceback
from typing import Optional, cast

class DocumentPortalException(Exception):
    """
    A DocumentPortalException class for capturing, normalizing, and formatting
    errors with detailed context such as file name, line number, and traceback.

    This exception can wrap any kind of error and provide additional
    debugging information that is more descriptive and industry-ready
    than standard Python exceptions.

    Attributes:
        file_name (str): The name of the file where the error occurred.
        lineno (int): The line number in the file where the error occurred.
        error_message (str): The normalized error message.
        traceback_str (str): A string representation of the full traceback, if available.
    """

    def __init__(self, error_message, error_details: Optional[object] = None):
        """
        Initialize the DocumentPortalException.

        Args:
            error_message (str | BaseException):
                The error message or exception object. If it's an exception,
                it will be converted to a string.
            error_details (Optional[object], default=None):
                Additional details about the error. Can be:
                  - None: Will automatically capture current exception info using sys.exc_info().
                  - sys: Explicitly capture the current exception info.
                  - Exception object: Capture type, value, and traceback directly.

        Behavior:
            - Normalizes the error message.
            - Extracts exception type, value, and traceback.
            - Walks to the deepest traceback frame to get the actual error location.
            - Stores file name, line number, and formatted traceback for debugging.

        Raises:
            None directly. But is meant to be raised as a custom error.
        """
        # Normalize message
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)  # Could also include class name
        else:
            norm_msg = str(error_message)

        # Resolve exc_info (supports sys, Exception object, or current context)
        exc_type = exc_value = exc_tb = None

        # ----------------- Core start -----------------
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            if hasattr(error_details, "exc_info"):
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = (
                    type(error_details),
                    error_details,
                    error_details.__traceback__,
                )
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()
        # ----------------- Core end -------------------

        # Walk to the last frame to report the most relevant location
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Full pretty traceback (if available)
        if exc_type and exc_tb:
            self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        """
        Return a human-readable string representation of the error.

        Returns:
            str: A compact, logger-friendly error message including:
                 - file name
                 - line number
                 - error message
                 - traceback (if available)
        """
        base = f"Error in [{self.file_name}] at line[{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\n{self.traceback_str}"
        return base

    def __repr__(self):
        """
        Return a developer-friendly string representation of the error.

        Returns:
            str: A concise representation of the exception object,
                 useful for debugging and logging.
        """
        return (
            f"DocumentPortalException(file={self.file_name}, "
            f"lineno={self.lineno}, message={self.error_message})"
        )


if __name__ == "__main__":
    # Demo Usage
    try:
        a = int("Str")  # This will fail
    except Exception as e:
        raise DocumentPortalException("Conversion Failed", e) from e
