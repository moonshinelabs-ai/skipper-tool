"""
Model profiling and instrumentation utilities.

This module provides a simple context manager for profiling model calls
using loguru for structured logging.
"""

import time
from contextlib import contextmanager
from typing import Any, Optional

from loguru import logger

# Global variable to store the current debug folder
_debug_folder = None


def set_debug_folder(folder_path):
    """Set the debug folder for model call logging."""
    global _debug_folder
    _debug_folder = folder_path

    # Setup model call logging if debug folder is provided
    if _debug_folder:
        setup_model_logging()

    # Also setup inference logging
    from skipper_tool.inference_logger import set_inference_logger

    # Only set up inference logger if folder_path is not None
    if folder_path:
        set_inference_logger(folder_path)


def setup_model_logging():
    """Setup model logging to use the same log file as main debug log."""
    # Model calls will now go to the same debug.log file
    # No separate file needed since main.py already sets up the debug log
    pass


@contextmanager
def profile(operation: str, model_type: str, **metadata: Any):
    """
    Context manager for profiling model calls.

    Args:
        operation: Description of the operation (e.g., "HTML reduction", "YOLO prediction")
        model_type: Type of model being used (e.g., "gemini-2.5-flash", "yolo", "blip2")
        **metadata: Additional metadata to log (e.g., input_size, batch_size, url)

    Usage:
        with profile("HTML reduction", "gemini-2.5-flash", input_chars=len(prompt)):
            response = client.chat.completions.create(...)
    """
    start_time = time.time()

    # Log operation start
    logger.debug(
        "Model call started",
        operation=operation,
        model_type=model_type,
        timestamp=start_time,
        **metadata,
    )

    try:
        yield

        # Calculate duration and log success
        duration = time.time() - start_time

        logger.debug(
            "Model call completed",
            operation=operation,
            model_type=model_type,
            duration=duration,
            **metadata,
        )

        # Human readable INFO log
        logger.info(f"✓ {operation} ({model_type}) - {duration:.2f}s")

    except Exception as e:
        # Calculate duration and log error
        duration = time.time() - start_time

        logger.error(
            "Model call failed",
            operation=operation,
            model_type=model_type,
            duration=duration,
            error=str(e),
            error_type=type(e).__name__,
            **metadata,
        )

        # Human readable ERROR log
        logger.error(f"✗ {operation} ({model_type}) failed after {duration:.2f}s: {e}")

        raise


def log_model_io(
    operation: str,
    input_data: Optional[str] = None,
    output_data: Optional[str] = None,
    **metadata,
):
    """
    Log model input/output data for debugging.

    Args:
        operation: Description of the operation
        input_data: Input data (will be truncated if too long)
        output_data: Output data (will be truncated if too long)
        **metadata: Additional metadata
    """

    def truncate_text(text: str, max_length: int = 1000) -> str:
        """Truncate text for logging."""
        if text and len(text) > max_length:
            return text[:max_length] + f"... [truncated from {len(text)} chars]"
        return text

    log_data = {"operation": operation, **metadata}

    if input_data:
        log_data["input"] = truncate_text(input_data)
        log_data["input_length"] = len(input_data)

    if output_data:
        log_data["output"] = truncate_text(output_data)
        log_data["output_length"] = len(output_data)

    logger.debug("Model I/O", **log_data)
