"""
Model inference logging and result storage.

This module provides functionality to save model inference results including
inputs, outputs, runtime information, and model names for later analysis.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger


class InferenceLogger:
    """Logger for model inference runs with structured data storage."""

    def __init__(self, output_dir: str | Path):
        """
        Initialize the inference logger.

        Args:
            output_dir: Directory to save inference results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different types of inferences
        (self.output_dir / "llm_calls").mkdir(exist_ok=True)
        (self.output_dir / "yolo_predictions").mkdir(exist_ok=True)
        (self.output_dir / "vision_calls").mkdir(exist_ok=True)

    def log_llm_inference(
        self,
        model_name: str,
        operation: str,
        input_prompt: str,
        output_response: str,
        runtime_seconds: float,
        metadata: dict | None = None,
    ) -> str:
        """
        Log an LLM inference run.

        Args:
            model_name: Name of the model used
            operation: Description of the operation
            input_prompt: Input prompt sent to the model
            output_response: Response from the model
            runtime_seconds: Runtime in seconds
            metadata: Additional metadata (e.g., temperature, max_tokens)

        Returns:
            Path to the saved inference file
        """
        timestamp = datetime.now().isoformat()
        inference_id = f"{operation}_{int(time.time() * 1000)}"

        inference_data = {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "operation": operation,
            "runtime_seconds": runtime_seconds,
            "input": {"prompt": input_prompt, "prompt_length": len(input_prompt)},
            "output": {
                "response": output_response,
                "response_length": len(output_response) if output_response else 0,
            },
            "metadata": metadata or {},
        }

        # Save to file
        filename = f"llm_{inference_id}.json"
        filepath = self.output_dir / "llm_calls" / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(inference_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"LLM inference logged to {filepath}")
        return str(filepath)

    def log_vision_inference(
        self,
        model_name: str,
        operation: str,
        input_prompt: str,
        image_data: dict,
        output_response: str,
        runtime_seconds: float,
        metadata: dict | None = None,
    ) -> str:
        """
        Log a vision model inference run.

        Args:
            model_name: Name of the model used
            operation: Description of the operation
            input_prompt: Text prompt sent to the model
            image_data: Dict with image info (size, format, etc.)
            output_response: Response from the model
            runtime_seconds: Runtime in seconds
            metadata: Additional metadata

        Returns:
            Path to the saved inference file
        """
        timestamp = datetime.now().isoformat()
        inference_id = f"{operation}_{int(time.time() * 1000)}"

        inference_data = {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "operation": operation,
            "runtime_seconds": runtime_seconds,
            "input": {
                "prompt": input_prompt,
                "prompt_length": len(input_prompt),
                "image": image_data,
            },
            "output": {
                "response": output_response,
                "response_length": len(output_response) if output_response else 0,
            },
            "metadata": metadata or {},
        }

        # Save to file
        filename = f"vision_{inference_id}.json"
        filepath = self.output_dir / "vision_calls" / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(inference_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Vision inference logged to {filepath}")
        return str(filepath)

    def log_yolo_inference(
        self,
        model_name: str,
        operation: str,
        image_info: dict,
        predictions: dict,
        runtime_seconds: float,
        metadata: dict | None = None,
    ) -> str:
        """
        Log a YOLO model inference run.

        Args:
            model_name: Name/path of the YOLO model used
            operation: Description of the operation
            image_info: Dict with image info (size, format, etc.)
            predictions: Dict with boxes, confidence scores, etc.
            runtime_seconds: Runtime in seconds
            metadata: Additional metadata (thresholds, etc.)

        Returns:
            Path to the saved inference file
        """
        timestamp = datetime.now().isoformat()
        inference_id = f"{operation}_{int(time.time() * 1000)}"

        inference_data = {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "operation": operation,
            "runtime_seconds": runtime_seconds,
            "input": {"image": image_info},
            "output": {
                "predictions": predictions,
                "num_detections": len(predictions.get("boxes", [])),
            },
            "metadata": metadata or {},
        }

        # Save to file
        filename = f"yolo_{inference_id}.json"
        filepath = self.output_dir / "yolo_predictions" / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(inference_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"YOLO inference logged to {filepath}")
        return str(filepath)

    def get_inference_summary(self) -> dict:
        """
        Get a summary of all logged inferences.

        Returns:
            Dict with counts and summary statistics
        """
        summary = {
            "llm_calls": 0,
            "vision_calls": 0,
            "yolo_predictions": 0,
            "total_runtime_seconds": 0.0,
        }

        # Count files in each directory
        for subdir in ["llm_calls", "vision_calls", "yolo_predictions"]:
            dir_path = self.output_dir / subdir
            if dir_path.exists():
                json_files = list(dir_path.glob("*.json"))
                summary[subdir] = len(json_files)

                # Sum runtime from all files
                for file_path in json_files:
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            summary["total_runtime_seconds"] += data.get(
                                "runtime_seconds", 0.0
                            )
                    except (json.JSONDecodeError, KeyError):
                        continue

        return summary


# Global inference logger instance
_inference_logger: InferenceLogger | None = None


def get_inference_logger() -> InferenceLogger | None:
    """Get the global inference logger instance."""
    return _inference_logger


def set_inference_logger(output_dir: str | Path) -> InferenceLogger:
    """
    Set up the global inference logger.

    Args:
        output_dir: Directory to save inference results

    Returns:
        InferenceLogger instance
    """
    global _inference_logger
    _inference_logger = InferenceLogger(output_dir)
    logger.info(f"Inference logging enabled - saving to {output_dir}")
    return _inference_logger


def log_llm_call(
    model_name: str,
    operation: str,
    input_prompt: str,
    output_response: str,
    runtime_seconds: float,
    **metadata,
) -> str | None:
    """
    Convenience function to log an LLM call.

    Returns path to saved file or None if logging not enabled.
    """
    if _inference_logger:
        return _inference_logger.log_llm_inference(
            model_name,
            operation,
            input_prompt,
            output_response,
            runtime_seconds,
            metadata,
        )
    return None


def log_vision_call(
    model_name: str,
    operation: str,
    input_prompt: str,
    image_data: dict,
    output_response: str,
    runtime_seconds: float,
    **metadata,
) -> str | None:
    """
    Convenience function to log a vision model call.

    Returns path to saved file or None if logging not enabled.
    """
    if _inference_logger:
        return _inference_logger.log_vision_inference(
            model_name,
            operation,
            input_prompt,
            image_data,
            output_response,
            runtime_seconds,
            metadata,
        )
    return None


def log_yolo_call(
    model_name: str,
    operation: str,
    image_info: dict,
    predictions: dict,
    runtime_seconds: float,
    **metadata,
) -> str | None:
    """
    Convenience function to log a YOLO call.

    Returns path to saved file or None if logging not enabled.
    """
    if _inference_logger:
        return _inference_logger.log_yolo_inference(
            model_name, operation, image_info, predictions, runtime_seconds, metadata
        )
    return None
