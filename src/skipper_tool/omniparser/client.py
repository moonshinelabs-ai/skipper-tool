#!/usr/bin/env python3
"""
Unified OmniParser client with automatic local/remote selection.

This module provides:
1. Single OmniParser class that automatically selects local vs remote
2. Clean factory pattern - no mode switching logic in predict()
3. Consistent interface regardless of backend
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from skipper_tool.omniparser.remote import RemoteOmniParser


class OmniParser:
    """
    Unified OmniParser client that automatically selects local or remote execution.

    Usage:
    # Auto-selection (tries local first, falls back to remote)
    parser = OmniParser()

    # Force local mode (if YOLO model path provided)
    parser = OmniParser(yolo_model_path="/path/to/model.pt")

    # Force remote mode
    parser = OmniParser(force_remote=True)
    """

    def __init__(
        self,
        yolo_model_path: str | None = None,
        force_remote: bool = False,
    ):
        """
        Initialize the OmniParser client with automatic mode selection.

        Args:
            yolo_model_path: Optional path to local YOLO model. If provided, will prefer local execution.
            force_remote: If True, force remote mode even if local model is available
        """
        self.yolo_model_path = yolo_model_path
        self.force_remote = force_remote

        self._parser = None
        self._mode = None

        # Initialize the appropriate parser
        self._initialize_parser()

    def _initialize_parser(self):
        """Initialize the appropriate parser (local or remote)."""
        # Get API configuration with hardcoded defaults
        self.api_key = os.getenv("SKIPPER_API_KEY")
        self.api_url = os.getenv(
            "SKIPPER_API_URL", "https://nate-3--omni-parser-api-fastapi-app.modal.run"
        ).rstrip("/")

        # Try local mode first if not forced remote and model path is provided
        if not self.force_remote and self.yolo_model_path:
            try:
                from .local import LocalOmniParser

                self._parser = LocalOmniParser(som_model_path=self.yolo_model_path)
                self._mode = "local"
                logger.info(
                    f"OmniParser initialized in local mode with model: {self.yolo_model_path}"
                )
                return
            except ImportError:
                logger.warning(
                    "Local OmniParser dependencies not available. Falling back to remote API."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize local OmniParser: {e}. Falling back to remote API."
                )

        # Fall back to remote mode
        if not self.api_key:
            raise ValueError(
                "No Skipper API key found and local mode unavailable. "
                "Set SKIPPER_API_KEY environment variable or provide a valid yolo_model_path."
            )

        self._parser = RemoteOmniParser(api_key=self.api_key, api_url=self.api_url)
        self._mode = "remote"
        logger.info(f"OmniParser initialized in remote mode with API: {self.api_url}")

    @property
    def mode(self) -> str:
        """Get the current execution mode (local or remote)."""
        return self._mode

    def predict(
        self,
        image: str | Image.Image | np.ndarray,
        conf: float = 0.05,
        imgsz: int | None = 640,
        iou_threshold: float = 0.1,
        output_coord_in_ratio: bool = False,
        **annotation_kwargs,
    ) -> dict:
        """
        Predict UI elements in the image and return annotated image.

        Args:
            image: Input image as file path, PIL Image, or numpy array
            conf: Confidence threshold for detections
            imgsz: Image size for processing
            iou_threshold: IoU threshold for non-maximum suppression
            output_coord_in_ratio: Whether to output coordinates as ratios
            **annotation_kwargs: Additional arguments for annotation (thickness, text_scale, etc.)

        Returns:
            Dictionary containing:
            - 'boxes': List of bounding boxes in format [x1, y1, x2, y2]
            - 'labels': List of element labels/descriptions
            - 'coordinates': Dictionary mapping element IDs to coordinates
            - 'parsed_content': List of parsed content descriptions
            - 'image_size': Tuple of (width, height)
            - 'annotated_image': PIL Image with annotations
        """
        logger.info(
            f"Predicting with {self._mode} mode, imgsz: {imgsz}, iou_threshold: {iou_threshold}, output_coord_in_ratio: {output_coord_in_ratio}"
        )

        # Pass all parameters to both local and remote parsers
        return self._parser.predict(
            image,
            conf=conf,
            imgsz=imgsz,
            iou_threshold=iou_threshold,
            output_coord_in_ratio=output_coord_in_ratio,
            **annotation_kwargs,
        )

    def get_element_at_point(self, predictions: dict, x: int, y: int) -> Optional[dict]:
        """
        Get the UI element at a specific point in the image.

        Args:
            predictions: Predictions from predict() method
            x, y: Pixel coordinates

        Returns:
            Dictionary with element information if found, None otherwise
        """
        return self._parser.get_element_at_point(predictions, x, y)

    def save_annotated_image(self, image: Image.Image, filename: str):
        """
        Save annotated image to file.

        Args:
            image: PIL Image to save
            filename: Output filename
        """
        self._parser.save_annotated_image(image, filename)
