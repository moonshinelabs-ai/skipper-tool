#!/usr/bin/env python3
"""
Local OmniParser implementation using YOLO models.

This module provides:
1. LocalOmniParser class for local YOLO execution
2. Integration with engine and annotation modules
3. Focused only on local model inference
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from skipper_tool.omniparser.annotation import (
    annotate_image,
    format_predictions_for_api_compatibility,
)
from skipper_tool.omniparser.engine import get_yolo_model, predict_ui_elements


class LocalOmniParser:
    """
    Local OmniParser implementation for UI element detection and annotation.

    This class uses local YOLO models to detect and annotate UI elements in images.
    """

    def __init__(
        self,
        som_model_path: str = "/Users/nharada/Models/omniparser/model.pt",
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        imgsz: int = 640,
        use_ocr: bool = True,
        ocr_text_threshold: float = 0.9,
    ):
        """
        Initialize the LocalOmniParser with model configurations.

        Args:
            som_model_path: Path to the YOLO model file
            box_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            imgsz: Image size for processing
            use_ocr: Whether to enable OCR text detection
            ocr_text_threshold: Confidence threshold for OCR text detection
        """
        # Set configuration as instance variables
        self.som_model_path = som_model_path
        self.box_threshold = box_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.use_ocr = use_ocr
        self.ocr_text_threshold = ocr_text_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize models
        self._load_models()

        logging.debug(f"LocalOmniParser initialized on {self.device}!")

    def _load_models(self):
        """Load the required models."""
        # Load YOLO model for icon detection
        self.yolo_model = get_yolo_model(model_path=self.som_model_path)

    def predict(
        self,
        image: str | Image.Image | np.ndarray,
        conf: float | None = None,
        imgsz: int | None = None,
        iou_threshold: float | None = None,
        output_coord_in_ratio: bool = False,
        **annotation_kwargs,
    ) -> dict:
        """
        Predict UI elements in the image and return annotated image.

        Args:
            image: Input image as file path, PIL Image, or numpy array
            conf: Confidence threshold for detections (overrides config if provided)
            imgsz: Image size for processing (overrides config if provided)
            iou_threshold: IoU threshold for non-maximum suppression (overrides config if provided)
            output_coord_in_ratio: Whether to output coordinates as ratios (currently ignored for local mode)
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
        # Use provided parameters or fall back to instance variable defaults
        box_threshold = conf if conf is not None else self.box_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold
        img_size = imgsz if imgsz is not None else self.imgsz

        # Get UI element predictions
        predictions = predict_ui_elements(
            image=image,
            model=self.yolo_model,
            box_threshold=box_threshold,
            iou_threshold=iou_thresh,
            imgsz=img_size,
            use_ocr=self.use_ocr,
            ocr_text_threshold=self.ocr_text_threshold,
        )

        # Format predictions for API compatibility
        result = format_predictions_for_api_compatibility(
            predictions, predictions["image_size"]
        )

        # Add annotated image
        annotated_image = annotate_image(image, predictions, **annotation_kwargs)
        result["annotated_image"] = annotated_image

        return result

    def get_element_at_point(self, predictions: dict, x: int, y: int) -> dict | None:
        """
        Get the UI element at a specific point in the image.

        Args:
            predictions: Predictions from predict() method
            x, y: Pixel coordinates

        Returns:
            Dictionary with element information if found, None otherwise
        """
        for i, box in enumerate(predictions["boxes"]):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                return {
                    "index": i,
                    "box": box,
                    "label": predictions["labels"][i],
                    "content": predictions["parsed_content"][i]
                    if i < len(predictions["parsed_content"])
                    else None,
                }
        return None

    def save_annotated_image(self, image: Image.Image, filename: str):
        """
        Save annotated image to file.

        Args:
            image: PIL Image to save
            filename: Output filename
        """
        image.save(filename)
        logging.debug(f"Annotated image saved to: {filename}")
