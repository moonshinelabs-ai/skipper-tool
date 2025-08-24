#!/usr/bin/env python3
"""
OCR utilities for text detection in UI screenshots.

This module provides PaddleOCR-based text detection functionality
adapted from the Magma UI agent demo.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Union, Tuple, List

# Initialize PaddleOCR with optimized settings from Magma demo
from paddleocr import PaddleOCR

_paddle_ocr = PaddleOCR(
    use_textline_orientation=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    lang="en",
)


def get_xyxy_from_paddle_result(
    coords: List[List[List[float]]],
) -> Tuple[int, int, int, int]:
    """Convert PaddleOCR coordinate format to xyxy bounding box.

    Args:
        coords: PaddleOCR coordinate format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates
    """
    x_coords = [point[0] for point in coords]
    y_coords = [point[1] for point in coords]
    x1, y1 = int(min(x_coords)), int(min(y_coords))
    x2, y2 = int(max(x_coords)), int(max(y_coords))
    return x1, y1, x2, y2


def check_ocr_box(
    image_source: Union[str, Image.Image],
    text_threshold: float = 0.5,
    output_bb_format: str = "xyxy",
) -> Tuple[Tuple[List[str], List[Tuple[int, int, int, int]]], None]:
    """
    Detect text in image using PaddleOCR.

    Args:
        image_source: Input image as file path or PIL Image
        text_threshold: Confidence threshold for text detection
        output_bb_format: Output bounding box format ('xyxy' or 'xywh')

    Returns:
        Tuple containing:
        - (text_list, bbox_list): Lists of detected text and bounding boxes
        - None: Goal filtering placeholder (for compatibility)

    Raises:
        ImportError: If PaddleOCR is not installed
        ValueError: If image format is not supported
    """
    if _paddle_ocr is None:
        raise ImportError(
            "PaddleOCR is not installed. Install it with: pip install paddleocr"
        )

    # Convert input to PIL Image if needed
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    elif not isinstance(image_source, Image.Image):
        raise ValueError("Input must be a file path or PIL Image")

    # Convert RGBA to RGB if needed
    if image_source.mode == "RGBA":
        image_source = image_source.convert("RGB")

    # Store original dimensions for scaling bounding boxes
    original_width, original_height = image_source.size
    scale_factor = 1.0

    # Resize image if larger than 1024px on any side
    max_size = 1024
    if max(original_width, original_height) > max_size:
        if original_width > original_height:
            new_width = max_size
            new_height = int((original_height * max_size) / original_width)
            scale_factor = original_width / new_width
        else:
            new_height = max_size
            new_width = int((original_width * max_size) / original_height)
            scale_factor = original_height / new_height

        image_source = image_source.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )

    # Convert to numpy array for PaddleOCR
    image_np = np.array(image_source)

    # Run OCR detection using latest PaddleOCR API
    result = _paddle_ocr.predict(image_np)

    if not result or len(result) == 0:
        return ([], []), None

    # Extract text and scores from new API format
    ocr_data = result[0]
    texts = ocr_data.get("rec_texts", [])
    scores = ocr_data.get("rec_scores", [])
    coordinates = ocr_data.get("dt_polys", [])

    if not texts or not scores or not coordinates:
        return ([], []), None

    # Filter results by confidence threshold
    filtered_indices = [i for i, score in enumerate(scores) if score > text_threshold]

    # Extract filtered coordinates and text
    coords = [coordinates[i] for i in filtered_indices]
    text = [texts[i] for i in filtered_indices]

    # Convert coordinates to desired format
    if output_bb_format == "xyxy":
        bbox = [get_xyxy_from_paddle_result(coord) for coord in coords]
        # Scale bounding boxes back to original image size if image was resized
        if scale_factor != 1.0:
            bbox = [
                (
                    int(x1 * scale_factor),
                    int(y1 * scale_factor),
                    int(x2 * scale_factor),
                    int(y2 * scale_factor),
                )
                for x1, y1, x2, y2 in bbox
            ]
    elif output_bb_format == "xywh":
        xyxy_boxes = [get_xyxy_from_paddle_result(coord) for coord in coords]
        # Scale bounding boxes back to original image size if image was resized
        if scale_factor != 1.0:
            xyxy_boxes = [
                (
                    int(x1 * scale_factor),
                    int(y1 * scale_factor),
                    int(x2 * scale_factor),
                    int(y2 * scale_factor),
                )
                for x1, y1, x2, y2 in xyxy_boxes
            ]
        bbox = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in xyxy_boxes]
    else:
        raise ValueError("output_bb_format must be 'xyxy' or 'xywh'")

    return (text, bbox), None


def is_paddleocr_available() -> bool:
    """Check if PaddleOCR is available."""
    return _paddle_ocr is not None
