#!/usr/bin/env python3
"""
Remote OmniParser implementation using API calls.

This module provides:
1. RemoteOmniParser class for API-based execution
2. Image upload and request handling utilities
3. Response formatting for consistency with local mode
"""

from __future__ import annotations

import base64
import io
import json
from typing import Optional

import numpy as np
import requests
from loguru import logger
from PIL import Image

from skipper_tool.omniparser.annotation import ensure_pil_image


def _prepare_image_for_upload(
    image: str | Image.Image | np.ndarray,
) -> tuple[dict, Image.Image]:
    """Convert image to format suitable for API upload. Returns (files_dict, pil_image)."""
    pil_image = ensure_pil_image(image)

    # Convert image to bytes for upload
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("image.png", img_bytes, "image/png")}
    return files, pil_image


def _make_api_request(
    api_url: str,
    endpoint: str,
    api_key: str,
    files: dict,
    params: dict,
    timeout: int = 30,
) -> dict:
    """Make a POST request to the API and return the JSON response."""
    headers = {"X-API-Key": api_key}

    try:
        response = requests.post(
            f"{api_url}/{endpoint}",
            files=files,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Remote API request failed: {e}")
        raise Exception(f"Failed to get response from remote API: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse API response: {e}")
        raise Exception(f"Invalid response from remote API: {e}")


class RemoteOmniParser:
    """
    Remote OmniParser implementation using API calls.

    This class communicates with a remote OmniParser service to perform
    UI element detection and annotation.
    """

    def __init__(self, api_key: str, api_url: str):
        """
        Initialize the RemoteOmniParser with API configuration.

        Args:
            api_key: API key for authentication
            api_url: Base URL for the remote API service
        """
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")

        logger.info(f"RemoteOmniParser initialized with API: {self.api_url}")

    def predict(
        self,
        image: str | Image.Image | np.ndarray,
        conf: float = 0.05,
        imgsz: int | None = 640,
        iou_threshold: float = 0.9,
        output_coord_in_ratio: bool = False,
        **annotation_kwargs,
    ) -> dict:
        """
        Predict UI elements in the image using remote API.

        Args:
            image: Input image as file path, PIL Image, or numpy array
            conf: Confidence threshold for detections
            imgsz: Image size for processing
            iou_threshold: IoU threshold for non-maximum suppression
            output_coord_in_ratio: Whether to output coordinates as ratios
            **annotation_kwargs: Additional arguments for annotation

        Returns:
            Dictionary containing:
            - 'boxes': List of bounding boxes in format [x1, y1, x2, y2]
            - 'labels': List of element labels/descriptions
            - 'coordinates': Dictionary mapping element IDs to coordinates
            - 'parsed_content': List of parsed content descriptions
            - 'image_size': Tuple of (width, height)
            - 'annotated_image': PIL Image with annotations (if available)
        """
        # Prepare image for upload
        files, pil_image = _prepare_image_for_upload(image)

        # Prepare request parameters
        params = {
            "conf": conf,
            "iou_threshold": iou_threshold,
            "output_coord_in_ratio": output_coord_in_ratio,
        }

        # Add annotation parameters
        for key, value in annotation_kwargs.items():
            params[key] = value

        # Add imgsz if specified
        if imgsz is not None:
            params["imgsz"] = imgsz

        result = _make_api_request(self.api_url, "predict", self.api_key, files, params)

        # Format the result
        formatted_result = {
            "boxes": result.get("boxes", []),
            "labels": result.get("labels", []),
            "coordinates": result.get("coordinates", {}),
            "parsed_content": result.get("parsed_content", []),
            "image_size": result.get("image_size", list(pil_image.size)),
        }

        # Add annotated image if requested and available
        if "annotated_image" in result:
            try:
                image_bytes = base64.b64decode(result["annotated_image"])
                annotated_image = Image.open(io.BytesIO(image_bytes))
                formatted_result["annotated_image"] = annotated_image
            except Exception as e:
                logger.error(f"Failed to decode annotated image: {e}")

        return formatted_result

    def get_element_at_point(self, predictions: dict, x: int, y: int) -> Optional[dict]:
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
        logger.info(f"Annotated image saved to: {filename}")
