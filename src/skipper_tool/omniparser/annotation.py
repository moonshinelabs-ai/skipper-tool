#!/usr/bin/env python3
"""
Image annotation utilities for UI element visualization.

This module provides functions for:
1. Annotating images with bounding boxes and labels
2. Converting between coordinate formats
3. Image processing utilities

Reusable across both local and remote modes.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
import torch
from PIL import Image

from skipper_tool.annotate import BoxAnnotator


def ensure_pil_image(image: str | Image.Image | np.ndarray) -> Image.Image:
    """Convert input to PIL Image if needed. Reusable utility function.
    
    Args:
        image: Input image as file path, PIL Image, or numpy array
        
    Returns:
        PIL Image in RGB format
        
    Raises:
        ValueError: If input format is not supported
    """
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    else:
        raise ValueError("Input must be a file path, PIL Image, or numpy array")


def annotate_image(
    image: str | Image.Image | np.ndarray,
    predictions: dict,
    box_color: tuple[int, int, int] = (255, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 3,
    text_scale: float = 0.8,
    text_padding: int = 5,
) -> Image.Image:
    """Annotate image with UI element bounding boxes using supervision.

    Args:
        image: Input image as file path, PIL Image, or numpy array
        predictions: Predictions dict containing boxes and image_size
        box_color: RGB color for bounding boxes
        text_color: RGB color for text labels
        thickness: Line thickness for boxes
        text_scale: Scale factor for text
        text_padding: Padding around text

    Returns:
        PIL Image with annotations
    """
    # Convert input to PIL Image
    image = ensure_pil_image(image)

    # Convert PIL to numpy for annotation
    image_np = np.array(image)

    # Get boxes from predictions
    if "boxes" not in predictions:
        return image

    # Convert normalized boxes back to pixel coordinates
    w, h = predictions["image_size"]
    boxes_tensor = predictions["boxes"]

    # Convert from normalized xyxy to pixel xyxy
    boxes_pixel = boxes_tensor * torch.Tensor([w, h, w, h])
    boxes_array = boxes_pixel.numpy()

    # Create detections for supervision
    detections = sv.Detections(xyxy=boxes_array)

    # Create labels with element IDs
    labels = [f"{i}" for i in range(len(boxes_array))]

    # Use BoxAnnotator for consistent styling
    box_annotator = BoxAnnotator(
        thickness=thickness,
        text_scale=text_scale,
        text_padding=text_padding,
        text_thickness=max(1, thickness // 2),
    )

    # Annotate the image
    annotated_frame = box_annotator.annotate(
        scene=image_np.copy(),
        detections=detections,
        labels=labels,
        image_size=image.size,
    )

    return Image.fromarray(annotated_frame)


def format_predictions_for_api_compatibility(
    predictions: dict, 
    image_size: tuple[int, int]
) -> dict:
    """Format predictions dict for compatibility with API responses.
    
    Args:
        predictions: Raw predictions from engine
        image_size: Image dimensions (width, height)
        
    Returns:
        Formatted predictions dict with boxes, labels, coordinates, etc.
    """
    w, h = image_size
    boxes_tensor = predictions["boxes"]
    
    # Convert boxes to pixel coordinates
    boxes_pixel = boxes_tensor * torch.Tensor([w, h, w, h])
    boxes = boxes_pixel.tolist()

    # Create labels and coordinates for compatibility
    labels = [f"Element {i}" for i in range(len(boxes))]
    coordinates = {}
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        coordinates[i] = [x1, y1, x2 - x1, y2 - y1]  # xywh format

    # Format parsed content
    parsed_content = []
    for i, element in enumerate(predictions["filtered_elements"]):
        if element.get("content"):
            parsed_content.append(element["content"])
        else:
            parsed_content.append(f"Element {i}")

    return {
        "boxes": boxes,
        "labels": labels,
        "coordinates": coordinates,
        "parsed_content": parsed_content,
        "image_size": image_size,
    }