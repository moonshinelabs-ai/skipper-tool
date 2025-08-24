#!/usr/bin/env python3
"""
Core YOLO prediction engine for UI element detection.

This module provides pure functions for:
1. Loading YOLO models
2. Predicting UI elements in images
3. Processing detection results

No class dependencies - just core prediction logic.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from skipper_tool.omniparser.ocr import check_ocr_box
from skipper_tool.boxes import int_box_area, remove_overlap_new
from skipper_tool.profiling import profile


def get_yolo_model(model_path: str):
    """Load YOLO model from path.

    Args:
        model_path: Path to the YOLO model file

    Returns:
        Loaded YOLO model
    """
    from ultralytics import YOLO

    return YOLO(model_path)


def predict_ui_elements(
    image: str | Image.Image | np.ndarray,
    model,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    imgsz: int | None = None,
    use_ocr: bool = True,
    ocr_text_threshold: float = 0.5,
) -> dict:
    """Predict UI elements in an image using YOLO model and OCR.

    Args:
        image: Input image as file path, PIL Image, or numpy array
        model: YOLO model for prediction
        box_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for overlap removal
        imgsz: Image size for processing (optional)
        use_ocr: Whether to include OCR text detection
        ocr_text_threshold: Confidence threshold for OCR text detection

    Returns:
        Dictionary containing:
        - 'boxes': Tensor of bounding boxes in xyxy format
        - 'confidence': Tensor of confidence scores
        - 'filtered_elements': List of filtered box elements with metadata
        - 'image_size': Tuple of (width, height)
    """
    # Convert input to PIL Image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Input must be a file path, PIL Image, or numpy array")

    w, h = image.size
    if not imgsz:
        imgsz = (h, w)

    start_time = time.time()

    # OCR text detection
    ocr_elements = []
    if use_ocr:
        with profile(
            "PaddleOCR",
            "paddleocr",
            image_size=image.size,
            text_threshold=ocr_text_threshold,
        ):
            (text_list, bbox_list), _ = check_ocr_box(
                image, text_threshold=ocr_text_threshold, output_bb_format="xyxy"
            )

            # Convert OCR results to normalized coordinates and element format
            for text, bbox in zip(text_list, bbox_list):
                x1, y1, x2, y2 = bbox
                # Normalize coordinates
                norm_bbox = [x1 / w, y1 / h, x2 / w, y2 / h]
                # Check if box has positive area
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    ocr_elements.append(
                        {
                            "type": "text",
                            "bbox": norm_bbox,
                            "interactivity": False,
                            "content": text,
                        }
                    )

    # YOLO prediction with profiling
    with profile(
        "YOLO prediction",
        "yolo",
        image_size=image.size,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    ):
        result = model.predict(
            source=image,
            conf=box_threshold,
            imgsz=imgsz,
            iou=iou_threshold,
        )

    boxes = result[0].boxes.xyxy
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    # Log inference results
    runtime = time.time() - start_time
    from skipper_tool.inference_logger import log_yolo_call

    image_info = {
        "size": image.size,
        "format": "PIL_Image",
    }

    predictions = {
        "boxes": boxes.tolist() if hasattr(boxes, "tolist") else boxes,
        "confidence": conf.tolist() if hasattr(conf, "tolist") else conf,
        "phrases": phrases,
        "num_detections": len(boxes),
    }

    log_yolo_call(
        model_name=getattr(model, "model_name", "yolo_model"),
        operation="YOLO prediction",
        image_info=image_info,
        predictions=predictions,
        runtime_seconds=runtime,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )

    # Normalize boxes to ratio coordinates
    xyxy = boxes / torch.Tensor([w, h, w, h]).to(boxes.device)

    # Process detections into element format
    xyxy_elem = [
        {"type": "icon", "bbox": box, "interactivity": True, "content": None}
        for box in xyxy.tolist()
        if int_box_area(box, w, h) > 0
    ]

    # Remove overlapping boxes
    filtered_boxes = remove_overlap_new(
        boxes=xyxy_elem,
        iou_threshold=iou_threshold,
        ocr_bbox=ocr_elements,
    )

    # Sort boxes so content-less ones are at the end
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x["content"] is None)

    # Convert back to tensor format
    filtered_boxes_tensor = torch.tensor([box["bbox"] for box in filtered_boxes_elem])

    return {
        "boxes": filtered_boxes_tensor,
        "confidence": conf,
        "filtered_elements": filtered_boxes_elem,
        "image_size": (w, h),
    }
