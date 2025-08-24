#!/usr/bin/env python3
"""
Box utilities for UI element detection and processing.

This module contains all box-related functionality including:
- Box area calculations
- Intersection and IoU computations
- Overlap removal algorithms
- Box coordinate conversions
"""

import cv2
import numpy as np
import torch
from torchvision.transforms import ToPILImage

from skipper_tool.profiling import profile


def box_area(box):
    """Calculate the area of a bounding box."""
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1, box2):
    """Calculate the intersection area between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def IoU(box1, box2, return_max=True):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    intersection = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection
    if box_area(box1) > 0 and box_area(box2) > 0:
        ratio1 = intersection / box_area(box1)
        ratio2 = intersection / box_area(box2)
    else:
        ratio1, ratio2 = 0, 0
    if return_max:
        return max(intersection / union, ratio1, ratio2)
    else:
        return intersection / union


def __calculate_iou(box1, box2, return_area=False):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    y1, x1, h1, w1 = box1
    y2, x2, h2, w2 = box2

    # Calculate the intersection area
    y_min = max(y1, y2)
    x_min = max(x1, x2)
    y_max = min(y1 + h1, y2 + h2)
    x_max = min(x1 + w1, x2 + w2)

    intersection_area = max(0, y_max - y_min) * max(0, x_max - x_min)

    # Compute the area of both bounding boxes
    box1_area = h1 * w1
    box2_area = h2 * w2

    # Calculate the IoU
    iou = intersection_area / (min(box1_area, box2_area) + 0.0001)

    if return_area:
        return iou, intersection_area
    return iou


def __calculate_nearest_corner_distance(box1, box2):
    """Calculate the distance between the nearest edge or corner of two bounding boxes."""
    y1, x1, h1, w1 = box1
    y2, x2, h2, w2 = box2
    corners1 = np.array([[y1, x1], [y1, x1 + w1], [y1 + h1, x1], [y1 + h1, x1 + w1]])
    corners2 = np.array([[y2, x2], [y2, x2 + w2], [y2 + h2, x2], [y2 + h2, x2 + w2]])
    # Calculate pairwise distances between corners
    distances = np.linalg.norm(corners1[:, np.newaxis] - corners2, axis=2)

    # Find the minimum distance
    min_distance = np.min(distances)
    return min_distance


def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    """Remove overlapping boxes based on IoU threshold."""
    assert ocr_bbox is None or isinstance(ocr_bbox, list)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)

    for i, box1 in enumerate(boxes):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            # keep the smaller box
            if (
                i != j
                and IoU(box1, box2) > iou_threshold
                and box_area(box1) > box_area(box2)
            ):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                # only add the box if it does not overlap with any ocr bbox
                if not any(
                    IoU(box1, box3) > iou_threshold and not is_inside(box1, box3)
                    for k, box3 in enumerate(ocr_bbox)
                ):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    """Remove overlapping boxes with new format."""
    assert ocr_bbox is None or isinstance(ocr_bbox, list)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)

    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem["bbox"]
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem["bbox"]
            if (
                i != j
                and IoU(box1, box2) > iou_threshold
                and box_area(box1) > box_area(box2)
            ):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                # keep yolo boxes + prioritize ocr label
                box_added = False
                ocr_labels = ""
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem["bbox"]
                        if is_inside(box3, box1):  # ocr inside icon
                            try:
                                # gather all ocr labels
                                ocr_labels += box3_elem["content"] + " "
                                filtered_boxes.remove(box3_elem)
                            except (KeyError, ValueError):
                                continue
                        elif is_inside(
                            box1, box3
                        ):  # icon inside ocr, don't added this icon box
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append(
                            {
                                "type": "icon",
                                "bbox": box1_elem["bbox"],
                                "interactivity": True,
                                "content": ocr_labels,
                            }
                        )
                    else:
                        filtered_boxes.append(
                            {
                                "type": "icon",
                                "bbox": box1_elem["bbox"],
                                "interactivity": True,
                                "content": None,
                            }
                        )
            else:
                filtered_boxes.append(box1_elem)
    return filtered_boxes


def int_box_area(box, w, h):
    """Calculate box area in integer pixels."""
    x1, y1, x2, y2 = box
    int_box = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area


@torch.inference_mode()
def get_parsed_content_icon(
    filtered_boxes,
    starting_idx,
    image_source,
    caption_model_processor,
    prompt=None,
    batch_size=None,
):
    """Get parsed content for icon elements using caption model."""
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = (
                int(coord[0] * image_source.shape[1]),
                int(coord[2] * image_source.shape[1]),
            )
            ymin, ymax = (
                int(coord[1] * image_source.shape[0]),
                int(coord[3] * image_source.shape[0]),
            )
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except (ValueError, IndexError):
            continue

    model, processor = (
        caption_model_processor["model"],
        caption_model_processor["processor"],
    )
    if not prompt:
        if "florence" in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i : i + batch_size]
        if model.device.type == "cuda":
            inputs = processor(
                images=batch,
                text=[prompt] * len(batch),
                return_tensors="pt",
                do_resize=False,
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch, text=[prompt] * len(batch), return_tensors="pt"
            ).to(device=device)
        model_name = (
            model.config.name_or_path
            if hasattr(model.config, "name_or_path")
            else "unknown"
        )
        with profile(
            "Caption generation",
            model_name,
            batch_size=len(batch),
            prompt=prompt,
            device=str(device),
        ):
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                num_beams=1,
                do_sample=False,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts
