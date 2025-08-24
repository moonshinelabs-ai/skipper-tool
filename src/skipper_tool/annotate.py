#!/usr/bin/env python3
"""
Annotation utilities for UI element visualization.

This module contains all annotation-related functionality including:
- BoxAnnotator class for drawing bounding boxes
- MarkHelper class for Set-of-Mark (SOM) visualization
- Label positioning utilities
- Image annotation functions
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageDraw, ImageFont
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from torchvision.ops import box_convert


# BoxAnnotator class
class BoxAnnotator:
    """
    A class for drawing bounding boxes on an image using detections provided.
    """

    def __init__(
        self,
        color: Color | ColorPalette = ColorPalette.DEFAULT,
        thickness: int = 3,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 2,
        text_padding: int = 10,
        avoid_overlap: bool = True,
    ):
        self.color: Color | ColorPalette = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.avoid_overlap: bool = avoid_overlap

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[list[str]] = None,
        skip_label: bool = False,
        image_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes on the frame using the detections provided."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            text = (
                f"{class_id}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            if not self.avoid_overlap:
                text_x = x1 + self.text_padding
                text_y = y1 - self.text_padding

                text_background_x1 = x1
                text_background_y1 = y1 - 2 * self.text_padding - text_height

                text_background_x2 = x1 + 2 * self.text_padding + text_width
                text_background_y2 = y1
            else:
                (
                    text_x,
                    text_y,
                    text_background_x1,
                    text_background_y1,
                    text_background_x2,
                    text_background_y2,
                ) = get_optimal_label_pos(
                    self.text_padding,
                    text_width,
                    text_height,
                    x1,
                    y1,
                    x2,
                    y2,
                    detections,
                    image_size,
                )

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            box_color = color.as_rgb()
            luminance = (
                0.299 * box_color[0] + 0.587 * box_color[1] + 0.114 * box_color[2]
            )
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=text_color,
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


# Helper functions for BoxAnnotator
def get_optimal_label_pos(
    text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size
):
    """Check overlap of text and background detection box, and get optimal label position."""
    from .boxes import IoU

    def get_is_overlap(
        detections,
        text_background_x1,
        text_background_y1,
        text_background_x2,
        text_background_y2,
        image_size,
    ):
        is_overlap = False
        for i in range(len(detections)):
            detection = detections.xyxy[i].astype(int)
            if (
                IoU(
                    [
                        text_background_x1,
                        text_background_y1,
                        text_background_x2,
                        text_background_y2,
                    ],
                    detection,
                )
                > 0.3
            ):
                is_overlap = True
                break
        # check if the text is out of the image
        if (
            text_background_x1 < 0
            or text_background_x2 > image_size[0]
            or text_background_y1 < 0
            or text_background_y2 > image_size[1]
        ):
            is_overlap = True
        return is_overlap

    # Try top left position
    text_x = x1 + text_padding
    text_y = y1 - text_padding
    text_background_x1 = x1
    text_background_y1 = y1 - 2 * text_padding - text_height
    text_background_x2 = x1 + 2 * text_padding + text_width
    text_background_y2 = y1
    is_overlap = get_is_overlap(
        detections,
        text_background_x1,
        text_background_y1,
        text_background_x2,
        text_background_y2,
        image_size,
    )
    if not is_overlap:
        return (
            text_x,
            text_y,
            text_background_x1,
            text_background_y1,
            text_background_x2,
            text_background_y2,
        )

    # Try outer left position
    text_x = x1 - text_padding - text_width
    text_y = y1 + text_padding + text_height
    text_background_x1 = x1 - 2 * text_padding - text_width
    text_background_y1 = y1
    text_background_x2 = x1
    text_background_y2 = y1 + 2 * text_padding + text_height
    is_overlap = get_is_overlap(
        detections,
        text_background_x1,
        text_background_y1,
        text_background_x2,
        text_background_y2,
        image_size,
    )
    if not is_overlap:
        return (
            text_x,
            text_y,
            text_background_x1,
            text_background_y1,
            text_background_x2,
            text_background_y2,
        )

    # Try outer right position
    text_x = x2 + text_padding
    text_y = y1 + text_padding + text_height
    text_background_x1 = x2
    text_background_y1 = y1
    text_background_x2 = x2 + 2 * text_padding + text_width
    text_background_y2 = y1 + 2 * text_padding + text_height
    is_overlap = get_is_overlap(
        detections,
        text_background_x1,
        text_background_y1,
        text_background_x2,
        text_background_y2,
        image_size,
    )
    if not is_overlap:
        return (
            text_x,
            text_y,
            text_background_x1,
            text_background_y1,
            text_background_x2,
            text_background_y2,
        )

    # Try top right position
    text_x = x2 - text_padding - text_width
    text_y = y1 - text_padding
    text_background_x1 = x2 - 2 * text_padding - text_width
    text_background_y1 = y1 - 2 * text_padding - text_height
    text_background_x2 = x2
    text_background_y2 = y1
    is_overlap = get_is_overlap(
        detections,
        text_background_x1,
        text_background_y1,
        text_background_x2,
        text_background_y2,
        image_size,
    )
    if not is_overlap:
        return (
            text_x,
            text_y,
            text_background_x1,
            text_background_y1,
            text_background_x2,
            text_background_y2,
        )

    return (
        text_x,
        text_y,
        text_background_x1,
        text_background_y1,
        text_background_x2,
        text_background_y2,
    )


# MarkHelper class and related functions for SOM visualization
font_path = "util/arial.ttf"


class MarkHelper:
    def __init__(self):
        self.markSize_dict = {}
        self.font_dict = {}
        self.min_font_size = 20
        self.max_font_size = 30
        self.max_font_proportion = 0.04

    def __get_markSize(self, text, image_height, image_width, font):
        im = Image.new("RGB", (image_width, image_height))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return height, width

    def _setup_new_font(self, image_height, image_width):
        key = f"{image_height}_{image_width}"

        # setup the font
        fontsize = self.min_font_size
        try:
            font = ImageFont.truetype(font_path, fontsize)
        except (OSError, ImportError):
            font = ImageFont.load_default()

        while min(self.__get_markSize("555", image_height, image_width, font)) < min(
            self.max_font_size,
            self.max_font_proportion * min(image_height, image_width),
        ):
            fontsize += 1
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except (OSError, ImportError):
                font = ImageFont.load_default()
        self.font_dict[key] = font

        # setup the markSize dict
        markSize_3digits = self.__get_markSize("555", image_height, image_width, font)
        markSize_2digits = self.__get_markSize("55", image_height, image_width, font)
        markSize_1digit = self.__get_markSize("5", image_height, image_width, font)
        self.markSize_dict[key] = {
            1: markSize_1digit,
            2: markSize_2digits,
            3: markSize_3digits,
        }

    def get_font(self, image_height, image_width):
        key = f"{image_height}_{image_width}"
        if key not in self.font_dict:
            self._setup_new_font(image_height, image_width)
        return self.font_dict[key]

    def get_mark_size(self, text_str, image_height, image_width):
        """Get the font size for the given image dimensions."""
        key = f"{image_height}_{image_width}"
        if key not in self.markSize_dict:
            self._setup_new_font(image_height, image_width)

        largest_size = self.markSize_dict[key].get(3, None)
        text_h, text_w = self.markSize_dict[key].get(len(text_str), largest_size)
        return text_h, text_w


def _find_least_overlapping_corner(bbox, bboxes, drawn_boxes, text_size, image_size):
    """Find the corner with the least overlap with other bboxes."""
    from .boxes import __calculate_iou, __calculate_nearest_corner_distance

    y, x, h, w = bbox
    h_text, w_text = text_size
    image_height, image_width = image_size
    corners = [
        # top-left
        (y - h_text, x),
        # top-right
        (y - h_text, x + w - w_text),
        # right-top
        (y, x + w),
        # right-bottom
        (y + h - h_text, x + w),
        # bottom-right
        (y + h, x + w - w_text),
        # bottom-left
        (y + h, x),
        # left-bottom
        (y + h - h_text, x - w_text),
        # left-top
        (y, x - w_text),
    ]
    best_corner = corners[0]
    max_flag = float("inf")

    for corner in corners:
        corner_bbox = (corner[0], corner[1], h_text, w_text)
        # if the corner is out of the image, skip
        if (
            corner[0] < 0
            or corner[1] < 0
            or corner[0] + h_text > image_height
            or corner[1] + w_text > image_width
        ):
            continue
        max_iou = -(image_width + image_height)
        # given the current corner, find the largest iou with other bboxes.
        for other_bbox in bboxes + drawn_boxes:
            if np.array_equal(bbox, other_bbox):
                continue
            iou = __calculate_iou(corner_bbox, other_bbox, return_area=True)[1]
            max_iou = max(
                max_iou,
                iou
                - 0.0001 * __calculate_nearest_corner_distance(corner_bbox, other_bbox),
            )
        # the smaller the max_IOU, the better the corner
        if max_iou < max_flag:
            max_flag = max_iou
            best_corner = corner

    return best_corner


def plot_boxes_with_marks(
    image: Image.Image,
    bboxes,  # (y, x, h, w)
    mark_helper: MarkHelper,
    linewidth=2,
    alpha=0,
    edgecolor=None,
    fn_save=None,
    normalized_to_pixel=True,
    add_mark=True,
) -> np.ndarray:
    """Plot bounding boxes on an image with marks attached to edges where no overlap occurs."""
    # Then modify the drawing code
    draw = ImageDraw.Draw(image)

    # draw boxes on the image
    image_width, image_height = image.size

    if normalized_to_pixel:
        bboxes = [
            (
                int(y * image_height),
                int(x * image_width),
                int(h * image_height),
                int(w * image_width),
            )
            for y, x, h, w in bboxes
        ]

    for box in bboxes:
        y, x, h, w = box
        draw.rectangle([x, y, x + w, y + h], outline=edgecolor, width=linewidth)

    # Draw the bounding boxes with index at the least overlapping corner
    drawn_boxes = []
    for idx, bbox in enumerate(bboxes):
        text = str(idx)
        text_h, text_w = mark_helper.get_mark_size(text, image_height, image_width)
        corner_y, corner_x = _find_least_overlapping_corner(
            bbox, bboxes, drawn_boxes, (text_h, text_w), (image_height, image_width)
        )

        # Define the index box (y, x, y + h, x + w)
        text_box = (corner_y, corner_x, text_h, text_w)

        if add_mark:
            # Draw the filled index box and text
            draw.rectangle(
                [corner_x, corner_y, corner_x + text_w, corner_y + text_h], fill="red"
            )
            font = mark_helper.get_font(image_height, image_width)
            draw.text((corner_x, corner_y), text, fill="white", font=font)

        # Update the list of drawn boxes
        drawn_boxes.append(np.array(text_box))

    if fn_save is not None:
        image.save(fn_save)
    return image


def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: list[str],
    text_scale: float,
    text_padding=5,
    text_thickness=2,
    thickness=3,
) -> np.ndarray:
    """Annotate image with bounding boxes and labels."""
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(
        text_scale=text_scale,
        text_padding=text_padding,
        text_thickness=text_thickness,
        thickness=thickness,
    )
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels, image_size=(w, h)
    )

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates
