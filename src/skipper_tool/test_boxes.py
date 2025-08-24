#!/usr/bin/env python3
"""Tests for boxes.py module."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

import skipper_tool.boxes as boxes_module
from skipper_tool.boxes import (
    IoU,
    box_area,
    get_parsed_content_icon,
    int_box_area,
    intersection_area,
    remove_overlap,
    remove_overlap_new,
)


class TestBasicBoxFunctions:
    """Test basic box utility functions."""

    def test_box_area(self):
        """Test box area calculation."""
        # Standard box [x1, y1, x2, y2]
        box = [0, 0, 10, 5]
        assert box_area(box) == 50

        # Single point box
        box = [5, 5, 5, 5]
        assert box_area(box) == 0

        # Unit square
        box = [0, 0, 1, 1]
        assert box_area(box) == 1

    def test_intersection_area(self):
        """Test intersection area calculation."""
        # Overlapping boxes
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        assert intersection_area(box1, box2) == 25

        # Non-overlapping boxes
        box1 = [0, 0, 5, 5]
        box2 = [10, 10, 15, 15]
        assert intersection_area(box1, box2) == 0

        # Touching boxes (edge case)
        box1 = [0, 0, 5, 5]
        box2 = [5, 0, 10, 5]
        assert intersection_area(box1, box2) == 0

        # One box inside another
        box1 = [0, 0, 10, 10]
        box2 = [2, 2, 8, 8]
        assert intersection_area(box1, box2) == 36

    def test_iou(self):
        """Test IoU calculation."""
        # Identical boxes
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        assert IoU(box1, box2) == 1.0

        # Non-overlapping boxes
        box1 = [0, 0, 5, 5]
        box2 = [10, 10, 15, 15]
        assert IoU(box1, box2) == 0.0

        # Partially overlapping boxes
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = IoU(box1, box2)
        # Intersection = 25, Union = 100 + 100 - 25 = 175
        # But with return_max=True, it should be max(25/175, 25/100, 25/100) = 0.25
        assert abs(iou - 0.25) < 1e-6

        # Test return_max=False
        iou_basic = IoU(box1, box2, return_max=False)
        expected = 25 / 175  # intersection / union
        assert abs(iou_basic - expected) < 1e-6

        # Zero area boxes
        box1 = [0, 0, 0, 0]
        box2 = [1, 1, 2, 2]
        assert IoU(box1, box2) == 0.0


class TestPrivateHelperFunctions:
    """Test private helper functions."""

    def test_calculate_iou(self):
        """Test private __calculate_iou function."""
        # Format: [y, x, h, w]
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 10, 10]

        calculate_iou = getattr(boxes_module, "__calculate_iou")
        iou = calculate_iou(box1, box2)
        # Intersection area = 5*5 = 25
        # Min area = min(100, 100) = 100
        # IoU = 25 / (100 + 0.0001) ≈ 0.25
        assert abs(iou - 0.25) < 1e-3

        # Test with return_area=True
        iou, area = calculate_iou(box1, box2, return_area=True)
        assert area == 25
        assert abs(iou - 0.25) < 1e-3

    def test_calculate_nearest_corner_distance(self):
        """Test nearest corner distance calculation."""
        # Adjacent boxes
        box1 = [0, 0, 5, 5]  # [y, x, h, w]
        box2 = [0, 5, 5, 5]
        calculate_distance = getattr(
            boxes_module, "__calculate_nearest_corner_distance"
        )
        distance = calculate_distance(box1, box2)
        assert distance == 0.0  # They touch at corners

        # Separated boxes
        box1 = [0, 0, 5, 5]
        box2 = [10, 10, 5, 5]
        calculate_distance = getattr(
            boxes_module, "__calculate_nearest_corner_distance"
        )
        distance = calculate_distance(box1, box2)
        expected = np.sqrt(
            (10 - 5) ** 2 + (10 - 5) ** 2
        )  # Distance from (5,5) to (10,10)
        assert abs(distance - expected) < 1e-6


class TestRemoveOverlap:
    """Test overlap removal functions."""

    def test_remove_overlap_basic(self):
        """Test basic overlap removal."""
        # Two overlapping boxes - should keep the smaller one
        boxes = torch.tensor(
            [
                [0, 0, 10, 10],  # Area = 100
                [5, 5, 8, 8],  # Area = 9 (smaller, should be kept)
            ]
        )

        result = remove_overlap(boxes, iou_threshold=0.1)
        assert result.shape[0] == 1
        assert torch.equal(result[0], torch.tensor([5, 5, 8, 8]))

    def test_remove_overlap_with_ocr(self):
        """Test overlap removal with OCR bounding boxes."""
        boxes = torch.tensor(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
            ]
        )

        ocr_bbox = [
            [5, 5, 8, 8],  # Overlaps with first box
        ]

        result = remove_overlap(boxes, iou_threshold=0.1, ocr_bbox=ocr_bbox)
        # Should include OCR box and non-overlapping box
        assert result.shape[0] == 2

    def test_remove_overlap_no_overlap(self):
        """Test with non-overlapping boxes."""
        boxes = torch.tensor(
            [
                [0, 0, 5, 5],
                [10, 10, 15, 15],
                [20, 0, 25, 5],
            ]
        )

        result = remove_overlap(boxes, iou_threshold=0.1)
        assert result.shape[0] == 3  # All boxes should remain

    def test_remove_overlap_new_basic(self):
        """Test new format overlap removal."""
        boxes = [
            {"bbox": [0, 0, 10, 10], "type": "icon"},
            {"bbox": [5, 5, 8, 8], "type": "icon"},
        ]

        result = remove_overlap_new(boxes, iou_threshold=0.1)
        # Should keep smaller box
        assert len(result) == 1
        assert result[0]["bbox"] == [5, 5, 8, 8]

    def test_remove_overlap_new_with_ocr(self):
        """Test new format with OCR integration."""
        boxes = [
            {"bbox": [0, 0, 10, 10], "type": "icon"},
        ]

        ocr_bbox = [
            {"bbox": [2, 2, 4, 4], "content": "button", "type": "text"},
        ]

        result = remove_overlap_new(boxes, iou_threshold=0.1, ocr_bbox=ocr_bbox)
        # Should merge OCR content with icon
        assert len(result) == 1
        assert result[0]["content"] == "button "
        assert result[0]["type"] == "icon"


class TestIntBoxArea:
    """Test integer box area calculation."""

    def test_int_box_area(self):
        """Test integer box area calculation."""
        # Normalized coordinates [0-1] to pixel coordinates
        box = [0.1, 0.2, 0.5, 0.8]  # [x1, y1, x2, y2]
        w, h = 100, 50

        area = int_box_area(box, w, h)
        # Expected: (50 - 10) * (40 - 10) = 40 * 30 = 1200
        assert area == 1200

        # Edge case: zero area
        box = [0.5, 0.5, 0.5, 0.5]
        area = int_box_area(box, w, h)
        assert area == 0


class TestGetParsedContentIcon:
    """Test icon content parsing."""

    @patch("skipper_tool.boxes.cv2")
    @patch("skipper_tool.boxes.ToPILImage")
    def test_get_parsed_content_icon(self, mock_to_pil, mock_cv2):
        """Test icon content parsing with mocked dependencies."""
        # Mock setup
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mock_to_pil_instance = Mock()
        mock_to_pil.return_value = mock_to_pil_instance
        mock_cv2.resize.return_value = mock_image

        # Mock model and processor
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        mock_model.config.name_or_path = "test-model"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

        mock_processor = Mock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__.side_effect = lambda key: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.tensor([[[1.0]]]),
        }[key]
        mock_processor.return_value = mock_inputs
        mock_processor.batch_decode.return_value = ["test caption", "test caption"]

        caption_model_processor = {"model": mock_model, "processor": mock_processor}

        # Test data
        filtered_boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]])

        result = get_parsed_content_icon(
            filtered_boxes=filtered_boxes,
            starting_idx=None,
            image_source=mock_image,
            caption_model_processor=caption_model_processor,
            batch_size=2,
        )

        assert len(result) == 2
        assert all(isinstance(text, str) for text in result)

    @patch("skipper_tool.boxes.cv2")
    @patch("skipper_tool.boxes.ToPILImage")
    def test_get_parsed_content_icon_florence(self, mock_to_pil, mock_cv2):
        """Test with Florence model (different prompt)."""
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mock_to_pil_instance = Mock()
        mock_to_pil.return_value = mock_to_pil_instance
        mock_cv2.resize.return_value = mock_image

        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        mock_model.config.name_or_path = "florence-model"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

        mock_processor = Mock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__.side_effect = lambda key: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.tensor([[[1.0]]]),
        }[key]
        mock_processor.return_value = mock_inputs
        mock_processor.batch_decode.return_value = ["florence caption"]

        caption_model_processor = {"model": mock_model, "processor": mock_processor}

        filtered_boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5]])

        get_parsed_content_icon(
            filtered_boxes=filtered_boxes,
            starting_idx=None,
            image_source=mock_image,
            caption_model_processor=caption_model_processor,
            batch_size=1,
        )

        # Verify Florence-specific prompt was used
        mock_processor.assert_called()
        call_args = mock_processor.call_args[1]
        assert "<CAPTION>" in call_args["text"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_inputs(self):
        """Test functions with empty inputs."""
        # Empty tensor for remove_overlap
        empty_boxes = torch.tensor([]).reshape(0, 4)
        result = remove_overlap(empty_boxes, 0.5)
        assert result.shape[0] == 0

        # Empty list for remove_overlap_new
        result = remove_overlap_new([], 0.5)
        assert len(result) == 0

    def test_invalid_box_coordinates(self):
        """Test with invalid box coordinates."""
        # Negative area box (x2 < x1)
        box = [10, 10, 5, 5]
        area = box_area(box)
        # Function computes (5-10) * (5-10) = 25, doesn't validate inputs
        assert area == 25

        # Test IoU with invalid boxes
        box1 = [0, 0, 10, 10]
        box2 = [15, 15, 10, 10]  # Invalid: x2 < x1, y2 < y1
        iou = IoU(box1, box2)
        assert iou >= 0  # Should not crash

    def test_single_box_operations(self):
        """Test operations with single boxes."""
        boxes = torch.tensor([[0, 0, 10, 10]])
        result = remove_overlap(boxes, 0.5)
        assert result.shape[0] == 1
        assert torch.equal(result, boxes)


if __name__ == "__main__":
    pytest.main([__file__])
