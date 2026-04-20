from __future__ import annotations

import numpy as np

from app.services.rendering_service import RenderingService


def test_draw_boxes_returns_copy_and_keeps_input_unchanged():
    service = RenderingService()
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    original = image.copy()
    detections = [
        {
            "class_id": 0,
            "confidence": 0.88,
            "bbox": [5, 6, 20, 22],
            "alert": True,
            "color": (0, 255, 0),
        }
    ]

    output = service.draw_boxes(image, detections)

    assert np.array_equal(image, original)
    assert output.shape == image.shape
    assert output is not image
    assert not np.array_equal(output, image)


def test_draw_boxes_uses_class_name_fallback_and_alert_style():
    service = RenderingService()
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    detections = [
        {
            "class_id": 3,
            "confidence": 0.95,
            "bbox": [10, 10, 30, 30],
            "alert": True,
            "color": (10, 20, 30),
        }
    ]

    output = service.draw_boxes(image, detections)

    assert output[10, 10].tolist() == [10, 20, 30]
    assert output[10, 30].tolist() == [10, 20, 30]
    # the label banner should paint pixels above the box in the detection color
    assert output[1, 10].tolist() == [10, 20, 30]


def test_draw_boxes_supports_custom_class_name_for_unknown_ids():
    service = RenderingService()
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    detections = [
        {
            "class_id": 99,
            "class_name": "CustomObject",
            "confidence": 0.5,
            "bbox": [15, 15, 35, 35],
            "alert": False,
            "color": (255, 0, 0),
        }
    ]

    output = service.draw_boxes(image, detections)

    assert output.shape == image.shape
    assert output[15, 15].tolist() == [255, 0, 0]
    assert output[15, 35].tolist() == [255, 0, 0]
