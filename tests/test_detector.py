from __future__ import annotations

import numpy as np
import pytest

import app.detector as detector_module


class DummyDetectionService:
    def __init__(self, models_loaded: dict[str, bool], detections: list[dict]) -> None:
        self.models_loaded = models_loaded
        self._detections = detections
        self.detect_calls: list[np.ndarray] = []
        self.draw_calls: list[tuple[np.ndarray, list[dict]]] = []
        self.analyze_calls: list[list[dict]] = []

    def detect(self, image: np.ndarray) -> list[dict]:
        self.detect_calls.append(image)
        return self._detections

    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        self.draw_calls.append((image, detections))
        return image

    def analyze_scene(self, detections: list[dict]) -> dict:
        self.analyze_calls.append(detections)
        return {"status": "normal", "alert_count": 0, "alert_types": [], "normal_count": len(detections), "total": len(detections), "timestamp": "2026-04-24 00:00:00"}


def test_detector_returns_empty_when_no_models_loaded(monkeypatch):
    service = DummyDetectionService(models_loaded={"garbage": False, "fire": False, "smoke": False}, detections=[{"alert": True}])
    monkeypatch.setattr(detector_module, "get_detection_service", lambda: service)

    with pytest.deprecated_call():
        detector = detector_module.MyDetector()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    detections = detector.detect(image)

    assert detections == []
    assert service.detect_calls == []


def test_detector_delegates_to_detection_service_when_models_loaded(monkeypatch):
    expected = [{"class_id": 3, "alert": True, "bbox": [1, 1, 2, 2], "source_model": "fire"}]
    service = DummyDetectionService(models_loaded={"garbage": False, "fire": True, "smoke": False}, detections=expected)
    monkeypatch.setattr(detector_module, "get_detection_service", lambda: service)

    with pytest.deprecated_call():
        detector = detector_module.MyDetector()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    detections = detector.detect(image)

    assert detections == expected
    assert service.detect_calls == [image]
