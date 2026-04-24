from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from app.constants import ALL_CLASSES, ALERT_ID_SET, BIN_TYPES
from app.dependencies import get_detection_service
from app.utils import base64_to_frame, frame_to_base64


class MyDetector:
    def __init__(
        self,
        garbage_model_path=None,
        fire_model_path=None,
        smoke_model_path=None,
        conf_threshold=0.5,
        iou_threshold=0.3,
    ):
        del garbage_model_path, fire_model_path, smoke_model_path, conf_threshold, iou_threshold
        warnings.warn(
            "app.detector.MyDetector is deprecated; use app.services.detection_service.DetectionService instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._service = get_detection_service()

    @property
    def models_loaded(self) -> dict[str, bool]:
        return self._service.models_loaded

    def detect(self, img: np.ndarray) -> list[dict]:
        if not any(self.models_loaded.values()):
            return []
        return self._service.detect(img)

    def draw_boxes(self, img: np.ndarray, det_list: list[dict]) -> np.ndarray:
        return self._service.draw_boxes(img, det_list)

    def check_scene(self, det_list: list[dict]) -> dict:
        return self._service.analyze_scene(det_list)

    def draw_results(self, img: np.ndarray, det_list: list[dict]) -> np.ndarray:
        return self.draw_boxes(img, det_list)

    def analyze_scene(self, det_list: list[dict]) -> dict:
        return self.check_scene(det_list)


UnifiedDetector = MyDetector
GarbageDetector = MyDetector
GARBAGE_CLASSES = ALL_CLASSES

__all__ = [
    "ALERT_ID_SET",
    "ALL_CLASSES",
    "BIN_TYPES",
    "GARBAGE_CLASSES",
    "GarbageDetector",
    "MyDetector",
    "UnifiedDetector",
    "base64_to_frame",
    "frame_to_base64",
]
