from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.alert_policy_service import AlertPolicyService
from app.services.detection_service import DetectionService, DetectionServiceDeps
from app.services.inference_service import InferenceService
from app.services.rendering_service import RenderingService
from app.services.scene_service import SceneService


@dataclass
class DummyRegistry:
    loaded: dict[str, bool]

    def loaded_map(self) -> dict[str, bool]:
        return self.loaded


class DummyInferenceService:
    def __init__(self, detections: list[dict], loaded: dict[str, bool] | None = None) -> None:
        self._detections = detections
        self.registry = DummyRegistry(loaded or {"garbage": True})
        self.detect_calls: list[np.ndarray] = []

    def detect(self, image: np.ndarray) -> list[dict]:
        self.detect_calls.append(image)
        return self._detections


class DummyAlertPolicyService:
    def __init__(self, result: list[dict]) -> None:
        self.result = result
        self.calls: list[list[dict]] = []

    def apply_cooldown(self, detections: list[dict]) -> list[dict]:
        self.calls.append(detections)
        return self.result


class DummyRenderingService:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, list[dict]]] = []

    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        self.calls.append((image, detections))
        return np.full_like(image, 123)


class DummySceneService:
    def __init__(self, scene: dict) -> None:
        self.scene = scene
        self.calls: list[list[dict]] = []

    def analyze(self, detections: list[dict]) -> dict:
        self.calls.append(detections)
        return self.scene


def build_service(
    detections: list[dict],
    cooled: list[dict] | None = None,
    scene: dict | None = None,
    loaded: dict[str, bool] | None = None,
) -> tuple[DetectionService, DummyInferenceService, DummyAlertPolicyService, DummyRenderingService, DummySceneService]:
    inference = DummyInferenceService(detections=detections, loaded=loaded)
    alert = DummyAlertPolicyService(cooled if cooled is not None else detections)
    rendering = DummyRenderingService()
    scene_service = DummySceneService(scene or {"label": "mock-scene"})
    service = DetectionService(
        DetectionServiceDeps(
            inference_service=inference,
            scene_service=scene_service,
            alert_policy_service=alert,
            rendering_service=rendering,
        )
    )
    return service, inference, alert, rendering, scene_service


def test_models_loaded_delegates_to_registry():
    service, _, _, _, _ = build_service(detections=[], loaded={"garbage": True, "fire": False})

    assert service.models_loaded == {"garbage": True, "fire": False}


def test_detect_applies_alert_cooldown_after_inference():
    raw_detections = [{"class_id": 1, "alert": True, "bbox": [0, 0, 10, 10]}]
    cooled_detections = [{"class_id": 1, "alert": False, "bbox": [0, 0, 10, 10]}]
    service, inference, alert, _, _ = build_service(raw_detections, cooled=cooled_detections)

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    detections = service.detect(image)

    assert detections == cooled_detections
    assert len(inference.detect_calls) == 1
    assert alert.calls == [raw_detections]


def test_detect_raw_skips_alert_cooldown():
    raw_detections = [{"class_id": 3, "alert": True, "bbox": [1, 1, 4, 4]}]
    service, inference, alert, _, _ = build_service(raw_detections, cooled=[{"class_id": 3, "alert": False, "bbox": [1, 1, 4, 4]}])

    image = np.zeros((6, 6, 3), dtype=np.uint8)
    detections = service.detect_raw(image)

    assert detections == raw_detections
    assert len(inference.detect_calls) == 1
    assert alert.calls == []


def test_draw_boxes_delegates_to_rendering_service():
    service, _, _, rendering, _ = build_service(detections=[])
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    detections = [{"bbox": [0, 0, 1, 1]}]

    result = service.draw_boxes(image, detections)

    assert result.shape == image.shape
    assert np.all(result == 123)
    assert rendering.calls == [(image, detections)]


def test_analyze_scene_delegates_to_scene_service():
    service, _, _, _, scene_service = build_service(detections=[], scene={"status": "ok"})
    detections = [{"class_id": 0}]

    assert service.analyze_scene(detections) == {"status": "ok"}
    assert scene_service.calls == [detections]


def test_build_response_includes_scene_and_optional_image():
    service, _, _, rendering, scene_service = build_service(detections=[], scene={"status": "ok"})
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    detections = [{"bbox": [0, 0, 1, 1]}]

    payload = service.build_response(image, detections, with_image=True)

    assert payload["scene"] == {"status": "ok"}
    assert payload["detections"] == detections
    assert payload["result_image"] is not None
    assert rendering.calls == [(image, detections)]
    assert scene_service.calls == [detections]


def test_build_response_without_image_skips_rendering():
    service, _, _, rendering, scene_service = build_service(detections=[], scene={"status": "ok"})
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    detections = [{"bbox": [0, 0, 1, 1]}]

    payload = service.build_response(image, detections, with_image=False)

    assert payload["scene"] == {"status": "ok"}
    assert payload["result_image"] is None
    assert rendering.calls == []
    assert scene_service.calls == [detections]
