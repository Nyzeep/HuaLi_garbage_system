from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.constants import ALL_CLASSES
from app.infrastructure.ml.model_registry import ModelBundle, ModelDescriptor, ModelRegistry
from app.services.inference_service import InferenceService


@dataclass
class DummyPrediction:
    class_id: int
    confidence: float
    bbox: list[int]


class DummyBackend:
    def __init__(self, predictions: list[DummyPrediction], loaded: bool = True) -> None:
        self._predictions = predictions
        self.loaded = loaded
        self.calls: list[dict] = []
        self.batch_calls: list[dict] = []

    def predict(self, *, image, conf_threshold: float, iou_threshold: float):
        self.calls.append(
            {
                "image_shape": tuple(image.shape),
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
            }
        )
        return self._predictions

    def predict_batch(self, images, conf_threshold: float, iou_threshold: float):
        self.batch_calls.append(
            {
                "batch_size": len(images),
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
            }
        )
        return [self._predictions for _ in images]


def build_registry(*bundles: tuple[str, DummyBackend | None, dict[int, int] | None]) -> ModelRegistry:
    registry = ModelRegistry()
    for key, backend, class_mapping in bundles:
        descriptor = ModelDescriptor(
            key=key,
            onnx_path=__file__,
            pt_path=__file__,
            class_mapping=class_mapping or {},
        )
        registry.register(descriptor, backend)
    return registry


def test_registry_get_returns_bundle_for_existing_key():
    backend = DummyBackend([])
    registry = build_registry(("garbage", backend, {0: 2}))

    bundle = registry.get("garbage")

    assert bundle is not None
    assert bundle.descriptor.key == "garbage"
    assert bundle.backend is backend


def test_registry_get_returns_none_for_missing_key():
    registry = ModelRegistry()

    assert registry.get("missing") is None


def test_registry_get_supports_empty_key():
    backend = DummyBackend([])
    registry = build_registry(("", backend, {0: 0}))

    bundle = registry.get("")

    assert bundle is not None
    assert bundle.descriptor.key == ""


def test_registry_get_uses_latest_bundle_on_duplicate_key():
    old_backend = DummyBackend([], loaded=True)
    new_backend = DummyBackend([], loaded=False)
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(key="garbage", onnx_path=__file__, pt_path=__file__, class_mapping={0: 0}),
        old_backend,
    )
    registry.register(
        ModelDescriptor(key="garbage", onnx_path=__file__, pt_path=__file__, class_mapping={0: 2}),
        new_backend,
    )

    bundle = registry.get("garbage")

    assert bundle is not None
    assert bundle.backend is new_backend
    assert bundle.descriptor.class_mapping == {0: 2}
    assert len(registry.items()) == 1
    assert registry.loaded_map() == {"garbage": False}


def test_detect_returns_empty_when_no_loaded_models():
    service = InferenceService(ModelRegistry())

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert service.detect(image) == []


def test_detect_single_loaded_model_maps_fields_correctly():
    backend = DummyBackend(
        predictions=[
            DummyPrediction(class_id=0, confidence=0.91, bbox=[1, 2, 3, 4]),
        ]
    )
    registry = build_registry(("garbage", backend, {0: 2}))
    service = InferenceService(registry)

    image = np.zeros((12, 10, 3), dtype=np.uint8)
    detections = service.detect(image)

    assert backend.calls == [{"image_shape": (12, 10, 3), "conf_threshold": 0.35, "iou_threshold": 0.3}]
    assert detections == [
        {
            "class_id": 2,
            "class_name": ALL_CLASSES[2]["name"],
            "confidence": 0.91,
            "bbox": [1, 2, 3, 4],
            "alert": ALL_CLASSES[2]["alert"],
            "color": ALL_CLASSES[2]["color"],
            "icon": ALL_CLASSES[2]["icon"],
            "source_model": "garbage",
        }
    ]


def test_detect_multiple_loaded_models_merges_results():
    garbage_backend = DummyBackend([DummyPrediction(class_id=0, confidence=0.8, bbox=[1, 1, 5, 5])])
    fire_backend = DummyBackend([DummyPrediction(class_id=3, confidence=0.95, bbox=[6, 6, 9, 9])])
    registry = build_registry(
        ("garbage", garbage_backend, {0: 0}),
        ("fire", fire_backend, {3: 3}),
    )
    service = InferenceService(registry)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    detections = service.detect(image)

    assert len(detections) == 2
    assert {det["source_model"] for det in detections} == {"garbage", "fire"}
    assert {det["class_id"] for det in detections} == {0, 3}
    assert garbage_backend.calls[0]["conf_threshold"] == 0.35
    assert fire_backend.calls[0]["conf_threshold"] == 0.5
    assert all(call["iou_threshold"] == 0.3 for call in garbage_backend.calls + fire_backend.calls)


def test_detect_ignores_unloaded_backends():
    loaded_backend = DummyBackend([DummyPrediction(class_id=4, confidence=0.77, bbox=[2, 2, 6, 6])])
    unloaded_backend = DummyBackend([DummyPrediction(class_id=1, confidence=0.66, bbox=[0, 0, 1, 1])], loaded=False)
    registry = build_registry(
        ("smoke", loaded_backend, {4: 4}),
        ("unused", unloaded_backend, {1: 1}),
    )
    service = InferenceService(registry)

    image = np.zeros((9, 9, 3), dtype=np.uint8)
    detections = service.detect(image)

    assert len(detections) == 1
    assert detections[0]["class_id"] == 4
    assert unloaded_backend.calls == []


def test_detect_dedupes_cross_model_overlaps(monkeypatch):
    backend_a = DummyBackend([DummyPrediction(class_id=0, confidence=0.9, bbox=[0, 0, 10, 10])])
    backend_b = DummyBackend([DummyPrediction(class_id=3, confidence=0.8, bbox=[0, 0, 10, 10])])
    registry = build_registry(("garbage", backend_a, {0: 0}), ("fire", backend_b, {3: 3}))
    service = InferenceService(registry)

    monkeypatch.setattr(service, "_dedupe_cross_model_results", lambda detections: detections[:1])

    detections = service.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(detections) == 1


def test_detect_batch_returns_per_image_results():
    backend = DummyBackend([DummyPrediction(class_id=0, confidence=0.9, bbox=[0, 0, 10, 10])])
    registry = build_registry(("garbage", backend, {0: 0}))
    service = InferenceService(registry)

    images = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]
    results = service.detect_batch(images)

    assert len(results) == 2
    assert all(len(batch) == 1 for batch in results)
    assert backend.batch_calls == [{"batch_size": 2, "conf_threshold": 0.35, "iou_threshold": 0.3}]
    assert backend.calls == []


def test_detect_uses_adaptive_conf_threshold_for_garbage(monkeypatch):
    backend = DummyBackend([DummyPrediction(class_id=0, confidence=0.91, bbox=[1, 2, 3, 4])])
    registry = build_registry(("garbage", backend, {0: 2}))
    service = InferenceService(registry)
    monkeypatch.setattr(service._settings, "garbage_bin_conf_threshold", 0.42)
    monkeypatch.setattr(service._settings, "adaptive_conf_floor", 0.35)
    monkeypatch.setattr(service._settings, "adaptive_conf_ceiling", 0.7)

    service.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    assert backend.calls[0]["conf_threshold"] == 0.42


def test_dedupe_cross_model_results_keeps_highest_confidence_box(monkeypatch):
    registry = ModelRegistry()
    service = InferenceService(registry)

    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.9, "source_model": "a", "class_id": 0},
        {"bbox": [0, 0, 10, 10], "confidence": 0.8, "source_model": "b", "class_id": 3},
    ]

    monkeypatch.setattr(service, "_dedupe_cross_model_results", InferenceService._dedupe_cross_model_results.__get__(service, InferenceService))
    monkeypatch.setattr(
        "app.infrastructure.ml.rust_bridge.RustBridge.non_max_suppression",
        lambda self, boxes, threshold: [boxes[0]],
    )

    result = service._dedupe_cross_model_results(detections)

    assert result == [detections[0]]
