from __future__ import annotations

from app.services.video_service import VideoProcessingService


class DummyDetectionService:
    def detect_raw(self, image):
        return []

    def draw_boxes(self, image, detections):
        return image


class DummyRustBridge:
    def dedupe_events(self, events, cooldown_ms, iou_threshold):
        return None


def _service() -> VideoProcessingService:
    return VideoProcessingService(
        detection_service=DummyDetectionService(),
        rust_bridge=DummyRustBridge(),
    )


def test_prune_and_cap_alert_history_trims_expired_and_limits_per_class():
    service = _service()
    history = [
        {"class_id": 1, "bbox": [i, 0, i + 10, 10], "timestamp": 0.0 + i * 0.01}
        for i in range(200)
    ]
    history.extend(
        {"class_id": 3, "bbox": [i, 0, i + 10, 10], "timestamp": 9.5 + i * 0.001}
        for i in range(5)
    )

    service._prune_and_cap_alert_history(history, current_ts=10.0)

    assert len([item for item in history if item["class_id"] == 1]) <= service.MAX_ALERT_HISTORY_PER_CLASS
    assert len([item for item in history if item["class_id"] == 3]) == 5
    assert all(10.0 - item["timestamp"] <= service.VIDEO_GARBAGE_COOLDOWN_SECONDS for item in history)


def test_python_video_cooldown_caps_history_growth():
    service = _service()
    history: list[dict] = []

    for index in range(200):
        detections = [{"class_id": 1, "bbox": [index * 20, 0, index * 20 + 10, 10], "alert": True}]
        service._apply_video_alert_cooldown_python(detections, current_ts=1.0, alert_history=history)

    assert len(history) <= service.MAX_ALERT_HISTORY_PER_CLASS
