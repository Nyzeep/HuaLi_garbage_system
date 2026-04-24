from __future__ import annotations

from app.services.video_service import VideoProcessingService


class DummyDetectionService:
    def detect_raw(self, image):
        return []

    def draw_boxes(self, image, detections):
        return image


class DummyRustBridge:
    def dedupe_events(self, events, cooldown_ms, iou_threshold):
        return events

    def hamming_distance(self, previous_hash, current_hash):
        return 0

    def perceptual_hash(self, grayscale_pixels, width, height):
        return 0


def test_adaptive_micro_batch_size_switches_between_2_and_4():
    service = VideoProcessingService(DummyDetectionService(), rust_bridge=DummyRustBridge())
    assert service._adaptive_micro_batch_size(last_detection_count=0, last_alert_count=0) == 4
    assert service._adaptive_micro_batch_size(last_detection_count=5, last_alert_count=1) == 2
