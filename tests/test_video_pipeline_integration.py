from __future__ import annotations

from pathlib import Path

import cv2
import imageio
import numpy as np

from app.services.detection_service import DetectionService
from app.infrastructure.ml.model_registry import ModelDescriptor
from app.services.video_service import VideoProcessingService
from app.upgrade.alarm import AlarmEngine
from app.upgrade.detection import DetectionEngine
from app.upgrade.pipeline import UpgradePipeline
from app.upgrade.tracker import TrackEngine


class DummyPrediction:
    def __init__(self, class_id: int = 3, confidence: float = 0.97, bbox: list[int] | None = None) -> None:
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox or [10, 10, 40, 40]


class DummyBackend:
    loaded = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, *, image, conf_threshold: float, iou_threshold: float):
        self.calls.append({"shape": tuple(image.shape), "conf": conf_threshold, "iou": iou_threshold})
        return [DummyPrediction()]


class DummyRegistry:
    def __init__(self) -> None:
        self.backend = DummyBackend()
        self.descriptor = ModelDescriptor(
            key="fire",
            onnx_path=Path("dummy.onnx"),
            pt_path=Path("dummy.pt"),
            class_mapping={3: 3},
        )
        self.bundle = type("Bundle", (), {"backend": self.backend, "descriptor": self.descriptor})()

    def items(self):
        return [self.bundle]

    def loaded_map(self):
        return {"fire": True}


class DummyInferenceService:
    def __init__(self):
        self.registry = DummyRegistry()

    def detect(self, image):
        bundle = self.registry.items()[0]
        predictions = bundle.backend.predict(image=image, conf_threshold=0.5, iou_threshold=0.3)
        return [
            {
                "class_id": pred.class_id,
                "class_name": "FIRE",
                "confidence": pred.confidence,
                "bbox": pred.bbox,
                "alert": True,
                "color": (0, 0, 255),
                "icon": "🔥",
                "source_model": bundle.descriptor.key,
            }
            for pred in predictions
        ]


class DummyAlertPolicyService:
    def apply_cooldown(self, detections):
        return detections


class DummySceneService:
    def analyze(self, detections):
        alert_count = sum(1 for d in detections if d.get("alert"))
        return {"status": "warning" if alert_count else "normal", "alert_count": alert_count, "alert_types": [3] if alert_count else []}


class DummyRenderingService:
    def draw_boxes(self, image, detections):
        output = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(output, (x1, y1), (x2, y2), det.get("color", (0, 255, 0)), 2)
        return output


class FakeDetectionService(DetectionService):
    def __init__(self):
        super().__init__(
            deps=type(
                "Deps",
                (),
                {
                    "inference_service": DummyInferenceService(),
                    "scene_service": DummySceneService(),
                    "alert_policy_service": DummyAlertPolicyService(),
                    "rendering_service": DummyRenderingService(),
                },
            )()
        )


def make_video(path: Path, frames: int = 4, size: tuple[int, int] = (64, 64)) -> None:
    with imageio.get_writer(str(path), fps=10, codec="libx264", pixelformat="yuv420p") as writer:
        for i in range(frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            cv2.putText(frame, f"{i}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def test_video_processing_pipeline_end_to_end(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    make_video(input_path, frames=3)

    detection_service = FakeDetectionService()
    video_service = VideoProcessingService(detection_service)
    monkeypatch.setattr(video_service.rust_bridge, "available", lambda: False)

    stats = video_service.process_video(
        input_path=input_path,
        output_path=output_path,
        skip_frames=1,
    )

    assert stats["total_frames"] == 3
    assert stats["total_detections"] >= 1
    assert stats["total_alerts"] >= 1
    assert output_path.exists()


def test_video_processing_pipeline_with_skip_frames_and_progress(tmp_path, monkeypatch):
    input_path = tmp_path / "input_skip.mp4"
    output_path = tmp_path / "output_skip.mp4"
    make_video(input_path, frames=5)

    detection_service = FakeDetectionService()
    video_service = VideoProcessingService(detection_service)
    monkeypatch.setattr(video_service.rust_bridge, "available", lambda: False)

    progress_updates: list[tuple[int, int]] = []

    stats = video_service.process_video(
        input_path=input_path,
        output_path=output_path,
        skip_frames=2,
        progress_callback=lambda current, total: progress_updates.append((current, total)),
    )

    assert stats["total_frames"] == 5
    assert progress_updates
    assert progress_updates[-1][0] == 5
    assert output_path.exists()


def test_upgrade_pipeline_attaches_track_and_alarm_metadata():
    detection_service = FakeDetectionService()
    pipeline = UpgradePipeline(
        detection_engine=DetectionEngine(detection_service),
        track_engine=TrackEngine(),
        alarm_engine=AlarmEngine(min_consecutive_frames=1),
    )

    detections = [
        {
            "class_id": 3,
            "class_name": "FIRE",
            "confidence": 0.97,
            "bbox": [10, 10, 40, 40],
            "alert": True,
            "color": (0, 0, 255),
            "icon": "🔥",
            "source_model": "fire",
        }
    ]

    result = pipeline.run_detections(detections)

    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1
    assert result.alarms == [] or result.alarms[0].track_id == 1
