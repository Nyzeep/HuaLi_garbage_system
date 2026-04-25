from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import create_app


class DummyDetectionService:
    def __init__(self) -> None:
        self.detect_calls: list[np.ndarray] = []
        self.draw_calls: list[tuple[np.ndarray, list[dict]]] = []
        self.build_calls: list[tuple[np.ndarray, list[dict], bool]] = []
        self.models_loaded = {"garbage": True, "fire": True, "smoke": False}

    def detect(self, image: np.ndarray) -> list[dict]:
        self.detect_calls.append(image)
        return [
            {
                "class_id": 3,
                "class_name": "FIRE",
                "confidence": 0.91,
                "bbox": [1, 1, 8, 8],
                "alert": True,
                "color": (0, 0, 255),
                "icon": "🔥",
                "source_model": "fire",
            }
        ]

    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        self.draw_calls.append((image, detections))
        return np.full_like(image, 200)

    def build_response(self, image: np.ndarray, detections: list[dict], with_image: bool = True) -> dict:
        self.build_calls.append((image, detections, with_image))
        alert_count = sum(1 for item in detections if item.get("alert", False))
        return {
            "scene": {
                "status": "mock-scene",
                "alert_count": alert_count,
                "alert_types": [item.get("class_name", "") for item in detections if item.get("alert", False)],
                "normal_count": len(detections) - alert_count,
                "total": len(detections),
                "timestamp": "2026-01-01 00:00:00",
            },
            "detections": detections,
            "result_image": "base64-image" if with_image else None,
        }


class DummyRecordService:
    def __init__(self) -> None:
        self.alert_records: list[dict] = []
        self.video_tasks: dict[str, dict] = {}

    def create_alert_record(self, db, scene, detections, rendered, source):
        self.alert_records.append(
            {"scene": scene, "detections": detections, "source": source, "shape": tuple(rendered.shape)}
        )

    def upsert_video_task(self, db, **kwargs):
        self.video_tasks[kwargs["task_id"]] = dict(kwargs)

    def update_video_task(self, db, task_id, **kwargs):
        self.video_tasks.setdefault(task_id, {}).update(kwargs)

    def get_video_task(self, db, task_id):
        data = self.video_tasks.get(task_id)
        if data is None:
            return None
        return type(
            "TaskRecord",
            (),
            {
                "status": data.get("status", "pending"),
                "progress": data.get("progress", 0),
                "message": data.get("message", ""),
                "output_path": data.get("output_path"),
                "total_frames": data.get("total_frames"),
                "detected_frames": data.get("detected_frames"),
                "total_detections": data.get("total_detections"),
                "total_alerts": data.get("total_alerts"),
                "video_info": data.get("video_info"),
                "record_uid": task_id,
                "created_at": __import__("datetime").datetime.now(),
                "alert_types": [],
                "alert_count": 0,
                "source": data.get("source", "video"),
            },
        )()

    def list_alerts(self, db, page, per_page, status):
        return 0, []

    def get_video_alert_types(self, db, task_id):
        return None

    def get_alert_image_base64(self, db, record_uid):
        return "fake-image"

    def build_statistics(self, db, started_at):
        return {"started_at": started_at, "total": 1}

    def list_classes(self):
        return {"classes": [0, 1, 2, 3, 4]}


class DummyAsyncResult:
    def __init__(self):
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True


class DummyThread:
    def __init__(self, target, kwargs=None, daemon=None):
        self.target = target
        self.kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        self.target(**self.kwargs)


def make_png_base64() -> str:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("utf-8")


def make_client(monkeypatch, dummy_detection_service, dummy_record_service):
    monkeypatch.setattr(routes, "RecordService", lambda uploads_dir: dummy_record_service)
    monkeypatch.setattr(routes, "base64_to_frame", lambda _: np.zeros((12, 12, 3), dtype=np.uint8))
    monkeypatch.setattr(routes.celery_app.control, "ping", lambda timeout=0.8: True)
    monkeypatch.setattr(routes.process_video_task, "apply_async", lambda **kwargs: None)
    monkeypatch.setattr(routes.threading, "Thread", DummyThread)

    app = create_app()
    app.dependency_overrides[routes.get_detection_service] = lambda: dummy_detection_service
    return TestClient(app)


def test_detect_image_endpoint_returns_detections_and_records_alert(monkeypatch):
    dummy_detection_service = DummyDetectionService()
    dummy_record_service = DummyRecordService()
    client = make_client(monkeypatch, dummy_detection_service, dummy_record_service)

    image = np.zeros((12, 12, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    files = {"file": ("test.png", io.BytesIO(encoded.tobytes()), "image/png")}
    response = client.post("/api/detect/image", files=files)

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["scene"]["status"] == "mock-scene"
    assert payload["detections"][0]["source"] == "fire"
    assert payload["result_image"] == "base64-image"
    assert len(dummy_detection_service.detect_calls) == 1
    assert len(dummy_detection_service.draw_calls) == 1
    assert len(dummy_detection_service.build_calls) == 1
    assert dummy_record_service.alert_records[0]["source"] == "image"


def test_detect_base64_endpoint_uses_base64_parser_and_records_camera(monkeypatch):
    dummy_detection_service = DummyDetectionService()
    dummy_record_service = DummyRecordService()
    client = make_client(monkeypatch, dummy_detection_service, dummy_record_service)

    response = client.post("/api/detect/base64", json={"image": make_png_base64()})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["result_image"] == "base64-image"
    assert dummy_record_service.alert_records[0]["source"] == "camera"
    assert len(dummy_detection_service.detect_calls) == 1


def test_detect_video_endpoint_submits_task_and_status_can_be_queried(monkeypatch):
    dummy_detection_service = DummyDetectionService()
    dummy_record_service = DummyRecordService()
    client = make_client(monkeypatch, dummy_detection_service, dummy_record_service)

    files = {"file": ("demo.mp4", io.BytesIO(b"fake-video-bytes"), "video/mp4")}
    response = client.post("/api/detect/video", files=files, data={"skip_frames": "2"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    task_id = payload["task_id"]
    assert task_id in dummy_record_service.video_tasks
    assert dummy_record_service.video_tasks[task_id]["status"] == "pending"

    dummy_record_service.video_tasks[task_id].update(
        {
            "status": "completed",
            "progress": 100,
            "message": "done",
            "output_path": str(Path("C:/tmp/demo_detected.mp4")),
            "total_frames": 10,
            "detected_frames": 3,
            "total_detections": 4,
            "total_alerts": 2,
            "video_info": "1920x1080, 30.0fps",
        }
    )

    status_response = client.get(f"/api/tasks/{task_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["success"] is True
    assert status_payload["status"] == "completed"
    assert status_payload["stats"]["total_frames"] == 10
    assert status_payload["result_video"] == "demo_detected.mp4"
