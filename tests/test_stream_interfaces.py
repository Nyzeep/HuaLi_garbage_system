from __future__ import annotations

import base64
import io
from datetime import datetime

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import create_app


class DummyDetectionService:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def detect(self, image: np.ndarray) -> list[dict]:
        self.calls.append(image)
        return [
            {
                "class_id": 3,
                "class_name": "FIRE",
                "confidence": 0.93,
                "bbox": [1, 1, 8, 8],
                "alert": True,
                "color": (0, 0, 255),
                "icon": "🔥",
                "source_model": "fire",
            }
        ]

    def build_response(self, image: np.ndarray, detections: list[dict], with_image: bool = True) -> dict:
        return {
            "scene": {"status": "warning"},
            "detections": detections,
            "result_image": "base64-image" if with_image else None,
        }


class DummyTaskRecord:
    def __init__(self, **kwargs) -> None:
        self.status = kwargs.get("status", "pending")
        self.progress = kwargs.get("progress", 0)
        self.message = kwargs.get("message", "")
        self.output_path = kwargs.get("output_path")
        self.total_frames = kwargs.get("total_frames")
        self.detected_frames = kwargs.get("detected_frames")
        self.total_detections = kwargs.get("total_detections")
        self.total_alerts = kwargs.get("total_alerts")
        self.video_info = kwargs.get("video_info")
        self.record_uid = kwargs.get("record_uid", "uid")
        self.created_at = kwargs.get("created_at", datetime.now())
        self.alert_types = kwargs.get("alert_types", [])
        self.alert_count = kwargs.get("alert_count", 0)
        self.source = kwargs.get("source", "video")


class DummyRecordService:
    def __init__(self) -> None:
        self.tasks: dict[str, DummyTaskRecord] = {}
        self.alert_snapshot = {
            "total": 1,
            "page": 1,
            "per_page": 20,
            "records": [
                {
                    "id": "alert-1",
                    "time": "2026-04-22 10:00:00",
                    "status": "warning",
                    "types": ["fire"],
                    "total": 1,
                    "alert_count": 1,
                    "source": "image",
                }
            ],
        }
        self.statistics = {"total_detections": 1, "total_alerts": 1, "hourly_alerts": [0] * 24, "class_stats": []}

    def create_alert_record(self, *args, **kwargs):
        return None

    def upsert_video_task(self, db, **kwargs):
        self.tasks[kwargs["task_id"]] = DummyTaskRecord(**kwargs)
        return self.tasks[kwargs["task_id"]]

    def update_video_task(self, db, task_id, **kwargs):
        record = self.tasks.setdefault(task_id, DummyTaskRecord(task_id=task_id))
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record

    def get_video_task(self, db, task_id):
        return self.tasks.get(task_id)

    def list_alerts(self, db, page, per_page, status):
        return self.alert_snapshot["total"], [
            DummyTaskRecord(
                record_uid="alert-1",
                created_at=datetime.now(),
                status="warning",
                alert_types=["fire"],
                total_detections=1,
                alert_count=1,
                source="image",
            )
        ]

    def get_alert_image_base64(self, db, record_uid):
        return "fake-image"

    def build_statistics(self, db, started_at):
        data = dict(self.statistics)
        data["start_time"] = started_at
        return data

    def list_classes(self):
        return {"classes": [0, 1, 2, 3, 4], "bin_types": ["A", "B"]}


class DummyDB:
    def commit(self):
        return None


def build_png_base64() -> str:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("utf-8")


def make_client(monkeypatch):
    dummy_detection = DummyDetectionService()
    dummy_record = DummyRecordService()

    monkeypatch.setattr(routes, "RecordService", lambda uploads_dir: dummy_record)
    monkeypatch.setattr(routes.celery_app.control, "ping", lambda timeout=0.8: True)
    monkeypatch.setattr(routes.process_video_task, "apply_async", lambda **kwargs: None)
    monkeypatch.setattr(routes.threading, "Thread", lambda target, kwargs=None, daemon=None: type(
        "DummyThread",
        (),
        {"start": lambda self: target(**(kwargs or {}))},
    )())

    app = create_app()
    app.dependency_overrides[routes.get_detection_service] = lambda: dummy_detection
    app.dependency_overrides[routes.get_db] = lambda: DummyDB()
    client = TestClient(app)
    return client, dummy_detection, dummy_record


def test_task_status_sse_emits_completed_payload(monkeypatch):
    client, _, record_service = make_client(monkeypatch)
    task_id = "task-1"
    record_service.tasks[task_id] = DummyTaskRecord(
        task_id=task_id,
        status="completed",
        progress=100,
        message="视频处理完成",
        output_path="C:/uploads/videos/task-1_detected.mp4",
        total_frames=12,
        detected_frames=4,
        total_detections=8,
        total_alerts=3,
        video_info="1920x1080, 30.0fps",
    )

    with client.stream("GET", f"/api/tasks/{task_id}/stream") as response:
        assert response.status_code == 200
        line = next(response.iter_lines())
        assert line.startswith("data: ")
        payload = line.removeprefix("data: ")
        assert '"status": "completed"' in payload
        assert '"result_video": "task-1_detected.mp4"' in payload
        assert '"total_frames": 12' in payload


def test_alerts_and_statistics_sse_emit_initial_snapshot(monkeypatch):
    client, _, record_service = make_client(monkeypatch)

    with client.stream("GET", "/api/alerts/stream?page=1&per_page=20&status=all") as alerts_response:
        assert alerts_response.status_code == 200
        line = next(alerts_response.iter_lines())
        assert line.startswith("data: ")
        payload = line.removeprefix("data: ")
        assert '"total": 1' in payload
        assert '"id": "alert-1"' in payload

    with client.stream("GET", "/api/statistics/stream") as stats_response:
        assert stats_response.status_code == 200
        line = next(stats_response.iter_lines())
        assert line.startswith("data: ")
        payload = line.removeprefix("data: ")
        assert '"total_detections": 1' in payload
        assert '"start_time"' in payload


def test_camera_websocket_returns_detection_result(monkeypatch):
    client, detection_service, _ = make_client(monkeypatch)

    with client.websocket_connect("/api/ws/camera") as websocket:
        websocket.send_json({"image": build_png_base64()})
        payload = websocket.receive_json()

    assert payload["success"] is True
    assert payload["detections"][0]["source"] == "fire"
    assert payload["scene"] == {"status": "warning"}
    assert len(detection_service.calls) == 1


def test_camera_websocket_returns_error_for_bad_image(monkeypatch):
    client, detection_service, _ = make_client(monkeypatch)

    with client.websocket_connect("/api/ws/camera") as websocket:
        websocket.send_json({"image": "not-a-valid-image"})
        payload = websocket.receive_json()

    assert payload == {"error": "图片解析失败"}
    assert detection_service.calls == []
