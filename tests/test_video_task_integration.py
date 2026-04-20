from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import app.tasks as tasks


class DummySession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DummyRecordService:
    def __init__(self, uploads_dir: Path) -> None:
        self.uploads_dir = uploads_dir
        self.updated: list[tuple[str, dict]] = []

    def update_video_task(self, db, task_id: str, **kwargs):
        self.updated.append((task_id, kwargs))


class DummyVideoService:
    def __init__(self, stats: dict | None = None, should_fail: bool = False) -> None:
        self.stats = stats or {
            "total_frames": 4,
            "detected_frames": 2,
            "total_detections": 3,
            "total_alerts": 1,
            "video_info": "64x64, 10.0fps",
        }
        self.should_fail = should_fail
        self.calls: list[dict] = []

    def process_video(self, *, input_path, output_path, skip_frames, progress_callback=None):
        self.calls.append(
            {
                "input_path": input_path,
                "output_path": output_path,
                "skip_frames": skip_frames,
                "has_progress_callback": progress_callback is not None,
            }
        )
        if progress_callback is not None:
            progress_callback(1, 4)
            progress_callback(4, 4)
        if self.should_fail:
            raise RuntimeError("process failed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")
        return self.stats


class DummyDetectionService:
    pass


class DummyCeleryTask:
    def __init__(self) -> None:
        self.request = type("Req", (), {"id": "celery-task-1"})()
        self.states: list[tuple[str, dict]] = []

    def update_state(self, state: str, meta: dict) -> None:
        self.states.append((state, meta))


def make_mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    assert writer.isOpened()
    for _ in range(3):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture()
def patched_tasks(monkeypatch, tmp_path):
    uploads_dir = tmp_path / "uploads"
    input_dir = uploads_dir / "videos"
    input_dir.mkdir(parents=True, exist_ok=True)

    session = DummySession()
    record_service = DummyRecordService(uploads_dir)
    video_service = DummyVideoService()

    monkeypatch.setattr(tasks, "bootstrap_application", lambda: None)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: session)
    monkeypatch.setattr(tasks, "RecordService", lambda uploads_dir_arg: record_service)
    monkeypatch.setattr(tasks, "get_detection_service", lambda: DummyDetectionService())
    monkeypatch.setattr(tasks, "VideoProcessingService", lambda detection_service: video_service)
    monkeypatch.setattr(tasks.settings, "uploads_dir", uploads_dir)

    return {
        "uploads_dir": uploads_dir,
        "session": session,
        "record_service": record_service,
        "video_service": video_service,
    }


def test_run_video_task_success_updates_status_progress_and_removes_input(patched_tasks, tmp_path):
    input_path = tmp_path / "input.mp4"
    make_mp4(input_path)

    result = tasks.run_video_task(task_id="task-1", input_path=str(input_path), skip_frames=2)

    record_service = patched_tasks["record_service"]
    session = patched_tasks["session"]
    video_service = patched_tasks["video_service"]

    assert result["task_id"] == "task-1"
    assert result["status"] == "completed"
    assert input_path.exists() is False
    assert session.closed is True
    assert len(video_service.calls) == 1
    assert video_service.calls[0]["skip_frames"] == 2

    updates = record_service.updated
    assert updates[0][1]["status"] == "processing"
    assert updates[0][1]["progress"] == 5
    assert any(item[1].get("progress") == 25 for item in updates)
    assert updates[-1][1]["status"] == "completed"
    assert updates[-1][1]["progress"] == 100
    assert updates[-1][1]["total_frames"] == 4


def test_run_video_task_failure_marks_task_failed(patched_tasks, tmp_path):
    input_path = tmp_path / "input_fail.mp4"
    make_mp4(input_path)

    patched_tasks["video_service"].should_fail = True

    with pytest.raises(RuntimeError, match="process failed"):
        tasks.run_video_task(task_id="task-err", input_path=str(input_path), skip_frames=1)

    record_service = patched_tasks["record_service"]
    session = patched_tasks["session"]

    assert session.closed is True
    assert record_service.updated[0][1]["status"] == "processing"
    assert record_service.updated[-1][1]["status"] == "failed"
    assert record_service.updated[-1][1]["message"] == "视频处理失败"
    assert "process failed" in record_service.updated[-1][1]["error_detail"]
    assert input_path.exists() is True


def test_process_video_task_forwards_request_id_and_progress_updates(monkeypatch, patched_tasks, tmp_path):
    input_path = tmp_path / "input_task.mp4"
    make_mp4(input_path)

    original_run_video_task = tasks.run_video_task
    captured: dict = {}

    def fake_run_video_task(*, task_id, input_path, skip_frames, progress_callback=None):
        captured.update(
            {
                "task_id": task_id,
                "input_path": input_path,
                "skip_frames": skip_frames,
            }
        )
        if progress_callback is not None:
            progress_callback(37)
        return {"task_id": task_id, "status": "completed", "result_video": "out.mp4", "stats": {}}

    monkeypatch.setattr(tasks, "run_video_task", fake_run_video_task)
    celery_task = DummyCeleryTask()

    result = tasks.process_video_task(celery_task, str(input_path), skip_frames=3)

    assert result["task_id"] == "celery-task-1"
    assert captured == {"task_id": "celery-task-1", "input_path": str(input_path), "skip_frames": 3}
    assert celery_task.states == [("PROGRESS", {"progress": 37})]

    monkeypatch.setattr(tasks, "run_video_task", original_run_video_task)
