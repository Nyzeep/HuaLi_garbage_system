from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from app.constants import ALL_CLASSES
from app.database import Base, SQLITE_BUSY_TIMEOUT_MS, create_database_engine
from app.db_models import AlertRecord, DetectionRecord, VideoTaskRecord
from app.services.record_service import RecordService


def make_session(tmp_path: Path):
    db_path = tmp_path / "record_service.db"
    engine = create_database_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return Session(), engine


def seed_alert(db, *, status: str, alert_count: int, created_at: datetime, source: str = "image") -> AlertRecord:
    record = AlertRecord(
        record_uid=f"uid{alert_count}{int(created_at.timestamp())}",
        status=status,
        alert_types=["火焰"] if status != "normal" else [],
        total_detections=max(alert_count, 1),
        alert_count=alert_count,
        result_image_path=None,
        source=source,
        created_at=created_at,
    )
    db.add(record)
    db.flush()
    db.add(
        DetectionRecord(
            alert_record_id=record.id,
            class_id=3,
            class_name=ALL_CLASSES[3]["name"],
            confidence=0.91,
            bbox=[0, 0, 10, 10],
            is_alert=True,
            source_model="fire",
        )
    )
    db.commit()
    db.refresh(record)
    return record


def test_upsert_video_task_creates_and_updates_record(tmp_path):
    db, engine = make_session(tmp_path)
    try:
        service = RecordService(tmp_path)

        created = service.upsert_video_task(
            db,
            task_id="task-1",
            input_filename="demo.mp4",
            input_path="/tmp/demo.mp4",
            status="pending",
            message="waiting",
        )
        assert created.task_id == "task-1"
        assert created.status == "pending"
        assert created.message == "waiting"

        updated = service.upsert_video_task(
            db,
            task_id="task-1",
            input_filename="demo.mp4",
            input_path="/tmp/demo.mp4",
            status="processing",
            message="running",
        )
        assert updated.id == created.id
        assert updated.status == "processing"
        assert updated.message == "running"
    finally:
        db.close()
        engine.dispose()


def test_update_video_task_and_get_video_task(tmp_path):
    db, engine = make_session(tmp_path)
    try:
        service = RecordService(tmp_path)
        service.upsert_video_task(
            db,
            task_id="task-2",
            input_filename="demo.mp4",
            input_path="/tmp/demo.mp4",
            status="pending",
            message="waiting",
        )

        updated = service.update_video_task(
            db,
            "task-2",
            status="completed",
            progress=100,
            message="done",
            output_path="/tmp/result.mp4",
        )
        assert updated is not None
        assert updated.status == "completed"
        assert updated.progress == 100
        assert updated.output_path == "/tmp/result.mp4"
        assert service.get_video_task(db, "task-2").status == "completed"
    finally:
        db.close()
        engine.dispose()


def test_list_classes_includes_known_metadata(tmp_path):
    service = RecordService(tmp_path)
    payload = service.list_classes()

    assert "classes" in payload
    assert "bin_types" in payload
    assert any(item["id"] == 3 and item["alert"] is True for item in payload["classes"])


def test_list_alerts_filters_and_orders_records(tmp_path):
    db, engine = make_session(tmp_path)
    try:
        service = RecordService(tmp_path)
        now = datetime.utcnow()
        older = now - timedelta(hours=1)
        seed_alert(db, status="warning", alert_count=1, created_at=older, source="camera")
        seed_alert(db, status="fire", alert_count=2, created_at=now, source="image")
        seed_alert(db, status="normal", alert_count=0, created_at=now + timedelta(minutes=1), source="image")

        total, records = service.list_alerts(db, page=1, per_page=10, status="all")
        assert total == 3
        assert records[0].created_at >= records[1].created_at

        total_warning, warning_records = service.list_alerts(db, page=1, per_page=10, status="warning")
        assert total_warning == 2
        assert all(r.status != "normal" for r in warning_records)

        total_fire, fire_records = service.list_alerts(db, page=1, per_page=10, status="fire")
        assert total_fire == 1
        assert fire_records[0].status == "fire"
    finally:
        db.close()
        engine.dispose()


def test_create_database_engine_enables_sqlite_wal_and_busy_timeout(tmp_path):
    db_path = tmp_path / "wal.db"
    engine = create_database_engine(f"sqlite:///{db_path}")

    try:
        with engine.connect() as connection:
            journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar()
            busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar()

        assert str(journal_mode).lower() == "wal"
        assert busy_timeout == SQLITE_BUSY_TIMEOUT_MS
    finally:
        engine.dispose()


def test_commit_with_retry_retries_locked_sqlite_writes(tmp_path, monkeypatch, caplog):
    db, engine = make_session(tmp_path)
    service = RecordService(tmp_path)
    locked_error = OperationalError(
        "COMMIT",
        {},
        sqlite3.OperationalError("database is locked"),
    )
    calls: list[str] = []

    def flaky_commit() -> None:
        calls.append("commit")
        if len(calls) == 1:
            raise locked_error

    rollback_calls: list[str] = []
    sleep_calls: list[float] = []

    monkeypatch.setattr(db, "commit", flaky_commit)
    monkeypatch.setattr(db, "rollback", lambda: rollback_calls.append("rollback"))
    monkeypatch.setattr("app.services.record_service.time.sleep", lambda seconds: sleep_calls.append(seconds))

    try:
        with caplog.at_level("WARNING", logger="app.services.record_service"):
            service._commit_with_retry(db)
    finally:
        db.close()
        engine.dispose()

    assert calls == ["commit", "commit"]
    assert rollback_calls == ["rollback"]
    assert sleep_calls == [pytest.approx(0.1)]
    assert "sqlite write retry attempt=1" in caplog.text
    assert "delay_seconds=0.10" in caplog.text
