from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.constants import ALL_CLASSES
from app.database import Base
from app.db_models import AlertRecord, DetectionRecord
from app.services.record_service import RecordService


def make_session(tmp_path: Path):
    db_path = tmp_path / "statistics.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return Session(), engine


def seed_alert(db, *, status: str, alert_count: int, created_at: datetime, class_id: int = 3):
    record = AlertRecord(
        record_uid=f"uid-{status}-{created_at.timestamp()}".replace(".", ""),
        status=status,
        alert_types=[ALL_CLASSES[class_id]["name"]] if status != "normal" else [],
        total_detections=max(1, alert_count),
        alert_count=alert_count,
        result_image_path=None,
        source="image",
        created_at=created_at,
    )
    db.add(record)
    db.flush()
    db.add(
        DetectionRecord(
            alert_record_id=record.id,
            class_id=class_id,
            class_name=ALL_CLASSES[class_id]["name"],
            confidence=0.9,
            bbox=[0, 0, 10, 10],
            is_alert=status != "normal",
            source_model="fire",
        )
    )
    db.commit()


def test_build_statistics_counts_and_class_distribution(tmp_path):
    db, engine = make_session(tmp_path)
    try:
        service = RecordService(tmp_path)
        now = datetime.utcnow()
        seed_alert(db, status="warning", alert_count=1, created_at=now - timedelta(hours=2), class_id=3)
        seed_alert(db, status="fire", alert_count=2, created_at=now - timedelta(hours=1), class_id=4)
        seed_alert(db, status="normal", alert_count=0, created_at=now, class_id=0)

        stats = service.build_statistics(db, started_at="2026-04-22 10:00:00")

        assert stats["start_time"] == "2026-04-22 10:00:00"
        assert stats["total_detections"] == 3
        assert stats["total_alerts"] == 3
        assert stats["alert_record_count"] == 3
        assert len(stats["hourly_alerts"]) == 24
        assert stats["today_alerts"] >= 2
        assert any(item["class_id"] == 3 for item in stats["class_stats"])
        assert any(item["class_id"] == 4 for item in stats["class_stats"])
    finally:
        db.close()
        engine.dispose()


def test_build_statistics_sorts_class_stats_by_count(tmp_path):
    db, engine = make_session(tmp_path)
    try:
        service = RecordService(tmp_path)
        now = datetime.utcnow()
        seed_alert(db, status="warning", alert_count=1, created_at=now - timedelta(minutes=5), class_id=3)
        seed_alert(db, status="warning", alert_count=1, created_at=now - timedelta(minutes=4), class_id=3)
        seed_alert(db, status="fire", alert_count=1, created_at=now - timedelta(minutes=3), class_id=4)

        stats = service.build_statistics(db, started_at="2026-04-22 10:00:00")

        counts = [item["count"] for item in stats["class_stats"]]
        assert counts == sorted(counts, reverse=True)
    finally:
        db.close()
        engine.dispose()
