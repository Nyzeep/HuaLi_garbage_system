from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from app.database import Base, create_database_engine
from app.db_models import VideoTaskRecord
from app.services.record_service import RecordService


def make_session_factory(tmp_path: Path):
    db_path = tmp_path / "concurrency.db"
    engine = create_database_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return Session, engine


def test_concurrent_video_task_updates_are_serializable(tmp_path):
    Session, engine = make_session_factory(tmp_path)
    service = RecordService(tmp_path)

    try:
        db = Session()
        service.upsert_video_task(
            db,
            task_id="task-concurrent",
            input_filename="demo.mp4",
            input_path="/tmp/demo.mp4",
            status="pending",
            message="waiting",
        )
        db.close()

        errors: list[str] = []
        statuses: list[str] = []
        lock = threading.Lock()

        def worker(idx: int):
            try:
                session = Session()
                try:
                    updated = service.update_video_task(
                        session,
                        "task-concurrent",
                        status=f"processing-{idx}",
                        progress=idx,
                        message=f"worker-{idx}",
                    )
                    assert updated is not None
                    with lock:
                        statuses.append(updated.status)
                finally:
                    session.close()
            except Exception as exc:
                with lock:
                    errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(statuses) == 8

        verify = Session()
        try:
            record = verify.query(VideoTaskRecord).filter(VideoTaskRecord.task_id == "task-concurrent").first()
            assert record is not None
            assert record.message.startswith("worker-")
            assert record.status.startswith("processing-")
        finally:
            verify.close()
    finally:
        engine.dispose()
