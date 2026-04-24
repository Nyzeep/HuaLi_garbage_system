from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime
from pathlib import Path
import re
import sqlite3
import time

import cv2
from sqlalchemy import func
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, selectinload

from app.constants import ALL_CLASSES, BIN_TYPES
from app.db_models import AlertRecord, DetectionRecord, VideoTaskRecord
from app.utils import frame_to_base64, relative_to, save_image


SQLITE_WRITE_RETRY_ATTEMPTS = 3
SQLITE_WRITE_RETRY_DELAY_SECONDS = 0.1

logger = logging.getLogger(__name__)


class RecordService:
    def __init__(self, uploads_dir: Path):
        self.uploads_dir = uploads_dir

    def build_output_path_payload(self, output_path: str | None) -> str | None:
        if not output_path:
            return None
        try:
            return Path(output_path).relative_to(self.uploads_dir).as_posix()
        except ValueError:
            return Path(output_path).name

    @staticmethod
    def parse_suppressed_alerts(video_info: str | None) -> int:
        if not video_info:
            return 0
        match = re.search(r"suppressed=(\d+)", video_info)
        return int(match.group(1)) if match else 0

    def _commit_with_retry(self, db: Session) -> None:
        attempts = SQLITE_WRITE_RETRY_ATTEMPTS
        for attempt in range(1, attempts + 1):
            try:
                db.commit()
                return
            except OperationalError as exc:
                db.rollback()
                if not self._is_retryable_sqlite_write_error(exc) or attempt >= attempts:
                    raise
                delay_seconds = SQLITE_WRITE_RETRY_DELAY_SECONDS * attempt
                logger.warning(
                    "sqlite write retry attempt=%d max_attempts=%d delay_seconds=%.2f error=%s",
                    attempt,
                    attempts,
                    delay_seconds,
                    exc,
                )
                time.sleep(delay_seconds)

    @staticmethod
    def _is_retryable_sqlite_write_error(exc: OperationalError) -> bool:
        original = getattr(exc, "orig", None)
        if not isinstance(original, sqlite3.OperationalError):
            return False
        message = str(original).lower()
        return "database is locked" in message or "database table is locked" in message

    def create_alert_record(
        self,
        db: Session,
        scene: dict,
        detections: list[dict],
        rendered_image,
        source: str,
    ) -> AlertRecord | None:
        if scene["alert_count"] <= 0:
            return None

        image_path = save_image(rendered_image, self.uploads_dir / "alerts")
        record = AlertRecord(
            record_uid=image_path.stem[-8:],
            status=scene["status"],
            alert_types=scene["alert_types"],
            total_detections=scene["total"],
            alert_count=scene["alert_count"],
            result_image_path=relative_to(image_path, self.uploads_dir),
            source=source,
        )
        db.add(record)
        db.flush()

        for detection in detections:
            db.add(
                DetectionRecord(
                    alert_record_id=record.id,
                    class_id=detection["class_id"],
                    class_name=detection["class_name"],
                    confidence=detection["confidence"],
                    bbox=detection["bbox"],
                    is_alert=detection["alert"],
                    source_model=detection.get("source_model", ""),
                ),
            )

        self._commit_with_retry(db)
        db.refresh(record)
        return record

    def create_video_alert_summary_record(
        self,
        db: Session,
        task_id: str,
        stats: dict,
    ) -> AlertRecord | None:
        total_alerts = int(stats.get("total_alerts", 0) or 0)
        if total_alerts <= 0:
            return None

        alert_types = [str(x) for x in (stats.get("alert_types") or [])]
        status = "warning"
        if any("火" in t for t in alert_types):
            status = "fire"
        elif any("烟" in t for t in alert_types):
            status = "smoke"
        elif any("溢出" in t for t in alert_types):
            status = "overflow"

        record = AlertRecord(
            record_uid=f"v{task_id[:7]}",
            status=status,
            alert_types=alert_types,
            total_detections=int(stats.get("total_detections", 0) or 0),
            alert_count=total_alerts,
            result_image_path=f"video_task:{task_id}",
            source="video",
        )
        db.add(record)
        self._commit_with_retry(db)
        db.refresh(record)
        return record

    def get_alert_detail(self, db: Session, record_uid: str) -> dict | None:
        record = db.query(AlertRecord).filter(AlertRecord.record_uid == record_uid).first()
        if record is None:
            return None

        payload: dict = {
            "id": record.record_uid,
            "source": record.source,
            "status": record.status,
            "types": record.alert_types or [],
            "alert_count": int(record.alert_count or 0),
            "total_detections": int(record.total_detections or 0),
            "time": record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if record.source != "video":
            image_b64 = self.get_alert_image_base64(db, record_uid)
            payload["detail_type"] = "image"
            payload["image"] = image_b64
            return payload

        task_id = ""
        token = record.result_image_path or ""
        if token.startswith("video_task:"):
            task_id = token.split(":", 1)[1].strip()
        elif record.record_uid.startswith("v") and len(record.record_uid) > 1:
            prefix = record.record_uid[1:]
            row = (
                db.query(VideoTaskRecord)
                .filter(VideoTaskRecord.task_id.like(f"{prefix}%"))
                .order_by(VideoTaskRecord.updated_at.desc())
                .first()
            )
            if row is not None:
                task_id = row.task_id

        task = None
        if task_id:
            task = db.query(VideoTaskRecord).filter(VideoTaskRecord.task_id == task_id).first()

        result_video = None
        suppressed_alerts = 0
        video_info = ""
        stats = {
            "total_frames": 0,
            "detected_frames": 0,
            "total_detections": 0,
            "total_alerts": int(record.alert_count or 0),
            "suppressed_alerts": 0,
            "video_info": "",
        }
        if task is not None:
            result_video = self.build_output_path_payload(task.output_path)
            video_info = task.video_info or ""
            suppressed_alerts = self.parse_suppressed_alerts(video_info)
            stats = {
                "total_frames": int(task.total_frames or 0),
                "detected_frames": int(task.detected_frames or 0),
                "total_detections": int(task.total_detections or 0),
                "total_alerts": int(task.total_alerts or 0),
                "suppressed_alerts": suppressed_alerts,
                "video_info": video_info,
            }

        payload["detail_type"] = "video"
        payload["task_id"] = task_id or None
        payload["result_video"] = result_video
        payload["stats"] = stats
        return payload

    def list_alerts(self, db: Session, page: int, per_page: int, status: str) -> tuple[int, list[AlertRecord]]:
        query = db.query(AlertRecord).order_by(AlertRecord.created_at.desc())
        if status == "warning":
            query = query.filter(AlertRecord.status != "normal")
        elif status != "all":
            query = query.filter(AlertRecord.status == status)

        total = query.count()
        records = (
            query.options(selectinload(AlertRecord.detections))
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        return total, records

    def get_alert_image_base64(self, db: Session, record_uid: str) -> str | None:
        record = db.query(AlertRecord).filter(AlertRecord.record_uid == record_uid).first()
        if record is None or not record.result_image_path:
            return None

        image_path = self.uploads_dir / record.result_image_path
        if not image_path.exists():
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            return None
        return frame_to_base64(image)

    def build_statistics(self, db: Session, started_at: str) -> dict:
        total_detections = db.query(func.count(DetectionRecord.id)).scalar() or 0
        total_alerts = (
            db.query(func.coalesce(func.sum(AlertRecord.alert_count), 0))
            .filter(AlertRecord.status != "normal")
            .scalar()
            or 0
        )
        alert_record_count = db.query(func.count(AlertRecord.id)).scalar() or 0

        today_start = datetime.combine(date.today(), datetime.min.time())
        today_alerts = (
            db.query(func.count(AlertRecord.id))
            .filter(AlertRecord.created_at >= today_start)
            .filter(AlertRecord.status != "normal")
            .scalar()
            or 0
        )

        hourly_alerts = [0] * 24
        alert_rows = (
            db.query(AlertRecord.created_at)
            .filter(AlertRecord.status != "normal")
            .all()
        )
        for (created_at,) in alert_rows:
            hourly_alerts[created_at.hour] += 1

        counter = Counter()
        class_rows = db.query(DetectionRecord.class_id, func.count(DetectionRecord.id)).group_by(DetectionRecord.class_id).all()
        for class_id, count in class_rows:
            counter[class_id] = count

        class_stats = [
            {
                "class_id": class_id,
                "class_name": ALL_CLASSES[class_id]["name"],
                "count": count,
                "is_alert": ALL_CLASSES[class_id]["alert"],
            }
            for class_id, count in counter.items()
        ]
        class_stats.sort(key=lambda item: item["count"], reverse=True)

        return {
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "today_alerts": today_alerts,
            "hourly_alerts": hourly_alerts,
            "class_stats": class_stats,
            "start_time": started_at,
            "alert_record_count": alert_record_count,
        }

    def upsert_video_task(
        self,
        db: Session,
        task_id: str,
        input_filename: str,
        input_path: str,
        status: str,
        message: str = "",
    ) -> VideoTaskRecord:
        record = db.query(VideoTaskRecord).filter(VideoTaskRecord.task_id == task_id).first()
        if record is None:
            record = VideoTaskRecord(
                task_id=task_id,
                input_filename=input_filename,
                input_path=input_path,
                status=status,
                message=message,
            )
            db.add(record)
        else:
            record.status = status
            record.message = message
        self._commit_with_retry(db)
        db.refresh(record)
        return record

    def update_video_task(self, db: Session, task_id: str, **kwargs) -> VideoTaskRecord | None:
        record = db.query(VideoTaskRecord).filter(VideoTaskRecord.task_id == task_id).first()
        if record is None:
            return None
        for key, value in kwargs.items():
            setattr(record, key, value)
        self._commit_with_retry(db)
        db.refresh(record)
        return record

    def get_video_task(self, db: Session, task_id: str) -> VideoTaskRecord | None:
        return db.query(VideoTaskRecord).filter(VideoTaskRecord.task_id == task_id).first()

    def get_video_alert_types(self, db: Session, task_id: str) -> list[str]:
        token = f"video_task:{task_id}"
        record = (
            db.query(AlertRecord)
            .filter(AlertRecord.source == "video")
            .filter(AlertRecord.result_image_path == token)
            .order_by(AlertRecord.created_at.desc())
            .first()
        )
        if record is None:
            return []
        return [str(x) for x in (record.alert_types or [])]

    def list_classes(self) -> dict:
        classes = [
            {
                "id": class_id,
                "name": info["name"],
                "en": info["en"],
                "alert": info["alert"],
                "icon": info.get("icon", ""),
            }
            for class_id, info in ALL_CLASSES.items()
        ]
        return {"classes": classes, "bin_types": BIN_TYPES}
