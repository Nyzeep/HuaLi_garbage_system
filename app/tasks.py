from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from celery.utils.log import get_task_logger

from app.bootstrap import bootstrap_application
from app.celery_app import celery_app
from app.config import get_settings
from app.database import SessionLocal
from app.dependencies import get_video_processing_service
from app.services.record_service import RecordService


logger = get_task_logger(__name__)
settings = get_settings()


def run_video_task(
    task_id: str,
    input_path: str,
    skip_frames: int = 1,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    bootstrap_application()
    db = SessionLocal()
    record_service = RecordService(settings.uploads_dir)
    video_service = get_video_processing_service()

    input_file = Path(input_path)
    output_dir = settings.uploads_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_file.stem}_detected.mp4"

    def update_progress(current_frame: int, total_frames: int) -> None:
        progress = int((current_frame / max(total_frames, 1)) * 100)
        record_service.update_video_task(
            db,
            task_id,
            status="processing",
            progress=progress,
            message=f"视频处理中 {progress}%",
        )
        if progress_callback is not None:
            progress_callback(progress)

    try:
        record_service.update_video_task(
            db,
            task_id,
            status="processing",
            progress=5,
            message="视频任务开始处理",
        )

        stats = video_service.process_video(
            input_path=input_file,
            output_path=output_path,
            skip_frames=max(skip_frames, 1),
            progress_callback=update_progress,
        )

        record_service.update_video_task(
            db,
            task_id,
            status="completed",
            progress=100,
            message="视频处理完成",
            output_path=output_path.as_posix(),
            total_frames=stats["total_frames"],
            detected_frames=stats["detected_frames"],
            total_detections=stats["total_detections"],
            total_alerts=stats["total_alerts"],
            video_info=stats["video_info"],
            error_detail=None,
        )

        if input_file.exists():
            os.remove(input_file)

        return {"task_id": task_id, "status": "completed", "result_video": output_path.name, "stats": stats}
    except Exception as exc:
        logger.exception("video task failed: %s", task_id)
        record_service.update_video_task(
            db,
            task_id,
            status="failed",
            progress=0,
            message="视频处理失败",
            error_detail=str(exc),
        )
        raise
    finally:
        db.close()


@celery_app.task(name="app.tasks.process_video_task", bind=True)
def process_video_task(self, input_path: str, skip_frames: int = 1) -> dict:
    task_id = self.request.id
    return run_video_task(
        task_id=task_id,
        input_path=input_path,
        skip_frames=skip_frames,
        progress_callback=lambda progress: self.update_state(state="PROGRESS", meta={"progress": progress}),
    )
