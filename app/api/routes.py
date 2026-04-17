from __future__ import annotations

import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import Settings
from app.constants import ALLOWED_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.database import get_db
from app.dependencies import get_detection_service
from app.schemas import (
    AlertImageResponse,
    AlertListResponse,
    Base64ImageRequest,
    DetectImageResponse,
    StatisticsResponse,
    SystemStatusResponse,
    VideoTaskCreateResponse,
    VideoTaskStatusResponse,
)
from app.services.detection_service import DetectionService
from app.services.record_service import RecordService
from app.tasks import process_video_task, run_video_task
from app.utils import base64_to_frame


def build_api_router(settings: Settings, started_at: str) -> APIRouter:
    router = APIRouter(prefix=settings.api_prefix)
    record_service = RecordService(settings.uploads_dir)

    def validate_extension(filename: str, expected: set[str] | None = None) -> str:
        if "." not in filename:
            raise HTTPException(status_code=400, detail="文件缺少后缀")
        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="文件格式不支持")
        if expected is not None and ext not in expected:
            raise HTTPException(status_code=400, detail="文件类型不匹配")
        return ext

    async def read_image_file(file: UploadFile) -> np.ndarray:
        validate_extension(file.filename or "", IMAGE_EXTENSIONS)
        content = await file.read()
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="图片解析失败")
        return image

    def build_detect_response(payload: dict) -> dict:
        return {
            "success": True,
            "detections": [
                {
                    "class_id": item["class_id"],
                    "class_name": item["class_name"],
                    "confidence": item["confidence"],
                    "bbox": item["bbox"],
                    "alert": item["alert"],
                    "icon": item.get("icon", ""),
                    "source": item.get("source_model", ""),
                }
                for item in payload["detections"]
            ],
            "scene": payload["scene"],
            "result_image": payload.get("result_image"),
        }

    @router.post("/detect/image", response_model=DetectImageResponse)
    async def detect_image(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        detection_service: DetectionService = Depends(get_detection_service),
    ) -> dict:
        image = await read_image_file(file)
        detections = detection_service.apply_cooldown(detection_service.detect(image))
        rendered = detection_service.draw_boxes(image, detections)
        payload = detection_service.build_response(image, detections, with_image=True)
        record_service.create_alert_record(db, payload["scene"], detections, rendered, source="image")
        return build_detect_response(payload)

    @router.post("/detect/base64", response_model=DetectImageResponse)
    async def detect_base64(
        request: Base64ImageRequest,
        db: Session = Depends(get_db),
        detection_service: DetectionService = Depends(get_detection_service),
    ) -> dict:
        try:
            image = base64_to_frame(request.image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="图片解析失败") from exc

        detections = detection_service.apply_cooldown(detection_service.detect(image))
        rendered = detection_service.draw_boxes(image, detections)
        payload = detection_service.build_response(image, detections, with_image=True)
        record_service.create_alert_record(db, payload["scene"], detections, rendered, source="camera")
        return build_detect_response(payload)

    @router.post("/detect/video", response_model=VideoTaskCreateResponse)
    async def detect_video(
        file: UploadFile = File(...),
        skip_frames: int = Form(default=settings.video_default_skip_frames),
        db: Session = Depends(get_db),
    ) -> dict:
        validate_extension(file.filename or "", VIDEO_EXTENSIONS)
        input_dir = settings.uploads_dir / "videos"
        input_dir.mkdir(parents=True, exist_ok=True)

        task_id = uuid.uuid4().hex
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        input_filename = f"{task_id}{suffix}"
        input_path = input_dir / input_filename

        with input_path.open("wb") as output_file:
            output_file.write(await file.read())

        record_service.upsert_video_task(
            db,
            task_id=task_id,
            input_filename=file.filename or input_filename,
            input_path=input_path.as_posix(),
            status="pending",
            message="任务已提交，等待处理",
        )

        try:
            process_video_task.apply_async(
                kwargs={
                    "input_path": input_path.as_posix(),
                    "skip_frames": max(skip_frames, 1),
                },
                task_id=task_id,
            )
        except Exception:
            threading.Thread(
                target=run_video_task,
                kwargs={
                    "task_id": task_id,
                    "input_path": input_path.as_posix(),
                    "skip_frames": max(skip_frames, 1),
                },
                daemon=True,
            ).start()
            record_service.update_video_task(
                db,
                task_id,
                message="Redis 不可用，已切换为本地后台线程处理",
            )

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": "视频任务已进入队列",
        }

    @router.get("/tasks/{task_id}", response_model=VideoTaskStatusResponse)
    async def get_task_status(task_id: str, db: Session = Depends(get_db)) -> dict:
        record = record_service.get_video_task(db, task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="任务不存在")

        payload = {
            "success": True,
            "task_id": task_id,
            "status": record.status,
            "progress": record.progress,
            "message": record.message,
            "result_video": None,
            "stats": None,
        }
        if record.output_path:
            try:
                payload["result_video"] = Path(record.output_path).relative_to(settings.uploads_dir).as_posix()
            except ValueError:
                payload["result_video"] = Path(record.output_path).name
        if record.status == "completed":
            payload["stats"] = {
                "total_frames": record.total_frames,
                "detected_frames": record.detected_frames,
                "total_detections": record.total_detections,
                "total_alerts": record.total_alerts,
                "video_info": record.video_info,
            }
        return payload

    @router.get("/alerts", response_model=AlertListResponse)
    async def get_alerts(
        page: int = 1,
        per_page: int = 20,
        status: str = "all",
        db: Session = Depends(get_db),
    ) -> dict:
        total, records = record_service.list_alerts(db, page=page, per_page=per_page, status=status)
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "records": [
                {
                    "id": record.record_uid,
                    "time": record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": record.status,
                    "types": record.alert_types,
                    "total": record.total_detections,
                    "alert_count": record.alert_count,
                    "source": record.source,
                }
                for record in records
            ],
        }

    @router.get("/alerts/{record_uid}/image", response_model=AlertImageResponse)
    async def get_alert_image(record_uid: str, db: Session = Depends(get_db)) -> dict:
        image_b64 = record_service.get_alert_image_base64(db, record_uid)
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="记录不存在")
        return {"image": image_b64}

    @router.get("/statistics", response_model=StatisticsResponse)
    async def get_statistics(db: Session = Depends(get_db)) -> dict:
        return record_service.build_statistics(db, started_at=started_at)

    @router.get("/classes")
    async def get_classes() -> dict:
        return record_service.list_classes()

    @router.get("/status", response_model=SystemStatusResponse)
    async def get_status(detection_service: DetectionService = Depends(get_detection_service)) -> dict:
        models_loaded = detection_service.models_loaded
        return {
            "model_loaded": any(models_loaded.values()),
            "garbage_model": models_loaded.get("garbage", False),
            "fire_model": models_loaded.get("fire", False),
            "smoke_model": models_loaded.get("smoke", False),
            "mode": "正常检测" if any(models_loaded.values()) else "演示模式",
            "uptime": started_at,
            "class_count": 5,
            "version": settings.app_version,
            "name": settings.app_name,
        }

    return router
