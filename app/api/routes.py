from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.config import Settings
from app.constants import ALLOWED_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.core.exceptions import FileParseError, FileTypeError, ResourceNotFoundError
from app.core.validators import (
    validate_image_bytes,
    validate_pagination,
    validate_skip_frames,
    validate_upload_size,
)
from app.database import get_db
from app.dependencies import get_detection_service, get_rust_bridge, get_upgrade_pipeline
from app.infrastructure.ml.rust_bridge import RustBridge
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
from app.upgrade import UpgradePipeline
from app.utils import base64_to_frame


class _LazyTaskProxy:
    """Lazily resolves the Celery task object to avoid import-time side effects."""

    def __init__(self) -> None:
        self._task: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self._task is None:
            from app.tasks import process_video_task

            self._task = process_video_task
        return getattr(self._task, name)


process_video_task = _LazyTaskProxy()


def build_api_router(settings: Settings, started_at: str) -> APIRouter:
    from app.tasks import run_video_task

    router = APIRouter(prefix=settings.api_prefix)
    record_service = RecordService(settings.uploads_dir)

    def has_celery_worker() -> bool:
        if settings.celery_task_always_eager:
            return True
        try:
            return bool(celery_app.control.ping(timeout=0.8))
        except Exception:
            return False

    def start_local_video_task(task_id: str, input_path: Path, skip_frames: int) -> None:
        threading.Thread(
            target=run_video_task,
            kwargs={
                "task_id": task_id,
                "input_path": input_path.as_posix(),
                "skip_frames": skip_frames,
            },
            daemon=True,
        ).start()

    def validate_extension(filename: str, expected: set[str] | None = None) -> str:
        if "." not in filename:
            raise FileTypeError("文件缺少后缀")
        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise FileTypeError("文件格式不支持")
        if expected is not None and ext not in expected:
            raise FileTypeError("文件类型不匹配")
        return ext

    async def read_image_file(file: UploadFile) -> np.ndarray:
        validate_extension(file.filename or "", IMAGE_EXTENSIONS)
        content = await file.read()
        validate_upload_size(content, settings.max_upload_size_mb)
        return validate_image_bytes(content)

    def attach_track_ids(detections: list[dict], upgrade_pipeline: UpgradePipeline) -> list[dict]:
        """Attach track IDs to detections using the upgrade pipeline.
        
        Args:
            detections: List of detection dictionaries
            upgrade_pipeline: Injected UpgradePipeline instance from dependency
        
        Returns:
            List of detections with track_id field added
        """
        pipe_result = upgrade_pipeline.run_detections(detections)
        tracks = pipe_result.tracks
        out: list[dict] = []
        for idx, det in enumerate(detections):
            item = det.copy()
            if idx < len(tracks) and tracks[idx] is not None:
                item["track_id"] = int(tracks[idx].track_id)
            out.append(item)
        return out

    def normalize_output_path(output_path: str | None) -> str | None:
        if hasattr(record_service, "build_output_path_payload"):
            return record_service.build_output_path_payload(output_path)
        if not output_path:
            return None
        try:
            return Path(output_path).relative_to(settings.uploads_dir).as_posix()
        except ValueError:
            return Path(output_path).name

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
                    "track_id": item.get("track_id"),
                    "bin_color": item.get("bin_color"),
                    "bin_color_confidence": item.get("bin_color_confidence"),
                    "bin_type_key": item.get("bin_type_key"),
                    "bin_type_name": item.get("bin_type_name"),
                    "related_bin_type_key": item.get("related_bin_type_key"),
                    "related_bin_type_name": item.get("related_bin_type_name"),
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
        upgrade_pipeline: UpgradePipeline = Depends(get_upgrade_pipeline),
    ) -> dict:
        image = await read_image_file(file)
        # Image upload should alert on every request (no cooldown).
        detections = detection_service.detect(image)
        detections = attach_track_ids(detections, upgrade_pipeline)
        rendered = detection_service.draw_boxes(image, detections)
        payload = detection_service.build_response(image, detections, with_image=True)
        record_service.create_alert_record(db, payload["scene"], detections, rendered, source="image")
        return build_detect_response(payload)

    @router.post("/detect/base64", response_model=DetectImageResponse)
    async def detect_base64(
        request: Base64ImageRequest,
        db: Session = Depends(get_db),
        detection_service: DetectionService = Depends(get_detection_service),
        upgrade_pipeline: UpgradePipeline = Depends(get_upgrade_pipeline),
    ) -> dict:
        try:
            image = base64_to_frame(request.image)
        except Exception as exc:
            raise FileParseError("图片解析失败") from exc

        # Camera/base64 requests follow the same no-cooldown behavior as image upload.
        detections = detection_service.detect(image)
        detections = attach_track_ids(detections, upgrade_pipeline)
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
        safe_skip_frames = validate_skip_frames(skip_frames)
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

        dispatch_message = "任务已进入后台处理队列"
        if has_celery_worker():
            try:
                process_video_task.apply_async(
                    kwargs={
                        "input_path": input_path.as_posix(),
                        "skip_frames": safe_skip_frames,
                    },
                    task_id=task_id,
                )
            except Exception:
                start_local_video_task(task_id=task_id, input_path=input_path, skip_frames=safe_skip_frames)
                dispatch_message = "Celery 分发失败，已切换本地线程处理"
                record_service.update_video_task(db, task_id, message=dispatch_message)
        else:
            start_local_video_task(task_id=task_id, input_path=input_path, skip_frames=safe_skip_frames)
            dispatch_message = "未检测到 Celery worker，已切换本地线程处理"
            record_service.update_video_task(db, task_id, message=dispatch_message)

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": dispatch_message,
        }

    @router.get("/tasks/{task_id}", response_model=VideoTaskStatusResponse)
    async def get_task_status(task_id: str, db: Session = Depends(get_db)) -> dict:
        record = record_service.get_video_task(db, task_id)
        if record is None:
            raise ResourceNotFoundError("任务不存在")

        updated_at = getattr(record, "updated_at", None) or record.created_at
        elapsed_seconds = max(0, int((updated_at - record.created_at).total_seconds()))
        payload = {
            "success": True,
            "task_id": task_id,
            "status": record.status,
            "progress": record.progress,
            "message": record.message,
            "result_video": None,
            "stats": None,
            "elapsed_seconds": elapsed_seconds,
        }
        payload["result_video"] = normalize_output_path(record.output_path)
        if record.status == "completed":
            video_info = record.video_info or ""
            if hasattr(record_service, "parse_suppressed_alerts"):
                suppressed_alerts = record_service.parse_suppressed_alerts(video_info)
            else:
                match = __import__("re").search(r"suppressed=(\d+)", video_info)
                suppressed_alerts = int(match.group(1)) if match else 0
            alert_types = record_service.get_video_alert_types(db, task_id) or []
            payload["stats"] = {
                "total_frames": record.total_frames,
                "detected_frames": record.detected_frames,
                "total_detections": record.total_detections,
                "total_alerts": record.total_alerts,
                "suppressed_alerts": suppressed_alerts,
                "alert_types": alert_types,
                "video_info": video_info,
            }
        return payload
    @router.get("/alerts", response_model=AlertListResponse)
    async def get_alerts(
        page: int = 1,
        per_page: int = 20,
        status: str = "all",
        db: Session = Depends(get_db),
    ) -> dict:
        page, per_page = validate_pagination(page, per_page)
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
            raise ResourceNotFoundError("记录不存在")
        return {"image": image_b64}

    @router.get("/alerts/{record_uid}/detail")
    async def get_alert_detail(record_uid: str, db: Session = Depends(get_db)) -> dict:
        payload = record_service.get_alert_detail(db, record_uid)
        if payload is None:
            raise ResourceNotFoundError("记录不存在")
        return payload

    @router.get("/statistics", response_model=StatisticsResponse)
    async def get_statistics(db: Session = Depends(get_db)) -> dict:
        return record_service.build_statistics(db, started_at=started_at)

    @router.get("/classes")
    async def get_classes() -> dict:
        return record_service.list_classes()

    # ------------------------------------------------------------------
    # SSE: video task progress stream
    # ------------------------------------------------------------------
    @router.get("/tasks/{task_id}/stream")
    async def stream_task_status(task_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
        async def event_generator():
            last_state: tuple | None = None

            def build_payload(record) -> dict:
                payload: dict = {
                    "status": record.status,
                    "progress": record.progress,
                    "message": record.message,
                }
                if record.status == "completed":
                    result_video = None
                    if record.output_path:
                        try:
                            result_video = Path(record.output_path).relative_to(settings.uploads_dir).as_posix()
                        except ValueError:
                            result_video = Path(record.output_path).name
                    payload["result_video"] = result_video
                    payload["stats"] = {
                        "total_frames": record.total_frames,
                        "detected_frames": record.detected_frames,
                        "total_detections": record.total_detections,
                        "total_alerts": record.total_alerts,
                        "video_info": record.video_info,
                    }
                return payload

            while True:
                db.rollback()  # end stale read-tx so we see background-thread commits
                record = record_service.get_video_task(db, task_id)
                if record is None:
                    yield f"event: error\ndata: {json.dumps({'message': '任务不存在'}, ensure_ascii=False)}\n\n"
                    return
                state = (record.status, record.progress, record.message)
                payload = build_payload(record)
                if not payload.get("message") and record.status == "processing":
                    payload["message"] = "视频处理中，请稍候..."
                if state != last_state:
                    last_state = state
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if record.status in ("completed", "failed"):
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # WebSocket: real-time camera detection
    # ------------------------------------------------------------------
    @router.websocket("/ws/camera")
    async def camera_websocket(
        websocket: WebSocket,
        detection_service: DetectionService = Depends(get_detection_service),
    ) -> None:
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                image_b64 = data.get("image", "")
                try:
                    image = base64_to_frame(image_b64)
                except Exception:
                    await websocket.send_json({"error": "图片解析失败"})
                    continue
                detections = detection_service.detect(image)
                payload = detection_service.build_response(image, detections, with_image=True)
                await websocket.send_json(build_detect_response(payload))
        except WebSocketDisconnect:
            pass

    # ------------------------------------------------------------------
    # SSE: alerts auto-refresh stream
    # ------------------------------------------------------------------
    @router.get("/alerts/stream")
    async def stream_alerts(
        page: int = 1,
        per_page: int = 20,
        status: str = "all",
        db: Session = Depends(get_db),
    ) -> StreamingResponse:
        page, per_page = validate_pagination(page, per_page)

        async def event_generator():
            while True:
                db.rollback()
                total, records = record_service.list_alerts(db, page=page, per_page=per_page, status=status)
                payload = {
                    "total": total,
                    "page": page,
                    "per_page": per_page,
                    "records": [
                        {
                            "id": r.record_uid,
                            "time": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": r.status,
                            "types": r.alert_types,
                            "total": r.total_detections,
                            "alert_count": r.alert_count,
                            "source": r.source,
                        }
                        for r in records
                    ],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(10)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # SSE: statistics auto-refresh stream
    # ------------------------------------------------------------------
    @router.get("/statistics/stream")
    async def stream_statistics(db: Session = Depends(get_db)) -> StreamingResponse:
        async def event_generator():
            while True:
                db.rollback()
                data = record_service.build_statistics(db, started_at=started_at)
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(10)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/status", response_model=SystemStatusResponse)
    async def get_status(
        detection_service: DetectionService = Depends(get_detection_service),
        rust_bridge: RustBridge = Depends(get_rust_bridge),
    ) -> dict:
        models_loaded = detection_service.models_loaded
        rust_status = rust_bridge.health_check()
        detector_loaded = any(
            [
                models_loaded.get("garbage", False),
                models_loaded.get("fire", False),
                models_loaded.get("smoke", False),
            ],
        )
        return {
            "model_loaded": detector_loaded,
            "garbage_model": models_loaded.get("garbage", False),
            "fire_model": models_loaded.get("fire", False),
            "smoke_model": models_loaded.get("smoke", False),
            "bin_color_model": models_loaded.get("bin_color", False),
            "mode": "正常检测" if detector_loaded else "演示模式",
            "uptime": started_at,
            "class_count": 5,
            "version": settings.app_version,
            "name": settings.app_name,
            "rust": rust_status,
        }

    return router

