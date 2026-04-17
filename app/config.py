from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "垃圾分类检测系统"
    app_version: str = "2.0.0"
    debug: bool = False
    api_prefix: str = "/api"

    database_url: str = Field(
        default=f"sqlite:///{(PROJECT_DIR / 'garbage_system.db').as_posix()}",
    )
    redis_url: str = "redis://localhost:6379/0"
    celery_task_always_eager: bool = False

    max_upload_size_mb: int = 200
    video_default_skip_frames: int = 3

    models_dir: Path = BASE_DIR / "models"
    uploads_dir: Path = BASE_DIR / "uploads"
    templates_dir: Path = BASE_DIR / "templates"

    garbage_pt_model: Path = BASE_DIR / "models" / "garbage_yolov8.pt"
    fire_pt_model: Path = BASE_DIR / "models" / "fire_yolov8.pt"
    smoke_pt_model: Path = BASE_DIR / "models" / "smoke_yolov8.pt"

    garbage_onnx_model: Path = BASE_DIR / "models" / "garbage_yolov8.onnx"
    fire_onnx_model: Path = BASE_DIR / "models" / "fire_yolov8.onnx"
    smoke_onnx_model: Path = BASE_DIR / "models" / "smoke_yolov8.onnx"

    default_conf_threshold: float = 0.5
    default_iou_threshold: float = 0.3


@lru_cache
def get_settings() -> Settings:
    return Settings()

