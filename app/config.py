from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RUNTIME_DIR = PROJECT_DIR.parent / "HuaLi_garbage_runtime"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "社区垃圾与火情识别预警系统"
    app_version: str = "2.0.0"
    debug: bool = False
    api_prefix: str = "/api"

    database_url: str = Field(
        default=f"sqlite:///{(PROJECT_DIR / 'garbage_system.db').as_posix()}",
    )
    redis_url: str = "redis://localhost:6379/0"
    celery_task_always_eager: bool = False

    max_upload_size_mb: int = 200
    video_default_skip_frames: int = 1

    models_dir: Path = BASE_DIR / "models"
    uploads_dir: Path = RUNTIME_DIR / "uploads"
    templates_dir: Path = BASE_DIR / "templates"

    garbage_pt_model: Path = BASE_DIR / "models" / "garbege.pt"
    fire_pt_model: Path = BASE_DIR / "models" / "only_fire.pt"
    smoke_pt_model: Path = BASE_DIR / "models" / "fire_smoke.pt"

    garbage_onnx_model: Path = BASE_DIR / "models" / "garbege.onnx"
    fire_onnx_model: Path = BASE_DIR / "models" / "only_fire.onnx"
    smoke_onnx_model: Path = BASE_DIR / "models" / "fire_smoke.onnx"
    bin_color_resnet18_model: Path = BASE_DIR / "models" / "bin_color_resnet18.pt"
    bin_color_min_confidence: float = 0.4
    smoke_model_include_fire: bool = True
    smoke_model_fire_class_id: int = 1
    smoke_model_smoke_class_id: int = 0

    garbage_int8_onnx_model: Path = BASE_DIR / "models" / "garbage.int8.onnx"
    fire_int8_onnx_model: Path = BASE_DIR / "models" / "only_fire.int8.onnx"
    smoke_int8_onnx_model: Path = BASE_DIR / "models" / "fire_smoke.int8.onnx"

    prefer_onnx_gpu: bool = True
    onnx_gpu_device_id: int = 0
    adaptive_skip_min: int = 1
    adaptive_skip_max: int = 12
    video_micro_batch_size: int = 4
    video_micro_batch_size_max: int = 16

    default_conf_threshold: float = 0.5
    garbage_bin_conf_threshold: float = 0.4
    fire_conf_threshold: float = 0.15
    smoke_conf_threshold: float = 0.30
    default_iou_threshold: float = 0.3
    adaptive_conf_floor: float = 0.35
    adaptive_conf_ceiling: float = 0.7

    kalman_process_noise: float = 1e-2
    kalman_measurement_noise: float = 1e-1
    kalman_error_cov_post: float = 1.0

    @model_validator(mode="after")
    def _validate_kalman_params(self) -> "Settings":
        if self.kalman_process_noise <= 0:
            raise ValueError("kalman_process_noise must be positive")
        if self.kalman_measurement_noise <= 0:
            raise ValueError("kalman_measurement_noise must be positive")
        if self.kalman_error_cov_post <= 0:
            raise ValueError("kalman_error_cov_post must be positive")
        return self

    @model_validator(mode="after")
    def _normalize_legacy_model_paths(self) -> "Settings":
        # Keep runtime uploads outside the repository so uvicorn --reload does
        # not restart the server whenever video input/output files are written.
        # If a legacy in-repo uploads directory still exists, do not auto-fallback
        # to it; users can opt into a custom UPLOADS_DIR explicitly if needed.

        # Historical compatibility: some local environments still use the old
        # misspelled INT8 filename `garbege.int8.onnx`. Prefer the corrected
        # `garbage.int8.onnx`, but transparently fall back when only the legacy
        # file exists.
        legacy_garbage_int8 = self.models_dir / "garbege.int8.onnx"
        if not self.garbage_int8_onnx_model.exists() and legacy_garbage_int8.exists():
            self.garbage_int8_onnx_model = legacy_garbage_int8
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


