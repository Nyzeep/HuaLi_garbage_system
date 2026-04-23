from __future__ import annotations

import logging

from app.celery_app import celery_app
from app.config import get_settings
from app.database import Base, engine
from app.dependencies import get_detection_service, get_rust_bridge
from app.utils import ensure_dir


logger = logging.getLogger(__name__)
_BOOTSTRAP_LOGGED = False


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _probe_redis() -> dict:
    try:
        with celery_app.connection_for_read() as connection:
            connection.ensure_connection(max_retries=0)
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _log_startup_summary() -> None:
    global _BOOTSTRAP_LOGGED

    if _BOOTSTRAP_LOGGED:
        return

    settings = get_settings()
    detection_service = get_detection_service()
    rust_bridge = get_rust_bridge()
    rust_status = rust_bridge.health_check(force_refresh=True)
    redis_status = _probe_redis()

    registry = detection_service.inference_service.registry
    model_summaries = [
        {
            "key": bundle.descriptor.key,
            "loaded": bundle.loaded,
            "backend": type(bundle.backend).__name__ if bundle.backend is not None else None,
            "onnx_path": str(bundle.descriptor.onnx_path),
            "pt_path": str(bundle.descriptor.pt_path),
        }
        for bundle in registry.items()
    ]

    logger.info(
        "startup config app=%s version=%s debug=%s uploads_dir=%s redis_url=%s rust_mode=%s",
        settings.app_name,
        settings.app_version,
        settings.debug,
        settings.uploads_dir,
        settings.redis_url,
        rust_bridge.mode,
    )
    logger.info("startup models=%s", model_summaries)
    logger.info("startup rust_status=%s", rust_status)
    logger.info("startup redis_status=%s", redis_status)

    _BOOTSTRAP_LOGGED = True


def bootstrap_application() -> None:
    _configure_logging()

    settings = get_settings()
    ensure_dir(settings.uploads_dir)
    ensure_dir(settings.uploads_dir / "alerts")
    ensure_dir(settings.uploads_dir / "videos")
    Base.metadata.create_all(bind=engine)
    _log_startup_summary()
