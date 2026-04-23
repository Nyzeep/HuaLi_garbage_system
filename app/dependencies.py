from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings
from app.infrastructure.ml.backends import OnnxYoloBackend, UltralyticsBackend
from app.infrastructure.ml.model_registry import ModelDescriptor, ModelRegistry
from app.infrastructure.ml.rust_bridge import RustBridge
from app.services.alert_policy_service import AlertPolicyService
from app.services.detection_service import DetectionService, DetectionServiceDeps
from app.services.inference_service import InferenceService
from app.services.rendering_service import RenderingService
from app.services.scene_service import SceneService
from app.services.video_service import VideoProcessingService


logger = logging.getLogger(__name__)


def _build_backend(model_key, onnx_path, pt_path):
    onnx_backend = OnnxYoloBackend(onnx_path)
    if onnx_backend.loaded:
        logger.info("model backend loaded model=%s backend=onnx path=%s", model_key, onnx_path)
        return onnx_backend

    pt_backend = UltralyticsBackend(pt_path)
    if pt_backend.loaded:
        logger.info("model backend loaded model=%s backend=pt path=%s", model_key, pt_path)
        return pt_backend

    logger.warning(
        "model backend unavailable model=%s onnx_exists=%s pt_exists=%s",
        model_key,
        onnx_path.exists(),
        pt_path.exists(),
    )
    return None


@lru_cache
def get_rust_bridge() -> RustBridge:
    """Singleton RustBridge — always uses PyO3 in-process calls."""
    return RustBridge()


@lru_cache
def get_detection_service() -> DetectionService:
    settings = get_settings()

    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            key="garbage",
            onnx_path=settings.garbage_onnx_model,
            pt_path=settings.garbage_pt_model,
            class_mapping={0: 2, 1: 0, 2: 1},
        ),
        _build_backend("garbage", settings.garbage_onnx_model, settings.garbage_pt_model),
    )
    registry.register(
        ModelDescriptor(
            key="fire",
            onnx_path=settings.fire_onnx_model,
            pt_path=settings.fire_pt_model,
            class_mapping={0: 3},
        ),
        _build_backend("fire", settings.fire_onnx_model, settings.fire_pt_model),
    )

    inference_service = InferenceService(registry)
    scene_service = SceneService()
    alert_policy_service = AlertPolicyService()
    rendering_service = RenderingService()
    return DetectionService(
        DetectionServiceDeps(
            inference_service=inference_service,
            scene_service=scene_service,
            alert_policy_service=alert_policy_service,
            rendering_service=rendering_service,
        )
    )


@lru_cache
def get_video_processing_service() -> VideoProcessingService:
    return VideoProcessingService(
        detection_service=get_detection_service(),
        rust_bridge=get_rust_bridge(),
    )
