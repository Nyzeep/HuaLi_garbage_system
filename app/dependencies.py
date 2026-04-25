from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings
from app.infrastructure.ml.backends import OnnxYoloBackend, UltralyticsBackend
from app.infrastructure.ml.model_registry import ModelDescriptor, ModelRegistry
from app.infrastructure.ml.rust_bridge import RustBridge
from app.services.alert_policy_service import AlertPolicyService
from app.services.bin_color_service import ResNet18BinColorService
from app.services.detection_service import DetectionService, DetectionServiceDeps
from app.services.inference_service import InferenceService
from app.services.rendering_service import RenderingService
from app.services.scene_service import SceneService
from app.services.video_service import VideoProcessingService
from app.upgrade import AlarmEngine, DetectionEngine, TrackEngine, UpgradePipeline


logger = logging.getLogger(__name__)


def _build_backend(model_key, onnx_path, pt_path, int8_onnx_path=None):
    if int8_onnx_path is not None:
        int8_backend = OnnxYoloBackend(int8_onnx_path)
        if int8_backend.loaded:
            logger.info("model backend loaded model=%s backend=onnx-int8 path=%s", model_key, int8_onnx_path)
            return int8_backend

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
        _build_backend("garbage", settings.garbage_onnx_model, settings.garbage_pt_model, settings.garbage_int8_onnx_model),
    )
    smoke_mapping: dict[int, int] = {
        settings.smoke_model_smoke_class_id: 4,
    }
    if settings.smoke_model_include_fire:
        smoke_mapping[settings.smoke_model_fire_class_id] = 3
    registry.register(
        ModelDescriptor(
            key="smoke",
            onnx_path=settings.smoke_onnx_model,
            pt_path=settings.smoke_pt_model,
            class_mapping=smoke_mapping,
        ),
        _build_backend("smoke", settings.smoke_onnx_model, settings.smoke_pt_model, settings.smoke_int8_onnx_model),
    )
    if not settings.smoke_model_include_fire:
        registry.register(
            ModelDescriptor(
                key="fire",
                onnx_path=settings.fire_onnx_model,
                pt_path=settings.fire_pt_model,
                class_mapping={0: 3},
            ),
            _build_backend("fire", settings.fire_onnx_model, settings.fire_pt_model, settings.fire_int8_onnx_model),
        )

    inference_service = InferenceService(registry)
    scene_service = SceneService()
    alert_policy_service = AlertPolicyService()
    rendering_service = RenderingService()
    bin_color_service = ResNet18BinColorService(settings.bin_color_resnet18_model)
    return DetectionService(
        DetectionServiceDeps(
            inference_service=inference_service,
            scene_service=scene_service,
            alert_policy_service=alert_policy_service,
            rendering_service=rendering_service,
            bin_color_service=bin_color_service,
            bin_color_min_confidence=settings.bin_color_min_confidence,
        )
    )


@lru_cache
def get_video_processing_service() -> VideoProcessingService:
    return VideoProcessingService(
        detection_service=get_detection_service(),
        rust_bridge=get_rust_bridge(),
    )


@lru_cache
def get_upgrade_pipeline() -> UpgradePipeline:
    """Create UpgradePipeline with properly initialized DetectionEngine.
    
    The DetectionEngine is initialized with the real DetectionService from
    get_detection_service(), avoiding the None detector issue.
    """
    detection_service = get_detection_service()
    return UpgradePipeline(
        detection_engine=DetectionEngine(detection_service),
        track_engine=TrackEngine(),
        alarm_engine=AlarmEngine(min_consecutive_frames=2),
    )
