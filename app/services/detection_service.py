from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.alert_policy_service import AlertPolicyService
from app.services.inference_service import InferenceService
from app.services.rendering_service import RenderingService
from app.services.scene_service import SceneService
from app.utils import frame_to_base64


@dataclass
class DetectionServiceDeps:
    inference_service: InferenceService
    scene_service: SceneService
    alert_policy_service: AlertPolicyService
    rendering_service: RenderingService


class DetectionService:
    def __init__(self, deps: DetectionServiceDeps):
        self.inference_service = deps.inference_service
        self.scene_service = deps.scene_service
        self.alert_policy_service = deps.alert_policy_service
        self.rendering_service = deps.rendering_service

    @property
    def models_loaded(self) -> dict[str, bool]:
        registry = self.inference_service.registry
        return registry.loaded_map()

    def detect(self, image: np.ndarray) -> list[dict]:
        detections = self.inference_service.detect(image)
        detections = self.alert_policy_service.apply_cooldown(detections)
        return detections

    def detect_raw(self, image: np.ndarray) -> list[dict]:
        """Inference only — no global cooldown. Use this in the video pipeline
        which has its own per-object IoU-based cooldown."""
        return self.inference_service.detect(image)

    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        return self.rendering_service.draw_boxes(image, detections)

    def analyze_scene(self, detections: list[dict]) -> dict:
        return self.scene_service.analyze(detections)

    def build_response(self, image: np.ndarray, detections: list[dict], with_image: bool = True) -> dict:
        scene = self.analyze_scene(detections)
        result_image = None
        if with_image:
            rendered = self.draw_boxes(image, detections)
            result_image = frame_to_base64(rendered)

        return {
            "scene": scene,
            "detections": detections,
            "result_image": result_image,
        }
