from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.constants import BIN_TYPES
from app.services.alert_policy_service import AlertPolicyService
from app.services.bin_color_service import ResNet18BinColorService
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
    bin_color_service: ResNet18BinColorService | None = None
    bin_color_min_confidence: float = 0.4


class DetectionService:
    def __init__(self, deps: DetectionServiceDeps):
        self.inference_service = deps.inference_service
        self.scene_service = deps.scene_service
        self.alert_policy_service = deps.alert_policy_service
        self.rendering_service = deps.rendering_service
        self.bin_color_service = deps.bin_color_service
        self.bin_color_min_confidence = deps.bin_color_min_confidence

    @staticmethod
    def _enhance_image(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)

    @property
    def models_loaded(self) -> dict[str, bool]:
        registry = self.inference_service.registry
        loaded = registry.loaded_map()
        if self.bin_color_service is not None:
            loaded["bin_color"] = bool(self.bin_color_service.loaded)
        return loaded

    def detect(self, image: np.ndarray) -> list[dict]:
        enhanced = self._enhance_image(image)
        detections = self.inference_service.detect(enhanced)
        detections = self._attach_bin_color(image, detections)
        detections = self._attach_alert_bin_context(detections)
        detections = self.alert_policy_service.apply_cooldown(detections)
        return detections

    def detect_raw(self, image: np.ndarray) -> list[dict]:
        """Inference only — no global cooldown. Use this in the video pipeline.

        Keep the video path lightweight: skip CLAHE/denoise and per-frame bin
        color classification here to avoid adding fixed CPU cost to every
        keyframe. The primary video goal is timely anomaly detection rather
        than live bin subtype refresh."""
        detections = self.inference_service.detect(image)
        return self._attach_alert_bin_context(detections)

    def detect_raw_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        batch = self.inference_service.detect_batch(images)
        return [self._attach_alert_bin_context(detections) for detections in batch]

    def _attach_bin_color(self, image: np.ndarray, detections: list[dict]) -> list[dict]:
        if self.bin_color_service is None or not self.bin_color_service.loaded:
            return detections

        h, w = image.shape[:2]
        fused: list[dict] = []
        for det in detections:
            if det.get("class_id") != 0:
                fused.append(det)
                continue

            x1, y1, x2, y2 = map(int, det.get("bbox", [0, 0, 0, 0]))
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                fused.append(det)
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                fused.append(det)
                continue
            pred = self.bin_color_service.predict(crop)
            det2 = det.copy()
            if pred is not None and pred.confidence >= self.bin_color_min_confidence:
                color_label = str(pred.label).lower()
                normalized_key = "other" if color_label == "gray" else color_label
                type_info = {
                    "key": normalized_key if normalized_key in BIN_TYPES else "other_misc",
                    "name": BIN_TYPES.get(normalized_key, BIN_TYPES["other_misc"])["name"],
                }
                det2["bin_color"] = color_label
                det2["bin_color_confidence"] = round(pred.confidence, 3)
                det2["bin_type_key"] = type_info["key"]
                det2["bin_type_name"] = type_info["name"]
                det2["class_name"] = type_info["name"]
            fused.append(det2)
        return fused

    @staticmethod
    def _bbox_center(bbox: list[int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _attach_alert_bin_context(self, detections: list[dict]) -> list[dict]:
        bins = [d for d in detections if d.get("class_id") == 0 and d.get("bin_type_name")]
        if not bins:
            return detections

        updated: list[dict] = []
        for det in detections:
            cid = det.get("class_id")
            if cid not in (1, 2):
                updated.append(det)
                continue

            cx, cy = self._bbox_center(det["bbox"])
            nearest_bin = min(
                bins,
                key=lambda b: (self._bbox_center(b["bbox"])[0] - cx) ** 2 + (self._bbox_center(b["bbox"])[1] - cy) ** 2,
            )
            bin_type_name = nearest_bin.get("bin_type_name", "垃圾桶")
            bin_type_key = nearest_bin.get("bin_type_key", "other_misc")

            det2 = det.copy()
            det2["related_bin_type_name"] = bin_type_name
            det2["related_bin_type_key"] = bin_type_key
            if cid == 1:
                det2["class_name"] = f"{bin_type_name}溢出"
            else:
                det2["class_name"] = f"{bin_type_name}附近散落垃圾"
            updated.append(det2)
        return updated

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
