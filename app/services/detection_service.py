from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.constants import ALL_CLASSES
from app.services.bin_color_service import ResNet18BinColorService
from app.services.inference import InferenceBackend, OnnxYoloBackend, UltralyticsBackend
from app.utils import frame_to_base64


@dataclass
class DetectorBundle:
    key: str
    class_mapping: dict[int, int]
    backend: InferenceBackend | None = None


BIN_COLOR_TO_TYPE: dict[str, dict[str, str]] = {
    "blue": {"key": "recyclable", "name": "可回收垃圾桶"},
    "red": {"key": "hazardous", "name": "有害垃圾桶"},
    "gray": {"key": "other", "name": "其他垃圾桶"},
    "green": {"key": "kitchen", "name": "厨余垃圾桶"},
    "other": {"key": "other_misc", "name": "其他"},
}

EVENT_NAME_OVERRIDE: dict[int, str] = {
    3: "火",
    4: "烟",
}


class AlertCooldown:
    COOLDOWN_CONFIG = {
        "overflow": 15 * 60,
        "garbage": 15 * 60,
        "fire": 90,
        "smoke": 90,
    }

    def __init__(self) -> None:
        self._last_alert_time: dict[int, float] = {}

    def _get_cooldown_seconds(self, class_id: int) -> int:
        class_name = ALL_CLASSES.get(class_id, {}).get("en", "")
        return self.COOLDOWN_CONFIG.get(class_name, 15 * 60)

    def can_alert(self, class_id: int) -> bool:
        cooldown = self._get_cooldown_seconds(class_id)
        now = time.time()
        last = self._last_alert_time.get(class_id, 0.0)
        if now - last >= cooldown:
            self._last_alert_time[class_id] = now
            return True
        return False


class DetectionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cooldown = AlertCooldown()
        self.detectors = self._build_detectors()
        self.bin_color_classifier = ResNet18BinColorService(settings.bin_color_resnet18_model)

    def _select_backend(self, onnx_path: Path, pt_path: Path) -> InferenceBackend | None:
        onnx_backend = OnnxYoloBackend(onnx_path)
        if onnx_backend.loaded:
            return onnx_backend

        pt_backend = UltralyticsBackend(pt_path)
        if pt_backend.loaded:
            return pt_backend

        return None

    def _build_detectors(self) -> list[DetectorBundle]:
        detectors: list[DetectorBundle] = [
            DetectorBundle(
                key="garbage",
                # Garbage model class order: 0=garbage_bin, 1=overflow, 2=garbage
                class_mapping={0: 2, 1: 0, 2: 1},
                backend=self._select_backend(self.settings.garbage_onnx_model, self.settings.garbage_pt_model),
            ),
        ]

        smoke_backend = self._select_backend(self.settings.smoke_onnx_model, self.settings.smoke_pt_model)
        if smoke_backend is not None and smoke_backend.loaded:
            smoke_mapping: dict[int, int] = {
                self.settings.smoke_model_smoke_class_id: 4,  # smoke
            }
            if self.settings.smoke_model_include_fire:
                smoke_mapping[self.settings.smoke_model_fire_class_id] = 3  # fire
            detectors.append(
                DetectorBundle(
                    key="smoke",
                    class_mapping=smoke_mapping,
                    backend=smoke_backend,
                ),
            )
            if not self.settings.smoke_model_include_fire:
                detectors.append(
                    DetectorBundle(
                        key="fire",
                        class_mapping={0: 3},
                        backend=self._select_backend(self.settings.fire_onnx_model, self.settings.fire_pt_model),
                    ),
                )
        else:
            detectors.append(
                DetectorBundle(
                    key="fire",
                    class_mapping={0: 3},
                    backend=self._select_backend(self.settings.fire_onnx_model, self.settings.fire_pt_model),
                ),
            )

        return detectors

    def _detector_models_available(self) -> bool:
        return any(bundle.backend and bundle.backend.loaded for bundle in self.detectors)

    @property
    def models_loaded(self) -> dict[str, bool]:
        models = {bundle.key: bool(bundle.backend and bundle.backend.loaded) for bundle in self.detectors}
        models["bin_color"] = bool(self.bin_color_classifier.loaded)
        return models

    def detect(self, image: np.ndarray) -> list[dict]:
        if self._detector_models_available():
            detections = self._run_models(image)
            detections = self._attach_bin_color(image, detections)
            return self._attach_alert_bin_context(detections)
        return self._fake_detect(image)

    def _run_models(self, image: np.ndarray) -> list[dict]:
        result_list: list[dict] = []
        h, w = image.shape[:2]
        min_box_area = w * h * 0.005

        for bundle in self.detectors:
            if bundle.backend is None or not bundle.backend.loaded:
                continue

            if bundle.key == "fire":
                conf_threshold = 0.30
            elif bundle.key == "smoke":
                conf_threshold = 0.15
            else:
                conf_threshold = min(
                    self.settings.default_conf_threshold,
                    self.settings.garbage_bin_conf_threshold,
                )
            for prediction in bundle.backend.predict(
                image=image,
                conf_threshold=conf_threshold,
                iou_threshold=self.settings.default_iou_threshold,
            ):
                class_id = bundle.class_mapping.get(prediction.class_id)
                if class_id is None:
                    continue

                # Class-specific confidence:
                # garbage_bin (class_id=0) uses garbage_bin_conf_threshold,
                # other garbage-model classes keep default_conf_threshold.
                if class_id == 0:
                    if prediction.confidence < self.settings.garbage_bin_conf_threshold:
                        continue
                elif bundle.key == "garbage" and prediction.confidence < self.settings.default_conf_threshold:
                    continue

                x1, y1, x2, y2 = prediction.bbox
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue

                info = ALL_CLASSES[class_id]
                class_name = EVENT_NAME_OVERRIDE.get(class_id, info["name"])
                result_list.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(prediction.confidence, 3),
                        "bbox": [x1, y1, x2, y2],
                        "alert": info["alert"],
                        "color": info["color"],
                        "icon": info["icon"],
                        "source_model": bundle.key if bundle.backend else "unknown",
                    },
                )

        return self._suppress_garbage_when_fire_overlaps(result_list)

    @staticmethod
    def _iou(box_a: list[int], box_b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _suppress_garbage_when_fire_overlaps(self, detections: list[dict]) -> list[dict]:
        # If a scattered-garbage box heavily overlaps a fire box, prefer fire and drop garbage.
        overlap_threshold = 0.35
        fire_boxes = [d["bbox"] for d in detections if d["class_id"] == 3]
        if not fire_boxes:
            return detections

        filtered: list[dict] = []
        for det in detections:
            if det["class_id"] != 2:
                filtered.append(det)
                continue

            has_near_fire = any(self._iou(det["bbox"], fire_box) >= overlap_threshold for fire_box in fire_boxes)
            if not has_near_fire:
                filtered.append(det)

        return filtered

    def _attach_bin_color(self, image: np.ndarray, detections: list[dict]) -> list[dict]:
        if not self.bin_color_classifier.loaded:
            return detections

        h, w = image.shape[:2]
        fused: list[dict] = []
        for det in detections:
            if det["class_id"] != 0:
                fused.append(det)
                continue

            x1, y1, x2, y2 = map(int, det.get("bbox", [0, 0, 0, 0]))
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                fused.append(det)
                continue

            crop = image[y1:y2, x1:x2]
            pred = self.bin_color_classifier.predict(crop)
            det2 = det.copy()
            if pred is not None and pred.confidence >= self.settings.bin_color_min_confidence:
                color_label = str(pred.label).lower()
                type_info = BIN_COLOR_TO_TYPE.get(color_label, {"key": "other_misc", "name": "其他"})
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

    @staticmethod
    def _load_chinese_font(size: int) -> ImageFont.ImageFont:
        font_candidates = [
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf", # 黑体
            "C:/Windows/Fonts/simsun.ttc", # 宋体
        ]
        for font_path in font_candidates:
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _draw_label_text(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        bg_color_bgr: tuple[int, int, int],
    ) -> np.ndarray:
        # Use PIL to render Chinese text to avoid cv2 "????" issue.
        font = self._load_chinese_font(size=20)
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        pad_x, pad_y = 6, 4

        left = max(0, x)
        top = max(0, y - th - pad_y * 2)
        right = min(pil_img.width, left + tw + pad_x * 2)
        bottom = min(pil_img.height, top + th + pad_y * 2)

        bg_color_rgb = (int(bg_color_bgr[2]), int(bg_color_bgr[1]), int(bg_color_bgr[0]))
        draw.rectangle([left, top, right, bottom], fill=bg_color_rgb)
        draw.text((left + pad_x, top + pad_y), text, fill=(255, 255, 255), font=font)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _attach_alert_bin_context(self, detections: list[dict]) -> list[dict]:
        """
        Attach nearest bin subtype context for:
        - overflow (class_id=1): "<bin_type>溢出"
        - scattered garbage (class_id=2): "<bin_type>附近散落垃圾"
        """
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

    def _fake_detect(self, image: np.ndarray) -> list[dict]:
        h, w = image.shape[:2]
        seed_val = int(image[h // 2, w // 2, 0]) if image.ndim == 3 else 0
        rng = random.Random(seed_val + int(time.time() // 3))
        result_list: list[dict] = []
        chosen = rng.sample(list(ALL_CLASSES.keys()), rng.randint(1, 4))

        for class_id in chosen:
            info = ALL_CLASSES[class_id]
            margin = 30
            x1 = rng.randint(margin, max(margin + 1, w // 2))
            y1 = rng.randint(margin, max(margin + 1, h // 2))
            x2 = rng.randint(max(x1 + 1, w // 2), max(x1 + 2, w - margin))
            y2 = rng.randint(max(y1 + 1, h // 2), max(y1 + 2, h - margin))
            result_list.append(
                {
                    "class_id": class_id,
                    "class_name": info["name"],
                    "confidence": round(rng.uniform(0.55, 0.96), 3),
                    "bbox": [x1, y1, x2, y2],
                    "alert": info["alert"],
                    "color": info["color"],
                    "icon": info["icon"],
                    "source_model": "demo",
                },
            )

        return result_list

    def apply_cooldown(self, detections: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for detection in detections:
            if detection["alert"] and not self.cooldown.can_alert(detection["class_id"]):
                detection = detection.copy()
                detection["alert"] = False
            filtered.append(detection)
        return filtered

    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        output = image.copy()
        en_label_map = {
            0: "GarbageBin",
            1: "Overflow",
            2: "Garbage",
            3: "火",
            4: "烟",
        }


        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            box_color = detection.get("color", (0, 255, 0))
            default_name = ALL_CLASSES.get(detection["class_id"], {}).get("name", "")
            if detection.get("class_name") and detection["class_name"] != default_name:
                en_name = detection["class_name"]
            elif detection["class_id"] == 0 and detection.get("bin_type_name"):
                en_name = detection["bin_type_name"]
            else:
                en_name = en_label_map.get(detection["class_id"], detection["class_name"])
            label = f"{en_name} {detection['confidence']:.0%}"
            line_w = 3 if detection["alert"] else 2

            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, line_w)
            output = self._draw_label_text(output, label, x1, y1, box_color)
            if detection["alert"]:
                cv2.putText(
                    output,
                    "! ALERT",
                    (x1, min(output.shape[0] - 10, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )
        return output

    def analyze_scene(self, detections: list[dict]) -> dict:
        alert_list = [item for item in detections if item["alert"]]
        alert_types = list({item["class_name"] for item in alert_list})
        class_ids = {item["class_id"] for item in detections}

        status = "normal"
        if 3 in class_ids:
            status = "fire"
        elif 4 in class_ids:
            status = "smoke"
        elif 1 in class_ids:
            status = "overflow"
        elif alert_list:
            status = "warning"

        return {
            "status": status,
            "alert_count": len(alert_list),
            "alert_types": alert_types,
            "normal_count": len([item for item in detections if not item["alert"]]),
            "total": len(detections),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def build_response(self, image: np.ndarray, detections: list[dict], with_image: bool = True) -> dict:
        scene = self.analyze_scene(detections)
        result_image = None
        if with_image:
            rendered = self.draw_boxes(image, detections)
            result_image = frame_to_base64(rendered)
        return {"scene": scene, "detections": detections, "result_image": result_image}




