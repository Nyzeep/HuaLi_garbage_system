from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.config import get_settings
from app.infrastructure.ml.rust_bridge import RustBridge


@dataclass
class Prediction:
    class_id: int
    confidence: float
    bbox: list[int]


class InferenceBackend(Protocol):
    loaded: bool

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Prediction]:
        ...


class BackendLoadError(RuntimeError):
    pass


class BaseBackend:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.loaded = False


def _onnx_execution_providers() -> list[tuple[str, dict] | str]:
    settings = get_settings()
    providers: list[tuple[str, dict] | str] = []
    if settings.prefer_onnx_gpu:
        providers.extend(
            [
                ("CUDAExecutionProvider", {"device_id": settings.onnx_gpu_device_id}),
                ("DmlExecutionProvider", {"device_id": settings.onnx_gpu_device_id}),
            ]
        )
    providers.append("CPUExecutionProvider")
    return providers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _letterbox(
    image: np.ndarray,
    target: int = 640,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image to square with letterbox padding. Returns (padded, scale, (pad_w, pad_h))."""
    h, w = image.shape[:2]
    scale = min(target / h, target / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_h, target - new_h - pad_h, pad_w, target - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, (pad_w, pad_h)


def _preprocess(image: np.ndarray, input_size: int = 640) -> np.ndarray:
    """BGR image → float32 NCHW tensor normalised to [0, 1]."""
    padded, _, _ = _letterbox(image, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis]  # (1, 3, H, W)


def _postprocess(
    output: np.ndarray,
    orig_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    input_size: int = 640,
    rust_bridge: RustBridge | None = None,
) -> list[Prediction]:
    """
    Convert raw YOLOv8 ONNX output to Prediction list.
    Handles both (1, 4+C, N) and (1, N, 4+C) layouts.
    """
    arr = output[0]  # drop batch dim
    # Normalise to (num_boxes, 4+num_classes)
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # (4+C, N) → (N, 4+C)

    num_classes = arr.shape[1] - 4
    if num_classes <= 0:
        return []

    cx, cy, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    class_scores = arr[:, 4:]
    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = confidences >= conf_threshold
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(confidences) == 0:
        return []

    orig_h, orig_w = orig_shape
    scale = min(input_size / orig_h, input_size / orig_w)
    pad_w = (input_size - int(round(orig_w * scale))) // 2
    pad_h = (input_size - int(round(orig_h * scale))) // 2

    x1 = np.clip(((cx - bw / 2) - pad_w) / scale, 0, orig_w).astype(int)
    y1 = np.clip(((cy - bh / 2) - pad_h) / scale, 0, orig_h).astype(int)
    x2 = np.clip(((cx + bw / 2) - pad_w) / scale, 0, orig_w).astype(int)
    y2 = np.clip(((cy + bh / 2) - pad_h) / scale, 0, orig_h).astype(int)

    if rust_bridge is not None:
        converted: list[list[int]] = []
        for left, top, right, bottom in zip(x1, y1, x2, y2, strict=False):
            restored = rust_bridge.invert_letterbox_bbox(
                [int(left), int(top), int(right), int(bottom)],
                scale=scale,
                pad_w=float(pad_w),
                pad_h=float(pad_h),
                original_width=orig_w,
                original_height=orig_h,
            )
            converted.append(restored or [int(left), int(top), int(right), int(bottom)])
        x1 = np.array([item[0] for item in converted], dtype=int)
        y1 = np.array([item[1] for item in converted], dtype=int)
        x2 = np.array([item[2] for item in converted], dtype=int)
        y2 = np.array([item[3] for item in converted], dtype=int)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0:
        return []

    results: list[Prediction] = []
    for i in indices.flatten():
        results.append(Prediction(
            class_id=int(class_ids[i]),
            confidence=round(float(confidences[i]), 3),
            bbox=[int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
        ))
    return results


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------

class OnnxYoloBackend(BaseBackend):
    """ONNX Runtime backend for YOLOv8 models."""

    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path)
        self._session = None
        self._input_name: str = "images"
        self._input_size: int = 640
        self._rust_bridge = RustBridge()
        self._try_load()

    def _try_load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            import onnxruntime as ort  # noqa: PLC0415
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=_onnx_execution_providers(),
            )
            self._input_name = self._session.get_inputs()[0].name
            input_shape = self._session.get_inputs()[0].shape
            # Shape is (batch, 3, H, W) — use H if it's a fixed int
            if isinstance(input_shape[2], int) and input_shape[2] > 0:
                self._input_size = input_shape[2]
            self.loaded = True
        except Exception:
            pass

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Prediction]:
        if self._session is None:
            return []
        tensor = _preprocess(image, self._input_size)
        outputs = self._session.run(None, {self._input_name: tensor})
        return _postprocess(
            outputs[0],
            image.shape[:2],
            conf_threshold,
            iou_threshold,
            self._input_size,
            rust_bridge=self._rust_bridge,
        )


class UltralyticsBackend(BaseBackend):
    """Ultralytics YOLO backend (PyTorch)."""

    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path)
        self._model = None
        self._try_load()

    def _try_load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            self._model = YOLO(str(self.model_path))
            self.loaded = True
        except Exception:
            pass

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Prediction]:
        if self._model is None:
            return []
        results = self._model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        predictions: list[Prediction] = []
        for box in results.boxes:
            predictions.append(Prediction(
                class_id=int(box.cls[0]),
                confidence=round(float(box.conf[0]), 3),
                bbox=list(map(int, box.xyxy[0])),
            ))
        return predictions
