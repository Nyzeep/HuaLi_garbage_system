from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.config import get_settings
from app.infrastructure.ml.rust_bridge import RustBridge

logger = logging.getLogger(__name__)


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

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[list[Prediction]]:
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


def _preprocess_batch(images: list[np.ndarray], input_size: int = 640) -> np.ndarray:
    tensors = [_preprocess(image, input_size)[0] for image in images]
    return np.stack(tensors, axis=0)


def _postprocess(
    output: np.ndarray,
    orig_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    input_size: int = 640,
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
        self._supports_batch = False
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
            batch_dim = input_shape[0] if len(input_shape) > 0 else 1
            self._supports_batch = not isinstance(batch_dim, int) or batch_dim != 1
            # Shape is (batch, 3, H, W) — use H if it's a fixed int
            if isinstance(input_shape[2], int) and input_shape[2] > 0:
                self._input_size = input_shape[2]
            self.loaded = True
        except Exception:
            pass

    def _run_with_iobinding(self, tensor: np.ndarray) -> np.ndarray | None:
        """Run inference using ONNX Runtime IO binding for zero-copy I/O.
        
        Returns None on failure so callers can fall back to standard inference.
        """
        if self._session is None:
            return None
        io_binding = None
        try:
            io_binding = self._session.io_binding()
            io_binding.bind_cpu_input(self._input_name, tensor)
            output_name = self._session.get_outputs()[0].name
            io_binding.bind_output(output_name)
            self._session.run_with_iobinding(io_binding)
            outputs = io_binding.copy_outputs_to_cpu()
            if not outputs:
                logger.warning("io_binding returned empty outputs for %s", self._input_name)
                return None
            return outputs[0]
        except Exception as exc:
            logger.warning("io_binding inference failed for %s, falling back: %s", self._input_name, exc)
            return None
        finally:
            if io_binding is not None:
                try:
                    io_binding.synchronize_inputs()
                    io_binding.synchronize_outputs()
                except Exception:
                    pass

    def _predict_single(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Prediction]:
        if self._session is None:
            return []
        tensor = _preprocess(image, self._input_size)
        output = self._run_with_iobinding(tensor)
        if output is None:
            try:
                outputs = self._session.run(None, {self._input_name: tensor})
                if not outputs:
                    return []
                output = outputs[0]
            except Exception as exc:
                logger.error("Standard inference also failed for %s: %s", self._input_name, exc)
                return []
        return _postprocess(output, image.shape[:2], conf_threshold, iou_threshold, self._input_size)

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Prediction]:
        return self._predict_single(image, conf_threshold, iou_threshold)

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[list[Prediction]]:
        if self._session is None or not images:
            return [[] for _ in images]
        if not self._supports_batch:
            return [
                self._predict_single(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                for image in images
            ]
        tensor = _preprocess_batch(images, self._input_size)
        output = self._run_with_iobinding(tensor)
        if output is None:
            outputs = self._session.run(None, {self._input_name: tensor})
            output = outputs[0]
        if not isinstance(output, np.ndarray) or output.shape[0] != len(images):
            return [
                self._predict_single(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                for image in images
            ]
        return [
            _postprocess(output[index:index + 1], image.shape[:2], conf_threshold, iou_threshold, self._input_size)
            for index, image in enumerate(images)
        ]


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
        return self.predict_batch([image], conf_threshold, iou_threshold)[0]

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[list[Prediction]]:
        if self._model is None:
            return [[] for _ in images]
        outputs = self._model(images, conf=conf_threshold, iou=iou_threshold, verbose=False)
        all_predictions: list[list[Prediction]] = []
        for result in outputs:
            predictions: list[Prediction] = []
            for box in result.boxes:
                predictions.append(Prediction(
                    class_id=int(box.cls[0]),
                    confidence=round(float(box.conf[0]), 3),
                    bbox=list(map(int, box.xyxy[0])),
                ))
            all_predictions.append(predictions)
        return all_predictions
