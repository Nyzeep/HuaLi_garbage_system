from __future__ import annotations

import warnings

from app.core.geometry import compute_iou
from app.infrastructure.ml.backends import InferenceBackend, OnnxYoloBackend, Prediction as RawPrediction, UltralyticsBackend
from app.services.inference_service import InferenceService


warnings.warn(
    "app.services.inference is deprecated; use app.infrastructure.ml.backends and app.services.inference_service instead",
    DeprecationWarning,
    stacklevel=2,
)


def non_max_suppression(predictions: list[RawPrediction], iou_threshold: float) -> list[RawPrediction]:
    sorted_preds = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    kept: list[RawPrediction] = []
    while sorted_preds:
        current = sorted_preds.pop(0)
        kept.append(current)
        sorted_preds = [
            candidate
            for candidate in sorted_preds
            if candidate.class_id != current.class_id or compute_iou(candidate.bbox, current.bbox) < iou_threshold
        ]
    return kept


__all__ = [
    "InferenceBackend",
    "InferenceService",
    "OnnxYoloBackend",
    "RawPrediction",
    "UltralyticsBackend",
    "non_max_suppression",
]
