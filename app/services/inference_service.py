from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from app.config import get_settings
from app.constants import ALL_CLASSES
from app.infrastructure.ml.model_registry import ModelRegistry
from app.infrastructure.ml.rust_bridge import RustBridge


logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._settings = get_settings()
        self._rust_bridge = RustBridge()
        # Persistent executor — avoid per-call thread pool creation overhead.
        self._executor: ThreadPoolExecutor | None = None

    def _get_executor(self, n_workers: int) -> ThreadPoolExecutor:
        if self._executor is None or self._executor._max_workers != n_workers:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=n_workers)
        return self._executor

    def detect(self, image: np.ndarray) -> list[dict]:
        started_at = time.perf_counter()
        bundles = [
            b for b in self.registry.items()
            if b.backend is not None and b.backend.loaded
        ]
        if not bundles:
            logger.warning("inference skipped loaded_models=0")
            return []
        if len(bundles) == 1:
            results = self._run_bundle(bundles[0], image)
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "inference completed models=%d detections=%d duration_ms=%.2f",
                len(bundles),
                len(results),
                duration_ms,
            )
            return results

        results: list[dict] = []
        executor = self._get_executor(len(bundles))
        futures = {
            executor.submit(self._run_bundle, bundle, image): bundle
            for bundle in bundles
        }
        for future in as_completed(futures):
            results.extend(future.result())
        results = self._dedupe_cross_model_results(results)
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "inference completed models=%d detections=%d duration_ms=%.2f",
            len(bundles),
            len(results),
            duration_ms,
        )
        return results

    def _adaptive_conf_threshold(self, model_key: str) -> float:
        base = self._settings.garbage_bin_conf_threshold if model_key == "garbage" else self._settings.default_conf_threshold
        return max(self._settings.adaptive_conf_floor, min(base, self._settings.adaptive_conf_ceiling))

    def _dedupe_cross_model_results(self, detections: list[dict]) -> list[dict]:
        if len(detections) < 2:
            return detections
        order = sorted(range(len(detections)), key=lambda idx: detections[idx]["confidence"], reverse=True)
        scored_boxes = [
            {"bbox": detections[idx]["bbox"], "score": float(detections[idx]["confidence"])}
            for idx in order
        ]
        try:
            kept_boxes = self._rust_bridge.non_max_suppression(scored_boxes, self._settings.default_iou_threshold)
        except Exception as exc:
            logger.warning("Rust NMS failed, falling back to original order", exc_info=exc)
            kept_boxes = None
        if not kept_boxes:
            return [detections[idx] for idx in order]
        kept_counts: dict[tuple[tuple[int, ...], float], int] = {}
        for item in kept_boxes:
            key = (tuple(item["bbox"]), float(item["score"]))
            kept_counts[key] = kept_counts.get(key, 0) + 1
        deduped: list[dict] = []
        for idx in order:
            key = (tuple(detections[idx]["bbox"]), float(detections[idx]["confidence"]))
            if kept_counts.get(key, 0) > 0:
                deduped.append(detections[idx])
                kept_counts[key] -= 1
        return deduped

    def detect_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        if not images:
            return []
        bundles = [
            b for b in self.registry.items()
            if b.backend is not None and b.backend.loaded
        ]
        if not bundles:
            logger.warning("batch inference skipped loaded_models=0 batch_size=%d", len(images))
            return [[] for _ in images]

        logger.info("batch inference started batch_size=%d models=%d", len(images), len(bundles))
        per_image_results: list[list[dict]] = [[] for _ in images]
        for bundle in bundles:
            conf_threshold = self._adaptive_conf_threshold(bundle.descriptor.key)
            bundle_started_at = time.perf_counter()
            logger.info(
                "batch model inference started model=%s batch_size=%d conf_threshold=%.2f",
                bundle.descriptor.key,
                len(images),
                conf_threshold,
            )
            try:
                batch_predictions = bundle.backend.predict_batch(
                    images=images,
                    conf_threshold=conf_threshold,
                    iou_threshold=self._settings.default_iou_threshold,
                )
            except Exception:
                logger.exception("batch model inference failed model=%s; fallback=per-frame", bundle.descriptor.key)
                batch_predictions = [
                    bundle.backend.predict(
                        image=image,
                        conf_threshold=conf_threshold,
                        iou_threshold=self._settings.default_iou_threshold,
                    )
                    for image in images
                ]
            logger.info(
                "batch model inference completed model=%s batch_size=%d duration_ms=%.2f",
                bundle.descriptor.key,
                len(images),
                (time.perf_counter() - bundle_started_at) * 1000,
            )
            for image_index, predictions in enumerate(batch_predictions):
                per_image_results[image_index].extend(self._map_predictions(bundle, predictions))

        return [self._dedupe_cross_model_results(result) for result in per_image_results]

    def _map_predictions(self, bundle, predictions) -> list[dict]:
        out: list[dict] = []
        for pred in predictions:
            class_id = bundle.descriptor.class_mapping.get(pred.class_id, pred.class_id)
            info = ALL_CLASSES.get(class_id, {})
            out.append(
                {
                    "class_id": class_id,
                    "class_name": info.get("name", "unknown"),
                    "confidence": pred.confidence,
                    "bbox": pred.bbox,
                    "alert": info.get("alert", False),
                    "color": info.get("color", (0, 255, 0)),
                    "icon": info.get("icon", ""),
                    "source_model": bundle.descriptor.key,
                }
            )
        return out

    def _run_bundle(self, bundle, image: np.ndarray) -> list[dict]:
        started_at = time.perf_counter()
        predictions = bundle.backend.predict(
            image=image,
            conf_threshold=self._adaptive_conf_threshold(bundle.descriptor.key),
            iou_threshold=self._settings.default_iou_threshold,
        )
        out = self._map_predictions(bundle, predictions)
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "model inference completed model=%s predictions=%d duration_ms=%.2f",
            bundle.descriptor.key,
            len(out),
            duration_ms,
        )
        return out
