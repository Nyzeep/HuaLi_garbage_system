from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from app.core.constants import ALL_CLASSES
from app.infrastructure.ml.model_registry import ModelRegistry


logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
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
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "inference completed models=%d detections=%d duration_ms=%.2f",
            len(bundles),
            len(results),
            duration_ms,
        )
        return results

    def _run_bundle(self, bundle, image: np.ndarray) -> list[dict]:
        started_at = time.perf_counter()
        predictions = bundle.backend.predict(
            image=image,
            conf_threshold=0.5,
            iou_threshold=0.3,
        )
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
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "model inference completed model=%s predictions=%d duration_ms=%.2f",
            bundle.descriptor.key,
            len(out),
            duration_ms,
        )
        return out
