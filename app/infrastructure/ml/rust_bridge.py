from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

try:
    import requests
except ImportError:
    requests = None


logger = logging.getLogger(__name__)

try:
    import huali_garbage_core as _rust_native  # type: ignore[import-untyped]
except ImportError:
    _rust_native = None


@dataclass(frozen=True)
class _RustHttpConfig:
    base_url: str = "http://127.0.0.1:50051"
    timeout_seconds: float = 2.0


class RustBridge:
    """Rust core bridge with PyO3 preferred and HTTP fallback."""

    _probe_cache_ttl_seconds = 5.0

    def __init__(self, http_base_url: str | None = None, timeout_seconds: float = 2.0) -> None:
        self._http = _RustHttpConfig(
            base_url=(http_base_url or _RustHttpConfig.base_url).rstrip("/"),
            timeout_seconds=timeout_seconds,
        )
        self._mode = "pyo3" if _rust_native is not None else "http"
        self._pyo3_available: bool | None = None
        self._pyo3_probe_at: float | None = None
        logger.info("rust bridge mode=%s", self._mode)
        if _rust_native is not None:
            try:
                self._refresh_pyo3_probe(force=True)
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                # If PyO3 probe refresh raises, initialization must not fail.
                # Force HTTP fallback so the service remains available.
                self._pyo3_available = False
                self._pyo3_probe_at = time.perf_counter()
                self._mode = "http"
                logger.exception(
                    "rust pyo3 probe initialization failed at=%s error_type=%s error=%s; fallback=http",
                    datetime.now(timezone.utc).isoformat(),
                    type(exc).__name__,
                    exc,
                )
            if not self._pyo3_available:
                self._mode = "http"

    @property
    def mode(self) -> str:
        return self._mode

    def _probe_pyo3(self) -> bool:
        if _rust_native is None:
            return False
        started_at = time.perf_counter()
        try:
            result = _rust_native.iou_py((0, 0, 1, 1), (0, 0, 1, 1))
            probe_ok = isinstance(result, float)
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info("rust pyo3 probe completed ok=%s duration_ms=%.2f", probe_ok, duration_ms)
            return probe_ok
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.warning("rust pyo3 probe failed duration_ms=%.2f error=%s", duration_ms, exc)
            return False

    def _refresh_pyo3_probe(self, force: bool = False) -> bool:
        now = time.perf_counter()
        if (
            not force
            and self._pyo3_available is not None
            and self._pyo3_probe_at is not None
            and now - self._pyo3_probe_at < self._probe_cache_ttl_seconds
        ):
            return self._pyo3_available
        self._pyo3_available = self._probe_pyo3()
        self._pyo3_probe_at = now
        return self._pyo3_available

    def _can_use_pyo3(self) -> bool:
        if _rust_native is None:
            self._mode = "http"
            return False

        try:
            probe_ok = self._refresh_pyo3_probe()
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            self._pyo3_available = False
            self._pyo3_probe_at = time.perf_counter()
            self._mode = "http"
            logger.exception(
                "rust pyo3 probe refresh failed at=%s error_type=%s error=%s; fallback=http",
                datetime.now(timezone.utc).isoformat(),
                type(exc).__name__,
                exc,
            )
            return False

        self._mode = "pyo3" if probe_ok else "http"
        return probe_ok

    def available(self) -> bool:
        if self._can_use_pyo3():
            return True
        return self._health_check_http_only()

    def _health_check_http_only(self) -> bool:
        if self._mode != "http":
            return False
        result = self.health_check()
        return bool(result.get("healthy"))

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        if requests is None:
            raise RuntimeError("requests is required for HTTP fallback mode")
        url = f"{self._http.base_url}{path}"
        return requests.post(url, json=payload, timeout=self._http.timeout_seconds)

    def _http_call(self, path: str, payload: dict[str, Any], result_key: str) -> list[Any] | None:
        if requests is None:
            logger.error("rust http %s skipped error=requests module unavailable", path)
            return None
        started_at = time.perf_counter()
        try:
            response = self._post_json(path, payload)
            response.raise_for_status()
            data = response.json()
            result = data.get(result_key)
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "rust http %s completed duration_ms=%.2f",
                path,
                duration_ms,
            )
            return result if isinstance(result, list) else None
        except Exception as exc:
            logger.exception("rust http %s failed error=%s", path, exc)
            return None

    def filter_boxes(self, boxes: list[list[int]], threshold: float) -> list[list[int]] | None:
        started_at = time.perf_counter()
        try:
            if self._can_use_pyo3() and _rust_native is not None:
                result = _rust_native.filter_overlapping_boxes_py(boxes, threshold)
                duration_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "rust pyo3 filter_boxes completed boxes=%d kept=%d duration_ms=%.2f",
                    len(boxes),
                    len(result),
                    duration_ms,
                )
                return [list(t) for t in result]
            result = self._http_call("/v1/filter-boxes", {"boxes": boxes, "threshold": threshold}, "boxes")
            return [list(item) for item in result] if result is not None else None
        except Exception as exc:
            logger.exception("rust filter_boxes failed error=%s", exc)
            return None

    def dedupe_events(
        self,
        events: list[dict],
        cooldown_ms: int,
        iou_threshold: float,
    ) -> list[dict] | None:
        started_at = time.perf_counter()
        try:
            if self._can_use_pyo3() and _rust_native is not None:
                native_events = [
                    (e["class_id"], e["bbox"], e["timestamp_ms"])
                    for e in events
                ]
                result = _rust_native.dedupe_track_events_py(native_events, cooldown_ms, iou_threshold)
                duration_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "rust pyo3 dedupe_events completed events=%d kept=%d duration_ms=%.2f",
                    len(events),
                    len(result),
                    duration_ms,
                )
                return [
                    {"class_id": t[0], "bbox": list(t[1]), "timestamp_ms": t[2]}
                    for t in result
                ]
            result = self._http_call(
                "/v1/dedupe-events",
                {"events": events, "cooldown_ms": cooldown_ms, "iou_threshold": iou_threshold},
                "events",
            )
            return result if result is not None else None
        except Exception as exc:
            logger.exception("rust dedupe_events failed error=%s", exc)
            return None

    def health_check(self, force_refresh: bool = True) -> dict:
        if self._mode == "pyo3" and _rust_native is not None:
            return {
                "available": True,
                "healthy": True,
                "error": None,
                "latency_ms": 0.0,
                "mode": "pyo3",
            }
        if requests is None:
            return {
                "available": False,
                "healthy": False,
                "error": "requests module unavailable",
                "latency_ms": None,
                "mode": "http",
            }
        try:
            started_at = time.perf_counter()
            response = requests.get(f"{self._http.base_url}/health", timeout=self._http.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            payload.setdefault("available", True)
            payload.setdefault("healthy", True)
            payload.setdefault("error", None)
            payload["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            payload["mode"] = "http"
            return payload
        except Exception as exc:
            return {
                "available": False,
                "healthy": False,
                "error": str(exc),
                "latency_ms": None,
                "mode": "http",
            }

    def close(self) -> None:
        return None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
