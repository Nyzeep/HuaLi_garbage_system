from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the PyO3 native module built by maturin.
# When available, Rust functions run in-process — no HTTP / JSON overhead.
# ---------------------------------------------------------------------------
try:
    import huali_garbage_core as _rust_native  # type: ignore[import-untyped]

    _HAS_PYO3 = True
    logger.info("rust pyo3 module loaded — in-process calls enabled")
except ImportError:
    _rust_native = None  # type: ignore[assignment]
    _HAS_PYO3 = False
    logger.info("rust pyo3 module not available — will use HTTP bridge")


@dataclass
class RustBridgeResult:
    ok: bool
    data: Any = None
    error: str | None = None


class RustBridge:
    """
    Bridge to Rust core utilities.

    Preferred path: PyO3 in-process calls (zero-copy, no network overhead).
    Fallback path:  HTTP calls to the standalone Rust microservice.

    Set *prefer_pyo3=False* (or config ``rust_prefer_pyo3=false``) to force
    the HTTP path even when the native module is available.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:50051",
        timeout_seconds: float = 2.0,
        prefer_pyo3: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._health_cache: dict | None = None
        self._use_pyo3: bool = prefer_pyo3 and _HAS_PYO3

        if self._use_pyo3:
            logger.info("rust bridge mode=pyo3 (in-process)")
        else:
            logger.warning("rust bridge fallback=http base_url=%s", self.base_url)

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "pyo3" if self._use_pyo3 else "http"

    def available(self) -> bool:
        if self._use_pyo3:
            return True
        status = self.health_check(force_refresh=False)
        return bool(status.get("available") and status.get("healthy"))

    # ------------------------------------------------------------------
    #  filter_boxes
    # ------------------------------------------------------------------

    def filter_boxes(self, boxes: list[list[int]], threshold: float) -> list[list[int]] | None:
        if self._use_pyo3:
            return self._filter_boxes_pyo3(boxes, threshold)
        return self._filter_boxes_http(boxes, threshold)

    def _filter_boxes_pyo3(self, boxes: list[list[int]], threshold: float) -> list[list[int]] | None:
        started_at = time.perf_counter()
        try:
            result = _rust_native.filter_overlapping_boxes_py(boxes, threshold)
            if not isinstance(result, list):
                raise TypeError(f"unexpected pyo3 result type: {type(result)!r}")
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "rust pyo3 filter_boxes completed boxes=%d kept=%d duration_ms=%.2f",
                len(boxes),
                len(result),
                duration_ms,
            )
            return [list(t) for t in result]
        except Exception as exc:
            logger.exception("rust pyo3 filter_boxes failed error=%s", exc)
            return None

    def _filter_boxes_http(self, boxes: list[list[int]], threshold: float) -> list[list[int]] | None:
        payload = {
            "action": "filter_overlapping_boxes",
            "boxes": [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in boxes],
            "threshold": threshold,
        }
        result = self.call(payload)
        if not result.ok or not result.data:
            return None
        return [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in result.data.get("boxes", [])]

    # ------------------------------------------------------------------
    #  dedupe_events
    # ------------------------------------------------------------------

    def dedupe_events(
        self,
        events: list[dict],
        cooldown_ms: int,
        iou_threshold: float,
    ) -> list[dict] | None:
        if self._use_pyo3:
            return self._dedupe_events_pyo3(events, cooldown_ms, iou_threshold)
        return self._dedupe_events_http(events, cooldown_ms, iou_threshold)

    def _dedupe_events_pyo3(
        self,
        events: list[dict],
        cooldown_ms: int,
        iou_threshold: float,
    ) -> list[dict] | None:
        started_at = time.perf_counter()
        try:
            # Convert dicts → tuples accepted by PyO3 (class_id, bbox, timestamp_ms)
            try:
                native_events = [
                    (e["class_id"], e["bbox"], e["timestamp_ms"])
                    for e in events
                ]
            except KeyError as e:
                logger.error("Missing required field in event data: %s", str(e))
                return None

            result = _rust_native.dedupe_track_events_py(native_events, cooldown_ms, iou_threshold)
            if not isinstance(result, list):
                raise TypeError(f"unexpected pyo3 result type: {type(result)!r}")
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
        except Exception as exc:
            logger.exception("rust pyo3 dedupe_events failed error=%s", exc)
            return None

    def _dedupe_events_http(
        self,
        events: list[dict],
        cooldown_ms: int,
        iou_threshold: float,
    ) -> list[dict] | None:
        payload = {
            "action": "dedupe_track_events",
            "events": [
                {
                    "class_id": e["class_id"],
                    "bbox": {
                        "x1": e["bbox"][0],
                        "y1": e["bbox"][1],
                        "x2": e["bbox"][2],
                        "y2": e["bbox"][3],
                    },
                    "timestamp_ms": e["timestamp_ms"],
                }
                for e in events
            ],
            "cooldown_ms": cooldown_ms,
            "iou_threshold": iou_threshold,
        }
        result = self.call(payload)
        if not result.ok or not result.data:
            return None
        out = []
        for e in result.data.get("events", []):
            b = e["bbox"]
            out.append(
                {
                    "class_id": e["class_id"],
                    "bbox": [b["x1"], b["y1"], b["x2"], b["y2"]],
                    "timestamp_ms": e["timestamp_ms"],
                }
            )
        return out

    # ------------------------------------------------------------------
    #  Health check
    # ------------------------------------------------------------------

    def health_check(self, force_refresh: bool = True) -> dict:
        if self._use_pyo3:
            return {
                "available": True,
                "healthy": True,
                "error": None,
                "latency_ms": 0.0,
                "mode": "pyo3",
            }

        if not force_refresh and self._health_cache is not None:
            return dict(self._health_cache)

        t0 = time.perf_counter()
        result = self._request("GET", "/health")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        if not result.ok:
            status = {
                "available": False,
                "healthy": False,
                "error": result.error,
                "latency_ms": latency_ms,
                "mode": "http",
            }
            self._health_cache = status
            return dict(status)

        payload = result.data or {}
        status = {
            "available": bool(payload.get("available", True)),
            "healthy": bool(payload.get("healthy", True)),
            "error": payload.get("error"),
            "latency_ms": payload.get("latency_ms", latency_ms),
            "mode": "http",
        }
        self._health_cache = status
        return dict(status)

    # ------------------------------------------------------------------
    #  Generic HTTP call (kept for backward-compat & fallback)
    # ------------------------------------------------------------------

    def call(self, payload: dict) -> RustBridgeResult:
        action = payload.get("action")
        if action == "compute_iou":
            req_payload = {"a": payload.get("a"), "b": payload.get("b")}
            return self._request("POST", "/v1/iou", req_payload)
        if action == "filter_overlapping_boxes":
            req_payload = {
                "boxes": payload.get("boxes", []),
                "threshold": payload.get("threshold"),
            }
            return self._request("POST", "/v1/filter-boxes", req_payload)
        if action == "dedupe_track_events":
            req_payload = {
                "events": payload.get("events", []),
                "cooldown_ms": payload.get("cooldown_ms"),
                "iou_threshold": payload.get("iou_threshold"),
            }
            return self._request("POST", "/v1/dedupe-events", req_payload)
        return RustBridgeResult(ok=False, error=f"unknown action: {action}")

    def _request(self, method: str, path: str, payload: dict | None = None) -> RustBridgeResult:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = request.Request(url, data=data, headers=headers, method=method)
        started_at = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body) if body else {}
                duration_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "rust http request completed method=%s path=%s status=%s duration_ms=%.2f",
                    method,
                    path,
                    getattr(resp, "status", 200),
                    duration_ms,
                )
                return RustBridgeResult(ok=True, data=parsed)
        except error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8")
                parsed = json.loads(body) if body else {}
                message = parsed.get("message") or parsed.get("error") or str(exc)
            except Exception:
                message = str(exc)
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "rust http request failed method=%s path=%s status=%s duration_ms=%.2f error=%s",
                method,
                path,
                exc.code,
                duration_ms,
                message,
            )
            return RustBridgeResult(ok=False, error=message)
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            message = str(reason)
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "rust http request failed method=%s path=%s duration_ms=%.2f error=%s",
                method,
                path,
                duration_ms,
                message,
            )
            return RustBridgeResult(ok=False, error=message)
        except Exception as exc:
            message = str(exc)
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "rust http request failed method=%s path=%s duration_ms=%.2f error=%s",
                method,
                path,
                duration_ms,
                message,
            )
            return RustBridgeResult(ok=False, error=message)

    def close(self) -> None:
        return None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
