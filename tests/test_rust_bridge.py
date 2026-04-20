from __future__ import annotations

import json
from io import BytesIO
from urllib import error
from unittest.mock import patch

import pytest

from app.infrastructure.ml.rust_bridge import RustBridge


class FakeResponse:
    def __init__(self, payload: dict | str | bytes) -> None:
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class UrlopenRecorder:
    def __init__(self, response_payload: dict | str | bytes) -> None:
        self.response_payload = response_payload
        self.calls: list[dict] = []

    def __call__(self, req, timeout):
        self.calls.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "headers": dict(req.header_items()),
                "body": json.loads(req.data.decode("utf-8")) if req.data else None,
                "timeout": timeout,
            }
        )
        return FakeResponse(self.response_payload)


def test_available_uses_cached_health_status():
    recorder = UrlopenRecorder({"available": True, "healthy": True})
    bridge = RustBridge(base_url="http://127.0.0.1:50051", timeout_seconds=1.5)

    with patch("app.infrastructure.ml.rust_bridge.request.urlopen", side_effect=recorder):
        assert bridge.available() is True
        assert bridge.available() is True

    assert len(recorder.calls) == 1
    assert recorder.calls[0]["url"] == "http://127.0.0.1:50051/health"
    assert recorder.calls[0]["method"] == "GET"
    assert recorder.calls[0]["timeout"] == 1.5


def test_health_check_success_parses_response():
    bridge = RustBridge(base_url="http://rust-service:50051")

    with patch(
        "app.infrastructure.ml.rust_bridge.request.urlopen",
        return_value=FakeResponse({"available": True, "healthy": True, "latency_ms": 7.25}),
    ):
        result = bridge.health_check()

    assert result["available"] is True
    assert result["healthy"] is True
    assert result["error"] is None
    assert result["latency_ms"] == 7.25


def test_health_check_returns_unavailable_on_url_error():
    bridge = RustBridge(base_url="http://rust-service:50051")

    with patch(
        "app.infrastructure.ml.rust_bridge.request.urlopen",
        side_effect=error.URLError("connection refused"),
    ):
        result = bridge.health_check()

    assert result["available"] is False
    assert result["healthy"] is False
    assert "connection refused" in result["error"]
    assert result["latency_ms"] is not None


def test_call_maps_compute_iou_to_rest_endpoint():
    recorder = UrlopenRecorder({"value": 0.5})
    bridge = RustBridge(base_url="http://127.0.0.1:50051", timeout_seconds=2.0)

    with patch("app.infrastructure.ml.rust_bridge.request.urlopen", side_effect=recorder):
        result = bridge.call(
            {
                "action": "compute_iou",
                "a": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                "b": {"x1": 5, "y1": 5, "x2": 15, "y2": 15},
            }
        )

    assert result.ok is True
    assert result.data == {"value": 0.5}
    assert recorder.calls == [
        {
            "url": "http://127.0.0.1:50051/v1/iou",
            "method": "POST",
            "headers": {"Accept": "application/json", "Content-type": "application/json"},
            "body": {
                "a": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                "b": {"x1": 5, "y1": 5, "x2": 15, "y2": 15},
            },
            "timeout": 2.0,
        }
    ]


def test_call_returns_error_message_from_http_error_body():
    bridge = RustBridge(base_url="http://127.0.0.1:50051")
    http_error = error.HTTPError(
        url="http://127.0.0.1:50051/v1/iou",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=BytesIO(json.dumps({"message": "invalid bbox coordinates"}).encode("utf-8")),
    )

    with patch("app.infrastructure.ml.rust_bridge.request.urlopen", side_effect=http_error):
        result = bridge.call(
            {
                "action": "compute_iou",
                "a": {"x1": 10, "y1": 10, "x2": 0, "y2": 0},
                "b": {"x1": 5, "y1": 5, "x2": 15, "y2": 15},
            }
        )

    assert result.ok is False
    assert result.error == "invalid bbox coordinates"


def test_call_rejects_unknown_action():
    bridge = RustBridge()

    result = bridge.call({"action": "bad_action"})

    assert result.ok is False
    assert result.error == "unknown action: bad_action"


def test_filter_boxes_posts_to_rest_service_and_parses_boxes():
    recorder = UrlopenRecorder(
        {
            "boxes": [
                {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                {"x1": 50, "y1": 50, "x2": 60, "y2": 60},
            ]
        }
    )
    bridge = RustBridge(base_url="http://127.0.0.1:50051")

    with patch("app.infrastructure.ml.rust_bridge.request.urlopen", side_effect=recorder):
        result = bridge.filter_boxes([[0, 0, 10, 10], [50, 50, 60, 60]], threshold=0.5)

    assert result == [[0, 0, 10, 10], [50, 50, 60, 60]]
    assert recorder.calls[0]["url"] == "http://127.0.0.1:50051/v1/filter-boxes"
    assert recorder.calls[0]["body"] == {
        "boxes": [
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"x1": 50, "y1": 50, "x2": 60, "y2": 60},
        ],
        "threshold": 0.5,
    }


def test_filter_boxes_returns_none_on_request_failure():
    bridge = RustBridge(base_url="http://127.0.0.1:50051")

    with patch(
        "app.infrastructure.ml.rust_bridge.request.urlopen",
        side_effect=error.URLError("service unavailable"),
    ):
        result = bridge.filter_boxes([[0, 0, 10, 10]], threshold=0.5)

    assert result is None


def test_dedupe_events_posts_to_rest_service_and_parses_events():
    recorder = UrlopenRecorder(
        {
            "events": [
                {
                    "class_id": 1,
                    "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                    "timestamp_ms": 0,
                }
            ]
        }
    )
    bridge = RustBridge(base_url="http://127.0.0.1:50051")
    events = [{"class_id": 1, "bbox": [0, 0, 100, 100], "timestamp_ms": 0}]

    with patch("app.infrastructure.ml.rust_bridge.request.urlopen", side_effect=recorder):
        result = bridge.dedupe_events(events, cooldown_ms=1000, iou_threshold=0.3)

    assert result == [{"class_id": 1, "bbox": [0, 0, 100, 100], "timestamp_ms": 0}]
    assert recorder.calls[0]["url"] == "http://127.0.0.1:50051/v1/dedupe-events"
    assert recorder.calls[0]["body"] == {
        "events": [
            {
                "class_id": 1,
                "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                "timestamp_ms": 0,
            }
        ],
        "cooldown_ms": 1000,
        "iou_threshold": 0.3,
    }


def test_dedupe_events_returns_none_on_request_failure():
    bridge = RustBridge(base_url="http://127.0.0.1:50051")

    with patch(
        "app.infrastructure.ml.rust_bridge.request.urlopen",
        side_effect=error.URLError("service unavailable"),
    ):
        result = bridge.dedupe_events([], cooldown_ms=1000, iou_threshold=0.3)

    assert result is None


def test_dedupe_events_pyo3_returns_none_and_logs_missing_fields(caplog):
    bridge = RustBridge(prefer_pyo3=False)
    bridge._use_pyo3 = True
    with patch("app.infrastructure.ml.rust_bridge._rust_native", create=True) as rust_native:
        rust_native.dedupe_track_events_py = lambda *args, **kwargs: []
        with caplog.at_level("ERROR", logger="app.infrastructure.ml.rust_bridge"):
            result = bridge._dedupe_events_pyo3(
                [{"class_id": 1, "bbox": [0, 0, 10, 10]}],
                cooldown_ms=1000,
                iou_threshold=0.3,
            )

    assert result is None
    assert "Missing required field in event data" in caplog.text


def test_dedupe_events_pyo3_converts_valid_events_and_returns_results():
    bridge = RustBridge(prefer_pyo3=False)
    bridge._use_pyo3 = True
    with patch("app.infrastructure.ml.rust_bridge._rust_native", create=True) as rust_native:
        rust_native.dedupe_track_events_py.return_value = [
            (1, [0, 0, 10, 10], 1234),
        ]
        result = bridge._dedupe_events_pyo3(
            [{"class_id": 1, "bbox": [0, 0, 10, 10], "timestamp_ms": 1234}],
            cooldown_ms=1000,
            iou_threshold=0.3,
        )

    assert result == [{"class_id": 1, "bbox": [0, 0, 10, 10], "timestamp_ms": 1234}]


def test_close_is_a_noop():
    bridge = RustBridge()
    assert bridge.close() is None
