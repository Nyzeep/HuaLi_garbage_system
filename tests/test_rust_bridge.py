from __future__ import annotations

from unittest.mock import patch

from app.infrastructure.ml import rust_bridge as rust_bridge_module
from app.infrastructure.ml.rust_bridge import RustBridge


def test_mode_is_pyo3_and_available_true():
    bridge = RustBridge()
    if rust_bridge_module._rust_native is None:
        assert bridge.mode == "http"
    else:
        assert bridge.mode == "pyo3"
    assert bridge.available() is True


def test_pyo3_probe_failure_marks_bridge_unavailable():
    with patch.object(rust_bridge_module, "_rust_native", create=True) as rust_native:
        rust_native.iou_py.side_effect = RuntimeError("boom")
        bridge = RustBridge()
        assert bridge.available() is False
        assert bridge.mode == "http"


def test_pyo3_probe_success_marks_bridge_available():
    with patch.object(rust_bridge_module, "_rust_native", create=True) as rust_native:
        rust_native.iou_py.return_value = 1.0
        bridge = RustBridge()
        assert bridge.available() is True
        assert bridge.mode == "pyo3"


def test_init_probe_exception_falls_back_to_http_and_logs(caplog):
    with patch.object(rust_bridge_module, "_rust_native", create=True), patch.object(
        RustBridge,
        "_refresh_pyo3_probe",
        side_effect=RuntimeError("probe init error"),
    ), caplog.at_level("ERROR", logger="app.infrastructure.ml.rust_bridge"):
        bridge = RustBridge(http_base_url="http://rust.local:50051")

    assert bridge.mode == "http"
    assert "probe initialization failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "fallback=http" in caplog.text


def test_runtime_probe_exception_falls_back_to_http_and_logs(caplog):
    with patch.object(rust_bridge_module, "_rust_native", create=True), patch.object(
        RustBridge,
        "_refresh_pyo3_probe",
        side_effect=ValueError("probe runtime error"),
    ), patch.object(
        RustBridge,
        "health_check",
        return_value={"healthy": False},
    ), caplog.at_level("ERROR", logger="app.infrastructure.ml.rust_bridge"):
        bridge = RustBridge(http_base_url="http://rust.local:50051")
        assert bridge.available() is False

    assert bridge.mode == "http"
    assert "probe refresh failed" in caplog.text
    assert "ValueError" in caplog.text
    assert "fallback=http" in caplog.text


def test_pyo3_probe_is_cached_for_short_window():
    with patch.object(rust_bridge_module, "_rust_native", create=True) as rust_native:
        rust_native.iou_py.return_value = 1.0
        bridge = RustBridge()
        first = bridge.available()
        second = bridge.available()
        assert first is True
        assert second is True
        assert rust_native.iou_py.call_count == 1


def test_http_fallback_used_when_pyo3_missing():
    with patch.object(rust_bridge_module, "_rust_native", None), patch.object(
        RustBridge,
        "health_check",
        return_value={"healthy": True},
    ) as health_check:
        bridge = RustBridge(http_base_url="http://rust.local:50051")
        assert bridge.mode == "http"
        assert bridge.available() is True
        health_check.assert_called()


def test_http_health_check_false_when_backend_unhealthy():
    with patch.object(rust_bridge_module, "_rust_native", None), patch.object(
        RustBridge,
        "health_check",
        return_value={"healthy": False},
    ):
        bridge = RustBridge(http_base_url="http://rust.local:50051")
        assert bridge.available() is False


def test_http_call_returns_none_when_requests_unavailable(caplog):
    with patch.object(rust_bridge_module, "requests", None), caplog.at_level("ERROR", logger="app.infrastructure.ml.rust_bridge"):
        bridge = RustBridge(http_base_url="http://rust.local:50051")
        result = bridge._http_call("/v1/filter-boxes", {"boxes": [], "threshold": 0.5}, "boxes")

    assert result is None
    assert "requests module unavailable" in caplog.text


def test_health_check_returns_unhealthy_when_requests_unavailable():
    with patch.object(rust_bridge_module, "_rust_native", None), patch.object(rust_bridge_module, "requests", None):
        bridge = RustBridge(http_base_url="http://rust.local:50051")
        result = bridge.health_check()

    assert result["available"] is False
    assert result["healthy"] is False
    assert result["mode"] == "http"
    assert result["error"] == "requests module unavailable"


def test_filter_boxes_falls_back_to_http_when_pyo3_probe_fails():
    with patch.object(rust_bridge_module, "_rust_native", create=True) as rust_native:
        rust_native.iou_py.side_effect = RuntimeError("boom")
        bridge = RustBridge(http_base_url="http://rust.local:50051")
        with patch.object(bridge, "_http_call", return_value=[[1, 2, 3, 4]]) as http_call:
            result = bridge.filter_boxes([[1, 2, 3, 4]], threshold=0.5)

    assert result == [[1, 2, 3, 4]]
    http_call.assert_called_once_with("/v1/filter-boxes", {"boxes": [[1, 2, 3, 4]], "threshold": 0.5}, "boxes")


def test_dedupe_events_falls_back_to_http_when_pyo3_probe_fails():
    with patch.object(rust_bridge_module, "_rust_native", create=True) as rust_native:
        rust_native.iou_py.side_effect = RuntimeError("boom")
        bridge = RustBridge(http_base_url="http://rust.local:50051")
        payload = [{"class_id": 1, "bbox": [0, 0, 10, 10], "timestamp_ms": 1}]
        with patch.object(bridge, "_http_call", return_value=payload) as http_call:
            result = bridge.dedupe_events(payload, cooldown_ms=1000, iou_threshold=0.3)

    assert result == payload
    http_call.assert_called_once_with(
        "/v1/dedupe-events",
        {"events": payload, "cooldown_ms": 1000, "iou_threshold": 0.3},
        "events",
    )


def test_health_check_reports_pyo3_mode():
    bridge = RustBridge()
    result = bridge.health_check()
    assert result == {
        "available": True,
        "healthy": True,
        "error": None,
        "latency_ms": 0.0,
        "mode": "pyo3",
    }


def test_filter_boxes_uses_pyo3_and_returns_boxes():
    bridge = RustBridge()

    with patch("app.infrastructure.ml.rust_bridge._rust_native", create=True) as rust_native:
        rust_native.filter_overlapping_boxes_py.return_value = [
            (0, 0, 10, 10),
            (50, 50, 60, 60),
        ]
        result = bridge.filter_boxes([[0, 0, 10, 10], [50, 50, 60, 60]], threshold=0.5)

    assert result == [[0, 0, 10, 10], [50, 50, 60, 60]]


def test_filter_boxes_returns_none_when_pyo3_fails():
    bridge = RustBridge()

    with patch("app.infrastructure.ml.rust_bridge._rust_native", create=True) as rust_native:
        rust_native.filter_overlapping_boxes_py.side_effect = RuntimeError("boom")
        result = bridge.filter_boxes([[0, 0, 10, 10]], threshold=0.5)

    assert result is None


def test_dedupe_events_uses_pyo3_and_returns_events():
    bridge = RustBridge()

    with patch("app.infrastructure.ml.rust_bridge._rust_native", create=True) as rust_native:
        rust_native.dedupe_track_events_py.return_value = [
            (1, [0, 0, 10, 10], 1234),
        ]
        result = bridge.dedupe_events(
            [{"class_id": 1, "bbox": [0, 0, 10, 10], "timestamp_ms": 1234}],
            cooldown_ms=1000,
            iou_threshold=0.3,
        )

    assert result == [{"class_id": 1, "bbox": [0, 0, 10, 10], "timestamp_ms": 1234}]


def test_dedupe_events_returns_none_when_payload_missing_fields(caplog):
    bridge = RustBridge()

    with patch("app.infrastructure.ml.rust_bridge._rust_native", create=True) as rust_native:
        rust_native.dedupe_track_events_py.return_value = []
        with caplog.at_level("ERROR", logger="app.infrastructure.ml.rust_bridge"):
            result = bridge.dedupe_events(
                [{"class_id": 1, "bbox": [0, 0, 10, 10]}],
                cooldown_ms=1000,
                iou_threshold=0.3,
            )

    assert result is None
    assert "timestamp_ms" in caplog.text


def test_close_is_a_noop():
    bridge = RustBridge()
    assert bridge.close() is None

