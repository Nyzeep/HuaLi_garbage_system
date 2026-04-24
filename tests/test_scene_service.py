from __future__ import annotations

import pytest

from app.services.scene_service import SceneService


def make_det(class_id: int, alert: bool, class_name: str = "") -> dict:
    return {"class_id": class_id, "class_name": class_name or f"cls{class_id}", "alert": alert}


# ---------------------------------------------------------------------------
# Status priority: fire > smoke > overflow > generic-warning > normal
# ---------------------------------------------------------------------------

def test_empty_detections_is_normal():
    svc = SceneService()
    result = svc.analyze([])
    assert result["status"] == "normal"
    assert result["alert_count"] == 0
    assert result["total"] == 0


def test_fire_takes_highest_priority():
    svc = SceneService()
    dets = [
        make_det(0, False),   # garbage_bin — normal
        make_det(1, True),    # overflow
        make_det(3, True),    # fire
    ]
    assert svc.analyze(dets)["status"] == "fire"


def test_smoke_beats_overflow():
    svc = SceneService()
    dets = [make_det(1, True), make_det(4, True)]
    assert svc.analyze(dets)["status"] == "smoke"


def test_overflow_beats_generic_alert():
    svc = SceneService()
    dets = [make_det(2, True), make_det(1, True)]
    assert svc.analyze(dets)["status"] == "overflow"


def test_generic_alert_gives_warning():
    svc = SceneService()
    # class 2 (garbage) is alert but not fire/smoke/overflow(1)
    dets = [make_det(2, True, "散落垃圾")]
    assert svc.analyze(dets)["status"] == "warning"


def test_all_normal_is_normal():
    svc = SceneService()
    dets = [make_det(0, False), make_det(0, False)]
    assert svc.analyze(dets)["status"] == "normal"


# ---------------------------------------------------------------------------
# Counts and alert_types
# ---------------------------------------------------------------------------

def test_alert_count_counts_only_alert_dets():
    svc = SceneService()
    dets = [make_det(3, True), make_det(0, False), make_det(3, True)]
    result = svc.analyze(dets)
    assert result["alert_count"] == 2
    assert result["normal_count"] == 1
    assert result["total"] == 3


def test_alert_types_deduplicated():
    svc = SceneService()
    # Two detections of the same class → only one entry in alert_types
    dets = [make_det(3, True, "火焰"), make_det(3, True, "火焰")]
    result = svc.analyze(dets)
    assert result["alert_types"].count("火焰") == 1


def test_alert_types_excludes_non_alert():
    svc = SceneService()
    dets = [make_det(0, False, "垃圾桶"), make_det(3, True, "火焰")]
    result = svc.analyze(dets)
    assert "垃圾桶" not in result["alert_types"]
    assert "火焰" in result["alert_types"]


def test_alert_types_empty_when_no_alerts():
    svc = SceneService()
    result = svc.analyze([make_det(0, False)])
    assert result["alert_types"] == []


def test_timestamp_present():
    svc = SceneService()
    result = svc.analyze([])
    assert "timestamp" in result
    assert len(result["timestamp"]) == 19  # "YYYY-MM-DD HH:MM:SS"
