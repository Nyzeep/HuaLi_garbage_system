from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.services.alert_policy_service import AlertPolicyService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def svc(tmp_path) -> AlertPolicyService:
    return AlertPolicyService(state_file=tmp_path / "alert_policy_state.json")


def _det(class_id: int, alert: bool = True) -> dict:
    return {"class_id": class_id, "alert": alert}


# ---------------------------------------------------------------------------
# can_alert — basic gate
# ---------------------------------------------------------------------------

def test_can_alert_first_time(svc):
    assert svc.can_alert(3) is True


def test_cannot_alert_immediately_after(svc):
    svc.can_alert(3)
    assert svc.can_alert(3) is False


def test_can_alert_again_after_cooldown_expires(svc):
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(3)  # record first alert at t0
        # Fire cooldown is 90s
        mock_time.time.return_value = t0 + 91
        assert svc.can_alert(3) is True


def test_cannot_alert_just_before_cooldown_expires(svc):
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(3)
        mock_time.time.return_value = t0 + 89  # 89 < 90 → still blocked
        assert svc.can_alert(3) is False


# ---------------------------------------------------------------------------
# Per-class cooldown durations
# ---------------------------------------------------------------------------

def test_garbage_bin_class0_cooldown_15min(svc):
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(0)  # class 0 = garbage_bin, cooldown 15*60=900s
        mock_time.time.return_value = t0 + 899
        assert svc.can_alert(0) is False
        mock_time.time.return_value = t0 + 901
        assert svc.can_alert(0) is True


def test_fire_class3_cooldown_90s(svc):
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(3)
        mock_time.time.return_value = t0 + 89
        assert svc.can_alert(3) is False
        mock_time.time.return_value = t0 + 91
        assert svc.can_alert(3) is True


def test_smoke_class4_cooldown_90s(svc):
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(4)
        mock_time.time.return_value = t0 + 91
        assert svc.can_alert(4) is True


def test_different_classes_independent(svc):
    """Alerting class 3 should not affect class 4's cooldown."""
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(3)
        assert svc.can_alert(4) is True  # 4 never alerted yet


# ---------------------------------------------------------------------------
# apply_cooldown
# ---------------------------------------------------------------------------

def test_apply_cooldown_non_alert_passes_through(svc):
    dets = [_det(0, alert=False)]
    result = svc.apply_cooldown(dets)
    assert result[0]["alert"] is False


def test_apply_cooldown_alert_first_call_passes(svc):
    result = svc.apply_cooldown([_det(3, alert=True)])
    assert result[0]["alert"] is True


def test_apply_cooldown_alert_second_call_suppressed(svc):
    svc.apply_cooldown([_det(3)])
    result = svc.apply_cooldown([_det(3)])
    assert result[0]["alert"] is False


def test_apply_cooldown_preserves_other_fields(svc):
    det = {"class_id": 3, "alert": True, "confidence": 0.95, "bbox": [1, 2, 3, 4]}
    result = svc.apply_cooldown([det])
    assert result[0]["confidence"] == 0.95
    assert result[0]["bbox"] == [1, 2, 3, 4]


def test_apply_cooldown_mixed_detections(svc):
    """One passes, one is suppressed (already recorded), one is non-alert."""
    # First call records class 3
    svc.apply_cooldown([_det(3)])
    dets = [_det(3), _det(4), _det(0, False)]
    result = svc.apply_cooldown(dets)
    assert result[0]["alert"] is False  # 3 suppressed
    assert result[1]["alert"] is True   # 4 new
    assert result[2]["alert"] is False  # 0 was already non-alert


def test_apply_cooldown_does_not_mutate_input(svc):
    det = {"class_id": 3, "alert": True}
    svc.apply_cooldown([det])
    result = svc.apply_cooldown([det])
    assert det["alert"] is True  # original unchanged
    assert result[0]["alert"] is False  # returned copy is suppressed


def test_alert_policy_persists_state_across_restart(tmp_path):
    state_file = tmp_path / "alert_policy_state.json"
    t0 = 1_000_000.0

    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc = AlertPolicyService(state_file=state_file)
        assert svc.can_alert(3) is True

    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0 + 1
        restarted = AlertPolicyService(state_file=state_file)
        assert restarted.can_alert(3) is False

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["last_alert_time"]["3"] == pytest.approx(t0)


def test_alert_policy_prunes_expired_state_on_load(tmp_path):
    state_file = tmp_path / "alert_policy_state.json"
    state_file.write_text(
        json.dumps({"last_alert_time": {"3": 1_000_050.0, "0": 999_000.0}}, ensure_ascii=False),
        encoding="utf-8",
    )

    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = 1_000_100.0
        svc = AlertPolicyService(state_file=state_file)

    assert 3 in svc._last_alert_time
    assert 0 not in svc._last_alert_time


def test_alert_policy_disables_persistence_when_write_fails(tmp_path, monkeypatch):
    state_file = tmp_path / "alert_policy_state.json"
    svc = AlertPolicyService(state_file=state_file)

    original_mkdir = Path.mkdir

    def failing_mkdir(path, *args, **kwargs):
        if path == state_file.parent:
            raise OSError("disk error")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)

    assert svc.can_alert(3) is True
    assert svc._persistence_disabled is True


def test_alert_policy_concurrent_persistence_uses_valid_state_file(tmp_path):
    state_file = tmp_path / "alert_policy_state.json"
    service = AlertPolicyService(state_file=state_file)
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(class_id: int) -> None:
        try:
            service.can_alert(class_id)
        except Exception as exc:  # pragma: no cover - assertion target is no exception
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(3 if index % 2 == 0 else 4,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert set(payload["last_alert_time"]).issubset({"3", "4"})
