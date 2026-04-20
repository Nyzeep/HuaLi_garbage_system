from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.services.alert_policy_service import AlertPolicyService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(class_id: int, alert: bool = True) -> dict:
    return {"class_id": class_id, "alert": alert}


# ---------------------------------------------------------------------------
# can_alert — basic gate
# ---------------------------------------------------------------------------

def test_can_alert_first_time():
    svc = AlertPolicyService()
    assert svc.can_alert(3) is True


def test_cannot_alert_immediately_after():
    svc = AlertPolicyService()
    svc.can_alert(3)
    assert svc.can_alert(3) is False


def test_can_alert_again_after_cooldown_expires():
    svc = AlertPolicyService()
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(3)  # record first alert at t0
        # Fire cooldown is 90s
        mock_time.time.return_value = t0 + 91
        assert svc.can_alert(3) is True


def test_cannot_alert_just_before_cooldown_expires():
    svc = AlertPolicyService()
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(3)
        mock_time.time.return_value = t0 + 89  # 89 < 90 → still blocked
        assert svc.can_alert(3) is False


# ---------------------------------------------------------------------------
# Per-class cooldown durations
# ---------------------------------------------------------------------------

def test_garbage_bin_class0_cooldown_15min():
    svc = AlertPolicyService()
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(0)  # class 0 = garbage_bin, cooldown 15*60=900s
        mock_time.time.return_value = t0 + 899
        assert svc.can_alert(0) is False
        mock_time.time.return_value = t0 + 901
        assert svc.can_alert(0) is True


def test_fire_class3_cooldown_90s():
    svc = AlertPolicyService()
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(3)
        mock_time.time.return_value = t0 + 89
        assert svc.can_alert(3) is False
        mock_time.time.return_value = t0 + 91
        assert svc.can_alert(3) is True


def test_smoke_class4_cooldown_90s():
    svc = AlertPolicyService()
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(4)
        mock_time.time.return_value = t0 + 91
        assert svc.can_alert(4) is True


def test_different_classes_independent():
    """Alerting class 3 should not affect class 4's cooldown."""
    svc = AlertPolicyService()
    t0 = 1_000_000.0
    with patch("app.services.alert_policy_service.time") as mock_time:
        mock_time.time.return_value = t0
        svc.can_alert(3)
        assert svc.can_alert(4) is True  # 4 never alerted yet


# ---------------------------------------------------------------------------
# apply_cooldown
# ---------------------------------------------------------------------------

def test_apply_cooldown_non_alert_passes_through():
    svc = AlertPolicyService()
    dets = [_det(0, alert=False)]
    result = svc.apply_cooldown(dets)
    assert result[0]["alert"] is False


def test_apply_cooldown_alert_first_call_passes():
    svc = AlertPolicyService()
    result = svc.apply_cooldown([_det(3, alert=True)])
    assert result[0]["alert"] is True


def test_apply_cooldown_alert_second_call_suppressed():
    svc = AlertPolicyService()
    svc.apply_cooldown([_det(3)])
    result = svc.apply_cooldown([_det(3)])
    assert result[0]["alert"] is False


def test_apply_cooldown_preserves_other_fields():
    svc = AlertPolicyService()
    det = {"class_id": 3, "alert": True, "confidence": 0.95, "bbox": [1, 2, 3, 4]}
    result = svc.apply_cooldown([det])
    assert result[0]["confidence"] == 0.95
    assert result[0]["bbox"] == [1, 2, 3, 4]


def test_apply_cooldown_mixed_detections():
    """One passes, one is suppressed (already recorded), one is non-alert."""
    svc = AlertPolicyService()
    # First call records class 3
    svc.apply_cooldown([_det(3)])
    dets = [_det(3), _det(4), _det(0, False)]
    result = svc.apply_cooldown(dets)
    assert result[0]["alert"] is False  # 3 suppressed
    assert result[1]["alert"] is True   # 4 new
    assert result[2]["alert"] is False  # 0 was already non-alert


def test_apply_cooldown_does_not_mutate_input():
    svc = AlertPolicyService()
    det = {"class_id": 3, "alert": True}
    svc.apply_cooldown([det])
    result = svc.apply_cooldown([det])
    assert det["alert"] is True  # original unchanged
    assert result[0]["alert"] is False  # returned copy is suppressed
