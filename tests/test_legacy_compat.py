from __future__ import annotations

import pytest

import app.alert_cooldown as legacy_cooldown
import app.services.inference as legacy_inference
from app.infrastructure.ml.backends import Prediction


class DummyPolicyService:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self._last_alert_time: dict[int, float] = {}

    def can_alert(self, class_id: int) -> bool:
        self.calls.append(class_id)
        return class_id == 3


def test_alert_cooldown_delegates_fire_category(monkeypatch):
    service = DummyPolicyService()
    monkeypatch.setattr(legacy_cooldown, "AlertPolicyService", lambda: service)

    with pytest.deprecated_call():
        cooldown = legacy_cooldown.AlertCooldown()

    assert cooldown.can_alert("fire") is True
    assert cooldown.can_alert("overflow") is False
    assert service.calls == [3, 1]


def test_legacy_non_max_suppression_preserves_iou_filtering():
    kept = legacy_inference.non_max_suppression(
        [
            Prediction(class_id=1, confidence=0.9, bbox=[0, 0, 10, 10]),
            Prediction(class_id=1, confidence=0.8, bbox=[1, 1, 9, 9]),
            Prediction(class_id=3, confidence=0.7, bbox=[1, 1, 9, 9]),
        ],
        iou_threshold=0.3,
    )

    assert kept == [
        Prediction(class_id=1, confidence=0.9, bbox=[0, 0, 10, 10]),
        Prediction(class_id=3, confidence=0.7, bbox=[1, 1, 9, 9]),
    ]
