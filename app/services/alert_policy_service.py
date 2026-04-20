from __future__ import annotations

import time

from app.core.constants import ALL_CLASSES


class AlertPolicyService:
    COOLDOWN_CONFIG = {
        "overflow": 15 * 60,
        "garbage": 15 * 60,
        "fire": 90,
        "smoke": 90,
    }

    def __init__(self) -> None:
        self._last_alert_time: dict[int, float] = {}

    def _get_cooldown_seconds(self, class_id: int) -> int:
        class_name = ALL_CLASSES.get(class_id, {}).get("en", "")
        return self.COOLDOWN_CONFIG.get(class_name, 15 * 60)

    def can_alert(self, class_id: int) -> bool:
        cooldown = self._get_cooldown_seconds(class_id)
        now = time.time()
        last = self._last_alert_time.get(class_id, 0.0)
        if now - last >= cooldown:
            self._last_alert_time[class_id] = now
            return True
        return False

    def apply_cooldown(self, detections: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for detection in detections:
            if detection.get("alert") and not self.can_alert(int(detection.get("class_id", -1))):
                item = detection.copy()
                item["alert"] = False
                filtered.append(item)
            else:
                filtered.append(detection)
        return filtered
