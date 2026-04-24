from __future__ import annotations

import warnings
from typing import Any

from app.services.alert_policy_service import AlertPolicyService


class AlertCooldown:
    def __init__(self):
        warnings.warn(
            "app.alert_cooldown.AlertCooldown is deprecated; use app.services.alert_policy_service.AlertPolicyService instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._service = AlertPolicyService()

    def can_alert(self, category: str) -> bool:
        category_map = {
            "overflow": 1,
            "garbage": 2,
            "fire": 3,
            "smoke": 4,
        }
        class_id = category_map.get(category.lower())
        if class_id is None:
            return True
        return self._service.can_alert(class_id)

    def reset_category(self, category: str):
        category_map = {
            "overflow": 1,
            "garbage": 2,
            "fire": 3,
            "smoke": 4,
        }
        class_id = category_map.get(category.lower())
        if class_id is not None:
            self._service._last_alert_time.pop(class_id, None)



class _CooldownManagerProxy:
    def __init__(self) -> None:
        self._manager: AlertCooldown | None = None

    def _get_manager(self) -> AlertCooldown:
        if self._manager is None:
            self._manager = AlertCooldown()
        return self._manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_manager(), name)


cooldown_manager = _CooldownManagerProxy()

__all__ = ["AlertCooldown", "cooldown_manager"]
