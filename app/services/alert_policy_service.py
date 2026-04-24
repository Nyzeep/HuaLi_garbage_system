from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from app.config import get_settings
from app.constants import ALL_CLASSES


logger = logging.getLogger(__name__)


class AlertPolicyService:
    COOLDOWN_CONFIG = {
        "overflow": 15 * 60,
        "garbage": 15 * 60,
        "fire": 90,
        "smoke": 90,
    }
    STATE_FILE_NAME = "alert_policy_state.json"

    def __init__(self, state_file: Path | None = None) -> None:
        settings = get_settings()
        self._state_file = state_file or settings.uploads_dir / self.STATE_FILE_NAME
        self._lock = threading.Lock()
        self._last_alert_time: dict[int, float] = {}
        self._persistence_disabled = False
        self._load_state()

    def _get_cooldown_seconds(self, class_id: int) -> int:
        class_name = ALL_CLASSES.get(class_id, {}).get("en", "")
        return self.COOLDOWN_CONFIG.get(class_name, 15 * 60)

    def _load_state(self) -> None:
        with self._lock:
            if not self._state_file.exists():
                return
            try:
                payload = json.loads(self._state_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return
            raw_state = payload.get("last_alert_time", {})
            if not isinstance(raw_state, dict):
                return
            loaded: dict[int, float] = {}
            for key, value in raw_state.items():
                try:
                    loaded[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            self._last_alert_time = loaded
            self._prune_expired_locked(time.time())

    def _persist_state_locked(self) -> None:
        if self._persistence_disabled:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "last_alert_time": {str(class_id): timestamp for class_id, timestamp in self._last_alert_time.items()},
            }
            temp_file = self._state_file.with_name(
                f"{self._state_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            temp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp_file.replace(self._state_file)
        except OSError as exc:
            self._persistence_disabled = True
            logger.warning("alert policy persistence disabled state_file=%s error=%s", self._state_file, exc)

    def _prune_expired_locked(self, now: float) -> bool:
        original_size = len(self._last_alert_time)
        self._last_alert_time = {
            class_id: timestamp
            for class_id, timestamp in self._last_alert_time.items()
            if now - timestamp < self._get_cooldown_seconds(class_id)
        }
        return len(self._last_alert_time) != original_size

    def can_alert(self, class_id: int) -> bool:
        cooldown = self._get_cooldown_seconds(class_id)
        now = time.time()
        with self._lock:
            state_changed = self._prune_expired_locked(now)
            last = self._last_alert_time.get(class_id, 0.0)
            if now - last >= cooldown:
                self._last_alert_time[class_id] = now
                self._persist_state_locked()
                return True
            if state_changed:
                self._persist_state_locked()
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
