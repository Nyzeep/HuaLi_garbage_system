from __future__ import annotations

from datetime import datetime


class SceneService:
    def analyze(self, detections: list[dict]) -> dict:
        alert_list = [item for item in detections if item.get("alert", False)]
        class_ids = {item.get("class_id") for item in detections}

        status = "normal"
        if 3 in class_ids:
            status = "fire"
        elif 4 in class_ids:
            status = "smoke"
        elif 1 in class_ids:
            status = "overflow"
        elif alert_list:
            status = "warning"

        return {
            "status": status,
            "alert_count": len(alert_list),
            "alert_types": list({item.get("class_name") for item in alert_list if item.get("class_name")}),
            "normal_count": len([item for item in detections if not item.get("alert", False)]),
            "total": len(detections),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
