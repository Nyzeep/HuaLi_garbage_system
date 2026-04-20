from __future__ import annotations

import cv2
import numpy as np


class RenderingService:
    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        output = image.copy()
        en_label_map = {
            0: "GarbageBin",
            1: "Overflow",
            2: "Garbage",
            3: "FIRE",
            4: "SMOKE",
        }

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            box_color = detection.get("color", (0, 255, 0))
            en_name = en_label_map.get(detection["class_id"], detection.get("class_name", "Object"))
            label = f"{en_name} {detection.get('confidence', 0.0):.0%}"
            line_w = 3 if detection.get("alert", False) else 2

            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, line_w)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            top = max(0, y1 - text_h - 8)
            cv2.rectangle(output, (x1, top), (x1 + text_w + 6, y1), box_color, -1)
            cv2.putText(
                output,
                label,
                (x1 + 3, max(text_h, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return output
