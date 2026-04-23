from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_bbox(box: Sequence[float], name: str) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"{name} must contain exactly 4 coordinates")

    x1, y1, x2, y2 = box
    coords = (x1, y1, x2, y2)
    if not all(isinstance(value, (int, float)) for value in coords):
        raise ValueError(f"{name} must contain numeric coordinates")

    if not all(math.isfinite(value) for value in coords):
        raise ValueError(f"{name} must contain finite coordinates")

    if x1 > x2 or y1 > y2:
        raise ValueError(f"{name} must satisfy x1 <= x2 and y1 <= y2")

    return float(x1), float(y1), float(x2), float(y2)


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _validate_bbox(box_a, "box_a")
    bx1, by1, bx2, by2 = _validate_bbox(box_b, "box_b")

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0
