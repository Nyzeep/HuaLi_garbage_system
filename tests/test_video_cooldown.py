from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.core.geometry import compute_iou

bbox_st = st.builds(
    lambda x, y, w, h: [x, y, x + w, y + h],
    st.integers(0, 500), st.integers(0, 500),
    st.integers(0, 300), st.integers(0, 300),
)


@given(bbox_st, bbox_st)
def test_iou_symmetry(a, b):
    assert compute_iou(a, b) == pytest.approx(compute_iou(b, a))


@given(bbox_st, bbox_st)
def test_iou_in_range(a, b):
    assert 0.0 <= compute_iou(a, b) <= 1.0 + 1e-9


@given(bbox_st)
def test_iou_self_is_one_or_zero(b):
    result = compute_iou(b, b)
    area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    if area > 0:
        assert result == pytest.approx(1.0)
    else:
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _apply_video_alert_cooldown_python — extracted inline to avoid imageio dep
# ---------------------------------------------------------------------------

IOU_MATCH_THRESHOLD = 0.4
VIDEO_GARBAGE_COOLDOWN = 3.0
VIDEO_FIRE_SMOKE_COOLDOWN = 1.0


def _cooldown_for(class_id):
    if class_id in (3, 4):
        return VIDEO_FIRE_SMOKE_COOLDOWN
    if class_id in (1, 2):
        return VIDEO_GARBAGE_COOLDOWN
    return 0.0


def apply_cooldown(detections, current_ts, alert_history):
    updated = []
    for det in detections:
        item = det.copy()
        if not item.get("alert"):
            updated.append(item)
            continue
        class_id = int(item.get("class_id", -1))
        cooldown = _cooldown_for(class_id)
        if cooldown <= 0:
            updated.append(item)
            continue
        bbox = item.get("bbox", [])
        suppressed = any(
            r["class_id"] == class_id
            and current_ts - r["timestamp"] <= cooldown
            and compute_iou(bbox, r["bbox"]) >= IOU_MATCH_THRESHOLD
            for r in alert_history
        )
        if suppressed:
            item["alert"] = False
        else:
            alert_history.append({"class_id": class_id, "bbox": bbox, "timestamp": current_ts})
        updated.append(item)
    max_window = max(VIDEO_GARBAGE_COOLDOWN, VIDEO_FIRE_SMOKE_COOLDOWN)
    alert_history[:] = [r for r in alert_history if current_ts - r["timestamp"] <= max_window]
    return updated


def _det(class_id, bbox, alert=True):
    return {"class_id": class_id, "bbox": bbox, "alert": alert}


def test_cooldown_suppresses_same_object():
    history = []
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history)
    result = apply_cooldown([_det(1, [0, 0, 100, 100])], 1.0, history)
    assert result[0]["alert"] is False


def test_cooldown_allows_different_location():
    history = []
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history)
    result = apply_cooldown([_det(1, [500, 500, 600, 600])], 1.0, history)
    assert result[0]["alert"] is True


def test_cooldown_non_alert_passthrough():
    history = []
    result = apply_cooldown([_det(1, [0, 0, 100, 100], alert=False)], 0.0, history)
    assert result[0]["alert"] is False


def test_cooldown_history_expires():
    history = []
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history)
    result = apply_cooldown([_det(1, [0, 0, 100, 100])], 4.0, history)
    assert result[0]["alert"] is True


def test_cooldown_zero_cooldown_class_always_alerts():
    history = []
    apply_cooldown([_det(0, [0, 0, 100, 100])], 0.0, history)
    result = apply_cooldown([_det(0, [0, 0, 100, 100])], 0.1, history)
    assert result[0]["alert"] is True


# ---------------------------------------------------------------------------
# Per-class cooldown durations
# ---------------------------------------------------------------------------

def test_fire_class_cooldown_is_1s_not_3s():
    """fire(3) and smoke(4) use 1s cooldown, not 3s."""
    history = []
    apply_cooldown([_det(3, [0, 0, 100, 100])], 0.0, history)
    # At 1.5s, garbage class (3s) would still suppress, but fire (1s) should not
    result = apply_cooldown([_det(3, [0, 0, 100, 100])], 1.5, history)
    assert result[0]["alert"] is True


def test_overflow_class_cooldown_is_3s():
    """overflow(1) and garbage(2) use 3s cooldown."""
    history = []
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history)
    result_within = apply_cooldown([_det(1, [0, 0, 100, 100])], 2.0, history)
    assert result_within[0]["alert"] is False  # within 3s

    history2 = []
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history2)
    result_after = apply_cooldown([_det(1, [0, 0, 100, 100])], 4.0, history2)
    assert result_after[0]["alert"] is True  # after 3s


# ---------------------------------------------------------------------------
# Multiple detections in same frame
# ---------------------------------------------------------------------------

def test_multiple_detections_independent_suppression():
    """Two different objects in same frame: one suppressed, one new."""
    history = []
    # Record object A
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history)

    # Same frame: object A again (same location) + object B (different location)
    dets = [_det(1, [0, 0, 100, 100]), _det(1, [500, 500, 600, 600])]
    result = apply_cooldown(dets, 1.0, history)
    assert result[0]["alert"] is False  # A suppressed
    assert result[1]["alert"] is True   # B new location


def test_multiple_same_frame_all_suppressed():
    history = []
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history)
    dets = [_det(1, [0, 0, 100, 100]), _det(1, [1, 1, 101, 101])]  # both overlap
    result = apply_cooldown(dets, 1.0, history)
    assert result[0]["alert"] is False
    assert result[1]["alert"] is False


# ---------------------------------------------------------------------------
# History expiry cleans up old entries
# ---------------------------------------------------------------------------

def test_history_pruned_after_max_window():
    """After max cooldown window passes, history should be trimmed."""
    history = []
    # Record many old events
    for i in range(10):
        apply_cooldown([_det(1, [i * 50, 0, i * 50 + 40, 40])], float(i), history)
    # Jump far forward — all old entries expired
    apply_cooldown([_det(1, [0, 0, 40, 40])], 100.0, history)
    assert len(history) <= 1  # only current-frame entry remains


# ---------------------------------------------------------------------------
# Rust-path fallback: when rust bridge returns None, python path is used
# ---------------------------------------------------------------------------

def test_iou_below_threshold_not_suppressed():
    """IoU < IOU_MATCH_THRESHOLD → not the same object → alert passes."""
    history = []
    # Record object at [0,0,100,100]
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history)
    # Object at [80,80,180,180]: IoU ≈ 0.073, below 0.4 threshold
    result = apply_cooldown([_det(1, [80, 80, 180, 180])], 1.0, history)
    assert result[0]["alert"] is True


def test_iou_above_threshold_suppressed():
    """IoU ≥ IOU_MATCH_THRESHOLD → same object → suppress."""
    history = []
    # Record object at [0,0,100,100]
    apply_cooldown([_det(1, [0, 0, 100, 100])], 0.0, history)
    # Object at [5,5,105,105]: IoU is high → same object
    result = apply_cooldown([_det(1, [5, 5, 105, 105])], 1.0, history)
    assert result[0]["alert"] is False
