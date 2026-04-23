from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.geometry import compute_iou
from app.upgrade.tracker import TrackEngine


# ---------------------------------------------------------------------------
# _iou helper
# ---------------------------------------------------------------------------

def test_iou_identical():
    assert compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)


def test_iou_partial():
    # Two 100x100 boxes offset by 50 → 50x50 intersection, union=100*100*2-50*50=17500
    result = compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
    assert result == pytest.approx(2500 / 17500, rel=1e-6)


def test_iou_zero_area():
    assert compute_iou([5, 5, 5, 5], [5, 5, 5, 5]) == pytest.approx(0.0)


def test_iou_accepts_float_bbox():
    result = compute_iou([0.0, 0.0, 100.5, 100.5], [50.25, 50.25, 150.75, 150.75])
    expected_inter = 50.25 * 50.25
    expected_union = 100.5 * 100.5 * 2 - expected_inter
    assert result == pytest.approx(expected_inter / expected_union, rel=1e-6)


def test_iou_invalid_bbox_length_raises():
    with pytest.raises(ValueError, match="exactly 4"):
        compute_iou([0, 0, 1], [0, 0, 1, 1])


def test_iou_non_numeric_bbox_raises():
    with pytest.raises(ValueError, match="numeric"):
        compute_iou(cast(Any, [0, 0, "x", 1]), [0, 0, 1, 1])


def test_iou_non_finite_bbox_raises():
    with pytest.raises(ValueError, match="finite"):
        compute_iou([0, 0, float("nan"), 1], [0, 0, 1, 1])


def test_iou_invalid_bbox_coordinate_order_raises():
    with pytest.raises(ValueError, match="x1 <= x2 and y1 <= y2"):
        compute_iou([10, 0, 5, 5], [0, 0, 1, 1])


@given(
    st.integers(0, 100), st.integers(0, 100),
    st.integers(0, 100), st.integers(0, 100),
)
def test_iou_symmetry(x1, y1, x2, y2):
    a = [0, 0, 50, 50]
    b = [x1, y1, x1 + x2, y1 + y2]
    assert compute_iou(a, b) == pytest.approx(compute_iou(b, a))


@given(
    st.integers(0, 200), st.integers(0, 200),
    st.integers(0, 200), st.integers(0, 200),
)
def test_iou_range(x1, y1, w, h):
    a = [0, 0, 100, 100]
    b = [x1, y1, x1 + w, y1 + h]
    result = compute_iou(a, b)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Detection stub
# ---------------------------------------------------------------------------

@dataclass
class Det:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]


# ---------------------------------------------------------------------------
# TrackEngine
# ---------------------------------------------------------------------------

def make_det(bbox, class_id=0):
    return Det(class_id=class_id, class_name="cls", confidence=0.9, bbox=bbox)


def test_track_new_detection_gets_id():
    engine = TrackEngine()
    tracks = engine.update([make_det([0, 0, 100, 100])])
    assert len(tracks) == 1
    assert tracks[0].track_id >= 1


def test_track_same_box_keeps_id():
    engine = TrackEngine()
    t1 = engine.update([make_det([0, 0, 100, 100])])
    t2 = engine.update([make_det([0, 0, 100, 100])])
    assert t1[0].track_id == t2[0].track_id


def test_track_moved_box_keeps_id():
    engine = TrackEngine()
    t1 = engine.update([make_det([0, 0, 100, 100])])
    # Slightly moved — IoU still > 0.3
    t2 = engine.update([make_det([5, 5, 105, 105])])
    assert t1[0].track_id == t2[0].track_id


def test_track_cross_class_no_match():
    engine = TrackEngine()
    t1 = engine.update([make_det([0, 0, 100, 100], class_id=0)])
    t2 = engine.update([make_det([0, 0, 100, 100], class_id=1)])
    assert t1[0].track_id != t2[0].track_id


def test_track_lost_frames_retirement():
    engine = TrackEngine()
    engine.update([make_det([0, 0, 100, 100])])
    # Feed empty frames until track should be retired
    for _ in range(TrackEngine.MAX_LOST_FRAMES + 1):
        engine.update([])
    # New detection should get a fresh ID
    t_new = engine.update([make_det([0, 0, 100, 100])])
    assert t_new[0].track_id > 1


def test_track_empty_input():
    engine = TrackEngine()
    assert engine.update([]) == []


def test_track_multiple_detections():
    engine = TrackEngine()
    dets = [make_det([0, 0, 50, 50]), make_det([100, 100, 150, 150])]
    tracks = engine.update(dets)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2  # distinct IDs


# ---------------------------------------------------------------------------
# Confidence and bbox update on re-match
# ---------------------------------------------------------------------------

def test_track_bbox_updated_on_rematch():
    engine = TrackEngine()
    engine.update([make_det([0, 0, 100, 100])])
    new_bbox = [5, 5, 105, 105]
    tracks = engine.update([make_det(new_bbox)])
    assert tracks[0].bbox == new_bbox


def test_track_confidence_updated_on_rematch():
    engine = TrackEngine()
    engine.update([Det(class_id=0, class_name="cls", confidence=0.5, bbox=[0, 0, 100, 100])])
    tracks = engine.update([Det(class_id=0, class_name="cls", confidence=0.99, bbox=[0, 0, 100, 100])])
    assert tracks[0].confidence == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Lost-frame boundary: survive exactly MAX_LOST_FRAMES, retire at MAX_LOST_FRAMES+1
# ---------------------------------------------------------------------------

def test_track_survives_exactly_max_lost_frames():
    engine = TrackEngine()
    tracks_init = engine.update([make_det([0, 0, 100, 100])])
    original_id = tracks_init[0].track_id

    for _ in range(TrackEngine.MAX_LOST_FRAMES):
        engine.update([])

    # Should still be active — a re-appearing detection re-matches
    tracks_after = engine.update([make_det([0, 0, 100, 100])])
    assert tracks_after[0].track_id == original_id


def test_track_retired_one_frame_after_max():
    engine = TrackEngine()
    engine.update([make_det([0, 0, 100, 100])])
    for _ in range(TrackEngine.MAX_LOST_FRAMES + 1):
        engine.update([])
    # Track is now retired; fresh detection gets new ID
    new_tracks = engine.update([make_det([0, 0, 100, 100])])
    assert new_tracks[0].track_id > 1


# ---------------------------------------------------------------------------
# Multi-object: one disappears, the other persists
# ---------------------------------------------------------------------------

def test_track_one_disappears_other_persists():
    engine = TrackEngine()
    t_init = engine.update([
        make_det([0, 0, 50, 50]),
        make_det([200, 200, 250, 250]),
    ])
    id_a = t_init[0].track_id
    id_b = t_init[1].track_id

    # Only first object present
    for _ in range(3):
        tracks = engine.update([make_det([0, 0, 50, 50])])

    assert tracks[0].track_id == id_a
    # Object B should have been aged but not yet retired (3 frames < MAX_LOST)
    active_ids = {tr.track_id for tr in engine._active}
    assert id_b in active_ids


# ---------------------------------------------------------------------------
# Hypothesis: track IDs are always positive integers
# ---------------------------------------------------------------------------

@given(
    st.lists(
        st.tuples(
            st.integers(0, 4),          # class_id
            st.integers(0, 400),        # x1
            st.integers(0, 400),        # y1
        ),
        min_size=0,
        max_size=5,
    )
)
def test_track_ids_are_positive(det_specs):
    engine = TrackEngine()
    dets = [Det(class_id=c, class_name="c", confidence=0.9,
                bbox=[x, y, x + 50, y + 50]) for c, x, y in det_specs]
    tracks = engine.update(dets)
    for tr in tracks:
        assert tr.track_id >= 1
