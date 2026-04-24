from __future__ import annotations

import pytest

from app.upgrade.alarm import AlarmEngine
from app.upgrade.tracker import Track


def make_track(track_id=1, class_id=0):
    return Track(track_id=track_id, class_id=class_id, class_name="cls", confidence=0.9, bbox=[0, 0, 50, 50])


def test_alarm_fires_at_threshold():
    engine = AlarmEngine(min_consecutive_frames=3)
    tr = make_track()
    assert engine.evaluate([tr]) == []
    assert engine.evaluate([tr]) == []
    alarms = engine.evaluate([tr])
    assert len(alarms) == 1
    assert alarms[0].track_id == tr.track_id
    assert alarms[0].level == "medium"


def test_alarm_fires_every_frame_after_threshold():
    engine = AlarmEngine(min_consecutive_frames=2)
    tr = make_track()
    engine.evaluate([tr])
    for _ in range(5):
        alarms = engine.evaluate([tr])
        assert len(alarms) == 1


def test_alarm_independent_tracks():
    engine = AlarmEngine(min_consecutive_frames=2)
    t1 = make_track(track_id=1)
    t2 = make_track(track_id=2)
    engine.evaluate([t1])
    engine.evaluate([t2])
    alarms = engine.evaluate([t1, t2])
    fired_ids = {a.track_id for a in alarms}
    assert fired_ids == {1, 2}


def test_alarm_no_tracks():
    engine = AlarmEngine()
    assert engine.evaluate([]) == []


def test_alarm_min_consecutive_frames_1():
    engine = AlarmEngine(min_consecutive_frames=1)
    alarms = engine.evaluate([make_track()])
    assert len(alarms) == 1
