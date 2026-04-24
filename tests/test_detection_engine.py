from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from app.upgrade.alarm import AlarmEngine
from app.upgrade.detection import Detection, DetectionEngine
from app.upgrade.pipeline import PipelineResult, UpgradePipeline
from app.upgrade.tracker import TrackEngine


# ---------------------------------------------------------------------------
# DetectionEngine.adapt()
# ---------------------------------------------------------------------------

def test_adapt_full_dict():
    engine = DetectionEngine(detector=None)
    raw = [{"class_id": 2, "class_name": "散落垃圾", "confidence": 0.88, "bbox": [10, 20, 50, 60]}]
    result = engine.adapt(raw)
    assert len(result) == 1
    d = result[0]
    assert isinstance(d, Detection)
    assert d.class_id == 2
    assert d.class_name == "散落垃圾"
    assert d.confidence == pytest.approx(0.88)
    assert d.bbox == [10, 20, 50, 60]


def test_adapt_missing_fields_use_defaults():
    engine = DetectionEngine(detector=None)
    result = engine.adapt([{}])
    assert len(result) == 1
    d = result[0]
    assert d.class_id == -1
    assert d.class_name == "unknown"
    assert d.confidence == pytest.approx(0.0)
    assert d.bbox == []


def test_adapt_empty_list():
    engine = DetectionEngine(detector=None)
    assert engine.adapt([]) == []


def test_adapt_multiple():
    engine = DetectionEngine(detector=None)
    raw = [
        {"class_id": 0, "class_name": "a", "confidence": 0.5, "bbox": [0, 0, 10, 10]},
        {"class_id": 3, "class_name": "b", "confidence": 0.9, "bbox": [5, 5, 15, 15]},
    ]
    result = engine.adapt(raw)
    assert len(result) == 2
    assert result[0].class_id == 0
    assert result[1].class_id == 3


def test_adapt_class_id_cast_to_int():
    engine = DetectionEngine(detector=None)
    result = engine.adapt([{"class_id": "2"}])
    assert isinstance(result[0].class_id, int)
    assert result[0].class_id == 2


def test_infer_delegates_to_detector():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        {"class_id": 1, "class_name": "垃圾溢出", "confidence": 0.75, "bbox": [0, 0, 100, 100]}
    ]
    engine = DetectionEngine(detector=mock_detector)
    frame = object()
    result = engine.infer(frame)
    mock_detector.detect.assert_called_once_with(frame)
    assert len(result) == 1
    assert result[0].class_id == 1


# ---------------------------------------------------------------------------
# UpgradePipeline.run_detections()
# ---------------------------------------------------------------------------

def _make_pipeline(min_frames: int = 2) -> UpgradePipeline:
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    return UpgradePipeline(
        detection_engine=DetectionEngine(mock_detector),
        track_engine=TrackEngine(),
        alarm_engine=AlarmEngine(min_consecutive_frames=min_frames),
    )


def test_run_detections_empty():
    pipeline = _make_pipeline()
    result = pipeline.run_detections([])
    assert isinstance(result, PipelineResult)
    assert result.detections == []
    assert result.tracks == []
    assert result.alarms == []


def test_run_detections_single_det_no_alarm_yet():
    pipeline = _make_pipeline(min_frames=2)
    raw = [{"class_id": 3, "class_name": "火焰", "confidence": 0.9, "bbox": [0, 0, 50, 50]}]
    result = pipeline.run_detections(raw)
    assert len(result.tracks) == 1
    assert result.alarms == []  # min_consecutive_frames=2, first frame → no alarm


def test_run_detections_alarm_fires_after_min_frames():
    pipeline = _make_pipeline(min_frames=2)
    raw = [{"class_id": 3, "class_name": "火焰", "confidence": 0.9, "bbox": [0, 0, 50, 50]}]
    pipeline.run_detections(raw)           # frame 1 — no alarm
    result = pipeline.run_detections(raw)  # frame 2 — alarm fires
    assert len(result.alarms) == 1
    assert result.alarms[0].class_id == 3


def test_run_detections_returns_adapted_detections():
    pipeline = _make_pipeline()
    raw = [{"class_id": 1, "class_name": "垃圾溢出", "confidence": 0.8, "bbox": [1, 2, 3, 4]}]
    result = pipeline.run_detections(raw)
    assert isinstance(result.detections[0], Detection)
    assert result.detections[0].class_id == 1


def test_run_detections_track_ids_stable_across_calls():
    pipeline = _make_pipeline()
    raw = [{"class_id": 0, "class_name": "cls", "confidence": 0.9, "bbox": [0, 0, 100, 100]}]
    r1 = pipeline.run_detections(raw)
    r2 = pipeline.run_detections(raw)
    assert r1.tracks[0].track_id == r2.tracks[0].track_id


# ---------------------------------------------------------------------------
# UpgradePipeline.run_frame()
# ---------------------------------------------------------------------------

def test_run_frame_calls_infer():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    pipeline = UpgradePipeline(
        detection_engine=DetectionEngine(mock_detector),
        track_engine=TrackEngine(),
        alarm_engine=AlarmEngine(),
    )
    frame = object()
    result = pipeline.run_frame(frame)
    mock_detector.detect.assert_called_once_with(frame)
    assert isinstance(result, PipelineResult)
