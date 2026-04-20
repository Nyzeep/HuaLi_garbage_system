from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]


class TrackEngine:
    """IoU-based greedy tracker.

    Each frame's detections are matched to existing active tracks using IoU.
    Matched tracks keep their ID and get their bbox/confidence updated.
    Unmatched detections spawn new tracks.
    Tracks that go unmatched for more than MAX_LOST_FRAMES are retired.

    Cross-class matches are forbidden — a detection can only be matched to a
    track with the same class_id.
    """

    IOU_MATCH_THRESHOLD: float = 0.3
    MAX_LOST_FRAMES: int = 10

    def __init__(self) -> None:
        self._id_gen = count(1)
        self._active: list[Track] = []
        self._lost: dict[int, int] = {}  # track_id → consecutive lost frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list) -> list[Track]:
        """Match detections to active tracks and return the current track list."""
        if not detections:
            self._age_all()
            return []

        if not self._active:
            return self._init_tracks(detections)

        matched_track_ids: set[int] = set()
        result: list[Track] = []

        for det in detections:
            best_iou = 0.0
            best_track: Track | None = None

            for tr in self._active:
                if tr.track_id in matched_track_ids:
                    continue
                if tr.class_id != det.class_id:
                    continue
                score = _iou(det.bbox, tr.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = tr

            if best_track is not None and best_iou >= self.IOU_MATCH_THRESHOLD:
                best_track.bbox = det.bbox
                best_track.confidence = det.confidence
                matched_track_ids.add(best_track.track_id)
                self._lost.pop(best_track.track_id, None)
                result.append(best_track)
            else:
                new_track = Track(
                    track_id=next(self._id_gen),
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=list(det.bbox),
                )
                result.append(new_track)

        # Age out tracks that were not matched this frame.
        for tr in self._active:
            if tr.track_id not in matched_track_ids:
                self._lost[tr.track_id] = self._lost.get(tr.track_id, 0) + 1

        # Rebuild active list: matched + still-alive lost tracks.
        self._active = result + [
            tr for tr in self._active
            if tr.track_id not in matched_track_ids
            and self._lost.get(tr.track_id, 0) <= self.MAX_LOST_FRAMES
        ]

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _age_all(self) -> None:
        for tr in self._active:
            self._lost[tr.track_id] = self._lost.get(tr.track_id, 0) + 1
        self._active = [
            tr for tr in self._active
            if self._lost.get(tr.track_id, 0) <= self.MAX_LOST_FRAMES
        ]

    def _init_tracks(self, detections: list) -> list[Track]:
        self._active = [
            Track(
                track_id=next(self._id_gen),
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=list(d.bbox),
            )
            for d in detections
        ]
        return list(self._active)


def _iou(bbox1: list[int], bbox2: list[int]) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
