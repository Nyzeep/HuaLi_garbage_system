from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import cv2
import numpy as np

from app.config import get_settings
from app.core.geometry import compute_iou
from app.infrastructure.ml.rust_bridge import RustBridge


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]
    kalman: cv2.KalmanFilter | None = None


class TrackEngine:
    """IoU-based greedy tracker with Kalman filtering.

    Each frame's detections are matched to existing active tracks using IoU.
    Matched tracks keep their ID and get their bbox/confidence updated.
    Unmatched detections spawn new tracks.
    Tracks that go unmatched for more than MAX_LOST_FRAMES are retired.

    Cross-class matches are forbidden — a detection can only be matched to a
    track with the same class_id.

    Kalman filter parameters (process_noise, measurement_noise, error_cov_post)
    can be configured via environment variables or .env file:
        KALMAN_PROCESS_NOISE=0.01
        KALMAN_MEASUREMENT_NOISE=0.1
        KALMAN_ERROR_COV_POST=1.0
    """

    IOU_MATCH_THRESHOLD: float = 0.3
    MAX_LOST_FRAMES: int = 10

    def __init__(
        self,
        rust_bridge: RustBridge | None = None,
        *,
        kalman_process_noise: float | None = None,
        kalman_measurement_noise: float | None = None,
        kalman_error_cov_post: float | None = None,
    ) -> None:
        self._id_gen = count(1)
        self._active: list[Track] = []
        self._lost: dict[int, int] = {}  # track_id → consecutive lost frames
        self._rust_bridge = rust_bridge or RustBridge()

        settings = get_settings()
        self._kalman_process_noise = kalman_process_noise if kalman_process_noise is not None else settings.kalman_process_noise
        self._kalman_measurement_noise = kalman_measurement_noise if kalman_measurement_noise is not None else settings.kalman_measurement_noise
        self._kalman_error_cov_post = kalman_error_cov_post if kalman_error_cov_post is not None else settings.kalman_error_cov_post

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _to_kalman_state(bbox: list[int]) -> np.ndarray:
        return np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0], [0], [0]], dtype=np.float32)

    def _create_kalman(self, bbox: list[int]) -> cv2.KalmanFilter:
        kalman = cv2.KalmanFilter(8, 4)
        kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * self._kalman_process_noise
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * self._kalman_measurement_noise
        kalman.errorCovPost = np.eye(8, dtype=np.float32) * self._kalman_error_cov_post
        kalman.statePost = self._to_kalman_state(bbox)
        return kalman

    @staticmethod
    def _predict_bbox(track: Track) -> list[int]:
        if track.kalman is None:
            return track.bbox
        prediction = track.kalman.predict().flatten()
        return [int(prediction[0]), int(prediction[1]), int(prediction[2]), int(prediction[3])]

    def _correct_track(self, track: Track, bbox: list[int]) -> None:
        if track.kalman is None:
            track.kalman = self._create_kalman(bbox)
            return
        measurement = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], dtype=np.float32)
        track.kalman.correct(measurement)

    def update(self, detections: list) -> list[Track]:
        """Match detections to active tracks and return the current track list."""
        if not detections:
            self._age_all()
            return []

        if not self._active:
            return self._init_tracks(detections)

        matched_track_ids: set[int] = set()
        result: list[Track] = []
        predicted_active = [self._predict_bbox(track) for track in self._active]

        for det in detections:
            best_iou = 0.0
            best_track: Track | None = None

            same_class_tracks = [
                (idx, tr) for idx, tr in enumerate(self._active)
                if tr.track_id not in matched_track_ids and tr.class_id == det.class_id
            ]
            if same_class_tracks:
                left = [list(det.bbox)]
                right = [predicted_active[idx] for idx, _ in same_class_tracks]
                matches = self._rust_bridge.batch_iou_match(left, right, self.IOU_MATCH_THRESHOLD)
                if matches is not None and len(matches) > 0:
                    _, right_index, best_iou = max(matches, key=lambda item: item[2])
                    best_track = same_class_tracks[right_index][1]

            if best_track is None:
                for idx, tr in enumerate(self._active):
                    if tr.track_id in matched_track_ids:
                        continue
                    if tr.class_id != det.class_id:
                        continue
                    score = compute_iou(det.bbox, predicted_active[idx])
                    if score > best_iou:
                        best_iou = score
                        best_track = tr

            if best_track is not None and best_iou >= self.IOU_MATCH_THRESHOLD:
                best_track.bbox = det.bbox
                best_track.confidence = det.confidence
                self._correct_track(best_track, det.bbox)
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
                    kalman=self._create_kalman(list(det.bbox)),
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
                kalman=self._create_kalman(list(d.bbox)),
            )
            for d in detections
        ]
        return list(self._active)

