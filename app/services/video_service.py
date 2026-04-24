from __future__ import annotations
from pathlib import Path

import cv2
import imageio

from app.services.detection_service import DetectionService
from app.upgrade import AlarmEngine, DetectionEngine, TrackEngine, UpgradePipeline


class VideoProcessingError(RuntimeError):
    pass


class VideoProcessingService:
    VIDEO_IOU_MATCH_THRESHOLD = 0.4
    VIDEO_GARBAGE_COOLDOWN_SECONDS = 3.0  # overflow/garbage
    VIDEO_FIRE_SMOKE_COOLDOWN_SECONDS = 1.0  # fire/smoke
    VIDEO_FIRE_SMOKE_MIN_CONFIRM_FRAMES = 2
    VIDEO_FIRE_SMOKE_HIGH_CONF_BYPASS = 0.60

    def __init__(self, detection_service: DetectionService):
        self.detection_service = detection_service
        # New upgrade pipeline is integrated as a non-breaking sidecar layer.
        # It consumes existing detections and adds track/alarm metadata.
        self.upgrade_pipeline = UpgradePipeline(
            detection_engine=DetectionEngine(detection_service),
            track_engine=TrackEngine(),
            alarm_engine=AlarmEngine(min_consecutive_frames=2),
        )

    @staticmethod
    def _bgr_to_rgb(frame):
        # OpenCV uses BGR, while imageio/ffmpeg writer expects RGB arrays.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _compute_iou(box1: list[int], box2: list[int]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _cooldown_seconds_for_class(self, class_id: int) -> float:
        if class_id in (3, 4):  # fire / smoke
            return self.VIDEO_FIRE_SMOKE_COOLDOWN_SECONDS
        if class_id in (1, 2):  # overflow / garbage
            return self.VIDEO_GARBAGE_COOLDOWN_SECONDS
        return 0.0

    def _apply_video_alert_cooldown(
        self,
        detections: list[dict],
        current_ts: float,
        alert_history: list[dict],
    ) -> tuple[list[dict], int]:
        """
        Video-only cooldown:
        - overflow/garbage: same object won't re-alert within 3s
        - fire/smoke: same object won't re-alert within 1s
        - same object match priority: track_id (stable) > bbox IoU fallback
        """
        updated: list[dict] = []
        suppressed_count = 0

        for det in detections:
            item = det.copy()
            if not item.get("alert", False):
                updated.append(item)
                continue

            class_id = int(item.get("class_id", -1))
            cooldown = self._cooldown_seconds_for_class(class_id)
            if cooldown <= 0:
                updated.append(item)
                continue

            bbox = item.get("bbox", [])
            track_id = item.get("track_id")
            has_recent_same_object = False
            for rec in alert_history:
                if rec["class_id"] != class_id:
                    continue
                if current_ts - rec["timestamp"] > cooldown:
                    continue
                # Prefer tracker identity match to avoid repeated alerts caused by
                # bbox jitter on the same object across adjacent frames.
                rec_tid = rec.get("track_id")
                if (track_id is not None) and (rec_tid is not None):
                    if int(track_id) == int(rec_tid):
                        has_recent_same_object = True
                        break
                    continue
                if self._compute_iou(bbox, rec["bbox"]) >= self.VIDEO_IOU_MATCH_THRESHOLD:
                    has_recent_same_object = True
                    break

            if has_recent_same_object:
                item["alert"] = False
                suppressed_count += 1
            else:
                alert_history.append(
                    {
                        "class_id": class_id,
                        "bbox": bbox,
                        "timestamp": current_ts,
                        "track_id": int(track_id) if track_id is not None else None,
                    }
                )
            updated.append(item)

        # Keep alert history bounded to reduce growth.
        max_keep_window = max(
            self.VIDEO_GARBAGE_COOLDOWN_SECONDS,
            self.VIDEO_FIRE_SMOKE_COOLDOWN_SECONDS,
        )
        alert_history[:] = [
            rec for rec in alert_history if current_ts - rec["timestamp"] <= max_keep_window
        ]
        return updated, suppressed_count

    def _attach_upgrade_metadata(
        self,
        detections: list[dict],
        timestamp: float | None = None,
    ) -> tuple[list[dict], int]:
        """Attach track_id from upgrade pipeline without changing existing alert semantics."""
        pipe_result = self.upgrade_pipeline.run_detections(detections, timestamp=timestamp)
        tracks = pipe_result.tracks
        alarms = pipe_result.alarms

        # Keep index alignment with tracker output to avoid key-collision ambiguity.
        out: list[dict] = []
        for idx, det in enumerate(detections):
            item = det.copy()
            if idx < len(tracks):
                item["track_id"] = int(tracks[idx].track_id)
            out.append(item)

        return out, len(alarms)

    def _apply_fire_smoke_temporal_confirmation(
        self,
        detections: list[dict],
        streak_state: dict[tuple[int, int], int],
    ) -> list[dict]:
        """
        Keep fire/smoke alert only after the same tracked object appears
        for enough consecutive processed frames.
        """
        active_keys: set[tuple[int, int]] = set()
        out: list[dict] = []
        for det in detections:
            item = det.copy()
            cid = int(item.get("class_id", -1))
            if not item.get("alert", False) or cid not in (3, 4):
                out.append(item)
                continue

            tid = item.get("track_id")
            if tid is None:
                item["alert"] = False
                out.append(item)
                continue

            key = (cid, int(tid))
            active_keys.add(key)
            count = int(streak_state.get(key, 0)) + 1
            streak_state[key] = count
            conf = float(item.get("confidence", 0.0))
            high_conf_bypass = conf >= self.VIDEO_FIRE_SMOKE_HIGH_CONF_BYPASS
            if (not high_conf_bypass) and count < self.VIDEO_FIRE_SMOKE_MIN_CONFIRM_FRAMES:
                item["alert"] = False
            out.append(item)

        for key in list(streak_state.keys()):
            if key not in active_keys:
                streak_state.pop(key, None)
        return out

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        skip_frames: int,
        progress_callback=None,
    ) -> dict:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise VideoProcessingError("无法读取视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or fps > 120:
            fps = 30.0

        frame_count = 0
        total_detections = 0
        total_alerts = 0
        total_suppressed_alerts = 0
        alert_type_set: set[str] = set()
        alert_frames = 0
        prev_result = None
        effective_skip = max(skip_frames, 1)
        alert_history: list[dict] = []
        fire_smoke_streak: dict[tuple[int, int], int] = {}
        total_pipeline_alarms = 0

        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Detect on frame 1, 1+skip, 1+2*skip ...
                # When skip_frames=1, every frame is processed.
                if (frame_count - 1) % effective_skip != 0:
                    frame_to_write = prev_result if prev_result is not None else frame
                    writer.append_data(self._bgr_to_rgb(frame_to_write))
                    if progress_callback and total_frames:
                        progress_callback(frame_count, total_frames)
                    continue

                detections = self.detection_service.detect(frame)
                current_ts = (frame_count / fps) if fps > 0 else 0.0
                detections, pipeline_alarm_count = self._attach_upgrade_metadata(
                    detections,
                    timestamp=current_ts,
                )
                detections = self._apply_fire_smoke_temporal_confirmation(
                    detections=detections,
                    streak_state=fire_smoke_streak,
                )
                detections, suppressed_this_frame = self._apply_video_alert_cooldown(
                    detections=detections,
                    current_ts=current_ts,
                    alert_history=alert_history,
                )
                total_suppressed_alerts += int(suppressed_this_frame)
                total_pipeline_alarms += pipeline_alarm_count

                rendered = self.detection_service.draw_boxes(frame, detections)
                prev_result = rendered.copy()

                frame_alerts = sum(1 for item in detections if item.get("alert", False))
                for item in detections:
                    if item.get("alert", False):
                        alert_type_set.add(str(item.get("class_name", "unknown")))
                if frame_alerts > 0:
                    alert_frames += 1
                total_alerts += frame_alerts
                total_detections += len(detections)

                cv2.putText(
                    rendered,
                    f"Frame {frame_count}: {len(detections)} detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    rendered,
                    f"Upgrade alarms: {total_pipeline_alarms}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 255),
                    2,
                )
                cv2.putText(
                    rendered,
                    f"Suppressed: {total_suppressed_alerts}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (150, 220, 255),
                    2,
                )
                writer.append_data(self._bgr_to_rgb(rendered))

                if progress_callback and total_frames:
                    progress_callback(frame_count, total_frames)
        finally:
            writer.close()
            cap.release()

        return {
            "total_frames": frame_count,
            "detected_frames": alert_frames,
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "suppressed_alerts": total_suppressed_alerts,
            "alert_types": sorted(alert_type_set),
            "video_info": f"{width}x{height}, {fps:.1f}fps, suppressed={total_suppressed_alerts}",
        }
