from __future__ import annotations
import logging
import time
from pathlib import Path

import cv2
import imageio

from app.infrastructure.ml.rust_bridge import RustBridge
from app.services.detection_service import DetectionService
from app.upgrade import AlarmEngine, DetectionEngine, TrackEngine, UpgradePipeline


logger = logging.getLogger(__name__)


class VideoProcessingError(RuntimeError):
    pass


class VideoProcessingService:
    VIDEO_IOU_MATCH_THRESHOLD = 0.4
    VIDEO_GARBAGE_COOLDOWN_SECONDS = 3.0  # overflow/garbage
    VIDEO_FIRE_SMOKE_COOLDOWN_SECONDS = 1.0  # fire/smoke
    VIDEO_ENCODER_CANDIDATES = (
        "libx264",
        "h264_nvenc",
        "h264_qsv",
        "h264_amf",
    )

    def __init__(self, detection_service: DetectionService, rust_bridge: RustBridge | None = None):
        self.detection_service = detection_service
        self.rust_bridge = rust_bridge or RustBridge()
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

    # Class groups for per-cooldown Rust deduplication
    _FIRE_SMOKE_CLASS_IDS: frozenset[int] = frozenset({3, 4})
    _GARBAGE_CLASS_IDS: frozenset[int] = frozenset({0, 1, 2})

    def _apply_video_alert_cooldown(
        self,
        detections: list[dict],
        current_ts: float,
        alert_history: list[dict],
    ) -> list[dict]:
        """
        Video-only cooldown:
        - overflow/garbage: same object won't re-alert within 3s
        - fire/smoke: same object won't re-alert within 1s

        When the Rust binary is available, deduplication is performed as a
        single batch call per cooldown group.  Falls back to the pure-Python
        implementation transparently.
        """
        started_at = time.perf_counter()
        result = self._apply_video_alert_cooldown_rust(detections, current_ts, alert_history)
        if result is not None:
            logger.info(
                "video alert cooldown completed path=rust detections=%d duration_ms=%.2f",
                len(detections),
                (time.perf_counter() - started_at) * 1000,
            )
            return result
        result = self._apply_video_alert_cooldown_python(detections, current_ts, alert_history)
        logger.warning(
            "video alert cooldown completed path=python-fallback detections=%d duration_ms=%.2f",
            len(detections),
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    def _apply_video_alert_cooldown_rust(
        self,
        detections: list[dict],
        current_ts: float,
        alert_history: list[dict],
    ) -> list[dict] | None:
        """
        Rust-accelerated path.  Converts alert history and new alert detections
        to TrackEvents and calls Rust dedupe_track_events per cooldown group.
        Returns None to signal that the caller should fall back to Python.
        """
        current_ts_ms = int(current_ts * 1000)

        alert_indices = [
            i for i, d in enumerate(detections)
            if d.get("alert", False) and self._cooldown_seconds_for_class(int(d.get("class_id", -1))) > 0
        ]
        if not alert_indices:
            return detections

        # Convert history to ms-timestamped events.
        def hist_as_events(class_ids: frozenset[int]) -> list[dict]:
            return [
                {"class_id": r["class_id"], "bbox": r["bbox"], "timestamp_ms": int(r["timestamp"] * 1000)}
                for r in alert_history
                if r["class_id"] in class_ids
            ]

        def new_as_events(class_ids: frozenset[int]) -> list[tuple[int, dict]]:
            return [
                (i, {"class_id": int(detections[i].get("class_id", -1)),
                     "bbox": detections[i]["bbox"],
                     "timestamp_ms": current_ts_ms})
                for i in alert_indices
                if int(detections[i].get("class_id", -1)) in class_ids
            ]

        # Per-group deduplication with the correct cooldown window.
        suppressed_indices: set[int] = set()
        new_history: list[dict] = []

        for class_ids, cooldown_s in (
            (self._FIRE_SMOKE_CLASS_IDS, self.VIDEO_FIRE_SMOKE_COOLDOWN_SECONDS),
            (self._GARBAGE_CLASS_IDS, self.VIDEO_GARBAGE_COOLDOWN_SECONDS),
        ):
            group_new = new_as_events(class_ids)
            if not group_new:
                # Preserve unaffected history for this group.
                new_history.extend(hist_as_events(class_ids))
                continue

            group_hist = hist_as_events(class_ids)
            cooldown_ms = int(cooldown_s * 1000)
            all_events = group_hist + [e for _, e in group_new]

            kept = self.rust_bridge.dedupe_events(all_events, cooldown_ms, self.VIDEO_IOU_MATCH_THRESHOLD)
            if kept is None:
                return None  # signal fallback

            # Surviving events whose timestamp matches current frame are the non-suppressed new alerts.
            surviving_keys: set[tuple[int, tuple]] = {
                (e["class_id"], tuple(e["bbox"]))
                for e in kept
                if e["timestamp_ms"] == current_ts_ms
            }
            for idx, evt in group_new:
                key = (evt["class_id"], tuple(evt["bbox"]))
                if key not in surviving_keys:
                    suppressed_indices.add(idx)

            new_history.extend(kept)

        # Rebuild history from Rust-deduped result.
        alert_history[:] = [
            {"class_id": e["class_id"], "bbox": e["bbox"], "timestamp": e["timestamp_ms"] / 1000.0}
            for e in new_history
        ]

        # Apply suppression to detections list.
        result = []
        for i, det in enumerate(detections):
            if i in suppressed_indices:
                item = det.copy()
                item["alert"] = False
                result.append(item)
            else:
                result.append(det)
        return result

    def _apply_video_alert_cooldown_python(
        self,
        detections: list[dict],
        current_ts: float,
        alert_history: list[dict],
    ) -> list[dict]:
        """Pure-Python fallback — original O(n × history) implementation."""
        updated = []

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
            has_recent_same_object = False
            for rec in alert_history:
                if rec["class_id"] != class_id:
                    continue
                if current_ts - rec["timestamp"] > cooldown:
                    continue
                if self._compute_iou(bbox, rec["bbox"]) >= self.VIDEO_IOU_MATCH_THRESHOLD:
                    has_recent_same_object = True
                    break

            if has_recent_same_object:
                item["alert"] = False
            else:
                alert_history.append({"class_id": class_id, "bbox": bbox, "timestamp": current_ts})
            updated.append(item)

        max_keep_window = max(
            self.VIDEO_GARBAGE_COOLDOWN_SECONDS,
            self.VIDEO_FIRE_SMOKE_COOLDOWN_SECONDS,
        )
        alert_history[:] = [
            rec for rec in alert_history if current_ts - rec["timestamp"] <= max_keep_window
        ]
        return updated

    @staticmethod
    def _find_ffmpeg_output_path(error: Exception) -> Path | None:
        message = str(error)
        marker = "ffmpeg error: [Errno 2] No such file or directory: '"
        if marker not in message:
            return None
        tail = message.split(marker, 1)[1]
        candidate = tail.split("'", 1)[0]
        return Path(candidate) if candidate else None

    @staticmethod
    def _build_video_writer(output_path: Path, fps: float):
        for codec in VideoProcessingService.VIDEO_ENCODER_CANDIDATES:
            try:
                writer = imageio.get_writer(
                    str(output_path),
                    fps=fps,
                    codec=codec,
                    pixelformat="yuv420p",
                )
                return writer, codec
            except Exception:
                continue
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )
        return writer, "libx264"

    def _append_frame_with_encoder_fallback(
        self,
        writer,
        frame_rgb,
        output_path: Path,
        fps: float,
    ):
        try:
            writer.append_data(frame_rgb)
            return writer, output_path
        except Exception as exc:
            fallback_path = self._find_ffmpeg_output_path(exc)
            if fallback_path is None:
                raise
            writer.close()
            fallback_writer = imageio.get_writer(
                str(fallback_path),
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
            )
            fallback_writer.append_data(frame_rgb)
            return fallback_writer, fallback_path

    def _attach_upgrade_metadata(self, detections: list[dict]) -> tuple[list[dict], int]:
        """Attach track_id from upgrade pipeline without changing existing alert semantics."""
        pipe_result = self.upgrade_pipeline.run_detections(detections)
        tracks = pipe_result.tracks
        alarms = pipe_result.alarms

        # Map by (class_id, bbox) for deterministic binding in current frame.
        track_map: dict[tuple[int, tuple[int, int, int, int]], int] = {}
        for tr in tracks:
            key = (int(tr.class_id), tuple(int(v) for v in tr.bbox))
            track_map[key] = int(tr.track_id)

        out: list[dict] = []
        for det in detections:
            key = (int(det.get("class_id", -1)), tuple(int(v) for v in det.get("bbox", [])))
            item = det.copy()
            if key in track_map:
                item["track_id"] = track_map[key]
            out.append(item)

        return out, len(alarms)

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        skip_frames: int,
        progress_callback=None,
    ) -> dict:
        started_at = time.perf_counter()
        logger.info(
            "video processing started input=%s output=%s skip_frames=%d",
            input_path,
            output_path,
            skip_frames,
        )
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
        alert_frames = 0
        prev_result = None
        effective_skip = max(skip_frames, 1)
        alert_history: list[dict] = []
        total_pipeline_alarms = 0

        writer, encoder_used = self._build_video_writer(output_path, fps)
        current_output_path = output_path

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if (frame_count - 1) % effective_skip != 0:
                    frame_to_write = prev_result if prev_result is not None else frame
                    frame_rgb = self._bgr_to_rgb(frame_to_write)
                    writer, current_output_path = self._append_frame_with_encoder_fallback(
                        writer,
                        frame_rgb,
                        current_output_path,
                        fps,
                    )
                    if progress_callback and total_frames:
                        progress_callback(frame_count, total_frames)
                    continue

                frame_started_at = time.perf_counter()
                detections = self.detection_service.detect_raw(frame)
                detections = self._apply_video_alert_cooldown(
                    detections=detections,
                    current_ts=(frame_count / fps) if fps > 0 else 0.0,
                    alert_history=alert_history,
                )
                detections, pipeline_alarm_count = self._attach_upgrade_metadata(detections)
                total_pipeline_alarms += pipeline_alarm_count

                rendered = self.detection_service.draw_boxes(frame, detections)
                prev_result = rendered.copy()

                frame_alerts = sum(1 for item in detections if item.get("alert", False))
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
                frame_rgb = self._bgr_to_rgb(rendered)
                writer, current_output_path = self._append_frame_with_encoder_fallback(
                    writer,
                    frame_rgb,
                    current_output_path,
                    fps,
                )
                logger.info(
                    "video frame processed frame=%d/%d detections=%d alerts=%d duration_ms=%.2f",
                    frame_count,
                    total_frames,
                    len(detections),
                    frame_alerts,
                    (time.perf_counter() - frame_started_at) * 1000,
                )

                if progress_callback and total_frames:
                    progress_callback(frame_count, total_frames)
        finally:
            writer.close()
            cap.release()

        if encoder_used != "libx264" and current_output_path != output_path:
            try:
                if output_path.exists():
                    output_path.unlink()
                current_output_path.replace(output_path)
            except Exception:
                pass

        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "video processing completed frames=%d detected_frames=%d total_detections=%d total_alerts=%d duration_ms=%.2f",
            frame_count,
            alert_frames,
            total_detections,
            total_alerts,
            duration_ms,
        )
        return {
            "total_frames": frame_count,
            "detected_frames": alert_frames,
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "video_info": f"{width}x{height}, {fps:.1f}fps",
        }
