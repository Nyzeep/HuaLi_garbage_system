from __future__ import annotations
import logging
import time
from pathlib import Path
from queue import Queue
import threading

import cv2
import imageio

from app.config import get_settings
from app.core.geometry import compute_iou
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
    BIN_COLOR_REFRESH_SECONDS = 3.0
    TEMPORAL_CONFIRM_FRAMES = 2
    MAX_ALERT_HISTORY_PER_CLASS = 128
    VIDEO_ENCODER_CANDIDATES = (
        "libx264",
        "h264_nvenc",
        "h264_qsv",
        "h264_amf",
    )

    def __init__(self, detection_service: DetectionService, rust_bridge: RustBridge | None = None):
        self.detection_service = detection_service
        self.rust_bridge = rust_bridge or RustBridge()
        self._settings = get_settings()
        self._video_bin_color_cache: dict[int, dict] = {}
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

    def _cooldown_seconds_for_class(self, class_id: int) -> float:
        if class_id in (3, 4):  # fire / smoke
            return self.VIDEO_FIRE_SMOKE_COOLDOWN_SECONDS
        if class_id in (1, 2):  # overflow / garbage
            return self.VIDEO_GARBAGE_COOLDOWN_SECONDS
        return 0.0

    # Class groups for per-cooldown Rust deduplication
    _FIRE_SMOKE_CLASS_IDS: frozenset[int] = frozenset({3, 4})
    _GARBAGE_CLASS_IDS: frozenset[int] = frozenset({0, 1, 2})

    @staticmethod
    def _temporal_signature(detection: dict) -> tuple[int, tuple[int, int, int, int]]:
        bbox = detection.get("bbox", [0, 0, 0, 0])
        return int(detection.get("class_id", -1)), tuple(int(v) for v in bbox)

    def _apply_temporal_consistency(self, detections: list[dict], temporal_state: dict[tuple[int, tuple[int, int, int, int]], int]) -> list[dict]:
        updated: list[dict] = []
        active_keys: set[tuple[int, tuple[int, int, int, int]]] = set()
        for detection in detections:
            item = detection.copy()
            class_id = int(item.get("class_id", -1))
            if not item.get("alert", False) or self._cooldown_seconds_for_class(class_id) <= 0:
                updated.append(item)
                continue
            key = self._temporal_signature(item)
            active_keys.add(key)
            count = temporal_state.get(key, 0) + 1
            temporal_state[key] = count
            if count < self.TEMPORAL_CONFIRM_FRAMES:
                item["alert"] = False
            updated.append(item)
        stale_keys = [key for key in temporal_state if key not in active_keys]
        for key in stale_keys:
            temporal_state.pop(key, None)
        return updated

    def _prune_and_cap_alert_history(self, alert_history: list[dict], current_ts: float) -> None:
        max_keep_window = max(
            self.VIDEO_GARBAGE_COOLDOWN_SECONDS,
            self.VIDEO_FIRE_SMOKE_COOLDOWN_SECONDS,
        )
        recent_history = [
            rec for rec in alert_history if current_ts - rec["timestamp"] <= max_keep_window
        ]

        by_class: dict[int, list[dict]] = {}
        passthrough: list[dict] = []
        for rec in recent_history:
            class_id = int(rec.get("class_id", -1))
            if class_id < 0:
                passthrough.append(rec)
                continue
            by_class.setdefault(class_id, []).append(rec)

        capped_history = passthrough
        for class_history in by_class.values():
            capped_history.extend(class_history[-self.MAX_ALERT_HISTORY_PER_CLASS :])
        capped_history.sort(key=lambda item: item["timestamp"])
        alert_history[:] = capped_history

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
        self._prune_and_cap_alert_history(alert_history, current_ts)

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
        """Pure-Python fallback — original O(n x history) implementation."""
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
                if compute_iou(bbox, rec["bbox"]) >= self.VIDEO_IOU_MATCH_THRESHOLD:
                    has_recent_same_object = True
                    break

            if has_recent_same_object:
                item["alert"] = False
            else:
                alert_history.append({"class_id": class_id, "bbox": bbox, "timestamp": current_ts})
            updated.append(item)

        self._prune_and_cap_alert_history(alert_history, current_ts)
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

    def _attach_video_bin_color_cached(self, frame, detections: list[dict], current_ts: float) -> list[dict]:
        if self.detection_service.bin_color_service is None or not self.detection_service.bin_color_service.loaded:
            return detections

        cache = self._video_bin_color_cache
        updated: list[dict] = []
        for det in detections:
            item = det.copy()
            if item.get("class_id") != 0 or item.get("track_id") is None:
                updated.append(item)
                continue

            track_id = int(item["track_id"])
            cached = cache.get(track_id)
            needs_refresh = cached is None or current_ts - float(cached["updated_at"]) >= self.BIN_COLOR_REFRESH_SECONDS
            if needs_refresh:
                enriched = self.detection_service._attach_bin_color(frame, [item])[0]
                cache[track_id] = {
                    "updated_at": current_ts,
                    "bin_color": enriched.get("bin_color"),
                    "bin_color_confidence": enriched.get("bin_color_confidence"),
                    "bin_type_key": enriched.get("bin_type_key"),
                    "bin_type_name": enriched.get("bin_type_name"),
                    "class_name": enriched.get("class_name"),
                }
                item = enriched
            else:
                item.update({k: v for k, v in cached.items() if k != "updated_at" and v is not None})
            updated.append(item)

        active_track_ids = {int(det["track_id"]) for det in updated if det.get("class_id") == 0 and det.get("track_id") is not None}
        stale_track_ids = [track_id for track_id in cache if track_id not in active_track_ids]
        for track_id in stale_track_ids:
            cache.pop(track_id, None)
        return updated

    @staticmethod
    def _frame_perceptual_hash(frame) -> tuple[bytes, int, int]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.tobytes(), gray.shape[1], gray.shape[0]

    def _adaptive_micro_batch_size(self, last_detection_count: int, last_alert_count: int) -> int:
        min_batch = max(1, self._settings.video_micro_batch_size)
        max_batch = max(min_batch, self._settings.video_micro_batch_size_max)
        if last_alert_count > 0 or last_detection_count > 3:
            return min_batch
        if last_detection_count == 0:
            return max_batch
        return min(max_batch, max(min_batch, min_batch * 2))

    def _adaptive_skip_frames(
        self,
        requested_skip: int,
        last_detection_count: int,
        last_alert_count: int,
        current_hash: int | None = None,
        previous_hash: int | None = None,
    ) -> int:
        min_skip = max(1, self._settings.adaptive_skip_min)
        max_skip = max(min_skip, self._settings.adaptive_skip_max)
        if last_alert_count > 0 or last_detection_count > 3:
            return min_skip
        if current_hash is not None and previous_hash is not None:
            distance = self.rust_bridge.hamming_distance(previous_hash, current_hash)
            if distance is not None and distance <= 4:
                return max_skip
            if distance is not None and distance >= 16:
                return min_skip
        if last_detection_count == 0:
            return min(max_skip, max(min_skip, requested_skip + 1))
        return max(min_skip, min(max_skip, requested_skip))

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

        requested_skip = max(skip_frames, 1)
        writer, encoder_used = self._build_video_writer(output_path, fps)
        logger.info("video writer initialized encoder=%s output=%s", encoder_used, output_path)
        current_output_path = output_path

        decode_queue: Queue[tuple[int, any] | None] = Queue(maxsize=32)
        infer_queue: Queue[dict | None] = Queue(maxsize=32)
        post_queue: Queue[dict | None] = Queue(maxsize=32)

        state_lock = threading.Lock()
        stop_event = threading.Event()
        exception_holder: list[BaseException] = []
        supports_batch = all(
            getattr(bundle.backend, "_supports_batch", False)
            for bundle in self.detection_service.inference_service.registry.items()
            if bundle.backend is not None and bundle.backend.loaded
        )
        shared_state = {
            "effective_skip": requested_skip,
            "micro_batch_size": max(1, self._settings.video_micro_batch_size) if supports_batch else 1,
            "previous_frame_hash": None,
            "prev_result": None,
            "total_pipeline_alarms": 0,
        }
        stats = {
            "frame_count": 0,
            "total_detections": 0,
            "total_alerts": 0,
            "alert_frames": 0,
        }
        stage_metrics = {
            "decode_ms": 0.0,
            "infer_ms": 0.0,
            "post_ms": 0.0,
            "encode_ms": 0.0,
            "decoded_frames": 0,
            "inferred_frames": 0,
            "postprocessed_frames": 0,
            "encoded_frames": 0,
        }
        alert_history: list[dict] = []
        temporal_state: dict[tuple[int, tuple[int, int, int, int]], int] = {}

        def fail(exc: BaseException) -> None:
            logger.exception("video pipeline worker failed", exc_info=exc)
            if not exception_holder:
                exception_holder.append(exc)
            stop_event.set()
            for queue in (decode_queue, infer_queue, post_queue):
                try:
                    queue.put_nowait(None)
                except Exception:
                    pass

        def decode_worker() -> None:
            logger.info("video decode worker started")
            frame_index = 0
            try:
                while not stop_event.is_set():
                    decode_started_at = time.perf_counter()
                    ret, frame = cap.read()
                    stage_metrics["decode_ms"] += (time.perf_counter() - decode_started_at) * 1000
                    if not ret:
                        decode_queue.put(None)
                        return
                    frame_index += 1
                    stage_metrics["decoded_frames"] += 1
                    decode_queue.put((frame_index, frame))
            except BaseException as exc:
                fail(exc)

        def infer_worker() -> None:
            logger.info("video infer worker started")
            try:
                pending_batch: list[tuple[int, any]] = []
                while not stop_event.is_set():
                    item = decode_queue.get()
                    if item is None:
                        if pending_batch:
                            frame_indices = [frame_index for frame_index, _ in pending_batch]
                            frames = [frame for _, frame in pending_batch]
                            logger.info(
                                "video infer batch flush_on_eof batch_size=%d first_frame=%d last_frame=%d",
                                len(pending_batch),
                                frame_indices[0],
                                frame_indices[-1],
                            )
                            infer_started_at = time.perf_counter()
                            detections_batch = self.detection_service.detect_raw_batch(frames)
                            infer_duration_ms = (time.perf_counter() - infer_started_at) * 1000
                            for frame_index, frame, detections in zip(frame_indices, frames, detections_batch, strict=False):
                                stage_metrics["infer_ms"] += infer_duration_ms / max(len(pending_batch), 1)
                                stage_metrics["inferred_frames"] += 1
                                if progress_callback and total_frames:
                                    progress_callback(frame_index, total_frames)
                                logger.info("video infer enqueue frame=%d detections=%d", frame_index, len(detections))
                                infer_queue.put(
                                    {
                                        "frame_count": frame_index,
                                        "frame": frame,
                                        "skipped": False,
                                        "detections": detections,
                                        "infer_duration_ms": infer_duration_ms / max(len(pending_batch), 1),
                                    }
                                )
                            pending_batch.clear()
                        infer_queue.put(None)
                        return
                    frame_index, frame = item
                    with state_lock:
                        effective_skip = int(shared_state["effective_skip"])
                        micro_batch_size = int(shared_state["micro_batch_size"])
                    if (frame_index - 1) % effective_skip != 0:
                        infer_queue.put({"frame_count": frame_index, "frame": frame, "skipped": True})
                        continue
                    pending_batch.append((frame_index, frame))
                    if len(pending_batch) >= max(1, micro_batch_size):
                        frame_indices = [index for index, _ in pending_batch]
                        frames = [frame for _, frame in pending_batch]
                        logger.info(
                            "video infer batch dispatch batch_size=%d first_frame=%d last_frame=%d effective_skip=%d",
                            len(pending_batch),
                            frame_indices[0],
                            frame_indices[-1],
                            micro_batch_size,
                        )
                        infer_started_at = time.perf_counter()
                        detections_batch = self.detection_service.detect_raw_batch(frames)
                        infer_duration_ms = (time.perf_counter() - infer_started_at) * 1000
                        for frame_index, frame, detections in zip(frame_indices, frames, detections_batch, strict=False):
                            stage_metrics["infer_ms"] += infer_duration_ms / max(len(pending_batch), 1)
                            stage_metrics["inferred_frames"] += 1
                            if progress_callback and total_frames:
                                progress_callback(frame_index, total_frames)
                            logger.info("video infer enqueue frame=%d detections=%d", frame_index, len(detections))
                            infer_queue.put(
                                {
                                    "frame_count": frame_index,
                                    "frame": frame,
                                    "skipped": False,
                                    "detections": detections,
                                    "infer_duration_ms": infer_duration_ms / max(len(pending_batch), 1),
                                }
                            )
                        pending_batch.clear()
            except BaseException as exc:
                fail(exc)

        def post_worker() -> None:
            logger.info("video post worker started")
            try:
                while not stop_event.is_set():
                    item = infer_queue.get()
                    logger.info("video post dequeue item=%s", "none" if item is None else item.get("frame_count"))
                    if item is None:
                        post_queue.put(None)
                        return
                    frame_index = int(item["frame_count"])
                    frame = item["frame"]
                    if item.get("skipped"):
                        with state_lock:
                            prev_result = shared_state["prev_result"]
                        frame_to_write = prev_result if prev_result is not None else frame
                        post_queue.put({"frame_count": frame_index, "frame_rgb": self._bgr_to_rgb(frame_to_write), "stats": None})
                        continue

                    post_started_at = time.perf_counter()
                    logger.info(
                        "video postprocess frame=%d detections_in=%d",
                        frame_index,
                        len(item["detections"]),
                    )
                    detections = item["detections"]
                    detections = self._apply_temporal_consistency(detections, temporal_state)
                    detections = self._apply_video_alert_cooldown(
                        detections=detections,
                        current_ts=(frame_index / fps) if fps > 0 else 0.0,
                        alert_history=alert_history,
                    )
                    detections, pipeline_alarm_count = self._attach_upgrade_metadata(detections)
                    detections = self._attach_video_bin_color_cached(frame, detections, (frame_index / fps) if fps > 0 else 0.0)
                    detections = self.detection_service._attach_alert_bin_context(detections)

                    rendered = self.detection_service.draw_boxes(frame, detections)
                    frame_alerts = sum(1 for det in detections if det.get("alert", False))
                    last_detection_count = len(detections)
                    last_alert_count = frame_alerts
                    current_frame_hash: int | None = None
                    should_hash = last_alert_count == 0 and last_detection_count <= 3
                    with state_lock:
                        previous_frame_hash = shared_state["previous_frame_hash"] if should_hash else None
                    if should_hash:
                        grayscale_pixels, hash_width, hash_height = self._frame_perceptual_hash(frame)
                        current_frame_hash = self.rust_bridge.perceptual_hash(grayscale_pixels, hash_width, hash_height)
                    effective_skip = self._adaptive_skip_frames(
                        requested_skip,
                        last_detection_count,
                        last_alert_count,
                        current_hash=current_frame_hash,
                        previous_hash=previous_frame_hash,
                    )

                    with state_lock:
                        shared_state["effective_skip"] = effective_skip
                        if supports_batch:
                            shared_state["micro_batch_size"] = self._adaptive_micro_batch_size(last_detection_count, last_alert_count)
                        else:
                            shared_state["micro_batch_size"] = 1
                        shared_state["previous_frame_hash"] = current_frame_hash if should_hash else None
                        shared_state["total_pipeline_alarms"] = int(shared_state["total_pipeline_alarms"]) + pipeline_alarm_count
                        shared_state["prev_result"] = rendered.copy()
                        total_pipeline_alarms = int(shared_state["total_pipeline_alarms"])

                    cv2.putText(
                        rendered,
                        f"Frame {frame_index}: {len(detections)} detected",
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
                    post_duration_ms = (time.perf_counter() - post_started_at) * 1000
                    stage_metrics["post_ms"] += post_duration_ms
                    stage_metrics["postprocessed_frames"] += 1
                    post_queue.put(
                        {
                            "frame_count": frame_index,
                            "frame_rgb": self._bgr_to_rgb(rendered),
                            "stats": {
                                "detections": len(detections),
                                "alerts": frame_alerts,
                                "infer_ms": float(item["infer_duration_ms"]),
                                "post_ms": post_duration_ms,
                            },
                        }
                    )
            except BaseException as exc:
                fail(exc)

        def encode_worker() -> None:
            logger.info("video encode worker started")
            nonlocal current_output_path
            try:
                while not stop_event.is_set():
                    item = post_queue.get()
                    logger.info("video encode dequeue item=%s", "none" if item is None else item.get("frame_count"))
                    if item is None:
                        return
                    frame_count = int(item["frame_count"])
                    encode_started_at = time.perf_counter()
                    logger.info("video encode frame=%d", frame_count)
                    writer_ref, output_ref = self._append_frame_with_encoder_fallback(
                        writer,
                        item["frame_rgb"],
                        current_output_path,
                        fps,
                    )
                    encode_duration_ms = (time.perf_counter() - encode_started_at) * 1000
                    stage_metrics["encode_ms"] += encode_duration_ms
                    stage_metrics["encoded_frames"] += 1
                    if output_ref != current_output_path:
                        current_output_path = output_ref
                    item_stats = item.get("stats")
                    if item_stats is not None:
                        stats["total_detections"] += int(item_stats["detections"])
                        stats["total_alerts"] += int(item_stats["alerts"])
                        if int(item_stats["alerts"]) > 0:
                            stats["alert_frames"] += 1
                        logger.info(
                            "video frame processed frame=%d/%d detections=%d alerts=%d infer_ms=%.2f post_ms=%.2f encode_ms=%.2f",
                            frame_count,
                            total_frames,
                            int(item_stats["detections"]),
                            int(item_stats["alerts"]),
                            float(item_stats["infer_ms"]),
                            float(item_stats["post_ms"]),
                            encode_duration_ms,
                        )
                    stats["frame_count"] = frame_count
            except BaseException as exc:
                fail(exc)

        threads = [
            threading.Thread(target=decode_worker, daemon=True),
            threading.Thread(target=infer_worker, daemon=True),
            threading.Thread(target=post_worker, daemon=True),
            threading.Thread(target=encode_worker, daemon=True),
        ]
        for thread in threads:
            thread.start()

        try:
            for thread in threads:
                thread.join()
            if exception_holder:
                raise exception_holder[0]
        finally:
            writer.close()
            cap.release()

        frame_count = int(stats["frame_count"])
        total_detections = int(stats["total_detections"])
        total_alerts = int(stats["total_alerts"])
        alert_frames = int(stats["alert_frames"])

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
        logger.info(
            "video pipeline stage summary decoded_frames=%d inferred_frames=%d postprocessed_frames=%d encoded_frames=%d decode_ms=%.2f infer_ms=%.2f post_ms=%.2f encode_ms=%.2f",
            int(stage_metrics["decoded_frames"]),
            int(stage_metrics["inferred_frames"]),
            int(stage_metrics["postprocessed_frames"]),
            int(stage_metrics["encoded_frames"]),
            float(stage_metrics["decode_ms"]),
            float(stage_metrics["infer_ms"]),
            float(stage_metrics["post_ms"]),
            float(stage_metrics["encode_ms"]),
        )
        return {
            "total_frames": frame_count,
            "detected_frames": alert_frames,
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "suppressed_alerts": 0,
            "alert_types": [],
            "video_info": f"{width}x{height}, {fps:.1f}fps",
        }
