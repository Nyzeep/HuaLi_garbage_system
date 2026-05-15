from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw

from app.services.detection_service import DetectionService
from app.upgrade import AlarmEngine, DetectionEngine, TrackEngine, UpgradePipeline


class VideoProcessingError(RuntimeError):
    pass


@dataclass
class MemoryEvent:
    class_id: int
    class_name: str
    priority: int
    first_seen_ts: float
    last_triggered_ts: float
    track_id: int | None = None
    bbox: list[int] | None = None
    state: str = "remembered"


class VideoEventStateManager:
    VIDEO_IOU_MATCH_THRESHOLD = 0.4
    CLASS_PRIORITIES = {
        3: 0,  # fire
        4: 1,  # smoke
        2: 2,  # garbage
        1: 3,  # overflow
    }
    COOLDOWN_SECONDS = {
        3: 2.0,
        4: 2.0,
        2: 3.0,
        1: 4.0,
    }

    def __init__(self) -> None:
        self.memory_events: list[MemoryEvent] = []
        self.alert_history: list[dict] = []

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
        return self.COOLDOWN_SECONDS.get(class_id, 0.0)

    def _priority_for_class(self, class_id: int) -> int:
        return self.CLASS_PRIORITIES.get(class_id, 99)

    def _find_recent_match(self, detection: dict, current_ts: float) -> dict | None:
        class_id = int(detection.get("class_id", -1))
        cooldown = self._cooldown_seconds_for_class(class_id)
        if cooldown <= 0:
            return None

        bbox = detection.get("bbox", [])
        track_id = detection.get("track_id")
        for rec in self.alert_history:
            if int(rec.get("class_id", -1)) != class_id:
                continue
            if current_ts - float(rec.get("timestamp", 0.0)) > cooldown:
                continue
            rec_tid = rec.get("track_id")
            if (track_id is not None) and (rec_tid is not None):
                if int(track_id) == int(rec_tid):
                    return rec
                continue
            rec_bbox = rec.get("bbox") or []
            if bbox and rec_bbox and self._compute_iou(bbox, rec_bbox) >= self.VIDEO_IOU_MATCH_THRESHOLD:
                return rec
        return None

    def _remember_event(self, detection: dict, current_ts: float, state: str) -> None:
        class_id = int(detection.get("class_id", -1))
        class_name = str(detection.get("class_name", "unknown"))
        track_id = detection.get("track_id")
        bbox = detection.get("bbox") or None

        for event in self.memory_events:
            if event.class_id != class_id:
                continue
            if (track_id is not None) and (event.track_id is not None) and int(track_id) == int(event.track_id):
                event.class_name = class_name
                event.last_triggered_ts = current_ts
                event.bbox = bbox
                event.state = state
                return
            if bbox and event.bbox and self._compute_iou(bbox, event.bbox) >= self.VIDEO_IOU_MATCH_THRESHOLD:
                event.class_name = class_name
                event.last_triggered_ts = current_ts
                event.track_id = int(track_id) if track_id is not None else event.track_id
                event.bbox = bbox
                event.state = state
                return

        self.memory_events.append(
            MemoryEvent(
                class_id=class_id,
                class_name=class_name,
                priority=self._priority_for_class(class_id),
                first_seen_ts=current_ts,
                last_triggered_ts=current_ts,
                track_id=int(track_id) if track_id is not None else None,
                bbox=bbox,
                state=state,
            )
        )

    def apply(self, detections: list[dict], current_ts: float) -> tuple[list[dict], dict]:
        updated: list[dict] = []
        new_alert_count = 0
        suppressed_count = 0

        for det in detections:
            item = det.copy()
            item["new_alert"] = False
            item["active_memory"] = False

            if not item.get("alert", False):
                updated.append(item)
                continue

            class_id = int(item.get("class_id", -1))
            if class_id not in self.COOLDOWN_SECONDS:
                updated.append(item)
                continue

            recent = self._find_recent_match(item, current_ts)
            if recent is None:
                item["new_alert"] = True
                item["active_memory"] = True
                new_alert_count += 1
                self.alert_history.append(
                    {
                        "class_id": class_id,
                        "bbox": item.get("bbox", []),
                        "timestamp": current_ts,
                        "track_id": int(item["track_id"]) if item.get("track_id") is not None else None,
                    }
                )
                self._remember_event(item, current_ts=current_ts, state="new")
            else:
                item["alert"] = False
                item["active_memory"] = True
                suppressed_count += 1
                self._remember_event(item, current_ts=float(recent["timestamp"]), state="remembered")

            updated.append(item)

        max_keep_window = max(self.COOLDOWN_SECONDS.values(), default=0.0)
        self.alert_history[:] = [
            rec for rec in self.alert_history if current_ts - float(rec.get("timestamp", 0.0)) <= max_keep_window
        ]

        active_alerts = self.get_active_alerts()
        summary = {
            "active_alerts": active_alerts,
            "active_alert_count": len(active_alerts),
            "highest_priority_alert": active_alerts[0]["class_name"] if active_alerts else None,
            "new_alert_count": new_alert_count,
            "suppressed_alerts": suppressed_count,
        }
        return updated, summary

    def get_active_alerts(self) -> list[dict]:
        active = sorted(
            self.memory_events,
            key=lambda item: (item.priority, item.first_seen_ts),
        )
        return [
            {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "priority": item.priority,
                "state": item.state,
            }
            for item in active
        ]

    def mark_frame_complete(self) -> None:
        for event in self.memory_events:
            if event.state == "new":
                event.state = "remembered"


class VideoProcessingService:
    def __init__(self, detection_service: DetectionService):
        self.detection_service = detection_service
        self.upgrade_pipeline = UpgradePipeline(
            detection_engine=DetectionEngine(detection_service),
            track_engine=TrackEngine(),
            alarm_engine=AlarmEngine(min_consecutive_frames=1),
        )
        self.event_state = VideoEventStateManager()

    @staticmethod
    def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _attach_upgrade_metadata(
        self,
        detections: list[dict],
        timestamp: float | None = None,
    ) -> tuple[list[dict], int]:
        pipe_result = self.upgrade_pipeline.run_detections(detections, timestamp=timestamp)
        tracks = pipe_result.tracks
        alarms = pipe_result.alarms

        out: list[dict] = []
        for idx, det in enumerate(detections):
            item = det.copy()
            if idx < len(tracks):
                item["track_id"] = int(tracks[idx].track_id)
            out.append(item)
        return out, len(alarms)

    def _render_alert_panel(self, image: np.ndarray, active_alerts: list[dict]) -> np.ndarray:
        if not active_alerts:
            return image

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        font_title = self.detection_service._load_chinese_font(size=22)
        font_line = self.detection_service._load_chinese_font(size=18)

        width, height = pil_img.size
        line_height = 30
        panel_width = min(380, max(260, int(width * 0.38)))
        panel_height = 58 + len(active_alerts) * line_height
        left = max(12, width - panel_width - 18)
        top = max(12, height - panel_height - 18)
        right = width - 18
        bottom = height - 18

        draw.rounded_rectangle([left, top, right, bottom], radius=18, fill=(120, 0, 0, 180))
        draw.text((left + 16, top + 12), "报警事件", fill=(255, 255, 255), font=font_title)

        y = top + 44
        for idx, alert in enumerate(active_alerts, start=1):
            state_text = "新触发" if alert.get("state") == "new" else "持续提示"
            text = f"{idx}. {alert.get('class_name', '未知事件')}  {state_text}"
            draw.text((left + 16, y), text, fill=(255, 236, 236), font=font_line)
            y += line_height

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_video_frame(self, image: np.ndarray, detections: list[dict], active_alerts: list[dict]) -> np.ndarray:
        visual_detections: list[dict] = []
        for det in detections:
            item = det.copy()
            if item.get("active_memory", False):
                item["alert"] = True
                item["color"] = (0, 0, 255)
            visual_detections.append(item)

        rendered = self.detection_service.draw_boxes(image, visual_detections)
        return self._render_alert_panel(rendered, active_alerts)

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        skip_frames: int,
        progress_callback=None,
        status_callback=None,
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
        total_pipeline_alarms = 0
        alert_frames = 0
        alert_type_set: set[str] = set()
        prev_result = None
        effective_skip = max(skip_frames, 1)
        latest_runtime_state = {
            "active_alerts": [],
            "active_alert_count": 0,
            "highest_priority_alert": None,
            "new_alert_count": 0,
        }

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
                detections, runtime_state = self.event_state.apply(
                    detections=detections,
                    current_ts=current_ts,
                )

                total_pipeline_alarms += int(pipeline_alarm_count)
                total_suppressed_alerts += int(runtime_state.get("suppressed_alerts", 0))
                latest_runtime_state = {
                    "active_alerts": runtime_state.get("active_alerts", []),
                    "active_alert_count": int(runtime_state.get("active_alert_count", 0) or 0),
                    "highest_priority_alert": runtime_state.get("highest_priority_alert"),
                    "new_alert_count": int(runtime_state.get("new_alert_count", 0) or 0),
                }

                rendered = self._draw_video_frame(
                    frame,
                    detections=detections,
                    active_alerts=latest_runtime_state["active_alerts"],
                )
                prev_result = rendered.copy()

                frame_alerts = sum(1 for item in detections if item.get("new_alert", False))
                for item in detections:
                    if item.get("active_memory", False):
                        alert_type_set.add(str(item.get("class_name", "unknown")))
                if latest_runtime_state["active_alert_count"] > 0:
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

                if status_callback is not None:
                    status_callback(
                        {
                            **latest_runtime_state,
                            "processed_frames": frame_count,
                            "total_frames": total_frames,
                        }
                    )
                if progress_callback and total_frames:
                    progress_callback(frame_count, total_frames)

                self.event_state.mark_frame_complete()
        finally:
            writer.close()
            cap.release()

        final_active_alerts = self.event_state.get_active_alerts()
        return {
            "total_frames": frame_count,
            "detected_frames": alert_frames,
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "suppressed_alerts": total_suppressed_alerts,
            "alert_types": sorted(alert_type_set),
            "video_info": f"{width}x{height}, {fps:.1f}fps, suppressed={total_suppressed_alerts}",
            "active_alerts": final_active_alerts,
            "active_alert_count": len(final_active_alerts),
            "highest_priority_alert": final_active_alerts[0]["class_name"] if final_active_alerts else None,
            "new_alert_count": 0,
        }
