from __future__ import annotations

import logging
import os
import platform
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


_FONT_CANDIDATES: list[str] = []

_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    _FONT_CANDIDATES = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
elif _SYSTEM == "Darwin":
    _FONT_CANDIDATES = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
else:
    _FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]


_FONT_PROJECT_PATH = os.environ.get("HUALI_CHINESE_FONT") or os.environ.get("CHINESE_FONT_PATH")


@lru_cache(maxsize=1)
def _load_chinese_font() -> ImageFont.ImageFont:
    if _FONT_PROJECT_PATH and os.path.isfile(_FONT_PROJECT_PATH):
        try:
            font = ImageFont.truetype(_FONT_PROJECT_PATH, size=20)
            logger.info("chinese font loaded from env path=%s", _FONT_PROJECT_PATH)
            return font
        except Exception as exc:
            logger.warning("chinese font env path failed path=%s error=%s", _FONT_PROJECT_PATH, exc)

    for font_path in _FONT_CANDIDATES:
        if not os.path.isfile(font_path):
            continue
        try:
            font = ImageFont.truetype(font_path, size=20)
            logger.info("chinese font loaded path=%s", font_path)
            return font
        except Exception as exc:
            logger.warning("chinese font try failed path=%s error=%s", font_path, exc)
            continue

    logger.warning("no chinese font found, using default font; chinese text may render as tofu")
    return ImageFont.load_default()


class RenderingService:
    @staticmethod
    def _draw_label_text(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        bg_color_bgr: tuple[int, int, int],
    ) -> np.ndarray:
        font = _load_chinese_font()
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad_x, pad_y = 6, 4

        left = max(0, x)
        top = max(0, y - text_h - pad_y * 2)
        right = min(pil_img.width, left + text_w + pad_x * 2)
        bottom = min(pil_img.height, top + text_h + pad_y * 2)

        bg_color_rgb = (int(bg_color_bgr[2]), int(bg_color_bgr[1]), int(bg_color_bgr[0]))
        draw.rectangle([left, top, right, bottom], fill=bg_color_rgb)
        draw.text((left + pad_x, top + pad_y), text, fill=(255, 255, 255), font=font)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_boxes(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        output = image.copy()
        en_label_map = {
            0: "GarbageBin",
            1: "Overflow",
            2: "Garbage",
            3: "FIRE",
            4: "SMOKE",
        }

        # 1. Draw all rectangles with OpenCV (fast, native BGR)
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            box_color = detection.get("color", (0, 255, 0))
            line_w = 3 if detection.get("alert", False) else 2
            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, line_w)

        # 2. Batch-draw labels via a single PIL conversion to avoid repeated
        #    BGR->RGB->BGR round-trips for every detection.
        if detections:
            pil_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = _load_chinese_font()

            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                box_color = detection.get("color", (0, 255, 0))
                default_name = en_label_map.get(detection["class_id"], detection.get("class_name", "Object"))
                label_name = detection.get("class_name") or default_name
                label = f"{label_name} {detection.get('confidence', 0.0):.0%}"

                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                pad_x, pad_y = 6, 4

                left = max(0, x1)
                top = max(0, y1 - text_h - pad_y * 2)
                right = min(pil_img.width, left + text_w + pad_x * 2)
                bottom = min(pil_img.height, top + text_h + pad_y * 2)

                bg_color_rgb = (int(box_color[2]), int(box_color[1]), int(box_color[0]))
                draw.rectangle([left, top, right, bottom], fill=bg_color_rgb)
                draw.text((left + pad_x, top + pad_y), label, fill=(255, 255, 255), font=font)

            output = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return output
