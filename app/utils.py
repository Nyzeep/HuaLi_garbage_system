from __future__ import annotations

import base64
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def frame_to_base64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("failed to encode image")
    return base64.b64encode(buf).decode("utf-8")


def base64_to_frame(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image")
    return img


def save_image(image: np.ndarray, output_dir: Path, suffix: str = ".jpg") -> Path:
    ensure_dir(output_dir)
    file_path = output_dir / f"{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{suffix}"

    ext = suffix.lower()
    if not ext.startswith("."):
        ext = f".{ext}"

    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"failed to encode image for {file_path}")

    try:
        with file_path.open("wb") as f:
            f.write(buf.tobytes())
    except OSError as exc:
        raise ValueError(f"failed to save image to {file_path}") from exc

    return file_path


def relative_to(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()

