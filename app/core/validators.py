from __future__ import annotations

import cv2
import numpy as np

from app.core.exceptions import FileParseError, ValidationError


def validate_upload_size(content: bytes, max_mb: int) -> None:
    """文件大小超出限制时抛出 ValidationError。"""
    max_bytes = max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise ValidationError(f"文件大小超过限制（最大 {max_mb} MB）")


def validate_image_bytes(content: bytes) -> np.ndarray:
    """将字节解码为 BGR 图像，解码失败时抛出 FileParseError。"""
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise FileParseError("图片解析失败")
    return image


def validate_skip_frames(value: int) -> int:
    """裁剪 skip_frames 到合法区间 [1, 60]。"""
    return max(1, min(value, 60))


def validate_pagination(page: int, per_page: int) -> tuple[int, int]:
    """裁剪分页参数：page ≥ 1，per_page ∈ [1, 100]。"""
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    return page, per_page
