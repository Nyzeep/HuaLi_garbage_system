from __future__ import annotations

from app.infrastructure.ml.rust_bridge import RustBridge


class DummyRustBridge:
    def invert_letterbox_bbox(self, bbox, *, scale, pad_w, pad_h, original_width, original_height):
        return [
            round((bbox[0] - pad_w) / scale),
            round((bbox[1] - pad_h) / scale),
            round((bbox[2] - pad_w) / scale),
            round((bbox[3] - pad_h) / scale),
        ]

    def batch_iou_match(self, left, right, threshold):
        return [(0, 0, 1.0)] if left and right and threshold <= 1.0 else []


def test_dummy_invert_letterbox_bbox_behavior():
    bridge = DummyRustBridge()
    result = bridge.invert_letterbox_bbox(
        [30, 50, 70, 90],
        scale=2.0,
        pad_w=10.0,
        pad_h=10.0,
        original_width=100,
        original_height=100,
    )
    assert result == [10, 20, 30, 40]


def test_dummy_batch_iou_match_behavior():
    bridge = DummyRustBridge()
    result = bridge.batch_iou_match([[0, 0, 10, 10]], [[0, 0, 10, 10]], 0.3)
    assert result == [(0, 0, 1.0)]
