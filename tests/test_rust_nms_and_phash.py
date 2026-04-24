from __future__ import annotations

from app.infrastructure.ml.rust_bridge import RustBridge


class DummyRustBridge:
    def non_max_suppression(self, boxes, threshold):
        return [boxes[0]] if boxes else []

    def perceptual_hash(self, grayscale_pixels, width, height):
        return sum(grayscale_pixels) + width + height

    def hamming_distance(self, a, b):
        return abs(a - b)


def test_dummy_nms_keeps_first_box():
    bridge = DummyRustBridge()
    result = bridge.non_max_suppression(
        [{"bbox": [0, 0, 10, 10], "score": 0.9}, {"bbox": [1, 1, 9, 9], "score": 0.8}],
        0.5,
    )
    assert result == [{"bbox": [0, 0, 10, 10], "score": 0.9}]


def test_dummy_perceptual_hash_and_hamming_distance():
    bridge = DummyRustBridge()
    hash_a = bridge.perceptual_hash([0, 1, 2, 3], 2, 2)
    hash_b = bridge.perceptual_hash([0, 1, 2, 4], 2, 2)
    assert hash_a == 10
    assert hash_b == 11
    assert bridge.hamming_distance(hash_a, hash_b) == 1
