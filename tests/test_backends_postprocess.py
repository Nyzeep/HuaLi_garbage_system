from __future__ import annotations

import numpy as np

from app.infrastructure.ml.backends import _postprocess


def test_postprocess_maps_letterbox_coords_back_to_original_space():
    output = np.zeros((1, 6, 5), dtype=np.float32)
    output[0, 0, :] = [320.0, 320.0, 100.0, 100.0, 0.9]

    predictions = _postprocess(
        output,
        orig_shape=(640, 640),
        conf_threshold=0.5,
        iou_threshold=0.3,
        input_size=640,
    )

    assert len(predictions) == 1
    assert predictions[0].bbox == [270, 270, 370, 370]
