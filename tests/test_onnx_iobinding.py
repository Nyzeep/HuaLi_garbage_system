from __future__ import annotations

import numpy as np

from app.infrastructure.ml.backends import OnnxYoloBackend


class DummyIOBinding:
    def __init__(self) -> None:
        self.bound_inputs = []
        self.bound_outputs = []

    def bind_cpu_input(self, name, tensor):
        self.bound_inputs.append((name, tensor.shape))

    def bind_output(self, name):
        self.bound_outputs.append(name)

    def copy_outputs_to_cpu(self):
        output = np.zeros((1, 6, 5), dtype=np.float32)
        output[0, 0, :] = [320.0, 320.0, 100.0, 100.0, 0.9]
        return [output]


class DummySession:
    def __init__(self) -> None:
        self._outputs = [type("Output", (), {"name": "output0"})()]
        self._inputs = [type("Input", (), {"name": "images"})()]
        self.run_calls = 0

    def io_binding(self):
        return DummyIOBinding()

    def get_outputs(self):
        return self._outputs

    def get_inputs(self):
        return self._inputs

    def run_with_iobinding(self, io_binding):
        return None

    def run(self, *_args, **_kwargs):
        self.run_calls += 1
        raise AssertionError("fallback run should not be used when IOBinding succeeds")


def test_onnx_backend_uses_iobinding_when_available(tmp_path):
    backend = OnnxYoloBackend(tmp_path / "dummy.onnx")
    backend._session = DummySession()
    backend._input_name = "images"
    backend._input_size = 640
    backend.loaded = True

    predictions = backend.predict(np.zeros((640, 640, 3), dtype=np.uint8), 0.5, 0.3)

    assert len(predictions) == 1
    assert predictions[0].bbox == [270, 270, 370, 370]


def test_onnx_backend_falls_back_to_run_when_iobinding_fails(tmp_path):
    backend = OnnxYoloBackend(tmp_path / "dummy.onnx")

    class FailingSession(DummySession):
        def io_binding(self):
            raise RuntimeError("no binding")

        def run(self, *_args, **_kwargs):
            output = np.zeros((1, 6, 5), dtype=np.float32)
            output[0, 0, :] = [320.0, 320.0, 100.0, 100.0, 0.9]
            return [output]

    backend._session = FailingSession()
    backend._input_name = "images"
    backend._input_size = 640
    backend.loaded = True

    predictions = backend.predict(np.zeros((640, 640, 3), dtype=np.uint8), 0.5, 0.3)

    assert len(predictions) == 1


def test_onnx_backend_predict_batch_falls_back_per_frame_for_fixed_batch_model(tmp_path):
    backend = OnnxYoloBackend(tmp_path / "dummy.onnx")

    class FixedBatchSession(DummySession):
        def run(self, *_args, **_kwargs):
            raise AssertionError("session.run should not be called in fixed-batch fallback path")

    backend._session = FixedBatchSession()
    backend._input_name = "images"
    backend._input_size = 640
    backend._supports_batch = False
    backend.loaded = True

    images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(4)]
    predictions = backend.predict_batch(images, 0.5, 0.3)

    assert len(predictions) == 4
    assert all(len(batch) == 1 for batch in predictions)
