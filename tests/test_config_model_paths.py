from __future__ import annotations

from app.config import Settings


def test_garbage_int8_path_uses_correct_spelling():
    settings = Settings()
    assert settings.garbage_int8_onnx_model.name == "garbage.int8.onnx"


def test_garbage_int8_path_falls_back_to_legacy_spelling(tmp_path):
    legacy = tmp_path / "garbege.int8.onnx"
    legacy.write_bytes(b"legacy")

    settings = Settings()
    settings.models_dir = tmp_path
    settings.garbage_int8_onnx_model = tmp_path / "garbage.int8.onnx"

    normalized = settings._normalize_legacy_model_paths()

    assert normalized.garbage_int8_onnx_model == legacy
