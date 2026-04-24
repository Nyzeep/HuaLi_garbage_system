from __future__ import annotations

from app.config import RUNTIME_DIR, Settings


def test_uploads_dir_defaults_to_runtime_directory():
    settings = Settings()
    assert settings.uploads_dir == RUNTIME_DIR / "uploads"
