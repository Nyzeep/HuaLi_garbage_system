from __future__ import annotations

from app.celery_app import celery_app


def test_process_video_task_is_registered_after_autodiscovery():
    celery_app.loader.import_default_modules()
    assert "app.tasks.process_video_task" in celery_app.tasks
