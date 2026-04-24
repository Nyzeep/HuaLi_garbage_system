from __future__ import annotations

from celery import Celery

from app.config import get_settings


settings = get_settings()

celery_app = Celery(
    "garbage_system",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_always_eager=settings.celery_task_always_eager,
)

# Discover tasks lazily so Celery workers register `app.tasks.*` without
# forcing `app.tasks` to import during unrelated module imports in tests.
celery_app.autodiscover_tasks(["app"])

