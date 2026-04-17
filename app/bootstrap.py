from __future__ import annotations

from app.config import get_settings
from app.database import Base, engine
from app.utils import ensure_dir


def bootstrap_application() -> None:
    settings = get_settings()
    ensure_dir(settings.uploads_dir)
    ensure_dir(settings.uploads_dir / "alerts")
    ensure_dir(settings.uploads_dir / "videos")
    Base.metadata.create_all(bind=engine)
