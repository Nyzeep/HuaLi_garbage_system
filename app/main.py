from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.pages import build_pages_router
from app.api.routes import build_api_router
from app.bootstrap import bootstrap_application
from app.config import get_settings
from app.core.exceptions import AppError
from app.core.responses import error_response


settings = get_settings()
started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_app() -> FastAPI:
    bootstrap_application()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(message=exc.message, code=exc.code),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(message=str(exc.detail), code="HTTP_ERROR"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_response(
                message="请求参数校验失败",
                code="VALIDATION_ERROR",
                data=exc.errors(),
            ),
        )

    templates = Jinja2Templates(directory=str(settings.templates_dir))

    app.mount("/uploads", StaticFiles(directory=str(settings.uploads_dir)), name="uploads")
    app.include_router(build_pages_router(templates))
    app.include_router(build_api_router(settings=settings, started_at=started_at))
    return app


def create_asgi_app() -> FastAPI:
    """Factory function for ASGI servers (uvicorn, gunicorn, etc.).

    Usage:
        uvicorn app.main:create_asgi_app --factory --host 127.0.0.1 --port 8000
    """
    return create_app()


class _LazyASGIApp:
    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def _get_app(self) -> FastAPI:
        if self._app is None:
            self._app = create_app()
        return self._app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        await self._get_app()(scope, receive, send)


app = _LazyASGIApp()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
