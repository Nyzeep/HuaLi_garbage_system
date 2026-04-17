from __future__ import annotations

import uvicorn

try:
    from app.main import app
except ModuleNotFoundError:
    from main import app  # type: ignore


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
