from __future__ import annotations

from typing import Any


def success_response(
    data: Any = None,
    message: str = "success",
    code: str = "SUCCESS",
    trace_id: str | None = None,
) -> dict:
    payload = {
        "success": True,
        "code": code,
        "message": message,
        "data": data,
    }
    if trace_id is not None:
        payload["trace_id"] = trace_id
    return payload


def error_response(
    message: str,
    code: str = "ERROR",
    data: Any = None,
    trace_id: str | None = None,
) -> dict:
    payload = {
        "success": False,
        "code": code,
        "message": message,
        "data": data,
    }
    if trace_id is not None:
        payload["trace_id"] = trace_id
    return payload
