from __future__ import annotations


class AppError(Exception):
    """应用层基础异常。"""

    code: str = "APP_ERROR"
    status_code: int = 500

    def __init__(self, message: str = "系统错误", detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ValidationError(AppError):
    code = "VALIDATION_ERROR"
    status_code = 422


class FileTypeError(AppError):
    code = "FILE_TYPE_ERROR"
    status_code = 400


class FileParseError(AppError):
    code = "FILE_PARSE_ERROR"
    status_code = 400


class ResourceNotFoundError(AppError):
    code = "RESOURCE_NOT_FOUND"
    status_code = 404


class InferenceError(AppError):
    code = "INFERENCE_ERROR"
    status_code = 500


class TaskDispatchError(AppError):
    code = "TASK_DISPATCH_ERROR"
    status_code = 503


class ServiceUnavailableError(AppError):
    code = "SERVICE_UNAVAILABLE"
    status_code = 503
