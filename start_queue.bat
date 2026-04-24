@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0."

rem ============================================================
rem  HuaLi_garbage_system Windows launcher
rem
rem  Modes:
rem    start_queue.bat              -> setup + web + celery (prebuilt wheel preferred)
rem    start_queue.bat all          -> same as default
rem    start_queue.bat lite         -> setup + web only, no Redis required
rem    start_queue.bat web          -> setup + web only
rem    start_queue.bat worker       -> setup + celery only
rem    start_queue.bat setup        -> create venv + install deps + install Rust accel if available
rem    start_queue.bat rust-http    -> setup + optional Rust HTTP fallback service
rem    start_queue.bat check        -> import checks only
rem    start_queue.bat help         -> show help
rem ============================================================

set "MODE=%~1"
if not defined MODE set "MODE=all"
if /I "%MODE%"=="default" set "MODE=all"
if /I "%MODE%"=="demo" set "MODE=lite"

for %%I in ("%~dp0.") do set "ROOT_DIR=%%~fI"
set "LOG_DIR=%ROOT_DIR%\logs"
set "VENV_DIR=.venv"
set "VENV_PYTHON=%ROOT_DIR%\.venv\Scripts\python.exe"
set "RUST_HTTP_EXE=%ROOT_DIR%\rust\target\release\huali_garbage_server.exe"
set "WHEEL_DIR=%ROOT_DIR%\vendor"
if not defined GITHUB_REPO call :detect_github_repo
if not defined GITHUB_RELEASE_TAG set "GITHUB_RELEASE_TAG=latest"
if not defined WHEEL_ASSET_GLOB set "WHEEL_ASSET_GLOB=huali_garbage_core-*-win_amd64.whl"
set "WEB_URL=http://127.0.0.1:8000"
set "REDIS_HOST=127.0.0.1"
set "REDIS_PORT=6379"
set "RUST_SERVICE_URL=http://127.0.0.1:50051"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

if /I "%MODE%"=="help" goto :help

echo ============================================================
echo   HuaLi_garbage_system launcher
echo   Mode: %MODE%
echo   Root: %ROOT_DIR%
echo ============================================================
echo.

call :ensure_venv || goto :fail
call :ensure_python || goto :fail

if /I "%MODE%"=="setup" (
    call :install_python_deps || goto :fail
    call :ensure_rust_acceleration
    call :verify_runtime || goto :fail
    echo.
    echo [OK] Setup completed.
    pause
    exit /b 0
)

if /I "%MODE%"=="check" (
    call :verify_runtime || goto :fail
    echo.
    echo [OK] Runtime imports succeeded.
    pause
    exit /b 0
)

call :install_python_deps_if_needed || goto :fail
call :ensure_rust_acceleration
call :verify_runtime || goto :fail

if /I "%MODE%"=="web" (
    call :start_web || goto :fail
    echo.
    echo [OK] Web started.
    pause
    exit /b 0
)

if /I "%MODE%"=="lite" (
    call :start_web || goto :fail
    echo.
    echo [OK] Lite mode started.
    echo   - Web:    %WEB_URL%
    echo   - Worker: disabled
    echo   - Redis:  not required in lite mode
    echo   - Video tasks will fall back to local thread execution when no worker is available.
    pause
    exit /b 0
)

if /I "%MODE%"=="worker" (
    call :check_redis || goto :fail
    call :start_worker || goto :fail
    echo.
    echo [OK] Worker started.
    pause
    exit /b 0
)

if /I "%MODE%"=="rust-http" (
    call :build_rust_http_binary || goto :fail
    call :start_rust_http || goto :fail
    echo.
    echo [OK] Rust HTTP fallback service started.
    pause
    exit /b 0
)

rem default/all
call :check_redis || goto :fail
call :start_worker || goto :fail
call :start_web || goto :fail
echo.
echo [OK] Started successfully.
echo   - Web:    %WEB_URL%
echo   - Redis:  redis://%REDIS_HOST%:%REDIS_PORT%/0
echo   - Uploads: app default (see app.config / optional UPLOADS_DIR env)
echo   - Logs:   %LOG_DIR%
echo.
echo Tips:
echo   - Default runtime prefers a prebuilt PyO3 Rust wheel from vendor or this repo's GitHub Release.
echo   - Upload path follows the app default unless you set UPLOADS_DIR explicitly.
echo   - Without Rust acceleration, the app still runs with Python fallback.
echo   - For normal users without Redis, prefer: start_queue.bat lite
echo   - To run Rust HTTP fallback service separately: start_queue.bat rust-http
echo.
pause
exit /b 0

:help
echo Usage:
echo   start_queue.bat [all^|lite^|web^|worker^|setup^|check^|rust-http^|help]
echo.
echo Modes:
echo   all       Setup, install Rust wheel if available, then start FastAPI + Celery worker
echo   lite      Setup and start FastAPI only; no Redis required, best for normal users
echo   web       Setup, install Rust wheel if available, then start FastAPI only
echo   worker    Setup, install Rust wheel if available, then start Celery worker only
echo   setup     Create venv, install Python deps, install/download/build Rust acceleration if available
echo   check     Verify imports for FastAPI/Celery app runtime
echo   rust-http Build/start optional Rust HTTP fallback service
echo.
pause
exit /b 0

:detect_github_repo
for /f "usebackq delims=" %%R in (`"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -Command "$u=(git config --get remote.origin.url 2^> $null); if(-not $u){exit 1}; if($u -match 'github\.com[:/](?<repo>[^/]+/[^/.]+?)(?:\.git)?$'){ $matches['repo'] }"`) do set "GITHUB_REPO=%%R"
if defined GITHUB_REPO exit /b 0
exit /b 0

:ensure_venv
if exist "%VENV_PYTHON%" exit /b 0
echo [INFO] Creating virtual environment...
where py >nul 2>&1
if not errorlevel 1 (
    py -m venv "%VENV_DIR%" >nul 2>&1
)
if not exist "%VENV_PYTHON%" (
    python -m venv "%VENV_DIR%" >nul 2>&1
)
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)
exit /b 0

:ensure_python
if exist "%VENV_PYTHON%" exit /b 0
echo [ERROR] Python executable not found: %VENV_PYTHON%
exit /b 1

:install_python_deps_if_needed
call "%VENV_PYTHON%" -c "import fastapi, uvicorn, celery, redis, sqlalchemy, cv2" >nul 2>&1
if not errorlevel 1 exit /b 0
call :install_python_deps || exit /b 1
exit /b 0

:install_python_deps
echo [INFO] Installing Python dependencies...
call "%VENV_PYTHON%" -m pip install --upgrade pip >"%LOG_DIR%\pip_upgrade.log" 2>&1
if errorlevel 1 (
    echo [ERROR] pip upgrade failed. See logs\pip_upgrade.log
    exit /b 1
)
call "%VENV_PYTHON%" -m pip install -r requirements.txt >"%LOG_DIR%\pip_install.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See logs\pip_install.log
    exit /b 1
)
exit /b 0

:ensure_rust_acceleration
call "%VENV_PYTHON%" -c "import huali_garbage_core" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Rust acceleration already available.
    exit /b 0
)
call :install_prebuilt_rust_wheel
call "%VENV_PYTHON%" -c "import huali_garbage_core" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Rust acceleration installed from local wheel.
    exit /b 0
)
call :download_release_rust_wheel
call :install_prebuilt_rust_wheel
call "%VENV_PYTHON%" -c "import huali_garbage_core" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Rust acceleration installed from GitHub Release wheel.
    exit /b 0
)
call :build_pyo3_extension_if_possible
call "%VENV_PYTHON%" -c "import huali_garbage_core" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Rust acceleration built locally.
    exit /b 0
)
echo [WARN] Rust acceleration unavailable. Falling back to Python implementation.
exit /b 0

:install_prebuilt_rust_wheel
if not exist "%WHEEL_DIR%" mkdir "%WHEEL_DIR%" >nul 2>&1
echo [INFO] Looking for prebuilt Rust wheel...
for %%F in ("%WHEEL_DIR%\huali_garbage_core-*.whl") do (
    if exist "%%~fF" (
        echo [INFO] Installing prebuilt wheel: %%~nxF
        call "%VENV_PYTHON%" -m pip install --force-reinstall "%%~fF" >"%LOG_DIR%\pip_rust_wheel.log" 2>&1
        if not errorlevel 1 exit /b 0
        echo [WARN] Prebuilt wheel install failed. See logs\pip_rust_wheel.log
        exit /b 0
    )
)
exit /b 0

:download_release_rust_wheel
if not defined GITHUB_REPO exit /b 0
if not exist "%WHEEL_DIR%" mkdir "%WHEEL_DIR%" >nul 2>&1
echo [INFO] Looking for Rust wheel on GitHub Releases...
call "%VENV_PYTHON%" "%ROOT_DIR%\scripts\download_rust_wheel.py" --repo "%GITHUB_REPO%" --tag "%GITHUB_RELEASE_TAG%" --pattern "%WHEEL_ASSET_GLOB%" --wheel-dir "%WHEEL_DIR%" >"%LOG_DIR%\download_rust_wheel.log" 2>&1
if errorlevel 1 (
    echo [WARN] GitHub Release wheel not downloaded. See logs\download_rust_wheel.log
    exit /b 0
)
echo [INFO] Downloaded Rust wheel from GitHub Release.
exit /b 0

:build_pyo3_extension_if_possible
where cargo >nul 2>&1
if errorlevel 1 exit /b 0
where rustc >nul 2>&1
if errorlevel 1 exit /b 0
echo [INFO] Building Rust PyO3 extension with local toolchain...
call "%VENV_PYTHON%" -m pip install maturin >"%LOG_DIR%\pip_maturin.log" 2>&1
if errorlevel 1 (
    echo [WARN] Failed to install maturin. See logs\pip_maturin.log
    exit /b 0
)
call "%VENV_PYTHON%" -m maturin develop --manifest-path rust\Cargo.toml --release >"%LOG_DIR%\maturin_build.log" 2>&1
if errorlevel 1 (
    echo [WARN] Local PyO3 build failed. See logs\maturin_build.log
    exit /b 0
)
exit /b 0

:build_rust_http_binary
if exist "%RUST_HTTP_EXE%" exit /b 0
echo [INFO] Building Rust HTTP fallback binary...
cargo build --release --manifest-path "%ROOT_DIR%rust\Cargo.toml" >"%LOG_DIR%\rust_http_build.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Rust HTTP build failed. See logs\rust_http_build.log
    exit /b 1
)
if not exist "%RUST_HTTP_EXE%" (
    echo [ERROR] Rust HTTP executable not found after build.
    exit /b 1
)
exit /b 0

:verify_runtime
echo [INFO] Verifying runtime imports...
call "%VENV_PYTHON%" -c "import fastapi, uvicorn, celery, redis, sqlalchemy, cv2; import app.main; print('ok')" >"%LOG_DIR%\runtime_check.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Runtime verification failed. See logs\runtime_check.log
    exit /b 1
)
exit /b 0

:check_redis
echo [INFO] Checking Redis connectivity...
call "%VENV_PYTHON%" -c "import socket; s=socket.create_connection(('%REDIS_HOST%', %REDIS_PORT%), timeout=1); s.close(); print('ok')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Redis is not reachable at redis://%REDIS_HOST%:%REDIS_PORT%/0
    echo         Please start Redis first.
    exit /b 1
)
exit /b 0

:start_web
echo [INFO] Starting FastAPI web...
start "FastAPI Web" cmd /k "cd /d ""%ROOT_DIR%"" && set ""PYTHONUTF8=1"" && ""%VENV_PYTHON%"" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
exit /b 0

:start_worker
echo [INFO] Starting Celery worker...
start "Celery Worker" cmd /k "cd /d ""%ROOT_DIR%"" && set ""PYTHONUTF8=1"" && ""%VENV_PYTHON%"" -m celery -A app.celery_app worker --loglevel=info --pool=solo"
exit /b 0

:start_rust_http
echo [INFO] Starting Rust HTTP fallback service...
start "Rust HTTP Service" cmd /k "cd /d ""%ROOT_DIR%"" && set ""RUST_SERVICE_HOST=127.0.0.1"" && set ""RUST_SERVICE_PORT=50051"" && ""%RUST_HTTP_EXE%"""
exit /b 0

:fail
echo.
echo [ERROR] Startup failed.
echo         Please inspect logs under %LOG_DIR%
pause
exit /b 1
