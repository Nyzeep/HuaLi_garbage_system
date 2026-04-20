@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

rem ============================================================
rem  HuaLi_garbage_system lightweight launcher
rem
rem  Usage:
rem    start_queue.bat            -> all services
rem    start_queue.bat all        -> all services
rem    start_queue.bat web        -> only FastAPI
rem    start_queue.bat worker     -> Celery + FastAPI
rem    start_queue.bat rust       -> only Rust service
rem    start_queue.bat setup      -> create venv + install deps + build rust
rem    start_queue.bat check      -> run project checks only
rem ============================================================

set "MODE=%~1"
if not defined MODE set "MODE=all"
if /I "%MODE%"=="default" set "MODE=all"
if /I "%MODE%"=="test" set "MODE=check"

set "ROOT_DIR=%~dp0"
set "LOG_DIR=%ROOT_DIR%logs"
set "VENV_DIR="
set "VENV_PYTHON="
set "RUST_SERVICE_EXE=%ROOT_DIR%rust\target\release\huali_garbage_core.exe"
set "RUST_SERVICE_URL=http://127.0.0.1:50051"
set "WEB_URL=http://127.0.0.1:8000"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

call :banner
call :find_venv
if not defined VENV_DIR (
    echo [INFO] No virtual environment found, creating .venv...
    call :create_venv || goto :fail
) else (
    echo [INFO] Using virtual environment: !VENV_DIR!
)

call :resolve_python || goto :fail

if /I "%MODE%"=="setup" (
    call :ensure_dependencies || goto :fail
    call :ensure_rust_binary || goto :fail
    echo [OK] Setup completed.
    pause
    exit /b 0
)

if /I "%MODE%"=="check" (
    call :run_project_checks || goto :fail
    echo [OK] Project checks passed.
    pause
    exit /b 0
)

call :health_check_environment || goto :fail
call :run_project_checks || goto :fail

if /I "%MODE%"=="rust" (
    call :start_rust_service || goto :fail
    call :wait_http_ready "%RUST_SERVICE_URL%" 20 "Rust service" || goto :fail
    echo [OK] Rust service is ready.
    pause
    exit /b 0
)

if /I "%MODE%"=="web" (
    call :start_web_service || goto :fail
    call :wait_http_ready "%WEB_URL%/status" 30 "FastAPI" || goto :fail
    start "" "%WEB_URL%"
    echo [OK] FastAPI is ready.
    pause
    exit /b 0
)

if /I "%MODE%"=="worker" (
    call :start_rust_service || goto :fail
    call :wait_http_ready "%RUST_SERVICE_URL%" 20 "Rust service" || goto :fail
    call :start_celery_worker || goto :fail
    call :start_web_service || goto :fail
    call :wait_http_ready "%WEB_URL%/status" 30 "FastAPI" || goto :fail
    start "" "%WEB_URL%"
    echo [OK] Worker mode is ready.
    pause
    exit /b 0
)

rem default/all
call :start_rust_service || goto :fail
call :wait_http_ready "%RUST_SERVICE_URL%" 20 "Rust service" || goto :fail
call :start_celery_worker || goto :fail
call :start_web_service || goto :fail
call :wait_http_ready "%WEB_URL%/status" 30 "FastAPI" || goto :fail
start "" "%WEB_URL%"

echo.
echo [OK] Started successfully.
echo - Rust service: %RUST_SERVICE_URL%
echo - FastAPI:      %WEB_URL%
echo - Logs:         %LOG_DIR%
echo.
echo Close the service windows to stop the stack.
pause
exit /b 0

:banner
echo ============================================================
echo   HuaLi_garbage_system launcher
echo   Mode: %MODE%
echo ============================================================
exit /b 0

:find_venv
if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "VENV_DIR=%VIRTUAL_ENV%"
        exit /b 0
    )
)

for %%D in (.venv .venv311 venv) do (
    if not defined VENV_DIR (
        if exist "%%~D\Scripts\python.exe" if exist "%%~D\pyvenv.cfg" set "VENV_DIR=%%~D"
    )
)

exit /b 0

:create_venv
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -m venv .venv >nul 2>&1
)
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv >nul 2>&1
)
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Failed to create virtual environment.
    echo Please make sure Python 3.10+ is installed and available in PATH.
    exit /b 1
)
set "VENV_DIR=.venv"
exit /b 0

:resolve_python
for %%I in ("!VENV_DIR!\Scripts\python.exe") do set "VENV_PYTHON=%%~fI"
if not exist "!VENV_PYTHON!" (
    echo [ERROR] Python executable not found in virtual environment.
    exit /b 1
)
exit /b 0

:ensure_dependencies
if not defined VENV_PYTHON call :resolve_python || exit /b 1

echo [INFO] Installing dependencies...
call "!VENV_PYTHON!" -m pip install --upgrade pip >"%LOG_DIR%\pip_upgrade.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    echo         See %LOG_DIR%\pip_upgrade.log
    exit /b 1
)

call "!VENV_PYTHON!" -m pip install -r requirements.txt >"%LOG_DIR%\pip_install.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    echo         See %LOG_DIR%\pip_install.log
    exit /b 1
)

exit /b 0

:ensure_rust_binary
if exist "%RUST_SERVICE_EXE%" exit /b 0

echo [INFO] Rust binary not found, building release executable...
if not exist "%ROOT_DIR%rust\Cargo.toml" (
    echo [ERROR] rust\Cargo.toml not found.
    exit /b 1
)

cargo build --release --manifest-path "%ROOT_DIR%rust\Cargo.toml" >"%LOG_DIR%\rust_build.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Rust build failed.
    echo         See %LOG_DIR%\rust_build.log
    exit /b 1
)

if not exist "%RUST_SERVICE_EXE%" (
    echo [ERROR] Rust build completed but executable was not found.
    exit /b 1
)
exit /b 0

:health_check_environment
if not exist "!VENV_PYTHON!" (
    echo [ERROR] Python executable missing from venv.
    exit /b 1
)

if not exist "%ROOT_DIR%requirements.txt" (
    echo [ERROR] requirements.txt not found.
    exit /b 1
)

if not exist "%RUST_SERVICE_EXE%" (
    echo [ERROR] Rust executable not found:
    echo         %RUST_SERVICE_EXE%
    echo         Run: start_queue.bat setup
    exit /b 1
)

call "!VENV_PYTHON!" -c "import fastapi, uvicorn, celery, redis, sqlalchemy, cv2" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python dependencies are missing.
    echo         Run: start_queue.bat setup
    exit /b 1
)

exit /b 0

:run_project_checks
if not exist "%ROOT_DIR%requirements.txt" (
    echo [ERROR] requirements.txt not found.
    exit /b 1
)

if not exist "%ROOT_DIR%pyproject.toml" (
    echo [ERROR] pyproject.toml not found.
    exit /b 1
)

if not exist "!VENV_PYTHON!" (
    echo [ERROR] Python executable missing from venv.
    exit /b 1
)

if exist "%ROOT_DIR%.pre-commit-config.yaml" (
    echo [INFO] Running pre-commit...
    call "!VENV_PYTHON!" -m pre_commit run --all-files
    if errorlevel 1 (
        echo [ERROR] pre-commit checks failed.
        exit /b 1
    )
) else (
    echo [WARN] .pre-commit-config.yaml not found, skipping pre-commit.
)

if exist "%ROOT_DIR%rust\Cargo.toml" (
    echo [INFO] Running Rust formatting check...
    pushd "%ROOT_DIR%rust"
    cargo fmt --all --check
    if errorlevel 1 (
        popd
        echo [ERROR] cargo fmt failed.
        exit /b 1
    )
    echo [INFO] Running Rust clippy...
    cargo clippy --all-targets --all-features -- -D warnings
    if errorlevel 1 (
        popd
        echo [ERROR] cargo clippy failed.
        exit /b 1
    )
    popd
)

exit /b 0

:start_rust_service
echo [INFO] Starting Rust service...
start "Rust Service" cmd /k "cd /d \"%ROOT_DIR%\" && set RUST_SERVICE_HOST=127.0.0.1 && set RUST_SERVICE_PORT=50051 && \"%RUST_SERVICE_EXE%\" > \"%LOG_DIR%\rust.log\" 2>&1"
exit /b 0

:start_celery_worker
echo [INFO] Starting Celery worker...
start "Celery Worker" cmd /k "cd /d \"%ROOT_DIR%\" && set RUST_SERVICE_URL=%RUST_SERVICE_URL% && \"!VENV_PYTHON!\" -m celery -A app.celery_app worker --loglevel=info --pool=solo > \"%LOG_DIR%\celery.log\" 2>&1"
exit /b 0

:start_web_service
echo [INFO] Starting FastAPI web...
start "FastAPI Web" cmd /k "cd /d \"%ROOT_DIR%\" && set RUST_SERVICE_URL=%RUST_SERVICE_URL% && \"!VENV_PYTHON!\" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload > \"%LOG_DIR%\web.log\" 2>&1"
exit /b 0

:wait_http_ready
set "READY_URL=%~1"
set /a "MAX_WAIT=%~2"
set "SERVICE_NAME=%~3"
set /a "ELAPSED=0"

echo [INFO] Waiting for %SERVICE_NAME% ...
:wait_loop
call "!VENV_PYTHON!" -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=2).read(); print('ok')" "%READY_URL%" >nul 2>&1
if not errorlevel 1 exit /b 0

if !ELAPSED! geq !MAX_WAIT! (
    echo [ERROR] %SERVICE_NAME% did not become ready in time.
    echo         Please check logs in %LOG_DIR%
    exit /b 1
)

timeout /t 1 /nobreak >nul
set /a ELAPSED+=1
goto :wait_loop

:fail
echo.
echo [ERROR] Startup failed.
echo         Please inspect logs in %LOG_DIR%
pause
exit /b 1
