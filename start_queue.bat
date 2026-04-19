@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "VENV_DIR="
set "VENV_PYTHON="
set "NEED_INSTALL=0"

echo [1/4] Checking Python virtual environment...

if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "VENV_DIR=%VIRTUAL_ENV%"
    )
)

if not defined VENV_DIR (
    for %%D in (.venv .venv311 venv) do (
        if not defined VENV_DIR (
            if exist "%%~D\Scripts\python.exe" if exist "%%~D\pyvenv.cfg" (
                set "VENV_DIR=%%~D"
            )
        )
    )
)

if not defined VENV_DIR (
    for /d %%D in (*) do (
        if not defined VENV_DIR (
            if exist "%%~fD\Scripts\python.exe" if exist "%%~fD\pyvenv.cfg" (
                set "VENV_DIR=%%~D"
            )
        )
    )
)

if not defined VENV_DIR (
    echo No virtual environment found. Creating .venv...

    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.11 -m venv .venv >nul 2>&1
    )

    if not exist ".venv\Scripts\python.exe" (
        python -m venv .venv >nul 2>&1
    )

    if not exist ".venv\Scripts\python.exe" (
        echo [ERROR] Failed to create virtual environment.
        echo Please make sure Python is installed and available in PATH.
        pause
        exit /b 1
    )

    set "VENV_DIR=.venv"
    set "NEED_INSTALL=1"
    echo Virtual environment created: !VENV_DIR!
) else (
    echo Using virtual environment: !VENV_DIR!
)

for %%I in ("!VENV_DIR!\Scripts\python.exe") do set "VENV_PYTHON=%%~fI"

"!VENV_PYTHON!" -c "import fastapi, uvicorn, celery, redis, sqlalchemy, cv2" >nul 2>&1
if errorlevel 1 (
    set "NEED_INSTALL=1"
)

if "!NEED_INSTALL!"=="1" (
    echo Installing project dependencies...
    call "!VENV_PYTHON!" -m pip install --upgrade pip
    if errorlevel 1 (
        echo [ERROR] Failed to upgrade pip.
        pause
        exit /b 1
    )

    call "!VENV_PYTHON!" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies from requirements.txt.
        pause
        exit /b 1
    )
)

echo [2/3] Starting Celery Worker...
start "Celery Worker" cmd /k ""!VENV_PYTHON!" -m celery -A app.celery_app worker --loglevel=info --pool=solo"

timeout /t 2 /nobreak >nul

echo [3/3] Starting FastAPI Web...
start "FastAPI Web" cmd /k ""!VENV_PYTHON!" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8000

echo.
echo Started successfully:
echo - Virtual environment: !VENV_DIR!
echo - Celery Worker window
echo - FastAPI Web window
echo - Browser: http://127.0.0.1:8000
echo.
echo Close the two terminal windows to stop services.
pause
