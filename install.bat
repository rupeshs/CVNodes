@echo off
setlocal

cd /d "%~dp0backend"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3 and try again.
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment in backend\.venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo Installing dependencies from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    exit /b 1
)

echo.
echo Install complete. Run start.bat to launch the server.
endlocal
