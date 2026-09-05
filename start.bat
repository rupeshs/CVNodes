@echo off
setlocal

cd /d "%~dp0backend"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run install.bat first.
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Starting CVNodes at http://127.0.0.1:8000
uvicorn server:app --reload

endlocal
