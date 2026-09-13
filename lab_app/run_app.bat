@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] First run: creating a local Python environment and installing packages...
    echo       This needs internet access once; after that the app runs fully offline.
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not find Python. Install Python 3.10+ from https://www.python.org/downloads/
        echo and check "Add python.exe to PATH" during setup, then run this file again.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

echo [2/2] Starting GL261 Segmentation Assistant...
echo A browser tab will open automatically. Close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py

pause
