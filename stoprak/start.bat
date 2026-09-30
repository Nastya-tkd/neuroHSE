@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Откройте в браузере: http://localhost:8000  (остановить: Ctrl+C)
start "" http://localhost:8000
python -m http.server 8000 || py -m http.server 8000
pause
