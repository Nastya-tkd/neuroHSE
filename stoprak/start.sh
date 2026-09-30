#!/bin/sh
# Запуск СТОП-РАК (macOS / Linux): двойной клик не нужен — выполните ./start.sh
cd "$(dirname "$0")"
echo "Откройте в браузере: http://localhost:8000  (остановить: Ctrl+C)"
(sleep 1; (open http://localhost:8000 || xdg-open http://localhost:8000) >/dev/null 2>&1) &
python3 -m http.server 8000
