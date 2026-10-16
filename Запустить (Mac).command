#!/usr/bin/env bash
# Двойной клик в Finder: запускает веб-интерфейс и открывает его в браузере (http://127.0.0.1:8765).
cd "$(dirname "$0")" || exit 1
./start_reel_factory.sh "$@"
status=$?
if [[ $status -ne 0 ]]; then
  echo
  echo "Не удалось запустить. Если это первый раз — сначала запустите «Установить (Mac).command»."
  read -r -p "Нажмите Enter, чтобы закрыть окно…" _
else
  echo "Reel Factory работает в фоне: http://127.0.0.1:8765  (это окно можно закрыть)."
fi
exit $status
