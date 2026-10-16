#!/usr/bin/env bash
# Двойной клик в Finder: устанавливает всё необходимое (один раз; повторный запуск безопасен).
cd "$(dirname "$0")" || exit 1
chmod +x setup_mac.sh start_reel_factory.sh "Запустить (Mac).command" 2>/dev/null
./setup_mac.sh
status=$?
echo
if [[ $status -eq 0 ]]; then
  echo "Установка завершена. Теперь запускайте «Запустить (Mac).command»."
else
  echo "Установка завершилась с ошибкой (код $status). Прокрутите окно вверх, чтобы увидеть причину."
fi
read -r -p "Нажмите Enter, чтобы закрыть окно…" _
exit $status
