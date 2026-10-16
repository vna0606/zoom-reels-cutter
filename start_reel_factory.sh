#!/usr/bin/env bash
# Запуск Zoom Reels Cutter на macOS/Linux (локальное окружение .venv + .tools).
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
export PYTHONUTF8=1
export HF_HOME="$PWD/.tools/hf"

if [[ ! -x .venv/bin/python ]]; then
  echo "Окружение не найдено — запускаю ./setup_mac.sh"
  ./setup_mac.sh || exit 1
fi
exec .venv/bin/python launch_reel_factory.py "$@"
