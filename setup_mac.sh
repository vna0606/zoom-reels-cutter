#!/usr/bin/env bash
# Zoom Reels Cutter: идемпотентная установка окружения для macOS (Apple Silicon и Intel).
# Всё ставится локально в .tools/ и .venv/ — без sudo, brew и правки профилей оболочки.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TOOLS="$ROOT/.tools"
HYPERFRAMES_VERSION="0.8.4"
PYTHON_VERSION="3.11"
mkdir -p "$TOOLS" "$TOOLS/dl"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "setup_mac.sh рассчитан на macOS." >&2
  exit 1
fi
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
  FF_SUFFIX="arm"; NODE_ARCH="arm64"
else
  FF_SUFFIX="intel"; NODE_ARCH="x64"
fi

log() { printf '\n==> %s\n' "$*"; }
download() { curl -fL --retry 3 --progress-bar -o "$2" "$1"; }

# ---------------------------------------------------------------- uv + Python
log "uv и Python $PYTHON_VERSION"
export UV_PYTHON_INSTALL_DIR="$TOOLS/python-installs"
export UV_CACHE_DIR="$TOOLS/uv-cache"
UV="$TOOLS/uv/uv"
if [[ ! -x "$UV" ]]; then
  # Официальный установщик astral.sh; UV_UNMANAGED_INSTALL = не трогать PATH и профили.
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$TOOLS/uv" sh
fi
"$UV" --version

if [[ ! -x "$ROOT/.venv/bin/python" ]] || ! "$ROOT/.venv/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  rm -rf "$ROOT/.venv"
  "$UV" venv --python "$PYTHON_VERSION" "$ROOT/.venv"
fi
log "Python-зависимости (requirements.txt)"
"$UV" pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"

# ---------------------------------------------------------------- ffmpeg
log "FFmpeg + FFprobe (статическая сборка с libass)"
FFDIR="$TOOLS/ffmpeg"
if [[ ! -x "$FFDIR/ffmpeg" || ! -x "$FFDIR/ffprobe" ]] || ! "$FFDIR/ffmpeg" -hide_banner -filters 2>/dev/null | grep -q ' subtitles '; then
  mkdir -p "$FFDIR"
  # osxexperts.net — общеизвестный источник статических macOS-сборок (GPL, libass, x264, fontconfig).
  download "https://www.osxexperts.net/ffmpeg71${FF_SUFFIX}.zip" "$TOOLS/dl/ffmpeg.zip"
  download "https://www.osxexperts.net/ffprobe71${FF_SUFFIX}.zip" "$TOOLS/dl/ffprobe.zip"
  unzip -oq "$TOOLS/dl/ffmpeg.zip" ffmpeg -d "$FFDIR"
  unzip -oq "$TOOLS/dl/ffprobe.zip" ffprobe -d "$FFDIR"
  chmod +x "$FFDIR/ffmpeg" "$FFDIR/ffprobe"
  xattr -d com.apple.quarantine "$FFDIR/ffmpeg" "$FFDIR/ffprobe" 2>/dev/null || true
fi
"$FFDIR/ffmpeg" -hide_banner -filters 2>/dev/null | grep ' subtitles ' \
  || { echo "В сборке FFmpeg нет фильтра subtitles (libass)." >&2; exit 1; }

# ---------------------------------------------------------------- шрифты
log "Шрифты для субтитров (Arial с кириллицей)"
# libass сканирует весь fontsdir; в /System/Library/Fonts/Supplemental есть шрифты, на которых он
# пишет предупреждения, поэтому копируем только семейство Arial в отдельную папку.
mkdir -p "$TOOLS/fonts"
for f in "Arial.ttf" "Arial Bold.ttf" "Arial Italic.ttf" "Arial Bold Italic.ttf" "Arial Black.ttf" "Arial Unicode.ttf"; do
  for dir in /System/Library/Fonts/Supplemental /Library/Fonts; do
    if [[ -f "$dir/$f" ]]; then cp -f "$dir/$f" "$TOOLS/fonts/$f"; break; fi
  done
done
ls "$TOOLS/fonts"

# ---------------------------------------------------------------- Node.js
log "Node.js LTS ($NODE_ARCH)"
NODE_DIR="$TOOLS/node"
if [[ ! -x "$NODE_DIR/bin/node" ]]; then
  NODE_VERSION="$(curl -fsSL https://nodejs.org/dist/index.json | "$ROOT/.venv/bin/python" -c '
import json, sys
print(next(item["version"] for item in json.load(sys.stdin) if item["lts"]))')"
  echo "Node.js $NODE_VERSION"
  download "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-darwin-$NODE_ARCH.tar.gz" "$TOOLS/dl/node.tar.gz"
  rm -rf "$NODE_DIR" "$TOOLS/node-extract"
  mkdir -p "$TOOLS/node-extract"
  tar -xzf "$TOOLS/dl/node.tar.gz" -C "$TOOLS/node-extract"
  mv "$TOOLS/node-extract"/node-* "$NODE_DIR"
  rmdir "$TOOLS/node-extract"
fi
"$NODE_DIR/bin/node" --version

# ---------------------------------------------------------------- HyperFrames
log "HyperFrames CLI $HYPERFRAMES_VERSION"
HF_DIR="$TOOLS/broadcast-motion-runtime-full"   # именно здесь его ищет reel_factory/hyperframes_visual.py
HF_BIN="$HF_DIR/node_modules/hyperframes/bin/hyperframes.mjs"
export PATH="$NODE_DIR/bin:$PATH"
installed_hf=""
if [[ -f "$HF_DIR/node_modules/hyperframes/package.json" ]]; then
  installed_hf="$(node -p "require('$HF_DIR/node_modules/hyperframes/package.json').version")"
fi
if [[ "$installed_hf" != "$HYPERFRAMES_VERSION" ]]; then
  mkdir -p "$HF_DIR"
  [[ -f "$HF_DIR/package.json" ]] || echo '{"name":"hyperframes-runtime","private":true}' > "$HF_DIR/package.json"
  (cd "$HF_DIR" && npm install --no-audit --no-fund --save-exact "hyperframes@$HYPERFRAMES_VERSION")
fi
node "$HF_BIN" --version

log "Chromium для рендера графики HyperFrames (скачивается один раз, ~150 МБ)"
node "$HF_BIN" browser ensure

# ---------------------------------------------------------------- проверка
log "Проверка"
"$ROOT/.venv/bin/python" -c "import faster_whisper, cv2, yt_dlp, numpy; print('python deps ok')"
"$ROOT/.venv/bin/python" -c "from reel_factory import runtime_paths as r; print(r.describe())" 2>/dev/null || true

# ---------------------------------------------------------------- ИИ (Claude Code)
CLAUDE_BIN="$(command -v claude || true)"
for c in "$HOME/.local/bin/claude" "$HOME/.claude/local/claude" /opt/homebrew/bin/claude /usr/local/bin/claude; do
  [[ -z "$CLAUDE_BIN" && -x "$c" ]] && CLAUDE_BIN="$c"
done
if [[ -n "$CLAUDE_BIN" ]]; then
  echo "Claude Code найден: $CLAUDE_BIN"
  if ! "$CLAUDE_BIN" auth status >/dev/null 2>&1; then
    echo "Но вход не выполнен. Откройте Терминал и выполните:  claude auth login"
  fi
else
  cat <<'NOCLAUDE'

ВНИМАНИЕ: Claude Code (команда `claude`) не найден. Он нужен для смыслового монтажа и НЕ ставится автоматически.
  1) Установите:  curl -fsSL https://claude.ai/install.sh | bash
  2) Войдите в подписку Claude:  claude auth login
  Подробнее: https://docs.claude.com/en/docs/claude-code/setup
NOCLAUDE
fi

cat <<MSG

Готово. Запуск: двойной клик по «Запустить (Mac).command» (или ./start_reel_factory.sh).
Необязательно: ключ Deepgram для быстрого облачного распознавания — скопируйте .env.example в .env и впишите ключ.
MSG
