# Zoom Reels Cutter: идемпотентная установка окружения для Windows 10/11 (x64).
# Всё ставится локально в .tools\ и .venv\ — без прав администратора и без правки PATH/реестра.
# Файл сохранён в UTF-8 с BOM (нужно Windows PowerShell 5.1 для русского текста). Не пересохраняйте без BOM.
#
# ВНИМАНИЕ: этот скрипт написан без возможности запуска на Windows и ещё не проверен на реальной машине.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # без этого Invoke-WebRequest в PS 5.1 качает в разы медленнее
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Root  = $PSScriptRoot
Set-Location $Root
$Tools = Join-Path $Root '.tools'
$Dl    = Join-Path $Tools 'dl'
$HyperframesVersion = '0.8.4'
$PythonVersion = '3.11'
New-Item -ItemType Directory -Force -Path $Tools, $Dl | Out-Null

function Log([string]$msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }

function Download([string]$url, [string]$dest) {
    for ($i = 1; $i -le 3; $i++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
            return
        } catch {
            if ($i -eq 3) { throw "Не удалось скачать $url : $($_.Exception.Message)" }
            Write-Host "Повторная попытка ($i)…"
            Start-Sleep -Seconds 3
        }
    }
}

function Unzip([string]$zip, [string]$dest) {
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $dest)
}

# Запуск внешней программы: возвращает @{Code; Out}. Внутри ошибки stderr не считаются исключениями (особенность PS 5.1).
function Capture([string]$exe, [string[]]$arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $exe @arguments 2>&1 | ForEach-Object { "$_" }
        return @{ Code = $LASTEXITCODE; Out = ($out -join "`n") }
    } finally { $ErrorActionPreference = $old }
}

# Запуск с выводом на экран; при ненулевом коде завершения бросает исключение.
function Run([string]$exe, [string[]]$arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $exe @arguments; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "Команда завершилась с ошибкой ($code): $exe $($arguments -join ' ')" }
}

function Main {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Нужна 64-разрядная Windows 10/11.' }
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        Write-Host 'Предупреждение: Windows на ARM. Будут установлены x64-версии (работают через эмуляцию).' -ForegroundColor Yellow
    }
    if ($Root -match '[^\u0000-\u007F]') {
        Write-Host "Предупреждение: в пути к папке есть не-латинские символы ($Root). Если что-то не заработает, перенесите папку, например, в C:\zoom-reels-cutter." -ForegroundColor Yellow
    }
    if ($Root.Length -gt 70) {
        Write-Host 'Предупреждение: путь к папке длинный; у Windows лимит 260 символов на путь. Лучше держать проект поближе к корню диска.' -ForegroundColor Yellow
    }

    # ------------------------------------------------------------ uv + Python
    Log "uv и Python $PythonVersion"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python-installs'
    $env:UV_CACHE_DIR = Join-Path $Tools 'uv-cache'
    $uvDir = Join-Path $Tools 'uv'
    $uv = Join-Path $uvDir 'uv.exe'
    if (-not (Test-Path $uv)) {
        $zip = Join-Path $Dl 'uv.zip'
        Download 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' $zip
        $tmp = Join-Path $Dl 'uv-extract'
        Unzip $zip $tmp
        New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
        $found = Get-ChildItem -Path $tmp -Recurse -Filter 'uv.exe' | Select-Object -First 1
        if (-not $found) { throw 'В архиве uv не найден uv.exe.' }
        Copy-Item $found.FullName $uv -Force
        Remove-Item -Recurse -Force $tmp
    }
    Run $uv @('--version')

    $venvPython = Join-Path $Root '.venv\Scripts\python.exe'
    $needVenv = $true
    if (Test-Path $venvPython) {
        $check = Capture $venvPython @('-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)')
        if ($check.Code -eq 0) { $needVenv = $false }
    }
    if ($needVenv) {
        if (Test-Path (Join-Path $Root '.venv')) { Remove-Item -Recurse -Force (Join-Path $Root '.venv') }
        # На Windows ARM берём x64-Python: для arm64 нет колёс opencv-python.
        $pySpec = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { "cpython-$PythonVersion-windows-x86_64" } else { $PythonVersion }
        Run $uv @('venv', '--python', $pySpec, (Join-Path $Root '.venv'))
    }
    Log 'Python-зависимости (requirements.txt)'
    Run $uv @('pip', 'install', '--python', $venvPython, '-r', (Join-Path $Root 'requirements.txt'))

    # ------------------------------------------------------------ FFmpeg
    Log 'FFmpeg + FFprobe (сборка с libass)'
    $ffDir = Join-Path $Tools 'ffmpeg'
    $ffmpeg = Join-Path $ffDir 'ffmpeg.exe'
    $ffprobe = Join-Path $ffDir 'ffprobe.exe'
    $haveSubs = $false
    if ((Test-Path $ffmpeg) -and (Test-Path $ffprobe)) {
        $haveSubs = (Capture $ffmpeg @('-hide_banner', '-filters')).Out -match ' subtitles '
    }
    if (-not $haveSubs) {
        $sources = @(
            'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
            'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
        )
        $ok = $false
        foreach ($url in $sources) {
            try {
                Write-Host "Скачиваю $url"
                $zip = Join-Path $Dl 'ffmpeg.zip'
                Download $url $zip
                $tmp = Join-Path $Dl 'ffmpeg-extract'
                Unzip $zip $tmp
                $f1 = Get-ChildItem -Path $tmp -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
                $f2 = Get-ChildItem -Path $tmp -Recurse -Filter 'ffprobe.exe' | Select-Object -First 1
                if (-not $f1 -or -not $f2) { throw 'В архиве нет ffmpeg.exe/ffprobe.exe.' }
                New-Item -ItemType Directory -Force -Path $ffDir | Out-Null
                Copy-Item $f1.FullName $ffmpeg -Force
                Copy-Item $f2.FullName $ffprobe -Force
                Remove-Item -Recurse -Force $tmp
                if ((Capture $ffmpeg @('-hide_banner', '-filters')).Out -match ' subtitles ') { $ok = $true; break }
                Write-Host 'В этой сборке нет фильтра subtitles (libass), пробую другой источник.' -ForegroundColor Yellow
            } catch {
                Write-Host "Источник не подошёл: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        }
        if (-not $ok) { throw 'Не удалось установить FFmpeg с фильтром subtitles (libass).' }
    }
    Run $ffmpeg @('-version')

    # ------------------------------------------------------------ шрифты
    Log 'Шрифты для субтитров (Arial с кириллицей)'
    $fontsDir = Join-Path $Tools 'fonts'
    New-Item -ItemType Directory -Force -Path $fontsDir | Out-Null
    foreach ($name in @('arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf', 'ariblk.ttf')) {
        $src = Join-Path $env:WINDIR "Fonts\$name"
        if (Test-Path $src) { Copy-Item $src (Join-Path $fontsDir $name) -Force }
    }
    Get-ChildItem $fontsDir | ForEach-Object { $_.Name }

    # ------------------------------------------------------------ Node.js
    Log 'Node.js LTS (x64)'
    $nodeDir = Join-Path $Tools 'node'
    $node = Join-Path $nodeDir 'node.exe'
    $npm = Join-Path $nodeDir 'npm.cmd'
    if (-not (Test-Path $node)) {
        $index = Invoke-RestMethod -Uri 'https://nodejs.org/dist/index.json' -UseBasicParsing
        $version = ($index | Where-Object { $_.lts } | Select-Object -First 1).version
        if (-not $version) { throw 'Не удалось определить версию Node.js LTS.' }
        Write-Host "Node.js $version"
        $zip = Join-Path $Dl 'node.zip'
        Download "https://nodejs.org/dist/$version/node-$version-win-x64.zip" $zip
        $tmp = Join-Path $Dl 'node-extract'
        Unzip $zip $tmp
        $inner = Get-ChildItem -Path $tmp -Directory | Select-Object -First 1
        if (-not $inner) { throw 'Архив Node.js пуст.' }
        if (Test-Path $nodeDir) { Remove-Item -Recurse -Force $nodeDir }
        Move-Item $inner.FullName $nodeDir
        Remove-Item -Recurse -Force $tmp
    }
    Run $node @('--version')
    $env:PATH = "$nodeDir;$env:PATH"

    # ------------------------------------------------------------ HyperFrames
    Log "HyperFrames CLI $HyperframesVersion"
    # Именно здесь его ищет reel_factory/hyperframes_visual.py (ISOLATED_HYPERFRAMES_RUNTIME).
    $hfDir = Join-Path $Tools 'broadcast-motion-runtime-full'
    $hfPkg = Join-Path $hfDir 'node_modules\hyperframes\package.json'
    $hfBin = Join-Path $hfDir 'node_modules\hyperframes\bin\hyperframes.mjs'
    $installed = ''
    if (Test-Path $hfPkg) { $installed = (Get-Content $hfPkg -Raw -Encoding UTF8 | ConvertFrom-Json).version }
    if ($installed -ne $HyperframesVersion) {
        New-Item -ItemType Directory -Force -Path $hfDir | Out-Null
        $pj = Join-Path $hfDir 'package.json'
        if (-not (Test-Path $pj)) {
            # без BOM: npm не любит BOM в package.json
            [System.IO.File]::WriteAllText($pj, '{"name":"hyperframes-runtime","private":true}', (New-Object System.Text.UTF8Encoding($false)))
        }
        Push-Location $hfDir
        try { Run $npm @('install', '--no-audit', '--no-fund', '--save-exact', "hyperframes@$HyperframesVersion") } finally { Pop-Location }
    }
    Run $node @($hfBin, '--version')

    Log 'Chromium для рендера графики HyperFrames (скачивается один раз, ~150 МБ)'
    Run $node @($hfBin, 'browser', 'ensure')

    # ------------------------------------------------------------ проверка
    Log 'Проверка'
    Run $venvPython @('-c', "import faster_whisper, cv2, yt_dlp, numpy; print('python deps ok')")
    $env:PYTHONUTF8 = '1'
    $d = Capture $venvPython @('-c', "from reel_factory import runtime_paths as r; print(r.describe())")
    Write-Host $d.Out

    # ------------------------------------------------------------ ИИ (Claude Code)
    $claude = Get-Command claude -ErrorAction SilentlyContinue
    $claudePath = $null
    if ($claude) { $claudePath = $claude.Source }
    foreach ($c in @((Join-Path $HOME '.local\bin\claude.exe'), (Join-Path $env:APPDATA 'npm\claude.cmd'))) {
        if (-not $claudePath -and (Test-Path $c)) { $claudePath = $c }
    }
    if ($claudePath) {
        Write-Host "Claude Code найден: $claudePath"
        if ((Capture $claudePath @('auth', 'status')).Code -ne 0) {
            Write-Host 'Но вход не выполнен. Откройте PowerShell и выполните:  claude auth login' -ForegroundColor Yellow
        }
    } else {
        Write-Host ''
        Write-Host 'ВНИМАНИЕ: Claude Code (команда claude) не найден. Он нужен для смыслового монтажа и НЕ ставится автоматически.' -ForegroundColor Yellow
        Write-Host '  1) Установите Git for Windows (https://git-scm.com/download/win), если его ещё нет.'
        Write-Host '  2) В PowerShell выполните:  irm https://claude.ai/install.ps1 | iex'
        Write-Host '  3) Закройте и снова откройте PowerShell, затем войдите в подписку Claude:  claude auth login'
        Write-Host '  Подробнее: https://docs.claude.com/en/docs/claude-code/setup'
    }

    Write-Host ''
    Write-Host 'Готово. Запуск: двойной клик по «Запустить (Windows).cmd».' -ForegroundColor Green
    Write-Host 'Необязательно: ключ Deepgram для быстрого облачного распознавания — скопируйте .env.example в .env и впишите ключ.'
}

try {
    Main
    exit 0
} catch {
    Write-Host ''
    Write-Host "ОШИБКА: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
