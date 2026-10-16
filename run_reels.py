"""Одна команда: эфир целиком -> все сильные ролики (подкаст вертикальный + горизонтальный, сырой звук) + отчёт.

    .venv/bin/python run_reels.py <video_path> [--out DIR] [--provider claude|codex] [--model small|base]
                                  [--asr deepgram|local|auto] [--min 30 --max 90] [--audience "текст"] [--no-review] [--parallel 3]

Этапы: транскрипция (точная, со словарём терминов, кэш в <out>/transcript.json) -> смысловой монтаж (все ролики:
карта тем, компоновка, пропущенные моменты, критик, ремонт, главред, чистка) -> рендер каждого ролика и каждого
отсеянного кандидата (в rejected/) -> отсмотр готового видео по содержанию и звуку -> report.json и index.html.
Возобновляется с кэшей: повторный запуск с тем же --out продолжает с места остановки.
Журнал с таймингами этапов: <out>/run.log. По умолчанию --out = results/<имя-видео>/ внутри папки проекта.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="путь к видеофайлу эфира")
    parser.add_argument("--out", help="папка результата (по умолчанию results/<имя-видео>/)")
    parser.add_argument("--provider", choices=["claude", "codex"], default="claude", help="ИИ-провайдер (claude)")
    parser.add_argument("--model", choices=["small", "base"], default="small", help="модель распознавания: small — точная")
    parser.add_argument("--min", dest="min_duration", type=int, default=30, help="минимум секунд ролика (30)")
    parser.add_argument("--max", dest="max_duration", type=int, default=90, help="максимум секунд ролика (90)")
    parser.add_argument("--asr", choices=["deepgram", "local", "auto"], default="auto",
                        help="распознавание: deepgram (облако), local (faster-whisper), auto — deepgram, если есть ключ")
    parser.add_argument("--language", default="ru", choices=["ru", "en", "auto"])
    parser.add_argument("--audience", default=None, help="кто смотрит эфир (контекст для компоновки; не фильтр)")
    parser.add_argument("--parallel", type=int, default=3, help="параллельных запросов к ИИ (3)")
    parser.add_argument("--no-review", action="store_true", help="не отсматривать готовое видео (быстрее)")
    parser.add_argument("--no-coverage", action="store_true", help="не искать пропущенные сильные моменты")
    parser.add_argument("--mode", choices=["broadcast", "podcast"], default="broadcast",
                        help="вертикаль: broadcast — спикеры Zoom + графика HyperFrames (по умолчанию), podcast — кадрирование по лицу без графики")
    parser.add_argument("--speakers", type=int, choices=[1, 2], default=2, help="сколько спикеров показывать в режиме broadcast (2)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    video = Path(args.video).expanduser()
    if not video.is_file():
        print(f"Файл не найден: {video}", file=sys.stderr)
        return 2
    from reel_factory import batch
    out = Path(args.out).expanduser() if args.out else ROOT / "results" / video.stem
    overrides = {"provider": args.provider, "model": args.model, "min_duration": args.min_duration,
                 "max_duration": args.max_duration, "language": args.language, "parallel": args.parallel,
                 "asr_provider": args.asr, "project_mode": args.mode, "speaker_count": args.speakers}
    if args.audience is not None:
        overrides["audience"] = args.audience
    if args.no_review:
        overrides["video_review"] = False
    if args.no_coverage:
        overrides["coverage"] = False
    options = batch.build_options(**overrides)
    if options["max_duration"] <= options["min_duration"]:
        print("--max должен быть больше --min", file=sys.stderr)
        return 2
    batch.run(video, out, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
