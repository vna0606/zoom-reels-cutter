from __future__ import annotations

import html
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".tools" / "opencv4"))
sys.path.insert(1, str(ROOT / ".tools" / "python"))

import yt_dlp
import cv2

from . import assembly, editorial, llm, review as video_review
from .pipeline import FFMPEG, PipelineError, media_info, process_video_with_transcript, transcribe_video

ProgressCallback = Callable[[int, str], None]
TAIL_PAD = 0.3   # секунд «воздуха» после последнего слова ролика (иначе обрыв на звуке слова)


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def valid_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower().split(":")[0]
        return parsed.scheme in {"http", "https"} and (
            host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
            or host.endswith(".youtube.com")
        )
    except Exception:
        return False


def provider_status() -> dict:
    status = llm.provider_status()
    status["heuristic"] = {"installed": True, "ready": True}
    try:
        status["resolved"] = llm.resolve_provider("auto")
    except llm.LLMError:
        status["resolved"] = None
    return status


def _download_progress(callback: ProgressCallback):
    def hook(data):
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            downloaded = data.get("downloaded_bytes") or 0
            if total:
                callback(5 + min(24, int(downloaded / total * 24)), "Скачиваю видео с YouTube")
        elif status == "finished":
            callback(30, "Видео YouTube загружено")
    return hook


def download_youtube(url: str, input_dir: Path, subtitle_language: str,
                     callback: ProgressCallback) -> tuple[Path, dict, dict | None]:
    if not valid_youtube_url(url):
        raise PipelineError("Укажите корректную ссылку YouTube.")
    input_dir.mkdir(parents=True, exist_ok=True)
    requested_languages = [f"{subtitle_language}.*"] if subtitle_language in {"ru", "en"} else ["ru.*", "en.*"]
    options = {
        "outtmpl": str(input_dir / "source.%(ext)s"),
        # Progressive YouTube MP4 is commonly limited to 360p. Select a
        # separate HD video stream and merge it with the best audio instead.
        "format": (
            "bestvideo[height>=720][height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height>=720][height<=1080]+bestaudio/"
            "best[height>=720][height<=1080]"
        ),
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": requested_languages,
        "subtitlesformat": "json3/vtt/best",
        "progress_hooks": [_download_progress(callback)],
        "restrictfilenames": True,
    }

    def recover_completed_part(error: Exception) -> Path | None:
        error_text = str(error).lower()
        if "rename file" not in error_text and "winerror 32" not in error_text:
            return None
        for partial in input_dir.glob("source.mp4.part"):
            try:
                if partial.stat().st_size > 1_000_000 and media_info(partial)["duration"] > 1:
                    return partial
            except Exception:
                continue
        return None

    callback(3, "Получаю данные YouTube")
    info = {}
    video_path = None
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except Exception as exc:
        # YouTube can rate-limit one subtitle track after another track has
        # already been saved. A subtitle error must not cancel the video job:
        # retry the media itself and then use any requested subtitle that is
        # already present, or fall back to local speech recognition.
        video_path = recover_completed_part(exc)
        if video_path is None:
            callback(12, "Повторяю загрузку видео без дорожки субтитров")
            video_options = {
                key: value for key, value in options.items()
                if key not in {"writesubtitles", "writeautomaticsub", "subtitleslangs", "subtitlesformat"}
            }
            try:
                with yt_dlp.YoutubeDL(video_options) as downloader:
                    info = downloader.extract_info(url, download=True)
            except Exception as video_exc:
                video_path = recover_completed_part(video_exc)
                if video_path is None:
                    raise PipelineError(f"Не удалось загрузить видео YouTube: {video_exc}") from video_exc
    if video_path is None:
        candidates = [
            path for path in input_dir.glob("source.*")
            if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov", ".m4v"}
        ]
        if not candidates:
            raise PipelineError("YouTube-видео загружено, но видеофайл не найден.")
        video_path = max(candidates, key=lambda path: path.stat().st_size)
    subtitle_files = [
        path for path in input_dir.glob("source.*") if path.suffix.lower() in {".json3", ".vtt"}
    ]

    def subtitle_priority(path: Path) -> tuple[int, int, int]:
        name = path.name.lower()
        preferred = subtitle_language if subtitle_language in {"ru", "en"} else str(info.get("language") or "ru").lower()
        exact_language = 0 if re.search(rf"\.{re.escape(preferred)}(?:[-_.]|$)", name) else 1
        return exact_language, 0 if path.suffix.lower() == ".json3" else 1, len(path.name)

    subtitle_files.sort(key=subtitle_priority)
    transcript = None
    for subtitle_path in subtitle_files:
        try:
            selected_language = subtitle_language if subtitle_language not in {"auto", ""} else (info.get("language") or "unknown")
            transcript = parse_youtube_subtitles(subtitle_path, selected_language)
            if transcript.get("words"):
                transcript["source"] = video_path.name
                transcript["youtube_url"] = url
                transcript["youtube_title"] = info.get("title") or "YouTube video"
                break
        except Exception:
            continue
    metadata = {
        "title": info.get("title") or video_path.stem,
        "uploader": info.get("uploader") or "",
        "duration": info.get("duration"),
        "webpage_url": info.get("webpage_url") or url,
        "thumbnail": info.get("thumbnail") or "",
    }
    return video_path, metadata, transcript


def _split_token(text: str, start: float, end: float) -> list[dict]:
    tokens = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)*", text)
    if not tokens:
        return []
    duration = max(0.05, end - start)
    weights = [max(1, len(token)) for token in tokens]
    total = sum(weights)
    cursor = start
    words = []
    for index, (token, weight) in enumerate(zip(tokens, weights)):
        next_time = end if index == len(tokens) - 1 else cursor + duration * weight / total
        words.append({"text": token, "start": round(cursor, 3), "end": round(next_time, 3), "probability": 1.0})
        cursor = next_time
    return words


def parse_youtube_json3(path: Path, language: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    words: list[dict] = []
    segments: list[dict] = []
    seen = set()
    for event in payload.get("events", []):
        start_ms = event.get("tStartMs")
        duration_ms = event.get("dDurationMs") or 0
        cues = event.get("segs") or []
        if start_ms is None or not cues:
            continue
        event_start = float(start_ms) / 1000
        event_end = event_start + max(0.15, float(duration_ms) / 1000)
        text = "".join(str(cue.get("utf8") or "") for cue in cues).strip()
        text = re.sub(r"\s+", " ", text)
        key = (round(event_start, 2), text)
        if not text or key in seen:
            continue
        seen.add(key)
        segments.append({"start": round(event_start, 3), "end": round(event_end, 3), "text": text})
        for index, cue in enumerate(cues):
            cue_text = str(cue.get("utf8") or "").strip()
            if not cue_text:
                continue
            cue_start = event_start + float(cue.get("tOffsetMs") or 0) / 1000
            if index + 1 < len(cues):
                cue_end = event_start + float(cues[index + 1].get("tOffsetMs") or duration_ms) / 1000
            else:
                cue_end = event_end
            words.extend(_split_token(cue_text, cue_start, max(cue_start + 0.05, cue_end)))
    words.sort(key=lambda item: (item["start"], item["end"]))
    segments.sort(key=lambda item: item["start"])
    return {
        "language": language if language not in {"auto", ""} else "unknown",
        "language_probability": 1.0,
        "words": words,
        "segments": segments,
        "text": " ".join(segment["text"] for segment in segments),
    }


def parse_youtube_vtt(path: Path, language: str) -> dict:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    timestamp = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})")
    cue_pattern = re.compile(
        r"(?m)^(?P<start>(?:\d+:)?\d{2}:\d{2}[.,]\d{3})\s+-->\s+"
        r"(?P<end>(?:\d+:)?\d{2}:\d{2}[.,]\d{3})[^\n]*\n(?P<text>(?:(?!\n\s*\n).)+)", re.S
    )

    def seconds(value: str) -> float:
        match = timestamp.fullmatch(value.strip())
        if not match:
            return 0.0
        h, m, s, ms = match.groups()
        return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    segments = []
    words = []
    previous_text = ""
    for match in cue_pattern.finditer(content):
        start, end = seconds(match.group("start")), seconds(match.group("end"))
        text = re.sub(r"<[^>]+>", "", match.group("text"))
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        if not text or text == previous_text:
            continue
        previous_text = text
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
        words.extend(_split_token(text, start, end))
    return {
        "language": language if language not in {"auto", ""} else "unknown",
        "language_probability": 1.0,
        "words": words,
        "segments": segments,
        "text": " ".join(segment["text"] for segment in segments),
    }


def parse_youtube_subtitles(path: Path, language: str) -> dict:
    return parse_youtube_json3(path, language) if path.suffix.lower() == ".json3" else parse_youtube_vtt(path, language)


def heuristic_highlights(transcript: dict, options: dict) -> list[dict]:
    segments = transcript.get("segments") or []
    if not segments:
        raise PipelineError("В транскрипте нет фраз для анализа.")
    min_duration = int(options.get("min_duration", 25))
    max_duration = int(options.get("max_duration", 75))
    target = max(min_duration, min(max_duration, int((min_duration + max_duration) / 2)))
    windows = []
    start_index = 0
    while start_index < len(segments):
        start = float(segments[start_index]["start"])
        end_index = start_index
        while end_index + 1 < len(segments) and float(segments[end_index]["end"]) - start < target:
            end_index += 1
        end = float(segments[end_index]["end"])
        if end - start >= min_duration * 0.75:
            text = " ".join(str(item["text"]) for item in segments[start_index:end_index + 1])
            low = text.lower()
            score = 45
            score += min(18, text.count("?") * 5 + text.count("!") * 4)
            score += min(12, len(re.findall(r"\d+", text)) * 3)
            signals = ("почему", "главн", "никогда", "ошиб", "важн", "секрет", "деньги", "результат", "на самом деле", "представьте")
            score += min(20, sum(4 for signal in signals if signal in low))
            score += 5 if 30 <= end - start <= 65 else 0
            words = re.findall(r"[0-9A-Za-zА-Яа-яЁё-]+", text)
            title = " ".join(words[:7]).strip().capitalize() or "Сильный момент"
            description = " ".join(words[:22]).strip()
            windows.append({
                "start": round(start, 2), "end": round(end, 2),
                "title": title, "description": description,
                "hook": " ".join(words[:12]), "score": min(95, score),
            })
        start_index = max(start_index + 1, end_index - 1)
    selected = []
    for item in sorted(windows, key=lambda value: value["score"], reverse=True):
        if any(min(item["end"], other["end"]) - max(item["start"], other["start"]) > 5 for other in selected):
            continue
        selected.append(item)
    return sorted(selected, key=lambda item: item["start"])


def heuristic_to_contract(item: dict) -> dict:
    """Окно эвристики -> контракт редактора: один сегмент, без вырезов."""
    start, end = float(item["start"]), float(item["end"])
    return {
        "title": item.get("title", ""), "description": item.get("description", ""),
        "hook": item.get("hook", ""), "score": item.get("score", 50),
        "rationale": "Эвристический режим: окно по ключевым словам, без смыслового монтажа.",
        "critic": {},
        "segments": [{"start": start, "end": end, "role": "point", "sentence_from": "", "sentence_to": ""}],
        "cuts": [], "start": start, "end": end, "duration": end - start,
    }


def candidate_segments(candidate: dict) -> list[dict]:
    """Сегменты кандидата; для старых проектов без `segments` — один сегмент [start, end]."""
    segments = candidate.get("segments")
    if segments:
        return segments
    return [{"start": float(candidate["start"]), "end": float(candidate["end"]), "role": "point"}]


def candidate_playback(candidate: dict) -> dict:
    """Кандидат в форме, пригодной для assembly (старые проекты получают один сегмент)."""
    return {**candidate, "segments": candidate_segments(candidate), "cuts": candidate.get("cuts") or []}


def normalize_candidates(candidates: list[dict], duration: float, options: dict,
                         words: list[dict] | None = None) -> list[dict]:
    """Приводит ответ редактора к виду проекта: длительность по сборке, окно, сортировка, id."""
    minimum = float(options.get("min_duration", 30))
    maximum = float(options.get("max_duration", 90))
    max_clips = int(options.get("max_clips", 0))
    all_clips = bool(options.get("all_clips")) or max_clips <= 0   # режим «все ролики» (по умолчанию): без лимита
    words = words or []
    result = []
    rejected_result = []
    for item in candidates:
        rejected_info = item.get("rejected") if isinstance(item.get("rejected"), dict) else None
        try:
            segments = []
            for seg in item.get("segments") or [{"start": item["start"], "end": item["end"], "role": "point"}]:
                seg_start = max(0.0, float(seg["start"]))
                seg_end = min(duration, float(seg["end"])) if duration else float(seg["end"])
                if seg_end - seg_start < 0.2:
                    continue
                segments.append({
                    "start": round(seg_start, 3), "end": round(seg_end, 3),
                    "role": str(seg.get("role") or "point"),
                    "sentence_from": str(seg.get("sentence_from") or ""),
                    "sentence_to": str(seg.get("sentence_to") or ""),
                })
            cuts = []
            for cut in item.get("cuts") or []:
                cut_start, cut_end = float(cut["start"]), float(cut["end"])
                if cut_end > cut_start:
                    cuts.append({"start": round(cut_start, 3), "end": round(cut_end, 3),
                                 "reason": str(cut.get("reason") or "filler")})
        except (KeyError, TypeError, ValueError):
            continue
        if not segments:
            continue
        draft = {"segments": segments, "cuts": cuts}
        intervals = assembly.keep_intervals(draft, words)
        length = assembly.assembled_duration(intervals)
        fmt = str(item.get("format") or "thesis")
        fmt = fmt if fmt in editorial.FORMATS else "thesis"
        critic_in = item.get("critic") if isinstance(item.get("critic"), dict) else {}
        # list/story сильнее критиком — до max x 1.6, остальные форматы — до max (+5% допуска)
        high = editorial.final_cap(fmt, maximum, critic_in) * editorial.FINAL_OVERFLOW_TOLERANCE
        if rejected_info is not None:
            # отсеянные кандидаты остаются пригодными к рендеру («Всё равно сделать»): окно длительности не применяется
            if length < 8.0:
                continue
        elif length < max(8.0, minimum * editorial.FINAL_MIN_FACTOR) or length > high:
            continue
        removed = sum(c["end"] - c["start"] for c in cuts)
        try:
            score = max(0, min(100, int(float(item.get("score", 50)))))
        except (TypeError, ValueError):
            score = 50
        critic = item.get("critic") if isinstance(item.get("critic"), dict) else {}
        chief = item.get("chief_editor") if isinstance(item.get("chief_editor"), dict) else None
        (rejected_result if rejected_info is not None else result).append({
            "id": "",
            "start": round(min(s["start"] for s in segments), 3),
            "end": round(max(s["end"] for s in segments), 3),
            "duration": round(length, 3),
            "title": str(item.get("title") or "Сильный отрывок")[:120],
            "description": str(item.get("description") or "")[:320],
            "hook": str(item.get("hook") or "")[:260],
            # заголовок-хук для визуала (до 8 слов; не озвучивается и в звук не вставляется)
            "hook_text": editorial.clean_hook_text(item.get("hook_text") or item.get("hook"), item.get("title") or ""),
            "source": str(item.get("source") or ""),
            "moment": str(item.get("moment") or ""),
            "spinoff_of": str(item.get("spinoff_of") or ""),
            "split_of": str(item.get("split_of") or ""),
            "local_fixes": [x for x in (item.get("local_fixes") or []) if isinstance(x, dict)],
            "rationale": str(item.get("rationale") or "")[:600],
            "thesis": str(item.get("thesis") or "")[:400],
            "topic": str(item.get("topic") or "")[:200],
            "format": fmt,
            "seams": [x for x in (item.get("seams") or []) if isinstance(x, dict)],
            "score": score,
            "critic": critic,
            "chief_editor": chief,
            "rejected": ({"stage": str(rejected_info.get("stage") or ""), "reason": str(rejected_info.get("reason") or ""),
                          "scores": rejected_info.get("scores") if isinstance(rejected_info.get("scores"), dict) else {}}
                         if rejected_info is not None else None),
            "segments": segments,
            "cuts": cuts,
            "cuts_removed_seconds": round(removed, 2),
            "selected": False, "status": "suggested", "progress": 0,
            "message": "Готов к нарезке", "error": None, "output": None, "result": None,
        })
    # порядок главного редактора (rank) сохраняется; иначе hook_warning ниже, затем оценки критика; без критика — по score
    if result and all((item.get("chief_editor") or {}).get("rank") for item in result):
        result.sort(key=lambda item: int(item["chief_editor"]["rank"]))
    elif any(item.get("critic") for item in result):
        result.sort(key=lambda item: (bool(item["critic"].get("hook_warning")),
                                      -sum(int(item["critic"].get(k, 0)) for k in (
                                          "hook_strength", "completeness", "clarity", "standalone", "single_idea")),
                                      -item["score"]))
    else:
        result.sort(key=lambda item: (-item["score"], item["start"]))
    if not all_clips:
        result = result[:max_clips]
    result = result + rejected_result   # отсеянные — после основных, нумерация сквозная
    for index, item in enumerate(result, 1):
        item["id"] = f"c{index:03d}"
    return result


def analyze_transcript(transcript: dict, work_dir: Path, options: dict,
                       callback: ProgressCallback) -> tuple[list[dict], str, str | None]:
    """Return (candidates, provider_used, warning).

    Смысловой монтаж (editorial.compose_clips) работает через Claude/Codex. Сбой ИИ — ошибка:
    эвристика запускается только если пользователь выбрал её явно.
    """
    requested = options.get("provider", "auto")
    if requested == "heuristic":
        provider = "heuristic"
    else:
        provider = llm.resolve_provider(requested)  # raises LLMError (Russian message)
    callback(72, f"Ищу сильные моменты: {provider}")
    if provider == "heuristic":
        candidates = [heuristic_to_contract(item) for item in heuristic_highlights(transcript, options)]
    else:
        if not transcript.get("words"):
            raise PipelineError(
                "В транскрипте нет пословных таймкодов: смысловому монтажу нужны слова. "
                "Создайте проект заново."
            )

        def compose_progress(percent: int, message: str) -> None:
            callback(72 + int(max(0, min(100, percent)) * 0.21), message)

        composed = editorial.compose_clips(
            transcript, work_dir / "editorial", options, provider=provider, progress=compose_progress,
        )
        # отсеянные кандидаты (с куском, пригодным к рендеру) едут дальше с пометкой `rejected`
        candidates = list(composed) + list(getattr(composed, "rejected", []) or [])
    callback(94, f"Найдено вариантов: {len([c for c in candidates if not c.get('rejected')])}")
    return candidates, provider, None


def can_retry_analysis(project: dict, project_dir: Path | None = None) -> bool:
    transcript = project.get("transcript")
    if project.get("status") not in {"failed", "interrupted"} or not transcript:
        return False
    error_stage = project.get("error_stage")
    if error_stage is None and project.get("candidates"):
        return False
    if error_stage not in {None, "analysis", "thumbnails"}:
        return False
    if project_dir is None:
        return True
    return (project_dir / transcript).is_file() and bool(project.get("input"))


def prepare_analysis_retry(project: dict) -> dict:
    project["status"] = "queued"
    project["progress"] = 58
    project["phase"] = "analysis"
    project["phase_progress"] = 0
    project["message"] = "Повторно запускаю поиск сильных моментов"
    project["paused"] = False
    project["provider_used"] = None
    project["warning"] = None
    project["error"] = None
    project["error_stage"] = None
    project["candidates"] = []
    return project


def slice_transcript(transcript: dict, start: float, end: float) -> dict:
    words = []
    for word in transcript.get("words") or []:
        word_start, word_end = float(word["start"]), float(word["end"])
        if word_end <= start or word_start >= end:
            continue
        words.append({
            **word,
            "start": round(max(0, word_start - start), 3),
            "end": round(min(end - start, word_end - start), 3),
        })
    segments = []
    for segment in transcript.get("segments") or []:
        seg_start, seg_end = float(segment["start"]), float(segment["end"])
        if seg_end <= start or seg_start >= end:
            continue
        segments.append({
            "start": round(max(0, seg_start - start), 3),
            "end": round(min(end - start, seg_end - start), 3),
            "text": segment["text"],
        })
    return {
        "language": transcript.get("language", "unknown"),
        "language_probability": transcript.get("language_probability", 1.0),
        "words": words,
        "segments": segments,
        "text": " ".join(segment["text"] for segment in segments),
    }


def extract_source_clip(source: Path, destination: Path, start: float, end: float) -> None:
    """Cut a source fragment without filters, transcoding, or editorial processing."""
    duration = max(0.1, end - start)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
        "-avoid_negative_ts", "make_zero", str(destination),
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise PipelineError(
            "Не удалось вырезать исходный фрагмент без обработки: "
            + result.stderr.decode("utf-8", errors="replace")[-900:]
        )


def render_highlight(source: Path, transcript: dict, candidate: dict, output: Path,
                     work_dir: Path, options: dict, callback: ProgressCallback) -> dict:
    """Собирает горизонтальный клип из кусков кандидата и рендерит вертикальный ролик.

    Субтитры берутся из полного транскрипта, пересчитанного на таймлинию склейки, поэтому
    повторное распознавание нужно только если в транскрипте нет слов.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    playback = candidate_playback(candidate)
    intervals = assembly.keep_intervals(playback, transcript.get("words") or [], tail_pad=TAIL_PAD)
    if not intervals:
        raise PipelineError("В выбранном ролике не осталось фрагментов для сборки.")
    gains = None
    audio_mode = assembly.normalize_audio_mode(options.get("audio_mode"))
    if options.get("audio_polish", True):
        # границы резов по реальному звуку (без тихих краёв и «хвостов» чужих слов); в режиме leveled ещё и выравнивание уровней
        try:
            intervals = assembly.refine_intervals(FFMPEG, source, intervals)
            intervals = assembly.snap_to_quiet(FFMPEG, source, intervals, transcript.get("words") or [])
            if audio_mode == "leveled" and options.get("gain_matching", True):
                # уровень речи на каждом стыке выравнивается (замер на соседних 0.6 с слева/справа)
                gains = assembly.seam_gains(assembly.edge_levels(FFMPEG, source, intervals),
                                            max_step=16.0, max_total=float(options.get("gain_max", 24.0)))
        except Exception:   # полировка звука — надстройка: без неё сборка идёт как раньше
            gains = None
    horizontal = work_dir / "horizontal.mp4"

    def assemble_update(progress: int, message: str) -> None:
        callback(min(12, int(max(0, progress) * 0.12)), message)

    assembly.assemble_clip(FFMPEG, source, intervals, horizontal, callback=assemble_update, gains=gains,
                           audio_mode=audio_mode,
                           normalize=(assembly.DEFAULT_NORMALIZE if options.get("audio_polish", True) else None))
    callback(12, "Ролик собран из кусков")
    sliced = assembly.remap_transcript(transcript, intervals)
    (work_dir / "assembly.json").write_text(
        json.dumps({"intervals": intervals, "gains": gains, "audio_mode": audio_mode, "segments": playback["segments"], "cuts": playback["cuts"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    render_callback = callback
    if not sliced.get("words"):
        language = options.get("language", "ru")
        if language == "auto" and transcript.get("language") in {"ru", "en"}:
            language = transcript["language"]

        def accurate_update(progress: int, message: str):
            callback(12 + min(23, int(max(0, progress) / 36 * 23)), message)

        sliced = transcribe_video(
            horizontal,
            work_dir / "accurate_transcription",
            language,
            options.get("model", "base"),
            accurate_update,
            mode="accurate",
            asr_provider=options.get("asr_provider", "auto"),
        )

        def render_callback(progress: int, message: str):
            callback(35 + min(65, int(max(0, progress) / 100 * 65)), message)

    if options.get("project_mode") == "broadcast":
        from .broadcast import process_broadcast_with_transcript
        return process_broadcast_with_transcript(
            horizontal, output, work_dir / "vertical", options, sliced, render_callback,
            candidate=candidate,
        )
    return process_video_with_transcript(
        horizontal, output, work_dir / "vertical", options, sliced, render_callback,
    )


def _assembled(work_dir: Path) -> dict:
    """Фактические интервалы сборки (после подрезки тихих краёв), чтобы отсмотр считал стыки по реальной шкале."""
    try:
        data = json.loads((Path(work_dir) / "assembly.json").read_text(encoding="utf-8"))
        return {"intervals": [(float(a), float(b)) for a, b in data["intervals"]]}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _review_brief(review: dict | None) -> dict:
    review = review or {}
    return {"verdict": review.get("verdict"), "scores": review.get("scores") or {},
            "issues": len(review.get("issues") or []), "status": review.get("status")}


def render_highlight_reviewed(source: Path, transcript: dict, candidate: dict, output: Path,
                              work_dir: Path, options: dict, callback: ProgressCallback, *,
                              renderer=None, reviewer=None) -> dict:
    """Рендер + отсмотр готового ролика шеф-редактором (опция `video_review`) + не более одной авто-правки.

    Порядок: render -> review -> (needs_edit и правка выражается кусками) -> перерендер -> повторный review.
    Исправленная версия заменяет исходную, только если она не хуже (вердикт не reject и сумма оценок не ниже).
    Результат — словарь рендера с добавленными `review`, `auto_fix`, `horizontal` и (если правка принята)
    `candidate_update` с новыми кусками: вызывающий код обновляет кандидата.
    """
    renderer = renderer or render_highlight
    reviewer = reviewer or video_review.review_video
    output = Path(output)
    work_dir = Path(work_dir)
    if not options.get("video_review", False):
        return renderer(source, transcript, candidate, output, work_dir, options, callback)

    def stage(lo: int, hi: int) -> ProgressCallback:
        return lambda p, m: callback(lo + int(max(0, min(100, p)) * (hi - lo) / 100), m)

    result = dict(renderer(source, transcript, candidate, output, work_dir, options, stage(0, 72)) or {})
    result["horizontal"] = str(work_dir / "horizontal.mp4")
    try:
        review = reviewer(output, candidate=candidate, transcript=transcript, work_dir=work_dir / "review",
                          options=options, callback=stage(72, 88), **_assembled(work_dir))
    except Exception as exc:   # отсмотр — надстройка: ролик уже готов и не должен пропасть
        review = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "verdict": None, "scores": {},
                  "issues": [], "suggested_fix": None}
    result["review"] = review
    fix = review.get("suggested_fix")
    if review.get("verdict") != "needs_edit" or not options.get("video_review_autofix", True):
        return result
    if not video_review.fix_is_expressible(fix):
        result["auto_fix"] = {"applied": False, "reason": "правка не выражается в кусках ролика"}
        return result
    update = video_review.apply_suggested_fix(candidate, fix, transcript, options)
    if not update:
        result["auto_fix"] = {"applied": False, "reason": "правку нельзя применить к кускам (id, длительность)",
                              "description": fix.get("description", "")}
        return result
    fixed_candidate = {**candidate, **update}
    fix_dir = work_dir / "autofix"
    fixed_output = output.with_name(f"{output.stem}.autofix{output.suffix}")
    info = {"applied": True, "kept": False, "description": fix.get("description", ""), "edits": fix.get("edits", []),
            "before": _review_brief(review)}
    try:
        fixed_result = dict(renderer(source, transcript, fixed_candidate, fixed_output, fix_dir, options,
                                     stage(88, 96)) or {})
        review2 = reviewer(fixed_output, candidate=fixed_candidate, transcript=transcript, work_dir=fix_dir / "review",
                           options=options, task="video_review_2", callback=stage(96, 99), **_assembled(fix_dir))
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        fixed_output.unlink(missing_ok=True)
        result["auto_fix"] = info
        return result
    info["after"] = _review_brief(review2)
    better = (review2.get("status") == "ok" and review2.get("verdict") != "reject"
              and (review2.get("verdict") == "publish"
                   or video_review.score_sum(review2) >= video_review.score_sum(review)))
    if better and fixed_output.is_file():
        os.replace(fixed_output, output)
        info["kept"] = True
        fixed_result["horizontal"] = str(fix_dir / "horizontal.mp4")
        fixed_result.update({"review": review2, "auto_fix": info, "candidate_update": update,
                             "review_before_fix": review})
        return fixed_result
    fixed_output.unlink(missing_ok=True)
    result["auto_fix"] = info
    return result


def generate_thumbnail(source: Path, timestamp: float, destination: Path):
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    height, width = frame.shape[:2]
    target_width = 640
    target_height = 360
    scale = max(target_width / width, target_height / height)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    y = max(0, (resized.shape[0] - target_height) // 2)
    x = max(0, (resized.shape[1] - target_width) // 2)
    crop = resized[y:y + target_height, x:x + target_width]
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
