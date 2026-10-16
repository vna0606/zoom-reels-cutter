from __future__ import annotations

import argparse
import cgi
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import socket
import threading
import time
import traceback
import uuid
import webbrowser
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import formatdate
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from reel_factory.pipeline import FFMPEG, PipelineError, normalize_glossary, process_video
from reel_factory.runtime_identity import (
    BUILD_ID,
    WORKSPACE_ID,
    artifact_set_version,
    artifact_version,
    versioned_url,
)
from reel_factory import assembly, editorial
from reel_factory.highlights import (
    analyze_transcript,
    candidate_segments,
    can_retry_analysis,
    download_youtube,
    extract_source_clip,
    generate_thumbnail,
    media_info,
    normalize_candidates,
    provider_status,
    prepare_analysis_retry,
    render_highlight,
    render_highlight_reviewed,
    transcribe_video,
    valid_youtube_url,
)

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"
JOBS_DIR = ROOT / "workspace" / "jobs"
HIGHLIGHTS_DIR = ROOT / "workspace" / "highlights"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MAX_FILES = 30
CHUNK_SIZE = 1024 * 1024
BUSY_STATUSES = {"queued", "processing", "analyzing", "rendering"}
MAX_RENDER_ATTEMPTS = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_url(url: str, path: Path) -> str:
    try:
        version = artifact_version(path)
    except OSError:
        version = BUILD_ID
    return versioned_url(url, version)


def safe_display_name(name: str) -> str:
    name = Path(name.replace("\\", "/")).name.strip()
    name = re.sub(r"[\x00-\x1f]", "", name)
    return name[:180] or "video.mp4"


def atomic_write_json(path: Path, payload: dict, attempts: int = 6) -> None:
    """Persist state despite brief Windows antivirus/indexer file locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:10]}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(max(1, attempts)):
            try:
                os.replace(temp, path)
                return
            except OSError as exc:
                retryable = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32, 33}
                if not retryable or attempt + 1 >= max(1, attempts):
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


class JobManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reel-worker")
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        for state_path in JOBS_DIR.glob("*/job.json"):
            try:
                job = json.loads(state_path.read_text(encoding="utf-8"))
                job["paused"] = False
                if job.get("status") in {"queued", "processing"}:
                    job["status"] = "interrupted"
                    job["message"] = "Обработка была прервана. Загрузите файлы повторно."
                    for item in job.get("files", []):
                        if item.get("status") in {"queued", "processing"}:
                            item["status"] = "interrupted"
                self.jobs[job["id"]] = job
                self._persist(job)
            except Exception:
                continue

    def _persist(self, job: dict) -> None:
        job_dir = JOBS_DIR / job["id"]
        atomic_write_json(job_dir / "job.json", job)

    def create(self, fields: list, options: dict) -> dict:
        if not fields:
            raise ValueError("Добавьте хотя бы одно видео.")
        if len(fields) > MAX_FILES:
            raise ValueError(f"За один раз можно загрузить не более {MAX_FILES} видео.")
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOBS_DIR / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        work_dir = job_dir / "work"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()
        work_dir.mkdir()
        files = []
        try:
            for index, field in enumerate(fields, 1):
                display_name = safe_display_name(field.filename or f"video_{index}.mp4")
                extension = Path(display_name).suffix.lower()
                if extension not in ALLOWED_EXTENSIONS:
                    raise ValueError(f"Неподдерживаемый формат: {display_name}")
                file_id = f"v{index:03d}"
                stored_path = input_dir / f"{file_id}{extension}"
                size = 0
                with stored_path.open("wb") as destination:
                    while True:
                        chunk = field.file.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        size += len(chunk)
                        destination.write(chunk)
                if size == 0:
                    raise ValueError(f"Файл {display_name} пуст.")
                files.append({
                    "id": file_id,
                    "name": display_name,
                    "size": size,
                    "input": str(stored_path.relative_to(job_dir)),
                    "output": str((output_dir / f"{file_id}.mp4").relative_to(job_dir)),
                    "status": "queued",
                    "progress": 0,
                    "message": "В очереди",
                    "error": None,
                    "result": None,
                })
            normalized_options = {
                "language": options.get("language", "ru") if options.get("language") in {"ru", "en", "auto"} else "ru",
                "model": options.get("model", "base") if options.get("model") in {"base", "small"} else "base",
                "subtitle_style": options.get("subtitle_style", "pill")
                if options.get("subtitle_style") in {"pill", "clean", "yellow"} else "pill",
            }
            job = {
                "id": job_id,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "status": "queued",
                "progress": 0,
                "message": "Задание добавлено в очередь",
                "paused": False,
                "options": normalized_options,
                "files": files,
            }
            with self.lock:
                self.jobs[job_id] = job
                self._persist(job)
            self.executor.submit(self._process, job_id)
            return self.public(job)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def _process(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job["status"] = "processing"
            job["message"] = "Обработка началась"
            job["updated_at"] = utc_now()
            self._persist(job)
        self._pause_gate(job)
        total = len(job["files"])
        for index, item in enumerate(job["files"]):
            self._pause_gate(job, item)
            with self.lock:
                item["status"] = "processing"
                item["message"] = "Подготовка"
                item["render_attempt"] = 1
                item["max_render_attempts"] = MAX_RENDER_ATTEMPTS
                item["retry_errors"] = []
                self._persist(job)

            last_persist = [0.0]

            def update(progress: int, message: str) -> None:
                self._pause_gate(job, item)
                with self.lock:
                    item["progress"] = max(0, min(100, int(progress)))
                    item["message"] = message
                    job["progress"] = int(((index + item["progress"] / 100) / total) * 100)
                    job["message"] = f"{index + 1} из {total}: {message}"
                    job["updated_at"] = utc_now()
                    now = time.monotonic()
                    if now - last_persist[0] >= 0.7 or progress >= 100:
                        self._persist(job)
                        last_persist[0] = now

            job_dir = JOBS_DIR / job_id
            for attempt in range(1, MAX_RENDER_ATTEMPTS + 1):
                with self.lock:
                    item["render_attempt"] = attempt
                    if attempt > 1:
                        item["progress"] = 0
                        item["message"] = f"Повтор рендера {attempt - 1} из {MAX_RENDER_ATTEMPTS - 1}"
                    self._persist(job)
                try:
                    result = process_video(
                        job_dir / item["input"],
                        job_dir / item["output"],
                        job_dir / "work" / item["id"],
                        job["options"],
                        update,
                    )
                    self._pause_gate(job, item)
                    with self.lock:
                        item["status"] = "completed"
                        item["progress"] = 100
                        item["message"] = "Готово"
                        item["error"] = None
                        item["result"] = result
                    break
                except Exception as exc:
                    error = str(exc) if isinstance(exc, PipelineError) else f"{type(exc).__name__}: {exc}"
                    error_path = job_dir / "work" / item["id"] / f"error.attempt-{attempt}.log"
                    error_path.parent.mkdir(parents=True, exist_ok=True)
                    error_path.write_text(traceback.format_exc(), encoding="utf-8")
                    with self.lock:
                        item["retry_errors"].append({"attempt": attempt, "error": error})
                        if attempt < MAX_RENDER_ATTEMPTS:
                            item["message"] = f"Ошибка. Автоповтор {attempt} из {MAX_RENDER_ATTEMPTS - 1}"
                            item["error"] = error
                        else:
                            item["status"] = "failed"
                            item["progress"] = 100
                            item["message"] = "Ошибка после 2 автоповторов"
                            item["error"] = error
                        self._persist(job)
            with self.lock:
                job["progress"] = int(((index + 1) / total) * 100)
                self._persist(job)

        with self.lock:
            completed = sum(item["status"] == "completed" for item in job["files"])
            failed = total - completed
            job["status"] = "completed" if failed == 0 else ("partial" if completed else "failed")
            job["progress"] = 100
            job["message"] = f"Готово: {completed}. Ошибок: {failed}."
            job["updated_at"] = utc_now()
            self._persist(job)

    def _pause_gate(self, job: dict, item: dict | None = None) -> None:
        while True:
            with self.lock:
                if not job.get("paused", False):
                    return
                if job.get("message") != "Обработка на паузе":
                    job["message"] = "Обработка на паузе"
                    if item is not None and item.get("status") in {"queued", "processing"}:
                        item["message"] = "На паузе"
                    job["updated_at"] = utc_now()
                    self._persist(job)
            time.sleep(0.25)

    def set_paused(self, job_id: str, paused: bool) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise ValueError("Задание не найдено.")
            if job["status"] not in {"queued", "processing"}:
                raise ValueError("Это задание уже нельзя поставить на паузу.")
            job["paused"] = bool(paused)
            job["message"] = "Обработка на паузе" if paused else "Продолжаю обработку"
            if paused:
                for item in job.get("files", []):
                    if item.get("status") in {"queued", "processing"}:
                        item["message"] = "На паузе"
            job["updated_at"] = utc_now()
            self._persist(job)
            return self.public(job)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            return self.jobs.get(job_id)

    def is_busy(self) -> bool:
        with self.lock:
            return any(job.get("status") in BUSY_STATUSES for job in self.jobs.values())

    def public(self, job: dict) -> dict:
        data = json.loads(json.dumps(job, ensure_ascii=False))
        job_dir = JOBS_DIR / job["id"]
        completed_paths = []
        for source_item, item in zip(job["files"], data["files"]):
            item.pop("input", None)
            item.pop("output", None)
            if item["status"] == "completed":
                output_path = job_dir / source_item["output"]
                completed_paths.append(output_path)
                item["download_url"] = artifact_url(
                    f"/api/jobs/{job['id']}/files/{item['id']}", output_path,
                )
        if completed_paths:
            data["zip_url"] = versioned_url(
                f"/api/jobs/{job['id']}/download.zip",
                artifact_set_version(completed_paths),
            )
        return data

    def list_public(self) -> list[dict]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda job: job["created_at"], reverse=True)[:20]
            return [self.public(job) for job in jobs]


MANAGER = JobManager()


class HighlightsManager:
    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.lock = threading.RLock()
        self.projects: dict[str, dict] = {}
        self.executor = executor
        HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self):
        for state_path in HIGHLIGHTS_DIR.glob("*/project.json"):
            try:
                project = json.loads(state_path.read_text(encoding="utf-8"))
                options = project.setdefault("options", {})
                project_mode = project.get("project_mode") or options.get("project_mode") or "podcast"
                project_mode = project_mode if project_mode in {"podcast", "broadcast"} else "podcast"
                project["project_mode"] = project_mode
                options["project_mode"] = project_mode
                project["paused"] = False
                project.setdefault("phase", "idle")
                project.setdefault("phase_progress", 0)
                project.setdefault("processed_seconds", None)
                project.setdefault("total_seconds", project.get("source", {}).get("duration"))
                project.setdefault("eta_seconds", None)
                project.setdefault("error_stage", None)
                if (
                    project.get("status") == "failed"
                    and project.get("transcript")
                    and not project.get("candidates")
                    and not project.get("error_stage")
                ):
                    project["error_stage"] = "analysis"
                if project.get("status") in {"analyzing", "rendering", "queued"}:
                    project["status"] = "interrupted"
                    if project.get("transcript"):
                        project["error_stage"] = "analysis"
                        project["message"] = "Работа была прервана. Можно повторить поиск моментов."
                    else:
                        project["message"] = "Работа была прервана. Создайте проект повторно."
                self.projects[project["id"]] = project
                self._persist(project)
            except Exception:
                continue

    def _persist(self, project: dict):
        project_dir = HIGHLIGHTS_DIR / project["id"]
        atomic_write_json(project_dir / "project.json", project)

    @staticmethod
    def _options(raw: dict) -> dict:
        def to_int(key: str, default: int) -> int:
            try:
                return int(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        def to_bool(key: str, default: bool) -> bool:
            value = raw.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() not in {"0", "false", "no", "off", ""}
            return bool(value)

        minimum = max(15, min(120, to_int("min_duration", 30)))
        maximum = max(minimum + 5, min(180, to_int("max_duration", 90)))
        max_clips = max(0, min(20, to_int("max_clips", 0)))   # 0 = все ролики (по умолчанию)
        try:
            playback_speed = round(float(raw.get("playback_speed", 1.0)), 2)
        except (TypeError, ValueError):
            playback_speed = 1.0
        if playback_speed not in {1.0, 1.05, 1.1, 1.15, 1.2, 1.25}:
            playback_speed = 1.0
        try:
            speaker_count = int(raw.get("speaker_count", 2))
        except (TypeError, ValueError):
            speaker_count = 2
        if speaker_count not in {1, 2}:
            speaker_count = 2
        visual_preset = str(raw.get("visual_preset", "editorial_lime"))
        if visual_preset not in {"editorial_lime", "itbooster"}:
            visual_preset = "editorial_lime"
        return {
            "project_mode": raw.get("project_mode", "podcast")
            if raw.get("project_mode") in {"podcast", "broadcast"} else "podcast",
            "language": raw.get("language", "ru") if raw.get("language") in {"ru", "en", "auto"} else "ru",
            "subtitle_language": raw.get("subtitle_language", "ru") if raw.get("subtitle_language") in {"ru", "en", "auto"} else "ru",
            "model": raw.get("model", "base") if raw.get("model") in {"base", "small"} else "base",
            "provider": raw.get("provider", "auto") if raw.get("provider") in {"auto", "claude", "codex", "heuristic"} else "auto",
            "min_duration": minimum,
            "max_duration": maximum,
            "max_clips": max_clips,
            "all_clips": to_bool("all_clips", max_clips == 0) or max_clips == 0,
            # контекст «что интересно аудитории» для компоновки и главреда (не фильтр); пустая строка отключает
            "audience": (str(raw["audience"]).strip()[:600] if raw.get("audience") is not None
                         else editorial.DEFAULT_AUDIENCE) if "audience" in raw else editorial.DEFAULT_AUDIENCE,
            # Whisper сохраняет «э-э», «ну», повторы (initial_prompt); нужны монтажу, чтобы вырезать их осознанно
            "keep_disfluencies": to_bool("keep_disfluencies", True),
            # смысловая чистка: слова-паразиты, ложные старты, повторы, длинные паузы
            "tighten": to_bool("tighten", True),
            # шеф-редактор отсматривает готовые ролики (кадры + звук + ИИ) и при необходимости правит один раз
            "video_review": to_bool("video_review", True),
            # термины для подсказки Whisper; None -> IT-словарь по умолчанию для русского
            "asr_glossary": normalize_glossary(raw.get("asr_glossary"), "ru") if "asr_glossary" in raw
            and raw.get("asr_glossary") is not None else normalize_glossary(None, "ru"),
            # движок распознавания: deepgram (облако) | local (faster-whisper) | auto (deepgram, если есть ключ)
            "asr_provider": raw.get("asr_provider", "auto") if raw.get("asr_provider") in {"auto", "deepgram", "local"} else "auto",
            "subtitle_style": raw.get("subtitle_style", "pill") if raw.get("subtitle_style") in {"pill", "clean", "yellow"} else "pill",
            "playback_speed": playback_speed,
            "speaker_count": speaker_count,
            "visual_preset": visual_preset,
            # звук склейки: raw (по умолчанию, без выравнивания громкости) | leveled (v4: gain + loudnorm); скрытая опция
            "audio_mode": "leveled" if str(raw.get("audio_mode", "raw")).strip().lower() == "leveled" else "raw",
        }

    def create(self, source_kind: str, field, youtube_url: str, raw_options: dict) -> dict:
        normalized_options = self._options(raw_options)
        project_mode = normalized_options["project_mode"]
        if source_kind not in {"upload", "youtube"}:
            raise ValueError("Выберите загрузку видео или ссылку YouTube.")
        if project_mode == "broadcast" and source_kind != "upload":
            raise ValueError("Для режима «Эфир» загрузите исходную запись Zoom с компьютера.")
        if source_kind == "upload" and (field is None or not getattr(field, "filename", None)):
            raise ValueError("Добавьте длинное видео.")
        if source_kind == "youtube" and not valid_youtube_url(youtube_url):
            raise ValueError("Укажите корректную ссылку YouTube.")
        project_id = "h" + uuid.uuid4().hex[:11]
        project_dir = HIGHLIGHTS_DIR / project_id
        input_dir = project_dir / "input"
        output_dir = project_dir / "output"
        work_dir = project_dir / "work"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()
        work_dir.mkdir()
        try:
            input_path = None
            source_name = "YouTube"
            if source_kind == "upload":
                source_name = safe_display_name(field.filename)
                extension = Path(source_name).suffix.lower()
                if extension not in ALLOWED_EXTENSIONS:
                    raise ValueError(f"Неподдерживаемый формат: {source_name}")
                stored_path = input_dir / f"source{extension}"
                with stored_path.open("wb") as destination:
                    while True:
                        chunk = field.file.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        destination.write(chunk)
                if not stored_path.stat().st_size:
                    raise ValueError("Загруженный файл пуст.")
                input_path = str(stored_path.relative_to(project_dir))
            project = {
                "id": project_id,
                "created_at": utc_now(), "updated_at": utc_now(),
                "status": "queued", "progress": 0,
                "message": "Проект добавлен в очередь",
                "phase": "queued", "phase_progress": 0,
                "processed_seconds": None, "total_seconds": None, "eta_seconds": None,
                "paused": False,
                "project_mode": project_mode,
                "source_kind": source_kind,
                "youtube_url": youtube_url.strip() if source_kind == "youtube" else "",
                "input": input_path,
                "transcript": None,
                "source": {"name": source_name, "title": source_name, "duration": None, "uploader": ""},
                "options": normalized_options,
                "provider_used": None, "warning": None, "error": None, "error_stage": None,
                "candidates": [],
            }
            with self.lock:
                self.projects[project_id] = project
                self._persist(project)
            self.executor.submit(self._analyze, project_id)
            return self.public(project)
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise

    def _analyze(self, project_id: str, reuse_transcript: bool = False):
        with self.lock:
            project = self.projects[project_id]
            project["status"] = "analyzing"
            project["phase"] = "analysis" if reuse_transcript else "preparation"
            project["phase_progress"] = 0
            project["eta_seconds"] = None
            project["error"] = None
            project["error_stage"] = "analysis" if reuse_transcript else "preparation"
            project["message"] = (
                "Повторно ищу сильные моменты"
                if reuse_transcript else (
                    "Подготавливаю запись эфира"
                    if project.get("project_mode") == "broadcast"
                    else "Подготавливаю длинное видео"
                )
            )
            self._persist(project)
        self._pause_gate(project)
        project_dir = HIGHLIGHTS_DIR / project_id
        last_persist = [0.0]
        current_stage = ["analysis" if reuse_transcript else "preparation"]

        def persist_throttled(force: bool = False):
            now = time.monotonic()
            if force or now - last_persist[0] >= 0.7:
                self._persist(project)
                last_persist[0] = now

        def update(progress: int, message: str):
            self._pause_gate(project)
            with self.lock:
                normalized = min(99, max(0, int(progress)))
                project["progress"] = max(project.get("progress", 0), normalized)
                project["message"] = message
                if normalized < 58:
                    current_stage[0] = "transcription"
                    project["phase"] = "transcription"
                    project["phase_progress"] = max(
                        project.get("phase_progress", 0),
                        min(100, max(0, int((normalized - 6) / 30 * 100))),
                    )
                elif normalized < 94:
                    current_stage[0] = "analysis"
                    project["phase"] = "analysis"
                    project["phase_progress"] = min(99, max(0, int((normalized - 72) / 22 * 100)))
                else:
                    current_stage[0] = "thumbnails"
                    project["phase"] = "thumbnails"
                    project["phase_progress"] = min(99, max(0, int((normalized - 94) / 5 * 100)))
                project["error_stage"] = current_stage[0]
                project["updated_at"] = utc_now()
                persist_throttled()

        def update_telemetry(details: dict):
            self._pause_gate(project)
            with self.lock:
                if details.get("event") == "transcription_progress":
                    current_stage[0] = "transcription"
                    project["phase"] = "transcription"
                    project["phase_progress"] = int(details.get("percent") or 0)
                    processed = float(details.get("processed_seconds") or 0)
                    project["eta_confidence"] = (
                        "high" if processed >= 1200 else ("medium" if processed >= 300 else "low")
                    )
                for key in (
                    "phase", "phase_progress", "processed_seconds", "total_seconds",
                    "elapsed_seconds", "eta_seconds", "eta_confidence",
                ):
                    if key in details:
                        project[key] = details[key]
                runtime = details.get("runtime")
                if runtime:
                    project["transcription_runtime"] = runtime
                if details.get("overall_progress") is not None:
                    project["progress"] = max(
                        project.get("progress", 0),
                        min(99, int(details["overall_progress"])),
                    )
                project["updated_at"] = utc_now()
                persist_throttled()

        try:
            transcript = None
            self._pause_gate(project)
            if reuse_transcript:
                transcript_path = project_dir / str(project.get("transcript") or "")
                video_path = project_dir / str(project.get("input") or "")
                if not transcript_path.is_file() or not video_path.is_file():
                    raise PipelineError("Сохранённый транскрипт или исходное видео не найдены.")
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
                update(58, "Использую готовый транскрипт")
            elif project["source_kind"] == "youtube":
                video_path, metadata, transcript = download_youtube(
                    project["youtube_url"], project_dir / "input", project["options"]["subtitle_language"], update,
                )
                project["input"] = str(video_path.relative_to(project_dir))
                project["source"].update(metadata)
                # Субтитры YouTube не используются: их пословные таймкоды синтетические
                # (слово режется по длине), а смысловому монтажу нужны реальные границы слов
                # для склейки кусков и вырезания паразитов. Всегда распознаём локально.
                transcript = None
            else:
                video_path = project_dir / project["input"]

            media = media_info(video_path)
            project["source"]["duration"] = media["duration"]
            project["total_seconds"] = media["duration"]
            if transcript is None:
                current_stage[0] = "transcription"
                project["error_stage"] = "transcription"
                transcript = transcribe_video(
                    video_path, project_dir / "work" / "transcription",
                    project["options"]["language"], project["options"]["model"], update,
                    mode="accurate",  # слова нужны смысловому монтажу в обоих режимах
                    telemetry_callback=update_telemetry,
                    keep_disfluencies=bool(project["options"].get("keep_disfluencies", True)),
                    glossary=project["options"].get("asr_glossary"),
                    asr_provider=project["options"].get("asr_provider", "auto"),
                )
                update(58, "Транскрипция готова")

            transcript_path = project_dir / "work" / "transcript.json"
            transcript_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.lock:
                project["transcript"] = str(transcript_path.relative_to(project_dir))
                project["phase"] = "analysis"
                project["phase_progress"] = 0
                project["processed_seconds"] = media["duration"]
                project["eta_seconds"] = None
                project["error_stage"] = "analysis"
                current_stage[0] = "analysis"
                persist_throttled(force=True)

            raw_candidates, provider, warning = analyze_transcript(
                transcript, project_dir / "work", project["options"], update,
            )
            self._pause_gate(project)
            candidates = normalize_candidates(
                raw_candidates, media["duration"], project["options"], transcript.get("words") or [],
            )
            if not candidates:
                raise PipelineError("Не удалось найти подходящие по длительности отрывки.")

            current_stage[0] = "thumbnails"
            with self.lock:
                project["error_stage"] = "thumbnails"
                project["phase"] = "thumbnails"
                project["phase_progress"] = 0
            for index, candidate in enumerate(candidates, 1):
                self._pause_gate(project)
                thumbnail = project_dir / "thumbnails" / f"{candidate['id']}.jpg"
                hook_start = float((candidate.get("segments") or [{"start": candidate["start"]}])[0]["start"])
                generate_thumbnail(video_path, hook_start + 0.5, thumbnail)
                with self.lock:
                    project["phase_progress"] = int(index / len(candidates) * 100)
                    project["progress"] = 94 + min(5, int(index / len(candidates) * 5))
                    project["message"] = f"Готовлю превью: {index} из {len(candidates)}"
                    persist_throttled()

            with self.lock:
                project["provider_used"] = provider
                project["warning"] = warning
                project["candidates"] = candidates
                project["status"] = "awaiting_selection"
                project["progress"] = 100
                project["phase"] = "complete"
                project["phase_progress"] = 100
                project["eta_seconds"] = 0
                project["error_stage"] = None
                source_word = "эфира" if project.get("project_mode") == "broadcast" else "выпуска"
                found = len([c for c in candidates if not c.get("rejected")])
                project["message"] = f"Найдено {found} сильных отрывков {source_word}" + (
                    f" (ещё {len(candidates) - found} не прошли проверку)" if found != len(candidates) else "")
                project["updated_at"] = utc_now()
                self._persist(project)
        except Exception as exc:
            with self.lock:
                project["status"] = "failed"
                project["progress"] = 100
                project["phase"] = current_stage[0]
                project["eta_seconds"] = None
                project["message"] = "Ошибка анализа"
                project["error_stage"] = current_stage[0]
                project["error"] = str(exc) if isinstance(exc, PipelineError) else f"{type(exc).__name__}: {exc}"
                project["updated_at"] = utc_now()
                self._persist(project)
            (project_dir / "work" / "error.log").parent.mkdir(parents=True, exist_ok=True)
            (project_dir / "work" / "error.log").write_text(traceback.format_exc(), encoding="utf-8")

    def start_render(self, project_id: str, selected_ids: list[str], subtitle_style: str,
                     playback_speed: float = 1.0, speaker_count: int = 2,
                     visual_preset: str = "editorial_lime", video_review: bool | None = None) -> dict:
        with self.lock:
            project = self.projects.get(project_id)
            if not project:
                raise ValueError("Проект не найден.")
            if project["status"] not in {"awaiting_selection", "completed", "partial", "failed"}:
                raise ValueError("Проект пока не готов к нарезке.")
            selected = [item for item in project["candidates"] if item["id"] in set(selected_ids)]
            if not selected:
                raise ValueError("Выберите хотя бы один отрывок.")
            try:
                playback_speed = round(float(playback_speed), 2)
            except (TypeError, ValueError):
                playback_speed = 1.0
            if playback_speed not in {1.0, 1.05, 1.1, 1.15, 1.2, 1.25}:
                raise ValueError("Скорость должна быть от 1.0× до 1.25× с шагом 0.05.")
            try:
                speaker_count = int(speaker_count)
            except (TypeError, ValueError):
                speaker_count = 2
            if speaker_count not in {1, 2}:
                raise ValueError("Количество отображаемых спикеров должно быть 1 или 2.")
            visual_preset = str(visual_preset or "editorial_lime")
            if visual_preset not in {"editorial_lime", "itbooster"}:
                raise ValueError("Выбран неизвестный визуальный пресет.")
            if project.get("project_mode") == "broadcast":
                status = provider_status()
                if not (status.get("claude", {}).get("ready") or status.get("codex", {}).get("ready")):
                    raise ValueError(
                        "Для motion-роликов нужен авторизованный ИИ-провайдер: Claude CLI "
                        "(`claude auth login`) или Codex CLI (`codex login`). Войдите и повторите рендер."
                    )
            project["options"]["subtitle_style"] = subtitle_style if subtitle_style in {"pill", "clean", "yellow"} else "pill"
            project["options"]["playback_speed"] = playback_speed
            project["options"]["speaker_count"] = speaker_count
            project["options"]["visual_preset"] = visual_preset
            if video_review is not None:
                project["options"]["video_review"] = bool(video_review)
            for item in project["candidates"]:
                item["selected"] = item in selected
                if item in selected:
                    # Hide stale download/result state immediately, but retain the
                    # last good file on disk until its replacement is verified.
                    item.pop("output", None)
                    item.pop("download_url", None)
                    item.pop("artifact_version", None)
                    item.update({
                        "status": "queued", "progress": 0, "message": "В очереди",
                        "error": None, "result": None,
                    })
                    if item.get("rejected"):   # «Всё равно сделать»: кандидат остаётся с причиной отсева, но рендерится как обычный
                        item["rejected_forced"] = True
            project["status"] = "rendering"
            project["paused"] = False
            project["progress"] = 0
            project["phase"] = "rendering"
            project["phase_progress"] = 0
            project["error_stage"] = "rendering"
            project["message"] = f"Готовлю {len(selected)} отрывков"
            project["error"] = None
            project.pop("zip_url", None)
            project["updated_at"] = utc_now()
            self._persist(project)
        self.executor.submit(self._render, project_id, [item["id"] for item in selected])
        return self.public(project)

    def retry_analysis(self, project_id: str) -> dict:
        with self.lock:
            project = self.projects.get(project_id)
            if not project:
                raise ValueError("Проект не найден.")
            project_dir = HIGHLIGHTS_DIR / project_id
            if not can_retry_analysis(project, project_dir):
                raise ValueError("Для этого проекта нет сохранённого транскрипта для повтора.")
            prepare_analysis_retry(project)
            project["updated_at"] = utc_now()
            self._persist(project)
        self.executor.submit(self._analyze, project_id, True)
        return self.public(project)

    def _render(self, project_id: str, selected_ids: list[str]):
        project = self.projects[project_id]
        project_dir = HIGHLIGHTS_DIR / project_id
        source = project_dir / project["input"]
        transcript = json.loads((project_dir / project["transcript"]).read_text(encoding="utf-8"))
        selected = [item for item in project["candidates"] if item["id"] in selected_ids]
        total = len(selected)
        for index, candidate in enumerate(selected):
            self._pause_gate(project, candidate)
            with self.lock:
                candidate["status"] = "processing"
                candidate["message"] = "Вырезаю отрывок"
                candidate["render_attempt"] = 1
                candidate["max_render_attempts"] = MAX_RENDER_ATTEMPTS
                candidate["retry_errors"] = []
                self._persist(project)
            last_persist = [0.0]

            def update(progress: int, message: str):
                self._pause_gate(project, candidate)
                with self.lock:
                    candidate["progress"] = int(progress)
                    candidate["message"] = message
                    project["progress"] = int(((index + progress / 100) / total) * 100)
                    project["message"] = f"{index + 1} из {total}: {message}"
                    now = time.monotonic()
                    if now - last_persist[0] >= 0.7 or progress >= 100:
                        self._persist(project)
                        last_persist[0] = now

            output = project_dir / "output" / f"{candidate['id']}.mp4"
            for attempt in range(1, MAX_RENDER_ATTEMPTS + 1):
                staged_output: Path | None = output.with_name(
                    f".{output.stem}.{uuid.uuid4().hex[:10]}.rendering{output.suffix}"
                )
                with self.lock:
                    candidate["render_attempt"] = attempt
                    if attempt > 1:
                        candidate["progress"] = 0
                        candidate["message"] = f"Повтор рендера {attempt - 1} из {MAX_RENDER_ATTEMPTS - 1}"
                    self._persist(project)
                try:
                    result = render_highlight_reviewed(
                        source, transcript, candidate, staged_output,
                        project_dir / "work" / candidate["id"], project["options"], update,
                        renderer=render_highlight,
                    )
                    self._pause_gate(project, candidate)
                    os.replace(staged_output, output)
                    staged_output = None
                    with self.lock:
                        candidate["status"] = "completed"
                        candidate["progress"] = 100
                        candidate["message"] = "Готово"
                        candidate["error"] = None
                        candidate["output"] = str(output.relative_to(project_dir))
                        update_fields = (result or {}).get("candidate_update")
                        if update_fields:   # авто-правка шеф-редактора поменяла куски ролика
                            candidate.update(update_fields)
                        candidate["result"] = result
                        candidate["artifact_version"] = artifact_version(output)
                    break
                except Exception as exc:
                    if staged_output is not None:
                        try:
                            staged_output.unlink(missing_ok=True)
                        except OSError:
                            pass
                    error = str(exc) if isinstance(exc, PipelineError) else f"{type(exc).__name__}: {exc}"
                    with self.lock:
                        candidate["retry_errors"].append({"attempt": attempt, "error": error})
                        if attempt < MAX_RENDER_ATTEMPTS:
                            candidate["message"] = f"Ошибка. Автоповтор {attempt} из {MAX_RENDER_ATTEMPTS - 1}"
                            candidate["error"] = error
                        else:
                            candidate["status"] = "failed"
                            candidate["progress"] = 100
                            candidate["message"] = "Ошибка после 2 автоповторов"
                            candidate["error"] = error
                        self._persist(project)
            with self.lock:
                project["progress"] = int((index + 1) / total * 100)
                self._persist(project)
        with self.lock:
            completed = sum(item["status"] == "completed" for item in selected)
            any_completed = any(
                item.get("status") == "completed" for item in project.get("candidates", [])
            )
            project["status"] = "completed" if completed == total else ("partial" if any_completed else "failed")
            project["progress"] = 100
            project["phase"] = "complete" if completed else "failed"
            project["phase_progress"] = 100
            project["error_stage"] = None if completed else "rendering"
            project["message"] = f"Готово: {completed}. Ошибок: {total - completed}."
            project["updated_at"] = utc_now()
            self._persist(project)

    def _pause_gate(self, project: dict, candidate: dict | None = None) -> None:
        while True:
            with self.lock:
                if not project.get("paused", False):
                    return
                if project.get("message") != "Процесс на паузе":
                    project["message"] = "Процесс на паузе"
                    if candidate is not None and candidate.get("status") in {"queued", "processing"}:
                        candidate["message"] = "На паузе"
                    project["updated_at"] = utc_now()
                    self._persist(project)
            time.sleep(0.25)

    def set_paused(self, project_id: str, paused: bool) -> dict:
        with self.lock:
            project = self.projects.get(project_id)
            if not project:
                raise ValueError("Проект не найден.")
            if project["status"] not in {"queued", "analyzing", "rendering"}:
                raise ValueError("Этот проект сейчас нельзя поставить на паузу.")
            project["paused"] = bool(paused)
            project["message"] = "Процесс на паузе" if paused else "Продолжаю обработку"
            if paused:
                for candidate in project.get("candidates", []):
                    if candidate.get("status") in {"queued", "processing"}:
                        candidate["message"] = "На паузе"
            project["updated_at"] = utc_now()
            self._persist(project)
            return self.public(project)

    def get(self, project_id: str) -> dict | None:
        with self.lock:
            return self.projects.get(project_id)

    def is_busy(self) -> bool:
        with self.lock:
            return any(
                project.get("status") in BUSY_STATUSES
                for project in self.projects.values()
            )

    def public(self, project: dict) -> dict:
        data = json.loads(json.dumps(project, ensure_ascii=False))
        project_dir = HIGHLIGHTS_DIR / project["id"]
        transcript_path = project.get("transcript")
        data["transcript_saved"] = bool(
            transcript_path and (project_dir / transcript_path).is_file()
        )
        data["can_retry_analysis"] = can_retry_analysis(project, project_dir)
        data.pop("input", None)
        data.pop("transcript", None)
        data.pop("youtube_url", None)
        completed_paths = []
        source_path = project_dir / str(project.get("input") or "")
        source_version = artifact_version(source_path) if source_path.is_file() else None
        for source_candidate, candidate in zip(project["candidates"], data["candidates"]):
            candidate.pop("output", None)
            thumbnail_path = project_dir / "thumbnails" / f"{candidate['id']}.jpg"
            candidate["thumbnail_url"] = artifact_url(
                f"/api/highlights/{project['id']}/thumbnails/{candidate['id']}",
                thumbnail_path,
            )
            if source_version:
                start_ms = round(float(source_candidate["start"]) * 1000)
                end_ms = round(float(source_candidate["end"]) * 1000)
                candidate["source_download_url"] = versioned_url(
                    f"/api/highlights/{project['id']}/source-files/{candidate['id']}",
                    f"{source_version}-{start_ms:x}-{end_ms:x}",
                )
            if candidate["status"] == "completed" and source_candidate.get("output"):
                output_path = project_dir / source_candidate["output"]
                completed_paths.append(output_path)
                candidate["download_url"] = artifact_url(
                    f"/api/highlights/{project['id']}/files/{candidate['id']}",
                    output_path,
                )
        if completed_paths:
            data["zip_url"] = versioned_url(
                f"/api/highlights/{project['id']}/download.zip",
                artifact_set_version(completed_paths),
            )
        return data

    def list_public(self, project_mode: str | None = None) -> list[dict]:
        with self.lock:
            projects = self.projects.values()
            if project_mode in {"podcast", "broadcast"}:
                projects = [
                    item for item in projects
                    if item.get("project_mode", "podcast") == project_mode
                ]
            projects = sorted(projects, key=lambda item: item["created_at"], reverse=True)[:20]
            return [self.public(item) for item in projects]


HIGHLIGHTS = HighlightsManager(MANAGER.executor)


def runtime_busy() -> bool:
    return MANAGER.is_busy() or HIGHLIGHTS.is_busy()


class ReelFactoryHTTPServer(ThreadingHTTPServer):
    """One listener per port; Windows must not share a bound Reel Factory port."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ReelFactory/1.0"

    def log_message(self, format_string: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status=HTTPStatus.BAD_REQUEST):
        self.send_json({"error": message}, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/health":
            return self.send_json({
                "ok": True,
                "service": "Reel Factory",
                "build_id": BUILD_ID,
                "workspace_id": WORKSPACE_ID,
                "pid": os.getpid(),
                "busy": runtime_busy(),
            })
        if path == "/api/jobs":
            return self.send_json({"jobs": MANAGER.list_public()})
        if path in {"/api/highlights/providers", "/api/providers"}:
            return self.send_json(provider_status())
        if path == "/api/highlights":
            query_mode = parse_qs(parsed.query).get("project_mode", [None])[0]
            return self.send_json({"projects": HIGHLIGHTS.list_public(query_mode)})
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})", path)
        if match:
            job = MANAGER.get(match.group(1))
            return self.send_json(MANAGER.public(job)) if job else self.send_error_json("Задание не найдено", HTTPStatus.NOT_FOUND)
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/files/(v\d{3})", path)
        if match:
            return self.send_result_file(match.group(1), match.group(2))
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/download\.zip", path)
        if match:
            return self.send_zip(match.group(1))
        match = re.fullmatch(r"/api/highlights/(h[a-f0-9]{11})", path)
        if match:
            project = HIGHLIGHTS.get(match.group(1))
            return self.send_json(HIGHLIGHTS.public(project)) if project else self.send_error_json("Проект не найден", HTTPStatus.NOT_FOUND)
        match = re.fullmatch(r"/api/highlights/(h[a-f0-9]{11})/files/(c\d{3})", path)
        if match:
            return self.send_highlight_file(match.group(1), match.group(2))
        match = re.fullmatch(r"/api/highlights/(h[a-f0-9]{11})/source-files/(c\d{3})", path)
        if match:
            return self.send_highlight_source_file(match.group(1), match.group(2))
        match = re.fullmatch(r"/api/highlights/(h[a-f0-9]{11})/thumbnails/(c\d{3})", path)
        if match:
            return self.send_highlight_thumbnail(match.group(1), match.group(2))
        match = re.fullmatch(r"/api/highlights/(h[a-f0-9]{11})/download\.zip", path)
        if match:
            return self.send_highlights_zip(match.group(1))
        return self.send_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/shutdown":
            try:
                client_ip = ipaddress.ip_address(self.client_address[0])
                if not client_ip.is_loopback or self.headers.get("X-Reel-Factory-Launcher") != "1":
                    return self.send_error_json("Остановка разрешена только локальному launcher.", HTTPStatus.FORBIDDEN)
                if runtime_busy():
                    return self.send_error_json(
                        "Reel Factory занят обработкой. Дождитесь завершения перед обновлением.",
                        HTTPStatus.CONFLICT,
                    )
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError("Некорректный запрос остановки.")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if (
                    payload.get("pid") != os.getpid()
                    or payload.get("workspace_id") != WORKSPACE_ID
                    or payload.get("build_id") != BUILD_ID
                ):
                    return self.send_error_json(
                        "Экземпляр Reel Factory не совпадает.", HTTPStatus.CONFLICT,
                    )
                # Re-check immediately before shutdown: a task may have been
                # accepted between the launcher's health probe and this request.
                if runtime_busy():
                    return self.send_error_json(
                        "Reel Factory занят обработкой. Дождитесь завершения перед обновлением.",
                        HTTPStatus.CONFLICT,
                    )
                self.send_json({"ok": True, "stopping_pid": os.getpid()}, HTTPStatus.ACCEPTED)
                threading.Thread(
                    target=self.server.shutdown,
                    name="reel-factory-shutdown",
                    daemon=True,
                ).start()
                return None
            except (ValueError, json.JSONDecodeError):
                return self.send_error_json("Некорректный запрос остановки.")
        job_control = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/(pause|resume)", path)
        if job_control:
            try:
                job = MANAGER.set_paused(job_control.group(1), job_control.group(2) == "pause")
                return self.send_json(job, HTTPStatus.OK)
            except ValueError as exc:
                return self.send_error_json(str(exc))
        highlight_control = re.fullmatch(r"/api/highlights/(h[a-f0-9]{11})/(pause|resume)", path)
        if highlight_control:
            try:
                project = HIGHLIGHTS.set_paused(
                    highlight_control.group(1), highlight_control.group(2) == "pause"
                )
                return self.send_json(project, HTTPStatus.OK)
            except ValueError as exc:
                return self.send_error_json(str(exc))
        retry_match = re.fullmatch(r"/api/highlights/(h[a-f0-9]{11})/retry-analysis", path)
        if retry_match:
            try:
                project = HIGHLIGHTS.retry_analysis(retry_match.group(1))
                return self.send_json(project, HTTPStatus.ACCEPTED)
            except ValueError as exc:
                return self.send_error_json(str(exc))
            except Exception as exc:
                traceback.print_exc()
                return self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)
        render_match = re.fullmatch(r"/api/highlights/(h[a-f0-9]{11})/render", path)
        if render_match:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                project = HIGHLIGHTS.start_render(
                    render_match.group(1), payload.get("selected_ids") or [],
                    payload.get("subtitle_style", "pill"), payload.get("playback_speed", 1.0),
                    payload.get("speaker_count", 2), payload.get("visual_preset", "editorial_lime"),
                    payload.get("video_review"),
                )
                return self.send_json(project, HTTPStatus.ACCEPTED)
            except (ValueError, json.JSONDecodeError) as exc:
                return self.send_error_json(str(exc))
            except Exception as exc:
                traceback.print_exc()
                return self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)
        if path not in {"/api/jobs", "/api/highlights"}:
            return self.send_error_json("Маршрут не найден", HTTPStatus.NOT_FOUND)
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            return self.send_error_json("Ожидалась форма с видеофайлами")
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
                keep_blank_values=True,
            )
            options_raw = form.getfirst("options", "{}")
            options = json.loads(options_raw)
            if path == "/api/jobs":
                file_field = form["files"] if "files" in form else []
                fields = file_field if isinstance(file_field, list) else [file_field]
                fields = [field for field in fields if getattr(field, "filename", None)]
                job = MANAGER.create(fields, options)
                return self.send_json(job, HTTPStatus.CREATED)
            source_kind = form.getfirst("source_kind", "upload")
            youtube_url = form.getfirst("youtube_url", "")
            if "project_mode" not in options:
                options["project_mode"] = form.getfirst("project_mode", "podcast")
            field = form["file"] if "file" in form and getattr(form["file"], "filename", None) else None
            project = HIGHLIGHTS.create(source_kind, field, youtube_url, options)
            return self.send_json(project, HTTPStatus.CREATED)
        except (ValueError, json.JSONDecodeError) as exc:
            return self.send_error_json(str(exc))
        except Exception as exc:
            traceback.print_exc()
            return self.send_error_json(f"Не удалось принять файлы: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_result_file(self, job_id: str, file_id: str):
        job = MANAGER.get(job_id)
        if not job:
            return self.send_error_json("Задание не найдено", HTTPStatus.NOT_FOUND)
        item = next((item for item in job["files"] if item["id"] == file_id), None)
        if not item or item["status"] != "completed":
            return self.send_error_json("Файл ещё не готов", HTTPStatus.NOT_FOUND)
        path = JOBS_DIR / job_id / item["output"]
        stem = Path(item["name"]).stem
        return self.send_file(path, f"Reels_{stem}.mp4", "video/mp4")

    def send_zip(self, job_id: str):
        job = MANAGER.get(job_id)
        if not job:
            return self.send_error_json("Задание не найдено", HTTPStatus.NOT_FOUND)
        completed = [item for item in job["files"] if item["status"] == "completed"]
        if not completed:
            return self.send_error_json("Готовых файлов пока нет", HTTPStatus.NOT_FOUND)
        archive = JOBS_DIR / job_id / "Reels_batch.zip"
        if not archive.exists() or any((JOBS_DIR / job_id / item["output"]).stat().st_mtime > archive.stat().st_mtime for item in completed):
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                for item in completed:
                    bundle.write(JOBS_DIR / job_id / item["output"], arcname=f"Reels_{Path(item['name']).stem}.mp4")
        return self.send_file(archive, "Reels_batch.zip", "application/zip")

    def send_highlight_file(self, project_id: str, clip_id: str):
        project = HIGHLIGHTS.get(project_id)
        if not project:
            return self.send_error_json("Проект не найден", HTTPStatus.NOT_FOUND)
        candidate = next((item for item in project["candidates"] if item["id"] == clip_id), None)
        if not candidate or candidate["status"] != "completed" or not candidate.get("output"):
            return self.send_error_json("Отрывок ещё не готов", HTTPStatus.NOT_FOUND)
        path = HIGHLIGHTS_DIR / project_id / candidate["output"]
        prefix = "Ether" if project.get("project_mode") == "broadcast" else "Short"
        return self.send_file(path, f"{prefix}_{clip_id}_{candidate['title']}.mp4", "video/mp4")

    def send_highlight_source_file(self, project_id: str, clip_id: str):
        project = HIGHLIGHTS.get(project_id)
        if not project:
            return self.send_error_json("Проект не найден", HTTPStatus.NOT_FOUND)
        candidate = next((item for item in project["candidates"] if item["id"] == clip_id), None)
        source = HIGHLIGHTS_DIR / project_id / str(project.get("input") or "")
        if not candidate or not source.is_file():
            return self.send_error_json("Исходный фрагмент не найден", HTTPStatus.NOT_FOUND)

        start = float(candidate["start"])
        end = float(candidate["end"])
        extension = source.suffix.lower() if source.suffix.lower() in ALLOWED_EXTENSIONS else ".mp4"
        segments = candidate_segments(candidate)
        multi = len(segments) > 1
        if multi:
            extension = ".mp4"  # склейка из нескольких кусков всегда перекодируется
        source_version = artifact_version(source)
        start_ms, end_ms = round(start * 1000), round(end * 1000)
        output = HIGHLIGHTS_DIR / project_id / "source_clips" / (
            f"{clip_id}_{start_ms}_{end_ms}_{len(segments)}_{source_version}{extension}"
        )
        if not output.is_file():
            staged = output.with_name(f".{output.stem}.{uuid.uuid4().hex[:10]}.exporting{extension}")
            try:
                if multi:
                    # без вырезов и чистки: только куски в порядке воспроизведения
                    spans = [(float(seg["start"]), float(seg["end"])) for seg in segments]
                    assembly.assemble_clip(FFMPEG, source, spans, staged)
                else:
                    extract_source_clip(source, staged, start, end)
                os.replace(staged, output)
            except Exception as exc:
                staged.unlink(missing_ok=True)
                message = str(exc) if isinstance(exc, PipelineError) else f"{type(exc).__name__}: {exc}"
                return self.send_error_json(message, HTTPStatus.INTERNAL_SERVER_ERROR)

        safe_title = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", candidate["title"]).strip("_")[:60]
        download_name = f"Source_{clip_id}_{safe_title or 'clip'}{extension}"
        content_type = mimetypes.guess_type(download_name)[0] or "application/octet-stream"
        return self.send_file(output, download_name, content_type)

    def send_highlight_thumbnail(self, project_id: str, clip_id: str):
        project = HIGHLIGHTS.get(project_id)
        if not project or not any(item["id"] == clip_id for item in project["candidates"]):
            return self.send_error_json("Превью не найдено", HTTPStatus.NOT_FOUND)
        path = HIGHLIGHTS_DIR / project_id / "thumbnails" / f"{clip_id}.jpg"
        return self.send_file(path, f"{clip_id}.jpg", "image/jpeg", attachment=False)

    def send_highlights_zip(self, project_id: str):
        project = HIGHLIGHTS.get(project_id)
        if not project:
            return self.send_error_json("Проект не найден", HTTPStatus.NOT_FOUND)
        completed = [item for item in project["candidates"] if item["status"] == "completed" and item.get("output")]
        if not completed:
            return self.send_error_json("Готовых отрывков пока нет", HTTPStatus.NOT_FOUND)
        archive_name = "Ether_reels.zip" if project.get("project_mode") == "broadcast" else "Vertical_shorts.zip"
        archive = HIGHLIGHTS_DIR / project_id / archive_name
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for item in completed:
                safe_title = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", item["title"]).strip("_")[:60]
                bundle.write(
                    HIGHLIGHTS_DIR / project_id / item["output"],
                    arcname=f"{item['id']}_{safe_title or 'short'}.mp4",
                )
        return self.send_file(archive, archive_name, "application/zip")

    def send_file(self, path: Path, download_name: str, content_type: str, attachment: bool = True):
        if not path.is_file():
            return self.send_error_json("Файл не найден", HTTPStatus.NOT_FOUND)
        stat = path.stat()
        size = stat.st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                start = int(match.group(1) or 0)
                end = min(int(match.group(2) or size - 1), size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Last-Modified", formatdate(stat.st_mtime, usegmt=True))
        self.send_header("ETag", f'"{artifact_version(path)}"')
        safe_ascii = re.sub(r"[^A-Za-z0-9._-]", "_", download_name)
        self.send_header("Content-Disposition", f'{"attachment" if attachment else "inline"}; filename="{safe_ascii}"')
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                chunk = source.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_static(self, request_path: str):
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
            return self.send_error_json("Недопустимый путь", HTTPStatus.FORBIDDEN)
        if not candidate.is_file():
            candidate = STATIC_DIR / "index.html"
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local batch converter for podcast videos")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server = ReelFactoryHTTPServer((args.host, args.port), AppHandler)
    except OSError as exc:
        print(
            f"Reel Factory не запущен: порт {args.host}:{args.port} уже занят ({exc}).",
            flush=True,
        )
        return 1
    url = f"http://{args.host}:{server.server_port}"
    print(f"Reel Factory: {url}", flush=True)
    print("Для остановки нажмите Ctrl+C", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
