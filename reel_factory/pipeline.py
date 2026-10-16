from __future__ import annotations

import gc
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".tools" / "opencv4"))
sys.path.insert(1, str(ROOT / ".tools" / "python"))

# PyPI's NVIDIA wheels keep DLLs outside Python's default Windows search path.
# Handles must remain alive for as long as CTranslate2 can run inference.
_NVIDIA_DLL_HANDLES: list[object] = []
if os.name == "nt":
    nvidia_candidates = [
        ROOT / ".tools" / "python" / "nvidia" / "cublas" / "bin",
        ROOT / ".tools" / "python" / "nvidia" / "cudnn" / "bin",
    ]
    nvidia_bin_dirs: list[Path] = []
    for nvidia_candidate in nvidia_candidates:
        try:
            if nvidia_candidate.is_dir():
                nvidia_bin_dirs.append(nvidia_candidate)
        except OSError:
            continue
    for nvidia_bin_dir in nvidia_bin_dirs:
        try:
            _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(str(nvidia_bin_dir)))
        except (AttributeError, FileNotFoundError, OSError):
            pass
    existing_path = os.environ.get("PATH", "")
    new_paths = [str(path) for path in nvidia_bin_dirs]
    if new_paths:
        os.environ["PATH"] = os.pathsep.join(new_paths + [existing_path])

import cv2
import numpy as np
from faster_whisper import WhisperModel

from . import runtime_paths

ProgressCallback = Callable[[int, str], None]
TelemetryCallback = Callable[[dict], None]
FFMPEG = runtime_paths.ffmpeg_path()


class PipelineError(RuntimeError):
    pass


class ModelManager:
    """Keeps a single Whisper model in memory; jobs are processed sequentially."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_name: str | None = None
        self._model: WhisperModel | None = None
        self._runtime: dict = {
            "model": None,
            "device": None,
            "compute_type": None,
            "cpu_threads": None,
            "num_workers": 1,
            "gpu_probe": "not_run",
            "fallback_reason": None,
            "attempts": [],
        }
        self._cuda_disabled_reason: str | None = None

    @staticmethod
    def _cpu_threads() -> int:
        configured = os.environ.get("REEL_WHISPER_CPU_THREADS", "").strip()
        if configured:
            try:
                return max(1, min(32, int(configured)))
            except ValueError:
                pass
        return max(1, min(8, os.cpu_count() or 8))

    @staticmethod
    def _cuda_available() -> bool:
        """Cheap check so CPU-only machines (Apple Silicon, no NVIDIA) never probe CUDA."""
        if os.environ.get("REEL_WHISPER_DEVICE", "").strip().lower() == "cpu":
            return False
        try:
            import ctranslate2
            return int(ctranslate2.get_cuda_device_count()) > 0
        except Exception:
            return False

    @staticmethod
    def _short_error(error: BaseException) -> str:
        message = f"{type(error).__name__}: {error}".strip()
        return message[:700]

    @staticmethod
    def _probe_gpu(model: WhisperModel) -> None:
        """Force one encoder call: CUDA model construction alone is not a useful probe."""
        silence = np.zeros(16000, dtype=np.float32)
        segments, _ = model.transcribe(
            silence,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            word_timestamps=False,
            vad_filter=False,
        )
        # faster-whisper defers inference until the segment generator is consumed.
        list(segments)

    def _clear_model(self) -> None:
        self._model = None
        self._model_name = None
        gc.collect()

    def _load_cpu(self, model_name: str, fallback_reason: str | None,
                  attempts: list[dict] | None = None) -> WhisperModel:
        threads = self._cpu_threads()
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            num_workers=1,
        )
        self._model = model
        self._model_name = model_name
        self._runtime = {
            "model": model_name,
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": threads,
            "num_workers": 1,
            "gpu_probe": "failed" if fallback_reason else "skipped",
            "fallback_reason": fallback_reason,
            "attempts": list(attempts or []),
        }
        return model

    def get(self, model_name: str) -> WhisperModel:
        with self._lock:
            if self._model is not None and self._model_name == model_name:
                return self._model
            self._clear_model()
            attempts: list[dict] = []

            if self._cuda_disabled_reason is None and not self._cuda_available():
                # No CUDA device: go straight to CPU/int8 without a noisy probe.
                return self._load_cpu(model_name, None, [])

            if self._cuda_disabled_reason is None:
                preferred = os.environ.get("REEL_WHISPER_GPU_COMPUTE", "float16").strip()
                if preferred not in {"float16", "int8_float16"}:
                    preferred = "float16"
                compute_types = [preferred]
                alternate = "int8_float16" if preferred == "float16" else "float16"
                compute_types.append(alternate)
                for compute_type in compute_types:
                    candidate = None
                    started = time.monotonic()
                    try:
                        candidate = WhisperModel(
                            model_name,
                            device="cuda",
                            compute_type=compute_type,
                            num_workers=1,
                        )
                        self._probe_gpu(candidate)
                    except Exception as error:
                        reason = self._short_error(error)
                        attempts.append({
                            "device": "cuda",
                            "compute_type": compute_type,
                            "ok": False,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "error": reason,
                        })
                        candidate = None
                        gc.collect()
                        continue

                    attempts.append({
                        "device": "cuda",
                        "compute_type": compute_type,
                        "ok": True,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    })
                    self._model = candidate
                    self._model_name = model_name
                    self._runtime = {
                        "model": model_name,
                        "device": "cuda",
                        "compute_type": compute_type,
                        "cpu_threads": None,
                        "num_workers": 1,
                        "gpu_probe": "passed",
                        "fallback_reason": None,
                        "attempts": attempts,
                    }
                    return candidate

                failures = "; ".join(item["error"] for item in attempts)
                self._cuda_disabled_reason = failures or "CUDA probe failed"

            fallback_reason = self._cuda_disabled_reason
            if not attempts and fallback_reason:
                attempts.append({
                    "device": "cuda",
                    "ok": False,
                    "skipped": True,
                    "error": fallback_reason,
                })
            return self._load_cpu(model_name, fallback_reason, attempts)

    def fallback_to_cpu(self, model_name: str, error: BaseException | str) -> WhisperModel:
        """Replace a CUDA model after a real transcription failure and remember the fallback."""
        with self._lock:
            reason = str(error) if isinstance(error, str) else self._short_error(error)
            if self._model is not None and self._model_name == model_name \
                    and self._runtime.get("device") == "cpu":
                return self._model
            attempts = list(self._runtime.get("attempts") or [])
            attempts.append({
                "device": "cuda",
                "compute_type": self._runtime.get("compute_type"),
                "ok": False,
                "stage": "transcription",
                "error": reason,
            })
            self._cuda_disabled_reason = reason
            self._clear_model()
            return self._load_cpu(model_name, reason, attempts)

    def runtime_metadata(self) -> dict:
        """Return a detached, JSON-safe description of the active inference runtime."""
        with self._lock:
            return json.loads(json.dumps(self._runtime, ensure_ascii=False))


MODELS = ModelManager()


def clean_word(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[^0-9A-Za-zА-Яа-яЁё]+|[^0-9A-Za-zА-Яа-яЁё]+$", "", text)
    return text


DISFLUENCY_PROMPTS = {
    "ru": "Ну, э-э, вот. Значит, м-м, как бы мы, мы думаем… Типа, короче.",
    "en": "Um, uh, so, like, you know, I mean, we, we think… Basically, right.",
}


DEFAULT_ASR_GLOSSARY = {
    "ru": ["ИИ", "ИИшка", "нейросети", "фулстек", "бэкенд", "фронтенд", "сеньор", "мидл", "джун", "оффер",
           "собеседование", "HR", "вакансии", "найм", "рынок IT"],
}


def normalize_glossary(value, language: str | None = None) -> list[str]:
    """Список терминов: None -> словарь по умолчанию для языка; строка делится по запятым/переводам строк."""
    if value is None:
        return list(DEFAULT_ASR_GLOSSARY.get(language or "", []))
    if isinstance(value, str):
        value = re.split(r"[,;\n]+", value)
    terms: list[str] = []
    for item in value or []:
        term = str(item).strip()[:40]
        if term and term not in terms:
            terms.append(term)
    return terms[:60]


def disfluency_prompt(language: str | None, keep_disfluencies: bool = True,
                      glossary=None) -> str | None:
    """initial_prompt для Whisper: пример речи с паразитами и паузами + словарь терминов.

    Whisper по умолчанию «причёсывает» речь и выбрасывает «э-э», «ну», повторы; пример с
    такими словами и знаками препинания в начале контекста удерживает и пунктуацию,
    и оговорки (они нужны монтажу: editorial вырезает их осознанно, а не теряет заранее).
    Словарь (`asr_glossary`) подсказывает написание терминов (IT-лексика, бренды); по умолчанию
    для ru есть IT-словарь, пустой список его отключает. Для неизвестного языка (auto) пример не используется.
    """
    parts = []
    if keep_disfluencies and DISFLUENCY_PROMPTS.get(language or ""):
        parts.append(DISFLUENCY_PROMPTS[language])
    terms = normalize_glossary(glossary, language)
    if terms:
        parts.append("Термины: " + ", ".join(terms) + ".")
    return " ".join(parts) or None


def media_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError("Не удалось открыть видео. Проверьте формат файла.")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or not frames or not width or not height:
        raise PipelineError("Не удалось определить параметры видео.")
    if width <= height:
        raise PipelineError("Ожидалось горизонтальное видео: ширина должна быть больше высоты.")
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": frames / fps,
    }


def _normalize_transcription_mode(mode: str) -> str:
    normalized = str(mode or "accurate").strip().lower()
    if normalized == "fast":
        normalized = "coarse"
    if normalized not in {"coarse", "accurate"}:
        raise ValueError("mode must be 'coarse'/'fast' or 'accurate'")
    return normalized


def _format_clock(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _emit_telemetry(callback: TelemetryCallback | None, payload: dict) -> None:
    if callback is None:
        return
    try:
        callback(dict(payload))
    except Exception:
        # Telemetry is observational and must never abort a several-hour job.
        return


def _is_cuda_runtime_error(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    signals = (
        "cuda", "cublas", "cudnn", "nvrtc", "gpu", "out of memory",
        "driver version", "device kernel", "compute type",
    )
    return any(signal in message for signal in signals)


def _runtime_metadata() -> dict:
    try:
        return MODELS.runtime_metadata()
    except (AttributeError, TypeError):
        return {}


def _transcribe_once(
    model: WhisperModel,
    video_path: Path,
    selected_language: str | None,
    mode: str,
    report: ProgressCallback,
    telemetry_callback: TelemetryCallback | None,
    progress_state: dict,
    started: float,
    keep_disfluencies: bool = True,
    glossary=None,
):
    accurate = mode == "accurate"
    kwargs = {
        "language": selected_language,
        "beam_size": 5,
        "word_timestamps": accurate,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 350},
    }
    prompt = disfluency_prompt(selected_language, keep_disfluencies, glossary)
    if prompt:
        # condition_on_previous_text оставлен включённым (по умолчанию): в faster-whisper
        # initial_prompt остаётся в контексте последующих окон только при нём, при False
        # подсказка действует лишь на первое окно и стиль «с паразитами» дрейфует к чистому.
        kwargs["initial_prompt"] = prompt

    # Keep Whisper's timestamp tokens and previous-text context in coarse mode.
    # Only word alignment is skipped; highlight selection still needs reliable
    # segment boundaries and coherent long-form text.
    segments, info = model.transcribe(str(video_path), **kwargs)
    total = max(0.0, float(getattr(info, "duration", 0.0) or 0.0))
    progress_state["total_seconds"] = max(
        float(progress_state.get("total_seconds", 0.0)), total
    )
    total = float(progress_state["total_seconds"])

    def emit_progress(processed: float, force: bool = False) -> None:
        now = time.monotonic()
        processed = max(float(progress_state.get("processed_seconds", 0.0)), processed)
        progress_state["processed_seconds"] = processed
        fraction = min(1.0, processed / total) if total > 0 else 0.0
        outer_progress = 10 + min(25, int(fraction * 25))
        elapsed = max(0.001, now - started)
        eta = None
        if processed > 0 and total > processed:
            eta = max(0.0, (total - processed) / (processed / elapsed))
        should_emit = (
            force
            or outer_progress > int(progress_state.get("outer_progress", 9))
            or now - float(progress_state.get("last_emit", 0.0)) >= 2.0
        )
        if not should_emit:
            return
        progress_state["last_emit"] = now
        progress_state["outer_progress"] = max(
            outer_progress, int(progress_state.get("outer_progress", 9))
        )
        percent = int(round(fraction * 100))
        eta_text = f" · осталось ~{_format_clock(eta)}" if eta is not None else ""
        report(
            int(progress_state["outer_progress"]),
            f"Распознаю речь: {percent}% · {_format_clock(processed)} / "
            f"{_format_clock(total)}{eta_text}",
        )
        _emit_telemetry(telemetry_callback, {
            "event": "transcription_progress",
            "mode": mode,
            "processed_seconds": round(processed, 3),
            "total_seconds": round(total, 3),
            "progress": round(fraction, 6),
            "percent": percent,
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": None if eta is None else round(eta, 3),
            "runtime": _runtime_metadata(),
        })

    emit_progress(float(progress_state.get("processed_seconds", 0.0)), force=True)
    words: list[dict] = []
    text_parts: list[str] = []
    segment_items: list[dict] = []
    for segment in segments:
        text = segment.text.strip()
        text_parts.append(text)
        segment_items.append({
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": text,
        })
        if accurate:
            for word in segment.words or []:
                cleaned = clean_word(word.word)
                if not cleaned or word.start is None or word.end is None:
                    continue
                words.append({
                    "text": cleaned,
                    # исходный токен Whisper с пунктуацией: нужен для границ предложений
                    "raw": word.word.strip(),
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "probability": round(float(word.probability), 4),
                })
        emit_progress(float(segment.end))

    elapsed = max(0.001, time.monotonic() - started)
    return {
        "info": info,
        "words": words,
        "segments": segment_items,
        "text": " ".join(text_parts).strip(),
        "pipeline": "sequential",
        "batch_size": 1,
        "elapsed_seconds": elapsed,
        "emit_progress": emit_progress,
    }


def resolve_asr_provider(value) -> str:
    """'deepgram' | 'local' | 'auto' -> 'deepgram' или 'local' (auto = deepgram, если есть ключ)."""
    from . import asr_deepgram
    choice = str(value or "local").strip().lower()
    if choice == "deepgram":
        return "deepgram"
    if choice == "auto":
        return "deepgram" if asr_deepgram.key_available() else "local"
    return "local"


def _transcribe_deepgram(video_path: Path, work_dir: Path, language: str, report, mode: str,
                         glossary, keep_disfluencies: bool) -> dict:
    from . import asr_deepgram
    accurate = mode == "accurate"
    selected_language = None if language == "auto" else language
    terms = normalize_glossary(glossary, selected_language)
    try:
        result = asr_deepgram.transcribe(video_path, work_dir, selected_language, terms, FFMPEG, report,
                                         clean_word, accurate=accurate)
    except asr_deepgram.DeepgramError as error:
        raise PipelineError(f"Распознавание через Deepgram не удалось: {error}") from None
    if accurate and not result["words"]:
        raise PipelineError("В видео не удалось обнаружить речь для субтитров.")
    if not accurate and (not result["segments"] or not result["text"]):
        raise PipelineError("В видео не удалось обнаружить речь для поиска ключевых моментов.")
    duration = float(result["duration"])
    elapsed = float(result["elapsed_seconds"])
    payload = {
        "source": video_path.name,
        "language": result["language"],
        "language_probability": result["language_probability"],
        "words": result["words"],
        "segments": result["segments"],
        "text": result["text"],
        "runtime": {"provider": "deepgram", "model": asr_deepgram.MODEL, "device": "cloud"},
        "transcription": {
            "mode": mode,
            "provider": "deepgram",
            "model": asr_deepgram.MODEL,
            "asr_warning": None,
            "word_timestamps": accurate,
            "keep_disfluencies": True,
            "glossary": terms,
            "keyterms_sent": result["keyterms"],
            "cached_response": result["cached"],
            "duration_seconds": round(duration, 3),
            "elapsed_seconds": round(elapsed, 3),
            "rtf": round(elapsed / duration, 6) if duration > 0 else None,
        },
    }
    (work_dir / "transcript.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (work_dir / "transcript.txt").write_text(payload["text"], encoding="utf-8")
    if accurate:
        report(36, f"Речь распознана: {len(result['words'])} слов")
    else:
        report(36, f"Черновая транскрибация готова: {len(result['segments'])} сегментов")
    return payload


def transcribe_video(
    video_path: Path,
    work_dir: Path,
    language: str,
    model_name: str,
    callback: ProgressCallback,
    *,
    mode: str = "accurate",
    telemetry_callback: TelemetryCallback | None = None,
    keep_disfluencies: bool = True,
    glossary=None,
    asr_provider: str = "local",
) -> dict:
    """Transcribe a source in fast segment-level or accurate word-level mode.

    The original five positional arguments remain compatible. ``mode='fast'`` is
    accepted as an alias for ``'coarse'``. Structured progress is optional; the
    existing ``callback(percent, message)`` receives the same information as text.
    """
    normalized_mode = _normalize_transcription_mode(mode)
    accurate = normalized_mode == "accurate"
    work_dir.mkdir(parents=True, exist_ok=True)
    reported_progress = 0

    def report(progress: int, message: str) -> None:
        nonlocal reported_progress
        reported_progress = max(reported_progress, int(progress))
        callback(reported_progress, message)

    asr_warning = None
    provider = resolve_asr_provider(asr_provider)
    if provider == "deepgram":
        try:
            return _transcribe_deepgram(video_path, work_dir, language, report, normalized_mode,
                                        glossary, keep_disfluencies)
        except PipelineError as error:
            if str(asr_provider or "").strip().lower() == "deepgram":
                raise
            asr_warning = f"Deepgram недоступен, распознаю локально: {error}"
            report(reported_progress, "Внимание: " + asr_warning)

    report(6, "Загружаю модель распознавания")
    model = MODELS.get(model_name)
    detail = "пословные таймкоды" if accurate else "сегменты для поиска ключевых моментов"
    report(10, f"Распознаю речь и {detail}")
    selected_language = None if language == "auto" else language
    started = time.monotonic()
    progress_state = {
        "processed_seconds": 0.0,
        "total_seconds": 0.0,
        "outer_progress": 9,
        "last_emit": 0.0,
    }

    try:
        result = _transcribe_once(
            model, video_path, selected_language, normalized_mode, report,
            telemetry_callback, progress_state, started, keep_disfluencies, glossary,
        )
    except Exception as error:
        runtime = _runtime_metadata()
        if runtime.get("device") != "cuda" or not _is_cuda_runtime_error(error):
            raise
        report(reported_progress, "CUDA дала сбой; продолжаю распознавание на CPU")
        _emit_telemetry(telemetry_callback, {
            "event": "transcription_runtime_fallback",
            "mode": normalized_mode,
            "processed_seconds": round(float(progress_state["processed_seconds"]), 3),
            "total_seconds": round(float(progress_state["total_seconds"]), 3),
            "progress": (
                round(float(progress_state["processed_seconds"]) /
                      float(progress_state["total_seconds"]), 6)
                if progress_state["total_seconds"] else 0.0
            ),
            "percent": max(0, (reported_progress - 10) * 4),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "eta_seconds": None,
            "error": ModelManager._short_error(error),
            "runtime": runtime,
        })
        model = MODELS.fallback_to_cpu(model_name, error)
        result = _transcribe_once(
            model, video_path, selected_language, normalized_mode, report,
            telemetry_callback, progress_state, started, keep_disfluencies, glossary,
        )

    words = result["words"]
    segment_items = result["segments"]
    text = result["text"]
    if accurate and not words:
        raise PipelineError("В видео не удалось обнаружить речь для субтитров.")
    if not accurate and (not segment_items or not text):
        raise PipelineError("В видео не удалось обнаружить речь для поиска ключевых моментов.")

    info = result["info"]
    duration = max(0.0, float(getattr(info, "duration", 0.0) or 0.0))
    duration_after_vad = max(
        0.0, float(getattr(info, "duration_after_vad", duration) or 0.0)
    )
    elapsed = float(result["elapsed_seconds"])
    result["emit_progress"](max(duration, float(progress_state["processed_seconds"])), force=True)
    runtime = _runtime_metadata()
    payload = {
        "source": video_path.name,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "words": words,
        "segments": segment_items,
        "text": text,
        "runtime": runtime,
        "transcription": {
            "mode": normalized_mode,
            "provider": "local",
            "asr_warning": asr_warning,
            "beam_size": 5,
            "word_timestamps": accurate,
            "keep_disfluencies": bool(keep_disfluencies and DISFLUENCY_PROMPTS.get(selected_language or "")),
            "glossary": normalize_glossary(glossary, selected_language),
            "vad_filter": True,
            "pipeline": result["pipeline"],
            "batch_size": result["batch_size"],
            "duration_seconds": round(duration, 3),
            "duration_after_vad_seconds": round(duration_after_vad, 3),
            "elapsed_seconds": round(elapsed, 3),
            "rtf": round(elapsed / duration, 6) if duration > 0 else None,
        },
    }
    (work_dir / "transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (work_dir / "transcript.txt").write_text(payload["text"], encoding="utf-8")
    if accurate:
        report(36, f"Речь распознана: {len(words)} слов")
    else:
        report(36, f"Черновая транскрибация готова: {len(segment_items)} сегментов")
    return payload


def transcribe_selected_clip(
    video_path: Path,
    work_dir: Path,
    language: str,
    model_name: str,
    callback: ProgressCallback,
    *,
    telemetry_callback: TelemetryCallback | None = None,
) -> dict:
    """Create the word-accurate transcript used to render one selected clip."""
    return transcribe_video(
        video_path,
        work_dir,
        language,
        model_name,
        callback,
        mode="accurate",
        telemetry_callback=telemetry_callback,
    )


def detect_cuts(video_path: Path, info: dict, callback: ProgressCallback):
    cap = cv2.VideoCapture(str(video_path))
    diffs: list[tuple[int, float]] = []
    prev = None
    frame_idx = 0
    notify_every = max(1, info["frames"] // 20)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        if prev is not None:
            diffs.append((frame_idx, float(np.mean(cv2.absdiff(lab, prev)))))
        prev = lab
        frame_idx += 1
        if frame_idx % notify_every == 0:
            callback(38 + min(10, int(frame_idx / info["frames"] * 10)), "Ищу монтажные склейки")
    cap.release()
    values = np.array([diff for _, diff in diffs], dtype=np.float32)
    median = float(np.median(values)) if len(values) else 0.0
    mad = float(np.median(np.abs(values - median))) if len(values) else 0.0
    threshold = max(7.0, median + 8.0 * max(mad, 0.5))
    candidates = [(frame, diff) for frame, diff in diffs if diff >= threshold]
    cuts: list[tuple[int, float]] = []
    min_gap = max(6, int(info["fps"] * 0.35))
    for frame, diff in candidates:
        if cuts and frame - cuts[-1][0] < min_gap:
            if diff > cuts[-1][1]:
                cuts[-1] = (frame, diff)
        else:
            cuts.append((frame, diff))
    boundaries = [0] + [frame for frame, _ in cuts] + [info["frames"]]
    return threshold, sorted(set(boundaries))


def dedupe_faces(faces):
    result = []
    for x, y, w, h in sorted(faces, key=lambda box: box[2] * box[3], reverse=True):
        cx, cy = x + w / 2, y + h / 2
        if any(
            abs(cx - (rx + rw / 2)) < 0.35 * max(w, rw)
            and abs(cy - (ry + rh / 2)) < 0.35 * max(h, rh)
            for rx, ry, rw, rh in result
        ):
            continue
        result.append((int(x), int(y), int(w), int(h)))
    return result


def detect_faces(frame, frontal, profile, min_face: int):
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    args = dict(scaleFactor=1.08, minNeighbors=5, minSize=(min_face, min_face))
    faces = list(frontal.detectMultiScale(gray, **args))
    faces += list(profile.detectMultiScale(gray, **args))
    flipped = cv2.flip(gray, 1)
    for x, y, w, h in profile.detectMultiScale(flipped, **args):
        faces.append((frame.shape[1] - x - w, y, w, h))
    result = []
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    for x, y, w, h in dedupe_faces(faces):
        roi = ycrcb[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        skin = cv2.inRange(roi, (0, 133, 77), (255, 178, 135))
        if float(np.count_nonzero(skin)) / float(skin.size) >= 0.07:
            result.append((x, y, w, h))
    return result


def classify_shots(video_path: Path, boundaries: list[int], info: dict, callback: ProgressCallback):
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    cap = cv2.VideoCapture(str(video_path))
    shots = []
    width, height = info["width"], info["height"]
    crop_w = max(2, int(round(height * 9 / 16 / 2) * 2))
    min_face = max(42, int(width * 0.029))
    total = max(1, len(boundaries) - 1)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        length = end - start
        observations = []
        for position in (0.30, 0.50, 0.70):
            frame_no = min(max(start, start + int(length * position)), end - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if ok:
                observations.append(detect_faces(frame, frontal, profile, min_face))
        two_face_votes = 0
        pair_centers = []
        centers = []
        for faces in observations:
            main = [face for face in faces if face[2] >= min_face and face[3] >= min_face]
            if len(main) >= 2:
                pair = sorted(main, key=lambda face: face[2] * face[3], reverse=True)[:2]
                area_ratio = (pair[1][2] * pair[1][3]) / (pair[0][2] * pair[0][3])
                pair_x = sorted(pair, key=lambda face: face[0] + face[2] / 2)
                left_center = pair_x[0][0] + pair_x[0][2] / 2
                right_center = pair_x[1][0] + pair_x[1][2] / 2
                if area_ratio >= 0.28 and right_center - left_center > width * 0.22:
                    two_face_votes += 1
                    pair_centers.append((left_center, right_center))
            if main:
                face = max(main, key=lambda item: item[2] * item[3])
                centers.append(face[0] + face[2] / 2)
        center = float(np.median(centers)) if centers else width / 2
        crop_x = int(round(max(0, min(width - crop_w, center - crop_w / 2))))
        stack_left = float(np.median([pair[0] for pair in pair_centers])) if pair_centers else width * 0.25
        stack_right = float(np.median([pair[1] for pair in pair_centers])) if pair_centers else width * 0.75
        half_w = width // 2
        stack_left_x = int(round(max(0, min(width - half_w, stack_left - half_w / 2))))
        stack_right_x = int(round(max(0, min(width - half_w, stack_right - half_w / 2))))
        shots.append({
            "start_frame": start,
            "end_frame": end,
            "start": round(start / info["fps"], 3),
            "end": round(end / info["fps"], 3),
            "layout": "stack" if two_face_votes >= 1 else "crop",
            "crop_x": crop_x,
            "stack_crop_x": [stack_left_x, stack_right_x],
            "face_centers": [round(value, 1) for value in centers],
        })
        callback(49 + int((index + 1) / total * 12), "Удерживаю спикеров в кадре")
    cap.release()
    return shots, crop_w


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - math.floor(seconds)) * 100))
    if cs == 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(words: list[dict], path: Path, duration: float, style: str):
    profiles = {
        "pill": ("Arial", 82, "&H00FFFFFF", "&H7A000000", "&H00000000", 3, 0, 0, -2),
        "clean": ("Arial", 78, "&H00FFFFFF", "&H00000000", "&H00000000", 1, 4, 0, -2),
        "yellow": ("Arial", 86, "&H0000E9FF", "&H6F000000", "&H00000000", 3, 0, 0, -2),
        "caption-pill-karaoke": ("Arial", 84, "&H00FFFFFF", "&H78000000", "&H00000000", 3, 0, 0, -2),
        "caption-highlight": ("Arial", 88, "&H0000E9FF", "&H72000000", "&H00000000", 3, 0, 0, -2),
        "caption-editorial-emphasis": ("Georgia", 80, "&H00E8F1F4", "&H68000000", "&H00000000", 3, 0, 1, -1),
        "caption-kinetic-slam": ("Arial Black", 92, "&H00FFFFFF", "&H00000000", "&H00131810", 1, 5, 0, 0),
        "caption-weight-shift": ("Arial", 80, "&H00FFFFFF", "&H00000000", "&H00131810", 1, 4, 0, 1),
        "caption-neon-accent": ("Arial", 84, "&H00FFFFFF", "&H00000000", "&H004AF5C7", 1, 5, 0, 0),
        "caption-neon-glow": ("Arial", 84, "&H00FFFFFF", "&H00000000", "&H004AF5C7", 1, 6, 0, 0),
        "caption-glitch-rgb": ("Arial", 86, "&H00FFFFFF", "&H00000000", "&H00FF4A4A", 1, 5, 0, 1),
        "caption-clip-wipe": ("Arial", 84, "&H00FFFFFF", "&H70000000", "&H00000000", 3, 0, 0, 0),
    }
    font, size, primary, back, outline_color, border_style, outline, italic, spacing = profiles.get(
        style, profiles["pill"]
    )
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{font},{size},{primary},{primary},{outline_color},{back},-1,{italic},0,0,100,100,{spacing},0,{border_style},{outline},0,2,80,80,430,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for index, word in enumerate(words):
        start = float(word["start"])
        next_start = float(words[index + 1]["start"]) if index + 1 < len(words) else duration
        natural_end = max(float(word["end"]) + 0.08, start + 0.14)
        end = min(next_start, natural_end, start + 0.9)
        if end <= start:
            end = min(duration, start + 0.10)
        text = str(word["text"]).replace("{", "(").replace("}", ")")
        fade = 45 if end - start >= 0.18 else 20
        if style == "caption-kinetic-slam":
            entrance = "\\fscx132\\fscy132\\t(0,140,\\fscx100\\fscy100)"
        elif style == "caption-weight-shift":
            entrance = "\\fsp5\\t(0,180,\\fsp1)"
        elif style == "caption-glitch-rgb":
            entrance = "\\frz-1\\t(0,90,\\frz1)\\t(90,180,\\frz0)"
        elif style == "caption-clip-wipe":
            entrance = "\\fscx86\\t(0,150,\\fscx100)"
        else:
            entrance = ""
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Word,,0,0,0,,"
            f"{{\\fad({fade},{fade}){entrance}}}{text}\n"
        )
    path.write_text("".join(lines), encoding="utf-8-sig")


def build_filter(shots: list[dict], ass_path: Path, info: dict, crop_w: int) -> str:
    width, height = info["width"], info["height"]
    half_w = width // 2
    source_half_height = min(height, max(2, int(round(half_w / 1.125 / 2) * 2)))
    source_y = max(0, min(height - source_half_height, int(height * 0.083)))
    labels = "".join(f"[v{index}]" for index in range(len(shots)))
    parts = [f"[0:v]split={len(shots)}{labels}"]
    for index, shot in enumerate(shots):
        start, end = shot["start_frame"], shot["end_frame"]
        base = f"[v{index}]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
        if shot["layout"] == "stack":
            stack_x = shot.get("stack_crop_x") or [0, width - half_w]
            parts.append(base + f",split=2[w{index}l][w{index}r]")
            parts.append(
                f"[w{index}l]crop={half_w}:{source_half_height}:{stack_x[0]}:{source_y},"
                f"scale=1080:960:flags=lanczos[l{index}]"
            )
            parts.append(
                f"[w{index}r]crop={half_w}:{source_half_height}:{stack_x[1]}:{source_y},"
                f"scale=1080:960:flags=lanczos[r{index}]"
            )
            parts.append(
                f"[l{index}][r{index}]vstack=inputs=2,"
                f"drawbox=x=0:y=958:w=1080:h=4:color=black@0.9:t=fill,setsar=1[s{index}]"
            )
        else:
            parts.append(
                base + f",crop={crop_w}:{height}:{shot['crop_x']}:0,"
                f"scale=1080:1920:flags=lanczos,setsar=1[s{index}]"
            )
    concat_inputs = "".join(f"[s{index}]" for index in range(len(shots)))
    safe_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    parts.append(
        f"{concat_inputs}concat=n={len(shots)}:v=1:a=0,"
        f"subtitles=filename='{safe_ass}'{runtime_paths.subtitles_fontsdir_option()}[outv]"
    )
    return ";\n".join(parts) + ";\n"


def render_video(video_path: Path, output_path: Path, work_dir: Path, transcript: dict,
                 info: dict, shots: list[dict], crop_w: int, subtitle_style: str,
                 callback: ProgressCallback):
    ass_path = work_dir / "subtitles.ass"
    write_ass(transcript["words"], ass_path, info["duration"], subtitle_style)
    filter_path = work_dir / "render.filter.txt"
    filter_path.write_text(build_filter(shots, ass_path, info, crop_w), encoding="utf-8")
    log_path = work_dir / "ffmpeg.log"
    callback(64, "Собираю вертикальное видео")
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "warning", "-i", str(video_path),
        "-filter_complex_script", str(filter_path), "-map", "[outv]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats", str(output_path),
    ]
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=log_file,
            text=True, encoding="utf-8", errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            key, _, value = line.strip().partition("=")
            if key in {"out_time_ms", "out_time_us"}:
                try:
                    rendered = int(value) / 1_000_000
                    percent = 64 + min(34, int(rendered / info["duration"] * 34))
                    callback(percent, "Кодирую финальный MP4")
                except (ValueError, ZeroDivisionError):
                    pass
        return_code = process.wait()
    if return_code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1600:]
        raise PipelineError("FFmpeg не смог собрать видео. " + tail)
    callback(99, "Проверяю готовый файл")


def verify_output(output_path: Path, expected_duration: float):
    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        raise PipelineError("Готовый файл не открывается после рендера.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps if fps else 0
    if (width, height) != (1080, 1920):
        raise PipelineError(f"Получено неверное разрешение {width}×{height}.")
    if abs(duration - expected_duration) > 0.25:
        raise PipelineError("Длительность результата отличается от исходника.")
    return {"width": width, "height": height, "fps": fps, "duration": duration, "bytes": output_path.stat().st_size}


def process_video(video_path: Path, output_path: Path, work_dir: Path, options: dict,
                  callback: ProgressCallback) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    callback(2, "Проверяю исходное видео")
    info = media_info(video_path)
    transcript = transcribe_video(
        video_path, work_dir,
        options.get("language", "ru"),
        options.get("model", "base"),
        callback,
    )
    threshold, boundaries = detect_cuts(video_path, info, callback)
    shots, crop_w = classify_shots(video_path, boundaries, info, callback)
    report = {
        "source": video_path.name,
        "media": info,
        "cut_threshold": threshold,
        "shots": shots,
    }
    (work_dir / "shots.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    render_video(
        video_path, output_path, work_dir, transcript, info, shots, crop_w,
        options.get("subtitle_style", "pill"), callback,
    )
    result = verify_output(output_path, info["duration"])
    result.update({
        "words": len(transcript["words"]),
        "shots": len(shots),
        "stacked_shots": sum(shot["layout"] == "stack" for shot in shots),
        "language": transcript["language"],
    })
    callback(100, "Готово")
    return result


def process_video_with_transcript(video_path: Path, output_path: Path, work_dir: Path,
                                  options: dict, transcript: dict,
                                  callback: ProgressCallback) -> dict:
    """Render a video using already available word-level timestamps."""
    work_dir.mkdir(parents=True, exist_ok=True)
    callback(4, "Проверяю выбранный отрывок")
    info = media_info(video_path)
    words = transcript.get("words") or []
    if not words:
        raise PipelineError("Для выбранного отрывка не найдены пословные таймкоды.")
    (work_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    callback(20, f"Использую готовые субтитры: {len(words)} слов")
    threshold, boundaries = detect_cuts(video_path, info, callback)
    shots, crop_w = classify_shots(video_path, boundaries, info, callback)
    report = {
        "source": video_path.name,
        "media": info,
        "cut_threshold": threshold,
        "shots": shots,
    }
    (work_dir / "shots.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_video(
        video_path, output_path, work_dir, transcript, info, shots, crop_w,
        options.get("subtitle_style", "pill"), callback,
    )
    result = verify_output(output_path, info["duration"])
    result.update({
        "words": len(words),
        "shots": len(shots),
        "stacked_shots": sum(shot["layout"] == "stack" for shot in shots),
        "language": transcript.get("language", options.get("language", "auto")),
    })
    callback(100, "Готово")
    return result
