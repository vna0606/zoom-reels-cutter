"""Распознавание речи через Deepgram (облако, nova-3) в том же формате транскрипта, что и локальный Whisper.

Ключ берётся из переменной окружения DEEPGRAM_API_KEY, иначе из файла .env в корне проекта,
иначе из .secrets/deepgram.env (строка `DEEPGRAM_API_KEY=...`). Ключ нигде не логируется и не попадает в тексты ошибок:
любое сообщение проходит через `redact`. Сырой ответ Deepgram (без секретов) кэшируется рядом с
транскриптом, чтобы повторный запуск не тратил деньги заново.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SECRETS_FILE = ROOT / ".secrets" / "deepgram.env"
DOTENV_FILE = ROOT / ".env"
ENV_NAME = "DEEPGRAM_API_KEY"
API_URL = "https://api.deepgram.com/v1/listen"
MODEL = "nova-3"
TIMEOUT_SECONDS = 1800
RETRIES = 2
MAX_KEYTERMS = 50
CACHE_NAME = "deepgram_response.json"

Report = Callable[[int, str], None]


class DeepgramError(RuntimeError):
    """Сообщение уже на русском и очищено от ключа."""


def _key_from_file(path: Path) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith(ENV_NAME + "="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    except OSError:
        pass
    return None


def load_api_key() -> str | None:
    key = os.environ.get(ENV_NAME, "").strip().strip("'\"")
    if key:
        return key
    return _key_from_file(DOTENV_FILE) or _key_from_file(SECRETS_FILE)


def key_available() -> bool:
    return load_api_key() is not None


def redact(text: object, key: str | None = None) -> str:
    """Убирает ключ и заголовок Authorization из любого текста."""
    out = str(text)
    key = key or load_api_key()
    if key:
        out = out.replace(key, "***")
    return re.sub(r"(?i)(authorization\W{0,4}(?:token|bearer)?\s*)[A-Za-z0-9._\-]{8,}", r"\1***", out)


def build_params(language: str | None, glossary: list[str] | None) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = [("model", MODEL)]
    if language and language != "auto":
        params.append(("language", language))
    else:
        params.append(("detect_language", "true"))
    params += [("smart_format", "true"), ("punctuate", "true"), ("filler_words", "true"),
               ("utterances", "true"), ("paragraphs", "true")]
    seen: list[str] = []
    for term in glossary or []:
        term = str(term).strip()
        if term and term not in seen:
            seen.append(term)
    for term in seen[:MAX_KEYTERMS]:
        params.append(("keyterm", term))
    return params


def extract_audio(ffmpeg: str, video: Path, dest: Path) -> Path:
    """Моно, 16 кГц, mp3 48 кбит/с (≈55 МБ на 2,6 ч)."""
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "libmp3lame", "-b:a", "48k", str(dest)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as error:
        raise DeepgramError(f"Не удалось запустить ffmpeg для извлечения аудио: {error}") from error
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise DeepgramError("Не удалось извлечь аудио из видео: " + (proc.stderr or "").strip()[-300:])
    return dest


def _post(url: str, data: bytes, key: str, timeout: float) -> dict:
    request = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"Token {key}", "Content-Type": "audio/mpeg"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_deepgram(audio: bytes, params: list[tuple[str, str]], key: str, report: Report | None = None,
                     *, timeout: float = TIMEOUT_SECONDS, retries: int = RETRIES, sleep=time.sleep) -> dict:
    """POST аудио; 2 повтора на 5xx/таймауты. 4xx — сразу ошибка; 400 из-за keyterm — один повтор без keyterm."""
    attempt = 0
    current = list(params)
    dropped_keyterms = False
    while True:
        url = API_URL + "?" + urllib.parse.urlencode(current)
        try:
            return _post(url, audio, key, timeout)
        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode("utf-8", "replace")[:300]
            except Exception:
                body = ""
            status = error.code
            if status == 400 and not dropped_keyterms and any(k == "keyterm" for k, _ in current) \
                    and "keyterm" in body.lower():
                dropped_keyterms = True
                current = [p for p in current if p[0] != "keyterm"]
                if report:
                    report(10, "Deepgram не принял термины-подсказки; повторяю без них")
                continue
            if status in (401, 403):
                raise DeepgramError(f"Deepgram отклонил ключ доступа (HTTP {status}). "
                                    "Проверьте DEEPGRAM_API_KEY или .secrets/deepgram.env.") from None
            if status >= 500 and attempt < retries:
                attempt += 1
                sleep(2 * attempt)
                continue
            raise DeepgramError(redact(f"Deepgram вернул ошибку HTTP {status}: {body}", key)) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt < retries:
                attempt += 1
                sleep(2 * attempt)
                continue
            raise DeepgramError(redact(f"Нет связи с Deepgram ({type(error).__name__}: {error}).", key)) from None
        except ValueError:
            raise DeepgramError("Deepgram вернул некорректный ответ (не JSON).") from None


def map_response(data: dict, clean_word: Callable[[str], str], *, accurate: bool = True,
                 requested_language: str | None = None) -> dict:
    """Ответ Deepgram -> {words, segments, text, language, language_probability, duration}."""
    try:
        channel = data["results"]["channels"][0]
        alt = channel["alternatives"][0]
    except (KeyError, IndexError, TypeError):
        raise DeepgramError("В ответе Deepgram нет результата распознавания.") from None
    words: list[dict] = []
    for item in alt.get("words") or []:
        raw = str(item.get("punctuated_word") or item.get("word") or "").strip()
        cleaned = clean_word(raw)
        if not cleaned or item.get("start") is None or item.get("end") is None:
            continue
        words.append({
            "text": cleaned, "raw": raw,
            "start": round(float(item["start"]), 3), "end": round(float(item["end"]), 3),
            "probability": round(float(item.get("confidence", 0.0) or 0.0), 4),
        })
    # Deepgram иногда отдаёт слово, чей конец заходит за начало следующего: подрезаем,
    # иначе на склейке звук соседнего слова попадёт в оба куска.
    words.sort(key=lambda w: w["start"])
    for prev, nxt in zip(words, words[1:]):
        if nxt["start"] < prev["end"]:
            prev["end"] = round(max(prev["start"] + 0.02, nxt["start"]), 3)
            nxt["start"] = max(nxt["start"], prev["end"])
    segments: list[dict] = []
    for utt in (data.get("results") or {}).get("utterances") or []:
        text = str(utt.get("transcript") or "").strip()
        if text:
            segments.append({"start": round(float(utt["start"]), 3), "end": round(float(utt["end"]), 3), "text": text})
    if not segments:
        paragraphs = ((alt.get("paragraphs") or {}).get("paragraphs")) or []
        for para in paragraphs:
            for sent in para.get("sentences") or []:
                text = str(sent.get("text") or "").strip()
                if text:
                    segments.append({"start": round(float(sent["start"]), 3), "end": round(float(sent["end"]), 3),
                                     "text": text})
    if not segments and words:
        segments = [{"start": words[0]["start"], "end": words[-1]["end"],
                     "text": " ".join(w["raw"] for w in words)}]
    text = str(alt.get("transcript") or "").strip() or " ".join(s["text"] for s in segments)
    if requested_language and requested_language != "auto":
        language, probability = requested_language, 1.0
    else:
        language = str(channel.get("detected_language") or "ru")
        probability = float(channel.get("language_confidence") or 0.0)
    duration = float((data.get("metadata") or {}).get("duration") or (words[-1]["end"] if words else 0.0))
    return {"words": words if accurate else [], "segments": segments, "text": text,
            "language": language, "language_probability": probability, "duration": duration}


def _signature(video: Path, params: list[tuple[str, str]]) -> str:
    try:
        st = video.stat()
        ident = f"{video.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        ident = video.name
    return hashlib.sha256((ident + json.dumps(params)).encode("utf-8")).hexdigest()


def transcribe(video: Path, work_dir: Path, language: str | None, glossary: list[str] | None,
               ffmpeg: str, report: Report, clean_word: Callable[[str], str], *, accurate: bool = True) -> dict:
    """Полный цикл: кэш -> аудио -> POST -> маппинг. Возвращает результат `map_response` + служебные поля."""
    key = load_api_key()
    if not key:
        raise DeepgramError("Не найден ключ Deepgram: задайте DEEPGRAM_API_KEY или файл .secrets/deepgram.env.")
    work_dir.mkdir(parents=True, exist_ok=True)
    params = build_params(language, glossary)
    signature = _signature(Path(video), params)
    cache = work_dir / CACHE_NAME
    data = None
    cached = False
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("signature") == signature:
            data, cached = saved["response"], True
    except (OSError, ValueError, KeyError):
        pass
    started = time.monotonic()
    if data is None:
        report(8, "Извлекаю аудио для Deepgram")
        audio_path = extract_audio(ffmpeg, Path(video), work_dir / "audio_deepgram.mp3")
        audio = audio_path.read_bytes()
        report(10, f"Загружаю аудио в Deepgram ({len(audio) / 1_048_576:.1f} МБ)")
        report(12, "Deepgram распознаёт…")
        data = request_deepgram(audio, params, key, report)
        cache.write_text(json.dumps({"signature": signature, "response": data}, ensure_ascii=False),
                         encoding="utf-8")
    else:
        report(12, "Ответ Deepgram взят из кэша")
    result = map_response(data, clean_word, accurate=accurate, requested_language=language)
    result["elapsed_seconds"] = time.monotonic() - started
    result["cached"] = cached
    result["keyterms"] = [v for k, v in params if k == "keyterm"]
    return result
