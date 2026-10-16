"""Шеф-редактор по готовому видео: отсматривает вертикальный mp4 так, как это сделал бы человек.

Что делает (см. README, «Шеф-редактор по готовому видео»):
1. кадры (ТОЛЬКО при options["review_visual"]=True; по умолчанию выключено — картинку не оцениваем):
   по одному раз в ~2 с и по два (±0.3 с) на каждом стыке склейки -> контактные листы JPEG;
2. аудио: перераспознаёт ЗВУК ГОТОВОГО ролика (faster-whisper, пословно), сравнивает с ожидаемым
   текстом после монтажа (обрезанные/пропавшие слова на стыках, задвоения), ищет паузы > 0.6 с, обрыв в конце и
   скачки громкости на стыках;
3. один запрос ИИ на ролик (task `video_review`, effort high, картинки + расшифровка + автонаходки) -> вердикт
   publish | needs_edit | reject, оценки 0–10, замечания с таймкодами, предлагаемая правка в терминах кусков
   ролика и подпись для публикации;
4. `apply_suggested_fix` превращает правку в новые куски кандидата (одна итерация авто-правки на ролик).

Тяжёлые зависимости (cv2, numpy, faster-whisper) подключаются лениво: модуль импортируется без них.
"""
from __future__ import annotations

import difflib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

from . import assembly, editorial

VERDICTS = ["publish", "needs_edit", "reject"]
SCORE_KEYS = ("hook", "idea_disclosure", "flow", "audio")
VISUAL_SCORE_KEYS = SCORE_KEYS + ("visual",)
ISSUE_KINDS = ["hook", "idea", "flow", "jump", "audio", "clipped_word", "silence", "duration", "ending", "other"]
VISUAL_ISSUE_KINDS = ISSUE_KINDS + ["visual", "subtitles"]
REJECT_AT_OR_BELOW = 3         # любая оценка <= 3 -> reject
EDIT_BELOW = 6                 # иначе любая оценка < 6 -> needs_edit
FIX_OPS = ["drop_last_sentences", "drop_first_sentences", "extend_piece_start", "extend_piece_end",
           "trim_piece_start", "trim_piece_end", "drop_piece", "cut_phrase"]
FRAME_STEP = 2.0               # обычные кадры раз в 2 с
SPLICE_OFFSET = 0.3            # кадры на стыке: t-0.3 и t+0.3
MAX_FRAMES = 72                # потолок кадров на ролик (<= 8 листов)
MAX_SHEET_WIDTH = 1600
SILENCE_GAP = 0.6              # пауза в речи длиннее — замечание
LOUDNESS_JUMP_DB = 10.0        # скачок речевого уровня (p80 громкости окон 0.6 с слева/справа от стыка)
LOUD_WINDOW = 0.6
MAX_FIX_COUNT = 3

_REVIEW_SCHEMA_BASE = {
    "type": "object",
    "required": ["verdict", "scores", "issues", "summary"],
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "scores": {
            "type": "object",
            "required": list(SCORE_KEYS),
            "properties": {k: {"type": "integer", "minimum": 0, "maximum": 10} for k in SCORE_KEYS},
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["t", "kind", "note"],
                "properties": {
                    "t": {"type": "number", "minimum": 0},
                    "kind": {"type": "string", "enum": ISSUE_KINDS},
                    "note": {"type": "string"},
                },
            },
        },
        "summary": {"type": "string"},
        "suggested_fix": {
            "type": "object",
            "required": ["description", "edits"],
            "properties": {
                "description": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["op"],
                        "properties": {
                            "op": {"type": "string", "enum": FIX_OPS},
                            "piece": {"type": "integer", "minimum": 1},
                            "count": {"type": "integer", "minimum": 1, "maximum": MAX_FIX_COUNT},
                            "sentence": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                },
            },
        },
        "caption": {"type": "string"},
    },
}




def review_schema(visual: bool = False) -> dict:
    """Схема ответа отсмотра; оценка и замечания про картинку добавляются только при review_visual."""
    import copy
    schema = copy.deepcopy(_REVIEW_SCHEMA_BASE)
    keys = VISUAL_SCORE_KEYS if visual else SCORE_KEYS
    schema["properties"]["scores"]["required"] = list(keys)
    schema["properties"]["scores"]["properties"] = {k: {"type": "integer", "minimum": 0, "maximum": 10} for k in keys}
    schema["properties"]["issues"]["items"]["properties"]["kind"]["enum"] = list(
        VISUAL_ISSUE_KINDS if visual else ISSUE_KINDS)
    return schema


REVIEW_SCHEMA = review_schema(False)


def verdict_from_scores(scores: dict) -> str:
    """Пороги: reject — любая оценка <= 3; needs_edit — иначе любая < 6; publish — все >= 6."""
    values = list(scores.values())
    if not values:
        return "needs_edit"
    if min(values) <= REJECT_AT_OR_BELOW:
        return "reject"
    if min(values) < EDIT_BELOW:
        return "needs_edit"
    return "publish"


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #

def _ffmpeg() -> str:
    from .pipeline import FFMPEG
    return FFMPEG


def _cv2():
    import sys
    root = Path(__file__).resolve().parents[1]
    for extra in (root / ".tools" / "opencv4", root / ".tools" / "python"):
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.append(str(extra))
    import cv2
    return cv2


def _np():
    import numpy
    return numpy


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", str(text).lower().replace("ё", "е"))


def _clock(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def media_duration(video: Path, ffmpeg: str | None = None) -> float:
    """Длительность из stderr `ffmpeg -i` (ffprobe в сборке может отсутствовать)."""
    result = subprocess.run([ffmpeg or _ffmpeg(), "-hide_banner", "-i", str(video)],
                            capture_output=True, text=True, errors="replace")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
    if not match:
        raise RuntimeError(f"Не удалось определить длительность {video}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


# --------------------------------------------------------------------------- #
# Стыки
# --------------------------------------------------------------------------- #

def splice_points(intervals: list[tuple[float, float]], speed: float = 1.0) -> list[dict]:
    """Стыки склейки на шкале ГОТОВОГО ролика: [{index, t, kind, gap}].

    kind = "seam" — переход между далёкими/переставленными кусками записи, "cut" — вырез внутри непрерывной речи.
    """
    speed = speed if speed and speed > 0 else 1.0
    out, offset = [], 0.0
    for k, ((s0, e0), (s1, _e1)) in enumerate(zip(intervals, intervals[1:]), 1):
        offset += (e0 - s0)
        gap = s1 - e0
        out.append({"index": k, "t": round(offset / speed, 3), "gap": round(gap, 2),
                    "kind": "seam" if (gap < -0.05 or gap > 5.0) else "cut"})
    return out


# --------------------------------------------------------------------------- #
# Кадры и контактные листы
# --------------------------------------------------------------------------- #

def frame_times(duration: float, splices: list[dict], *, step: float = FRAME_STEP,
                max_frames: int = MAX_FRAMES) -> list[dict]:
    """Моменты кадров: сетка раз в `step` + последний кадр + по два на каждом стыке (±SPLICE_OFFSET).

    При избытке кадров сначала уходят «вырезы» (cut), затем равномерно прореживаются; стыки kind=seam сохраняются.
    """
    hi = max(0.0, duration - 0.05)
    grid = [{"t": round(min(hi, i * step), 3), "splice": None, "side": ""}
            for i in range(int(hi // step) + 1)]
    grid.append({"t": round(max(0.0, duration - 0.15), 3), "splice": None, "side": "end"})

    def pair(sp):
        return [{"t": round(min(hi, max(0.0, sp["t"] + d)), 3), "splice": sp["index"], "side": side,
                 "splice_kind": sp["kind"]}
                for d, side in ((-SPLICE_OFFSET, "-"), (SPLICE_OFFSET, "+"))]

    seams = [f for sp in splices if sp["kind"] == "seam" for f in pair(sp)]
    cuts = [sp for sp in splices if sp["kind"] != "seam"]
    chosen = seams + grid
    room = max(0, max_frames - len(chosen))
    if cuts and room >= 2:
        keep = min(len(cuts), room // 2)
        picked = [cuts[int(i * len(cuts) / keep)] for i in range(keep)] if keep < len(cuts) else cuts
        chosen += [f for sp in picked for f in pair(sp)]
    chosen.sort(key=lambda f: (f["t"], f["splice"] is None))
    out: list[dict] = []
    for f in chosen:      # кадры ближе 0.1 с друг к другу — один; стыковые кадры не вытесняются сеткой
        if out and f["t"] - out[-1]["t"] < 0.1:
            if out[-1]["splice"] is None and f["splice"] is not None:
                out[-1] = f
            continue
        out.append(f)
    if len(out) > max_frames:
        regular = [f for f in out if f["splice"] is None]
        spliced = [f for f in out if f["splice"] is not None]
        room = max(0, max_frames - len(spliced))
        regular = [regular[int(i * len(regular) / room)] for i in range(room)] if room and room < len(regular) else regular[:room]
        out = sorted(spliced + regular, key=lambda f: f["t"])[:max_frames]
    return out


def extract_frames(video: Path, out_dir: Path, times: list[dict], *, width: int = 540,
                   ffmpeg: str | None = None) -> list[dict]:
    """Вытаскивает кадры ffmpeg-ом (быстрый seek): [{t, path, splice, side}] без неудавшихся."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg or _ffmpeg()
    frames = []
    for n, item in enumerate(times):
        path = out_dir / f"f{n:03d}_{item['t']:.2f}.jpg"
        if not path.is_file():
            subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{item['t']:.3f}", "-i", str(video),
                 "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if path.is_file() and path.stat().st_size > 0:
            frames.append(dict(item, path=str(path)))
    return frames


def make_contact_sheets(frames: list[dict], out_dir: Path, *, max_width: int = MAX_SHEET_WIDTH,
                        name: str = "sheet", spans_out: list | None = None) -> list[Path]:
    """Склеивает кадры в листы по ~10–12 тайлов; у каждого тайла таймкод, стыки подсвечены красной плашкой.

    Вертикальное видео — сетка 5×2, горизонтальное — 4×3 (ширина листа <= max_width).
    """
    if not frames:
        return []
    cv2, np = _cv2(), _np()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(frames[0]["path"])
    if first is None:
        return []
    h0, w0 = first.shape[:2]
    cols, rows = (5, 2) if h0 > w0 else (4, 3)
    tile_w = max_width // cols
    tile_h = int(round(tile_w * h0 / w0))
    bar = 22
    per_sheet = cols * rows
    sheets: list[Path] = []
    for sheet_no, start in enumerate(range(0, len(frames), per_sheet), 1):
        canvas = np.full((rows * (tile_h + bar), cols * tile_w, 3), 24, dtype=np.uint8)
        for k, item in enumerate(frames[start:start + per_sheet]):
            img = cv2.imread(item["path"])
            if img is None:
                continue
            img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            r, c = divmod(k, cols)
            y, x = r * (tile_h + bar), c * tile_w
            canvas[y + bar:y + bar + tile_h, x:x + tile_w] = img
            spliced = item.get("splice") is not None
            color = (40, 40, 200) if spliced else (70, 70, 70)
            cv2.rectangle(canvas, (x, y), (x + tile_w - 1, y + bar - 1), color, -1)
            label = f"{item['t']:.1f}s"
            if spliced:
                label += f"  SPLICE {item['splice']} ({item['side']}0.3s)"
            cv2.putText(canvas, label, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x, y), (x + tile_w - 1, y + bar + tile_h - 1), (0, 0, 0), 1)
        path = out_dir / f"{name}{sheet_no:02d}.jpg"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        group = frames[start:start + per_sheet]
        if spans_out is not None:
            spans_out.append((group[0]["t"], group[-1]["t"]))
        sheets.append(path)
    return sheets


# --------------------------------------------------------------------------- #
# Аудио
# --------------------------------------------------------------------------- #

def load_audio(video: Path, *, sample_rate: int = 16000, ffmpeg: str | None = None):
    """Моно float32 из готового видео (через ffmpeg, без временных файлов)."""
    np = _np()
    proc = subprocess.run(
        [ffmpeg or _ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(sample_rate), "-f", "f32le", "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype="float32"), sample_rate
    return np.frombuffer(proc.stdout, dtype="<f4").copy(), sample_rate


def _db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-6))


def _rms_db(samples, sr: int, t0: float, t1: float) -> float:
    np = _np()
    a, b = max(0, int(t0 * sr)), min(len(samples), int(t1 * sr))
    if b - a < 8:
        return -120.0
    return _db(float(np.sqrt(np.mean(np.square(samples[a:b])))))


def _speech_level_db(samples, sr: int) -> float:
    """Медианная громкость «речевых» окон (выше шумового пола) — точка отсчёта для провалов и скачков."""
    np = _np()
    win = int(0.05 * sr)
    if len(samples) < win * 4:
        return -30.0
    n = len(samples) // win
    frames = samples[:n * win].reshape(n, win)
    levels = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(np.square(frames), axis=1)), 1e-6))
    floor = float(np.percentile(levels, 20))
    speech = levels[levels > max(floor + 10.0, -50.0)]
    return float(np.median(speech)) if len(speech) else float(np.median(levels))


def _loud_level_db(samples, sr: int, t0: float, t1: float) -> float:
    """Речевой уровень участка: 80-й перцентиль громкости окон по 20 мс (паузы внутри участка не считаются)."""
    np = _np()
    win = int(0.02 * sr)
    seg = samples[max(0, int(t0 * sr)):max(0, int(t1 * sr))]
    n = len(seg) // win
    if n < 3:
        return -120.0
    frames = seg[:n * win].reshape(n, win)
    levels = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(np.square(frames), axis=1)), 1e-6))
    return float(np.percentile(levels, 80))


def silence_runs(samples, sr: int, *, min_len: float = SILENCE_GAP) -> list[tuple[float, float]]:
    """Тихие участки длиннее min_len: громкость окна на >=28 дБ ниже речевой медианы (но не выше -45 дБ)."""
    np = _np()
    win = int(0.02 * sr)
    if len(samples) < win * 4:
        return []
    thr = min(_speech_level_db(samples, sr) - 28.0, -45.0)
    n = len(samples) // win
    frames = samples[:n * win].reshape(n, win)
    levels = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(np.square(frames), axis=1)), 1e-6))
    quiet = levels < thr
    runs, start = [], None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 0.02 >= min_len:
                runs.append((round(start * 0.02, 2), round(i * 0.02, 2)))
            start = None
    return runs


def audio_findings(samples, sr: int, splices: list[dict], actual_words: list[dict], duration: float) -> dict:
    """Автоматические находки по звуку: скачки громкости на стыках, паузы, обрыв в конце, начало с тишины."""
    findings: list[dict] = []
    if len(samples) == 0:
        return {"findings": [{"t": 0.0, "kind": "audio", "severity": "high", "note": "в ролике нет звуковой дорожки"}],
                "speech_level_db": None, "splice_levels": []}
    ref = _speech_level_db(samples, sr)
    levels = []
    for sp in splices:
        t = sp["t"]
        left, right = _rms_db(samples, sr, t - 0.12, t - 0.02), _rms_db(samples, sr, t + 0.02, t + 0.12)
        mid = _rms_db(samples, sr, t - 0.04, t + 0.04)
        loud_l = _loud_level_db(samples, sr, t - LOUD_WINDOW, t - 0.02)
        loud_r = _loud_level_db(samples, sr, t + 0.02, t + LOUD_WINDOW)
        levels.append({"index": sp["index"], "t": t, "left_db": round(left, 1), "right_db": round(right, 1),
                       "mid_db": round(mid, 1), "speech_left_db": round(loud_l, 1), "speech_right_db": round(loud_r, 1)})
        if min(loud_l, loud_r) > -90.0 and abs(loud_l - loud_r) > LOUDNESS_JUMP_DB:
            findings.append({"t": t, "kind": "audio", "severity": "medium",
                             "note": f"скачок громкости речи на стыке {sp['index']}: {loud_l:.0f} -> {loud_r:.0f} дБ"})
        # слово есть и слева, и справа, а в самой точке стыка провал: возможно, слово срезано
        near_left = any(0 <= t - w["end"] <= 0.12 for w in actual_words)
        near_right = any(0 <= w["start"] - t <= 0.12 for w in actual_words)
        if near_left and near_right and mid < ref - 22.0 and min(left, right) > ref - 20.0:
            findings.append({"t": t, "kind": "audio", "severity": "medium",
                             "note": f"провал громкости ровно на стыке {sp['index']} ({mid:.0f} дБ при речи {ref:.0f} дБ)"})
    gaps: list[tuple[float, float]] = [(a, b) for a, b in silence_runs(samples, sr) if a >= 0.05 and b <= duration - 0.05]
    for w0, w1 in zip(actual_words, actual_words[1:]):   # пауза между распознанными словами (словам верим меньше, чем звуку)
        if w1["start"] - w0["end"] > SILENCE_GAP and not any(abs(a - w0["end"]) < 0.4 for a, _ in gaps):
            gaps.append((w0["end"], w1["start"]))
    for a, b in sorted(gaps):
        findings.append({"t": round(a, 2), "kind": "silence", "severity": "medium",
                         "note": f"пауза {b - a:.1f} с ({_clock(a)}–{_clock(b)})"})
    if actual_words:
        first = actual_words[0]["start"]
        if first > 0.8:
            findings.append({"t": 0.0, "kind": "silence", "severity": "medium",
                             "note": f"ролик начинается с тишины: первое слово на {first:.1f} с"})
        last_end = actual_words[-1]["end"]
        tail = duration - last_end
        end_level = _rms_db(samples, sr, duration - 0.25, duration - 0.03)
        if tail < 0.12 and end_level > ref - 14.0:
            findings.append({"t": round(duration - 0.1, 2), "kind": "ending", "severity": "high",
                             "note": f"обрыв в конце: речь не затухает (последнее слово до {last_end:.1f} из {duration:.1f} с, "
                                     f"громкость {end_level:.0f} дБ)"})
    return {"findings": findings, "speech_level_db": round(ref, 1), "splice_levels": levels}


def compare_transcripts(expected: list[dict], actual: list[dict], splices: list[dict], *,
                        near: float = 0.8) -> dict:
    """Сравнивает ожидаемый (из монтажа) и распознанный из ГОТОВОГО видео текст.

    Находки: слова, пропавшие рядом со стыком (обрезаны), слова, добавленные/задвоенные, общий коэффициент совпадения.
    """
    exp = [(_norm(w.get("text", "")), float(w.get("start", 0)), float(w.get("end", 0))) for w in expected]
    act = [(_norm(w.get("text", "")), float(w.get("start", 0)), float(w.get("end", 0))) for w in actual]
    exp = [x for x in exp if x[0]]
    act = [x for x in act if x[0]]
    findings: list[dict] = []
    if not exp or not act:
        return {"findings": findings, "match_ratio": 0.0 if exp else 1.0}
    matcher = difflib.SequenceMatcher(None, [x[0] for x in exp], [x[0] for x in act], autojunk=False)
    times = [sp["t"] for sp in splices]

    def nearest(t: float):
        return min(times, key=lambda x: abs(x - t)) if times else None

    for tag, i0, i1, j0, j1 in matcher.get_opcodes():
        if tag in ("delete", "replace") and i1 - i0 <= 3:
            # пропало ≤3 слов: интересно, только если это рядом со стыком (там слово могли срезать)
            t = exp[i0][1]
            sp_t = nearest(t)
            if sp_t is not None and abs(sp_t - t) <= near and not (tag == "replace" and j1 - j0 >= i1 - i0):
                words = " ".join(x[0] for x in exp[i0:i1])
                findings.append({"t": round(t, 2), "kind": "clipped_word", "severity": "high",
                                 "note": f"возможно срезано слово «{words}» рядом со стыком на {sp_t:.1f} с "
                                         f"(в монтаже есть, в звуке не распознано)"})
    for k in range(len(act) - 1):   # задвоенное слово: одинаковые подряд в звуке, но не в монтаже
        a, b = act[k], act[k + 1]
        if a[0] == b[0] and len(a[0]) >= 2 and b[1] - a[2] < 0.5:
            allowed = any(exp[m][0] == exp[m + 1][0] == a[0] for m in range(len(exp) - 1))
            if not allowed:
                findings.append({"t": round(a[1], 2), "kind": "audio", "severity": "medium",
                                 "note": f"задвоенное слово «{a[0]}»"})
    return {"findings": findings, "match_ratio": round(matcher.ratio(), 3)}


def transcribe_output(video: Path, work_dir: Path, language: str, model_name: str,
                      callback: Callable[[int, str], None] | None = None) -> dict:
    """Пословная расшифровка звука готового ролика (тот же faster-whisper и модель проекта); кэш по размеру файла."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = work_dir / "output_transcript.json"
    stamp = {"size": Path(video).stat().st_size, "model": model_name, "language": language}
    if cache.is_file():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("stamp") == stamp:
                return saved["transcript"]
        except (OSError, ValueError, KeyError):
            pass
    from .pipeline import transcribe_video
    transcript = transcribe_video(Path(video), work_dir / "whisper", language, model_name,
                                  callback or (lambda p, m: None), mode="accurate")
    cache.write_text(json.dumps({"stamp": stamp, "transcript": transcript}, ensure_ascii=False), encoding="utf-8")
    return transcript


# --------------------------------------------------------------------------- #
# Запрос к ИИ
# --------------------------------------------------------------------------- #

def _transcript_lines(words: list[dict], max_words: int = 14, gap: float = 0.45) -> list[str]:
    lines, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > gap or len(cur) >= max_words):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return [f"[{_clock(l[0]['start'])}] " + " ".join(str(w.get("raw") or w.get("text", "")) for w in l) for l in lines]


def piece_layout(candidate: dict, words: list[dict], speed: float = 1.0) -> list[dict]:
    """Куски кандидата с их положением на шкале готового ролика: [{n, role, sentences, start, end}]."""
    speed = speed if speed and speed > 0 else 1.0
    offset, out = 0.0, []
    cuts = candidate.get("cuts") or []
    for n, seg in enumerate(candidate.get("segments") or [], 1):
        ivs = assembly.keep_intervals({"segments": [seg], "cuts": cuts}, words)
        length = assembly.assembled_duration(ivs)
        out.append({"n": n, "role": seg.get("role", "point"),
                    "sentences": [seg.get("sentence_from") or "", seg.get("sentence_to") or ""],
                    "start": round(offset / speed, 2), "end": round((offset + length) / speed, 2)})
        offset += length
    return out


def build_prompt(candidate: dict, layout: list[dict], actual_words: list[dict], splices: list[dict],
                 auto: list[dict], sheets: list[Path] | None, spans: list[tuple[float, float]], duration: float,
                 window: tuple[float, float] | None = None, visual: bool = False) -> str:
    sheets = sheets or []
    pieces = []
    for p in layout:
        a, b = p["sentences"]
        span = f"{a}–{b}" if a else "(без id предложений)"
        pieces.append(f"Кусок {p['n']} [{span}], роль {p['role']}: {_clock(p['start'])}–{_clock(p['end'])} в ролике")
    splice_lines = []
    for sp in splices:
        before = [w for w in actual_words if w["end"] <= sp["t"] + 0.05][-4:]
        after = [w for w in actual_words if w["start"] >= sp["t"] - 0.05][:4]
        kind = "стык кусков (скачок по записи)" if sp["kind"] == "seam" else "вырез внутри речи"
        splice_lines.append(f"- #{sp['index']} t={sp['t']:.2f} с, {kind}: …{' '.join(str(w['text']) for w in before)} ⟶ "
                            f"{' '.join(str(w['text']) for w in after)}…")
    auto_lines = [f"- [{_clock(f['t'])}] ({f.get('severity', 'low')}) {f['note']}" for f in auto] or [
        "- автоматические проверки ничего не нашли"]
    window_line = ""
    if window:
        flag = "" if window[0] <= duration <= window[1] else "  ВНЕ ОКНА — отметь kind=duration"
        window_line = f"ДОПУСТИМАЯ ДЛИТЕЛЬНОСТЬ: {window[0]:.0f}–{window[1]:.0f} с (сейчас {duration:.1f} с){flag}\n"
    sheet_block, visual_score, visual_rule = "", "", ""
    if visual and sheets:
        sheet_lines = []
        for n, path in enumerate(sheets, 1):
            span = f"{_clock(spans[n - 1][0])}–{_clock(spans[n - 1][1])}" if n - 1 < len(spans) else ""
            sheet_lines.append(f"{n}. {Path(path).name} (кадры {span})")
        sheet_block = ("КОНТАКТНЫЕ ЛИСТЫ КАДРОВ (изображения приложены; красная плашка SPLICE n — кадры за 0.3 с до и "
                       "после стыка n):\n" + "\n".join(sheet_lines) + "\n\n")
        visual_score = "- visual — спикер виден и скадрирован, нет чёрных/замороженных кадров, субтитры читаемы;\n"
    scores_line = "hook, idea_disclosure, flow, audio" + (", visual" if visual else "")
    return f"""Ты — шеф-редактор канала коротких видео (Reels). Ты ОТСМАТРИВАЕШЬ ГОТОВЫЙ ролик на слух, как человек перед публикацией. Картинку НЕ оцениваешь: у тебя только расшифровка звука готового ролика с таймкодами, моменты склеек, автоматические находки по звуку, название и тезис. Ролик собран из кусков эфира (монтаж по предложениям), поэтому главные риски — склейки, обрывы фраз и неровный звук.

РОЛИК: «{candidate.get('title', '')}»  | формат: {candidate.get('format', '')}  | длительность {duration:.1f} с
{window_line}ТЕЗИС: {candidate.get('thesis', '') or candidate.get('description', '')}
КУСКИ МОНТАЖА:
{chr(10).join(pieces) or '(один непрерывный отрезок)'}

{sheet_block}РАСШИФРОВКА ЗВУКА ГОТОВОГО РОЛИКА (получена распознаванием самого mp4 с таймкодами):
{chr(10).join(_transcript_lines(actual_words))}

СТЫКИ СКЛЕЙКИ (моменты в готовом ролике, где состыкованы разные места записи):
{chr(10).join(splice_lines) or '(стыков нет)'}

АВТОМАТИЧЕСКИЕ НАХОДКИ по звуку и синхронизации (проверь по тексту; ложные срабатывания возможны):
{chr(10).join(auto_lines)}

ЧТО ОЦЕНИТЬ ({scores_line}; оценки 0–10):
- hook — первые 3 секунды: цепляет ли первая фраза, понятно ли сразу о чём ролик, нет ли повтора слов и обрывка фразы в начале;
- idea_disclosure — раскрыта ли мысль до конца: есть ли развитие и явный вывод, не оборвано ли на середине, нет ли служебного хвоста после вывода («в чате написали», вопрос залу);
- flow — нет ли скачков и резких переходов, которые СЛЫШНО в речи: смена темы/героя на стыке, «рваная» речь (много склеек внутри одной фразы), повтор одних и тех же слов на шве, паузы длиннее 0.6 с;
- audio — нет срезанных слов, щелчков, скачков громкости на стыках, длинных пауз и резкого обрыва в конце;
{visual_score}issues — конкретные замечания: t — секунда в ролике, kind ({'|'.join(VISUAL_ISSUE_KINDS if visual else ISSUE_KINDS)}), note — по-русски, что именно. Не выдумывай замечаний: если всё хорошо, список пуст.

ВЕРДИКТ (его итог пересчитывается по оценкам: reject — любая оценка <= 3; needs_edit — иначе любая < 6; publish — все >= 6):
- publish — можно публиковать как есть;
- needs_edit — идея сильная, но есть поправимый дефект текста/монтажа: лишний хвост, слабый вход, оборванный конец, ненужное предложение в начале;
- reject — исправить правкой нельзя: мысль не раскрыта, ролик непонятен без контекста, звук испорчен.
suggested_fix — ТОЛЬКО если вердикт needs_edit и дефект лечится правкой кусков монтажа. description — что поправить словами; edits — конкретные операции над кусками (нумерация кусков — как в списке «КУСКИ МОНТАЖА», count — число предложений, 1–{MAX_FIX_COUNT}). Правка не должна делать ролик длиннее допустимой длительности:
  drop_last_sentences {{count}} — убрать последние предложения ролика (служебный хвост);
  drop_first_sentences {{count}} — убрать первые предложения ролика (слабый вход);
  extend_piece_start {{piece,count}} — добавить count предыдущих предложений в начало куска (кусок начинается на полуслове);
  extend_piece_end {{piece,count}} — добавить count следующих предложений в конец куска (мысль оборвана);
  trim_piece_start / trim_piece_end {{piece,count}} — убрать предложения с начала/конца куска;
  drop_piece {{piece}} — убрать кусок целиком;
  cut_phrase {{sentence,text}} — вырезать из речи дословную фразу внутри предложения (sentence — id предложения из списка кусков, text — слова ровно как в расшифровке): локальный обрывок, паразитная фраза, повисшая половина предложения на стыке.
ЛОКАЛЬНЫЙ ДЕФЕКТ — ПОДРЕЗАЙ, А НЕ БРАКУЙ: если мешает только обрывок или лишняя фраза, не занижай оценки до 3 и ниже, а верни needs_edit с cut_phrase / trim_*. Все правки идут в естественном порядке записи (extend_* расширяют кусок назад/вперёд по записи); переставлять куски местами нельзя.
Если правка не выражается этими операциями (проблема в звуке, самой идее), suggested_fix не возвращай.
caption — подпись к посту (1–2 коротких предложения по-русски, без хештегов-спама).
summary — вывод одним-двумя предложениями.
Текст распознан автоматически: ошибки написания слов — не дефект монтажа. Отвечай строго JSON по схеме.
"""


def _clean_review(data: dict, visual: bool = False) -> dict:
    keys = VISUAL_SCORE_KEYS if visual else SCORE_KEYS
    scores = {k: max(0, min(10, int((data.get("scores") or {}).get(k, 0)))) for k in keys}
    issues = []
    for x in data.get("issues") or []:
        try:
            issues.append({"t": round(float(x.get("t", 0)), 2), "kind": str(x.get("kind") or "other"),
                           "note": str(x.get("note") or "").strip()})
        except (TypeError, ValueError, AttributeError):
            continue
    issues.sort(key=lambda i: i["t"])
    fix = data.get("suggested_fix") if isinstance(data.get("suggested_fix"), dict) else None
    edits = []
    if fix:
        for e in fix.get("edits") or []:
            if isinstance(e, dict) and e.get("op") in FIX_OPS:
                item = {"op": e["op"], "piece": int(e["piece"]) if e.get("piece") else None,
                        "count": max(1, min(MAX_FIX_COUNT, int(e.get("count") or 1)))}
                if e.get("sentence") or e.get("text"):
                    item.update(sentence=str(e.get("sentence") or "").strip(), text=str(e.get("text") or "").strip())
                edits.append(item)
    verdict = verdict_from_scores(scores)
    if verdict != "needs_edit":
        fix = None
    return {
        "verdict": verdict,
        "scores": scores, "issues": issues, "summary": str(data.get("summary") or "").strip(),
        "suggested_fix": ({"description": str(fix.get("description") or "").strip(), "edits": edits} if fix else None),
        "caption": str(data.get("caption") or "").strip(),
    }


# --------------------------------------------------------------------------- #
# Главная функция отсмотра
# --------------------------------------------------------------------------- #

def review_video(video: Path, *, candidate: dict, transcript: dict, work_dir: Path, options: dict,
                 intervals: list[tuple[float, float]] | None = None, task: str = "video_review",
                 llm: Callable | None = None, transcribe: Callable | None = None,
                 callback: Callable[[int, str], None] | None = None) -> dict:
    """Отсматривает готовый ролик и возвращает отчёт (JSON-совместимый).

    llm и transcribe можно подменить (тесты). Сбой ИИ не роняет рендер: возвращается status="error" с автонаходками.
    """
    t_start = time.monotonic()
    emit = callback or (lambda p, m: None)
    video, work_dir = Path(video), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    speed = float((options or {}).get("playback_speed", 1.0) or 1.0)
    words_all = transcript.get("words") or []
    if intervals is None:
        intervals = assembly.keep_intervals(editorial_playback(candidate), words_all, tail_pad=0.3)
    expected = assembly.remap_transcript(transcript, intervals).get("words") or []
    if speed != 1.0:
        expected = [dict(w, start=w["start"] / speed, end=w["end"] / speed) for w in expected]
    duration = media_duration(video)
    splices = splice_points(intervals, speed)
    timings: dict[str, float] = {}

    visual = bool((options or {}).get("review_visual", False))
    frames, sheets, spans = [], [], []
    if visual:
        emit(5, "Шеф-редактор: достаю кадры")
        t0 = time.monotonic()
        frames = extract_frames(video, work_dir / "frames", frame_times(duration, splices))
        sheets = make_contact_sheets(frames, work_dir / "sheets", spans_out=spans)
        timings["frames"] = round(time.monotonic() - t0, 1)

    emit(25, "Шеф-редактор: слушаю звук ролика")
    t0 = time.monotonic()
    language = (options or {}).get("language", "ru")
    if language == "auto":
        language = transcript.get("language", "ru")
    do_transcribe = transcribe or (lambda v: transcribe_output(v, work_dir, language, (options or {}).get("model", "small")))
    actual = (do_transcribe(video) or {}).get("words") or []
    samples, sr = load_audio(video)
    audio = audio_findings(samples, sr, splices, actual, duration)
    compared = compare_transcripts(expected, actual, splices)
    auto = sorted(audio["findings"] + compared["findings"], key=lambda f: f["t"])
    timings["audio"] = round(time.monotonic() - t0, 1)

    emit(55, "Шеф-редактор: оценивает ролик")
    layout = piece_layout(candidate, words_all, speed)
    window = None
    if options and "min_duration" in options:
        window = (editorial.FINAL_MIN_FACTOR * float(options["min_duration"]),
                  editorial.final_cap(candidate.get("format") or "thesis", float(options.get("max_duration", 90)),
                                      candidate.get("critic")) * editorial.FINAL_OVERFLOW_TOLERANCE)
    prompt = build_prompt(candidate, layout, actual, splices, auto, sheets, spans, duration, window, visual)
    report: dict = {
        "status": "ok", "video": video.name, "duration": round(duration, 2), "frames": len(frames),
        "sheets": [p.name for p in sheets], "splices": splices, "auto_findings": auto,
        "match_ratio": compared["match_ratio"], "speech_level_db": audio["speech_level_db"],
        "pieces": layout, "task": task,
    }
    run = llm
    if run is None:
        from .llm import run_llm_json
        run = run_llm_json
    t0 = time.monotonic()
    try:
        kwargs = {"images": list(sheets)} if sheets else {}
        data = run(prompt, review_schema(visual), work_dir / "llm", task=task, provider=(options or {}).get("provider"),
                   effort="high", **kwargs)
        report.update(_clean_review(data, visual))
        report["provider"] = (options or {}).get("provider")
    except Exception as exc:   # отсмотр — надстройка: ролик уже готов
        report.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "verdict": None, "scores": {},
                       "issues": [{"t": f["t"], "kind": f["kind"], "note": f["note"]} for f in auto],
                       "suggested_fix": None, "summary": "", "caption": ""})
    timings["llm"] = round(time.monotonic() - t0, 1)
    timings["total"] = round(time.monotonic() - t_start, 1)
    report["timings"] = timings
    (work_dir / f"{task}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(100, "Шеф-редактор: готово")
    return report


def editorial_playback(candidate: dict) -> dict:
    """Кандидат в форме, пригодной для assembly (как highlights.candidate_playback, без импорта тяжёлого модуля)."""
    segments = candidate.get("segments") or [
        {"start": float(candidate["start"]), "end": float(candidate["end"]), "role": "point"}]
    return {**candidate, "segments": segments, "cuts": candidate.get("cuts") or []}


# --------------------------------------------------------------------------- #
# Авто-правка: suggested_fix -> новые куски кандидата
# --------------------------------------------------------------------------- #

def fix_is_expressible(fix: dict | None) -> bool:
    return bool(fix and fix.get("edits"))


def apply_suggested_fix(candidate: dict, fix: dict | None, transcript: dict, options: dict | None = None) -> dict | None:
    """Применяет правку к кускам кандидата (по предложениям S0001…). Возвращает обновление кандидата или None.

    Обновление: {segments, cuts, start, end, duration, cuts_removed_seconds}. None — если правку нельзя
    выразить (нет id предложений, неизвестный кусок, результат слишком короткий/длинный или пустой).
    """
    if not fix_is_expressible(fix):
        return None
    words = transcript.get("words") or []
    segments = [dict(s) for s in candidate.get("segments") or []]
    if not words or not segments or any(not s.get("sentence_from") or not s.get("sentence_to") for s in segments):
        return None
    sentences = editorial.build_sentences(transcript)
    idx = {s["id"]: i for i, s in enumerate(sentences)}
    pieces = []
    for seg in segments:
        a, b = idx.get(seg["sentence_from"]), idx.get(seg["sentence_to"])
        if a is None or b is None or a > b:
            return None
        if sentences[a]["w0"] < 0 or abs(float(words[sentences[a]["w0"]]["start"]) - float(seg["start"])) > 0.35:
            return None   # кандидат собран по другой разметке предложений: правка по id ненадёжна
        pieces.append({"a": a, "b": b, "role": seg.get("role", "point")})
    n_sent = len(sentences)
    phrase_cuts: list[tuple[int, int]] = []
    was_chronological = editorial.is_chronological(pieces)
    for edit in fix["edits"]:
        op, count = edit.get("op"), max(1, min(MAX_FIX_COUNT, int(edit.get("count") or 1)))
        n = edit.get("piece")
        if op == "cut_phrase":
            i = idx.get(str(edit.get("sentence") or "").strip().upper())
            loc = editorial._locate_phrase(sentences[i], words, str(edit.get("text") or "")) if i is not None else None
            if loc is None:
                return None   # фраза не найдена: правка ненадёжна
            phrase_cuts.append(loc)
            continue
        if op in ("drop_last_sentences", "drop_first_sentences"):
            for _ in range(count):
                if not pieces:
                    return None
                p = pieces[-1] if op == "drop_last_sentences" else pieces[0]
                if p["a"] == p["b"]:
                    pieces.remove(p)
                else:
                    if op == "drop_last_sentences":
                        p["b"] -= 1
                    else:
                        p["a"] += 1
            if op == "drop_last_sentences" and pieces and pieces[-1]["role"] != "payoff":
                pieces[-1]["role"] = "payoff"
            continue
        if not n or not 1 <= n <= len(pieces):
            return None
        p = pieces[n - 1]
        if op == "extend_piece_start":
            p["a"] = max(0, p["a"] - count)
        elif op == "extend_piece_end":
            p["b"] = min(n_sent - 1, p["b"] + count)
        elif op == "trim_piece_start":
            p["a"] += count
        elif op == "trim_piece_end":
            p["b"] -= count
        elif op == "drop_piece":
            p["a"], p["b"] = 1, 0   # метка удаления
        else:
            return None
    pieces = [p for p in pieces if p["a"] <= p["b"]]
    if not pieces:
        return None
    if was_chronological:
        # правка не должна нарушать естественный порядок (расширение куска назад могло наехать на соседний — склеим)
        pieces = editorial._natural_order(pieces)
        if pieces and pieces[-1]["role"] != "payoff" and candidate.get("segments") and \
                candidate["segments"][-1].get("role") == "payoff" and pieces[-1]["b"] == idx.get(candidate["segments"][-1]["sentence_to"]):
            pieces[-1]["role"] = "payoff"
    else:
        pieces = editorial._merge_adjacent(pieces)
    opts = options or {}
    low = max(8.0, 0.85 * float(opts.get("min_duration", 30)))
    high = editorial.final_cap(candidate.get("format") or "thesis", float(opts.get("max_duration", 90)),
                               candidate.get("critic")) * editorial.FINAL_OVERFLOW_TOLERANCE

    def build(pcs):
        new_segments = []
        for p in pcs:
            sa, sb = sentences[p["a"]], sentences[p["b"]]
            if sa["w0"] < 0:
                return None
            new_segments.append({"start": round(float(words[sa["w0"]]["start"]), 3),
                                 "end": round(float(words[sb["w1"]]["end"]), 3), "role": p["role"],
                                 "sentence_from": sa["id"], "sentence_to": sb["id"]})
        cuts = [c for c in candidate.get("cuts") or []
                if any(s["start"] - 1e-6 <= float(c["start"]) and float(c["end"]) <= s["end"] + 1e-6 for s in new_segments)]
        for w0, w1 in phrase_cuts:   # локальные вырезы по фразам из отсмотра
            cut = {"start": round(float(words[w0]["start"]), 3), "end": round(float(words[w1]["end"]), 3),
                   "reason": editorial.FORCED_REASON}
            if any(s["start"] - 1e-6 <= cut["start"] and cut["end"] <= s["end"] + 1e-6 for s in new_segments):
                cuts.append(cut)
        draft = {"segments": new_segments, "cuts": cuts}
        return new_segments, cuts, assembly.assembled_duration(assembly.keep_intervals(draft, words))

    built = build(pieces)
    if built is None:
        return None
    # правка не должна раздувать ролик: сверх окна режем context/example (hook/payoff не трогаем), иначе отказ
    for _ in range(3):
        if built[2] <= high:
            break
        speech = editorial._pieces_duration(pieces, sentences)
        ratio = built[2] / max(1e-6, speech)
        pieces = editorial._trim_to_window(pieces, sentences, high / max(0.5, ratio) * 0.98, low)
        if not pieces:
            return None
        built = build(pieces)
        if built is None:
            return None
    new_segments, cuts, duration = built
    if not low <= duration <= high:
        return None
    return {"segments": new_segments, "cuts": cuts,
            "start": round(min(s["start"] for s in new_segments), 3),
            "end": round(max(s["end"] for s in new_segments), 3),
            "duration": round(duration, 3),
            "cuts_removed_seconds": round(sum(float(c["end"]) - float(c["start"]) for c in cuts), 2)}


def score_sum(review: dict | None) -> int:
    return sum((review or {}).get("scores", {}).values()) if review else 0
