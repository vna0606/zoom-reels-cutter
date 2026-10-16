"""Детерминированная сборка клипа из нескольких фрагментов источника.

Модуль использует только стандартную библиотеку: тяжёлые зависимости
(faster_whisper, cv2) сюда не попадают.
"""
from __future__ import annotations

import array
import bisect
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

Interval = "tuple[float, float]"


class PipelineError(RuntimeError):
    """Локальная замена reel_factory.pipeline.PipelineError (без тяжёлых импортов)."""


def _pipeline_error_class():
    """Если pipeline уже импортирован, используем его класс, чтобы ошибки ловились единообразно."""
    import sys

    module = sys.modules.get("reel_factory.pipeline")
    cls = getattr(module, "PipelineError", None) if module else None
    return cls or PipelineError


def _fail(message: str):
    return _pipeline_error_class()(message)


# --------------------------------------------------------------------------- #
# Интервалы
# --------------------------------------------------------------------------- #

def source_spans(candidate: dict) -> list[tuple[float, float]]:
    """Сегменты кандидата как пары (start, end) в порядке воспроизведения."""
    return [(float(s["start"]), float(s["end"])) for s in candidate.get("segments", [])]


def assembled_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return float(sum(max(0.0, e - s) for s, e in intervals))


def _subtract(span: tuple[float, float], cuts: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    result = [span]
    for cs, ce in sorted(cuts):
        nxt = []
        for s, e in result:
            if ce <= s or cs >= e:
                nxt.append((s, e))
                continue
            if cs > s:
                nxt.append((s, cs))
            if ce < e:
                nxt.append((ce, e))
        result = nxt
    return result


# Пауза живой речи короче этого порога не трогается (каждая сжатая пауза = склейка); длиннее — сжимается до max_gap.
PAUSE_COMPRESS_FROM = 0.9


def keep_intervals(
    candidate: dict,
    words: list[dict],
    *,
    pad: float = 0.05,
    max_gap: float = 0.35,
    min_piece: float = 0.12,
    tail_pad: float = 0.0,
    compress_from: float = PAUSE_COMPRESS_FROM,
) -> list[tuple[float, float]]:
    """Интервалы источника в порядке воспроизведения, которые нужно оставить.

    tail_pad > 0 добавляет в конец последнего интервала «воздух» после последнего слова (не заходя в следующее слово):
    без него ролик обрывается на самом звуке слова, и отсмотр слышит резкий обрыв.
    """
    segments = candidate.get("segments") or []
    cuts = [(float(c["start"]), float(c["end"])) for c in candidate.get("cuts") or []]
    out: list[tuple[float, float]] = []

    if not words:
        for seg in segments:
            out.extend(_subtract((float(seg["start"]), float(seg["end"])), cuts))
        return _finalize(out, min_piece)

    starts = [float(w["start"]) for w in words]
    ends = [float(w["end"]) for w in words]
    eps = 1e-3

    def removed(idx: int) -> bool:
        mid = (starts[idx] + ends[idx]) / 2.0
        return any(cs - eps <= mid <= ce + eps for cs, ce in cuts)

    half_gap = max_gap / 2.0
    for seg in segments:
        s0, s1 = float(seg["start"]), float(seg["end"])
        lo = bisect.bisect_left(starts, s0 - eps)
        idxs = []
        i = lo
        while i < len(words) and starts[i] < s1 - eps:
            if ends[i] <= s1 + eps:
                idxs.append(i)
            i += 1
        if not idxs:
            continue
        kept = [i for i in idxs if not removed(i)]
        if not kept:
            continue

        def left_bound(first: int) -> float:
            """Левая граница: не заходим в предыдущее слово (берём середину паузы)."""
            start = starts[first] - pad
            if first > 0:
                start = max(start, (ends[first - 1] + starts[first]) / 2.0)
            return max(0.0, min(start, starts[first]))

        def right_bound(last: int) -> float:
            end = ends[last] + pad
            if last + 1 < len(words):
                end = min(end, (ends[last] + starts[last + 1]) / 2.0)
            return max(end, ends[last])

        cur_start = left_bound(kept[0])
        for a, b in zip(kept, kept[1:]):
            gap = starts[b] - ends[a]
            removed_between = b - a > 1
            if not removed_between and gap < compress_from:
                continue
            if removed_between:
                # граница - середина паузы рядом с вырезанным словом (там тишина)
                reach = pad if gap < compress_from else half_gap
                end_a = min(ends[a] + reach, (ends[a] + starts[a + 1]) / 2.0)
                start_b = max(starts[b] - reach, (ends[b - 1] + starts[b]) / 2.0)
            else:
                # длинная пауза: оставляем max_gap/2 с каждой стороны
                end_a = ends[a] + half_gap
                start_b = starts[b] - half_gap
            out.append((cur_start, max(end_a, ends[a])))
            cur_start = min(start_b, starts[b])
        out.append((cur_start, right_bound(kept[-1])))

    result = _finalize(out, min_piece)
    if tail_pad > 0 and result:
        last_s, last_e = result[-1]
        nxt = bisect.bisect_left(starts, last_e - eps)
        limit = starts[nxt] - 0.04 if nxt < len(starts) else last_e + tail_pad
        result[-1] = (last_s, max(last_e, min(last_e + tail_pad, limit)))
    return result


def _finalize(intervals: list[tuple[float, float]], min_piece: float) -> list[tuple[float, float]]:
    """Склеивает перекрывающиеся соседние в порядке воспроизведения, отбрасывает короткие."""
    merged: list[list[float]] = []
    for s, e in intervals:
        if e <= s:
            continue
        if merged and merged[-1][0] - 1e-6 <= s <= merged[-1][1] + 1e-6 and e >= merged[-1][0]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][0] = min(merged[-1][0], s)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged if b - a >= min_piece]


# --------------------------------------------------------------------------- #
# Границы резов по реальной громкости: кусок не начинается и не кончается тишиной
# --------------------------------------------------------------------------- #

FRAME_SEC = 0.01
EDGE_SILENCE_MAX = 0.15       # допустимая «тишина» на краю куска (с)
EDGE_SILENCE_DB = 28.0        # тишина = на столько дБ ниже речевой медианы
LEAD_KEEP = 0.07              # оставляем перед первым звуком (с)
BURST_MAX = 0.30             # короткий обрывок звука на краю куска (с) ...
BURST_PAUSE = 0.30           # ... после которого пауза не короче (с) — считаем чужим хвостом
TAIL_KEEP = 0.12              # оставляем после последнего звука (с): затухание слова


def _decode_levels(ffmpeg: str, source: Path, start: float, length: float) -> list[float]:
    """Громкость (дБ, RMS по окнам 10 мс) куска источника: моно 16 кГц через ffmpeg."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{length:.3f}",
           "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    except Exception:
        return []
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) // 2 * 2])
    win = int(16000 * FRAME_SEC)
    out = []
    for i in range(0, len(samples) - win + 1, win):
        chunk = samples[i:i + win]
        rms = math.sqrt(sum(x * x for x in chunk) / win) / 32768.0
        out.append(20.0 * math.log10(max(rms, 1e-6)))
    return out


def trim_edge_silence(levels: Sequence[float], t0: float, interval: tuple[float, float], ref_db: float | None = None,
                      *, trim_end: bool = True) -> tuple[float, float]:
    """Сдвигает границы интервала внутрь, если он начинается/кончается тишиной длиннее EDGE_SILENCE_MAX.

    levels — громкость окон по 10 мс, начиная с абсолютного времени t0 источника. Без речи в интервале — как есть.
    """
    s, e = interval
    i0 = max(0, int(round((s - t0) / FRAME_SEC)))
    i1 = min(len(levels), int(round((e - t0) / FRAME_SEC)))
    if i1 - i0 < 6:
        return interval
    window = list(levels[i0:i1])
    if ref_db is None:
        loud = sorted(window)
        ref_db = loud[int(len(loud) * 0.75)]
    thr = ref_db - EDGE_SILENCE_DB
    loud = [v > thr for v in window]
    first = next((k for k, v in enumerate(loud) if v), None)
    if first is None:
        return interval
    last = len(loud) - 1 - next(k for k, v in enumerate(reversed(loud)) if v)

    def runs(flags):
        """[(значение, начало, конец)] подряд идущих одинаковых флагов."""
        out, st = [], 0
        for k in range(1, len(flags) + 1):
            if k == len(flags) or flags[k] != flags[st]:
                out.append((flags[st], st, k))
                st = k
        return out

    rs = runs(loud)
    burst_max, pause_min = int(BURST_MAX / FRAME_SEC), int(BURST_PAUSE / FRAME_SEC)
    # обрывок звука перед паузой на самом краю («хвост» чужого слова из-за неточных таймкодов) — отрезаем вместе с паузой
    if len(rs) >= 3 and rs[0][0] and rs[0][2] - rs[0][1] <= burst_max and not rs[1][0] \
            and rs[1][2] - rs[1][1] >= pause_min:
        first = rs[1][2]
    if trim_end and len(rs) >= 3 and rs[-1][0] and rs[-1][2] - rs[-1][1] <= burst_max and not rs[-2][0] \
            and rs[-2][2] - rs[-2][1] >= pause_min:
        last = rs[-2][1] - 1
    new_s, new_e = s, e
    if first * FRAME_SEC > EDGE_SILENCE_MAX:
        new_s = s + max(0.0, first * FRAME_SEC - LEAD_KEEP)
    if trim_end and (len(window) - 1 - last) * FRAME_SEC > EDGE_SILENCE_MAX:
        new_e = s + (last + 1) * FRAME_SEC + TAIL_KEEP
    new_e = min(e, max(new_e, new_s + 0.3))
    if new_e - new_s < 0.3:
        return interval
    return (round(new_s, 3), round(new_e, 3))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def piece_gains(levels: Sequence[float | None], *, max_gain: float = 9.0, deadband: float = 1.5) -> list[float]:
    """Поправка громкости (дБ) для каждого куска к общему речевому уровню (медиана); None — без поправки."""
    known = [v for v in levels if v is not None]
    if not known:
        return [0.0] * len(levels)
    target = sorted(known)[len(known) // 2]
    out = []
    for v in levels:
        g = 0.0 if v is None else max(-max_gain, min(max_gain, target - v))
        out.append(round(g, 1) if abs(g) >= deadband else 0.0)
    return out


def refine_intervals(ffmpeg: str, source: Path, intervals: Sequence[tuple[float, float]], *,
                     keep_last_tail: bool = True, with_levels: bool = False):
    """Подрезает тихие края кусков по реальному звуку источника (последний кусок сохраняет «воздух» в конце).

    with_levels=True возвращает ещё и речевой уровень (75-й перцентиль громкости окон, дБ) каждого куска.
    """
    out: list[tuple[float, float]] = []
    speech: list[float | None] = []
    n = len(intervals)
    for idx, (s, e) in enumerate(intervals):
        margin = 0.05
        base = max(0.0, s - margin)
        levels = _decode_levels(ffmpeg, Path(source), base, (e - base) + margin)
        if not levels:
            out.append((s, e))
            speech.append(None)
            continue
        new = trim_edge_silence(levels, base, (s, e), None, trim_end=not (keep_last_tail and idx == n - 1))
        out.append(new)
        i0, i1 = int((new[0] - base) / FRAME_SEC), int((new[1] - base) / FRAME_SEC)
        window = levels[i0:i1] or levels
        speech.append(_percentile(window, 0.75) if len(window) >= 20 else None)
    return (out, speech) if with_levels else out


QUIET_WINDOW = 0.04           # ищем самые тихие 40 мс внутри паузы
START_REACH = 0.15            # начало куска не дальше этого от первого слова
END_REACH = 0.20              # конец куска не дальше этого от последнего слова


def quietest_point(levels: Sequence[float], t0: float, lo: float, hi: float, window: float = QUIET_WINDOW) -> float | None:
    """Центр самого тихого окна `window` секунд в [lo, hi] (levels — дБ окон по 10 мс от абсолютного времени t0)."""
    n = max(1, int(round(window / FRAME_SEC)))
    i0 = max(0, int(round((lo - t0) / FRAME_SEC)))
    i1 = min(len(levels), int(round((hi - t0) / FRAME_SEC)))
    if i1 - i0 < n:
        return None
    best, best_i = None, None
    for i in range(i0, i1 - n + 1):
        value = sum(levels[i:i + n]) / n
        if best is None or value < best - 1e-9:
            best, best_i = value, i
    return t0 + (best_i + n / 2.0) * FRAME_SEC


def snap_to_quiet(ffmpeg: str, source: Path, intervals: Sequence[tuple[float, float]], words: Sequence[dict], *,
                  keep_last_tail: bool = True) -> list[tuple[float, float]]:
    """Границы кусков переносятся в середину реальных пауз: самые тихие 40 мс между соседними словами.

    Начало куска — в паузе перед первым словом (не дальше START_REACH от него), конец — в паузе после последнего слова
    (не дальше END_REACH). Если рядом нет паузы (слова вплотную) — граница остаётся как была.
    """
    if not words:
        return [tuple(x) for x in intervals]
    starts = [float(w["start"]) for w in words]
    ends = [float(w["end"]) for w in words]
    out: list[tuple[float, float]] = []
    n = len(intervals)
    for idx, (s, e) in enumerate(intervals):
        new_s, new_e = s, e
        fw = bisect.bisect_left(starts, s - 0.001)
        lw = bisect.bisect_right(ends, e + 0.001) - 1
        if 0 < fw < len(words) and starts[fw] < e:
            lo = max(ends[fw - 1] + 0.005, starts[fw] - START_REACH)
            hi = starts[fw] - 0.02
            if hi - lo >= QUIET_WINDOW:
                base = lo - 0.05
                lv = _decode_levels(ffmpeg, Path(source), base, hi - base + 0.05)
                c = quietest_point(lv, base, lo, hi) if lv else None
                if c is not None:
                    new_s = round(c, 3)
        if lw >= 0 and lw + 1 < len(words) and ends[lw] > new_s and not (keep_last_tail and idx == n - 1):
            lo = ends[lw] + 0.02
            hi = min(starts[lw + 1] - 0.01, ends[lw] + END_REACH)
            if hi - lo >= QUIET_WINDOW:
                base = lo - 0.05
                lv = _decode_levels(ffmpeg, Path(source), base, hi - base + 0.05)
                c = quietest_point(lv, base, lo, hi) if lv else None
                if c is not None:
                    new_e = round(c, 3)
        if new_e - new_s < 0.3:
            new_s, new_e = s, e
        out.append((new_s, new_e))
    return out


EDGE_LEVEL_WINDOW = 0.6


def edge_levels(ffmpeg: str, source: Path, intervals: Sequence[tuple[float, float]]) -> list[tuple[float | None, float | None]]:
    """Речевой уровень (p80 громкости окон 10 мс, дБ) в первых и последних 0.6 с каждого куска (после подрезки)."""
    out: list[tuple[float | None, float | None]] = []
    for s, e in intervals:
        lv = _decode_levels(ffmpeg, Path(source), s, e - s)
        if len(lv) < 20:
            out.append((None, None))
            continue
        w = max(10, min(int(EDGE_LEVEL_WINDOW / FRAME_SEC), len(lv) // 2))
        out.append((_percentile(lv[:w], 0.8), _percentile(lv[-w:], 0.8)))
    return out


def seam_gains(edges: Sequence[tuple[float | None, float | None]], *, max_step: float = 10.0,
               max_total: float = 12.0, deadband: float = 1.0) -> list[float]:
    """Поправки громкости кусков так, чтобы уровень на каждом стыке совпал: g[i+1] = конец(i)+g[i] - начало(i+1).

    Шаг на стык ограничен max_step, накопленная поправка — max_total (после центрирования по медиане).
    """
    if not edges:
        return []
    gains = [0.0]
    for (_, end_prev), (start_next, _) in zip(edges, edges[1:]):
        if end_prev is None or start_next is None:
            gains.append(gains[-1])
            continue
        step = max(-max_step, min(max_step, end_prev - start_next))
        gains.append(gains[-1] + step)
    if not gains:
        return []
    ordered = sorted(gains)
    mid = ordered[len(ordered) // 2]
    out = [max(-max_total, min(max_total, g - mid)) for g in gains]
    return [round(g, 1) if abs(g) >= deadband else 0.0 for g in out]


def splice_loudness(samples: Sequence[float], sr: int, times: Sequence[float], window: float = 0.12) -> list[dict]:
    """Громкость (дБ RMS) слева и справа от каждого стыка: окна (t-window, t-0.02) и (t+0.02, t+window)."""
    def rms(a: float, b: float) -> float:
        i, j = max(0, int(a * sr)), min(len(samples), int(b * sr))
        if j - i < 8:
            return -120.0
        seg = samples[i:j]
        return 20.0 * math.log10(max(math.sqrt(sum(x * x for x in seg) / len(seg)), 1e-6))
    return [{"t": t, "left_db": round(rms(t - window, t - 0.02), 1), "right_db": round(rms(t + 0.02, t + window), 1)}
            for t in times]


# --------------------------------------------------------------------------- #
# Транскрипт на новой временной шкале
# --------------------------------------------------------------------------- #

def remap_transcript(transcript: dict, intervals: Sequence[tuple[float, float]]) -> dict:
    words = transcript.get("words") or []
    segs = transcript.get("segments") or []
    seg_starts = [float(s["start"]) for s in segs]

    def seg_index(t: float) -> int:
        if not segs:
            return 0
        i = bisect.bisect_right(seg_starts, t) - 1
        return max(0, i)

    out_words: list[dict] = []
    word_seg: list[int] = []
    offset = 0.0
    for s, e in intervals:
        for w in words:
            ws, we = float(w["start"]), float(w["end"])
            mid = (ws + we) / 2.0
            if not (s <= mid <= e):
                continue
            nw = dict(w)
            nw["start"] = round(offset + max(ws, s) - s, 3)
            nw["end"] = round(offset + min(we, e) - s, 3)
            if nw["end"] < nw["start"]:
                nw["end"] = nw["start"]
            out_words.append(nw)
            word_seg.append(seg_index(mid))
        offset += e - s

    out_segments: list[dict] = []
    group: list[int] = []

    def flush() -> None:
        if not group:
            return
        chunk = [out_words[i] for i in group]
        out_segments.append({
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join(str(w["text"]) for w in chunk).strip(),
        })

    for i, w in enumerate(out_words):
        if group:
            prev = out_words[group[-1]]
            if word_seg[i] != word_seg[group[-1]] or w["start"] - prev["end"] > 0.8 or w["start"] < prev["start"]:
                flush()
                group = []
        group.append(i)
    flush()

    result = {k: v for k, v in transcript.items() if k not in {"words", "segments", "text"}}
    result["words"] = out_words
    result["segments"] = out_segments
    result["text"] = " ".join(str(w["text"]) for w in out_words).strip()
    return result


# --------------------------------------------------------------------------- #
# ffmpeg
# --------------------------------------------------------------------------- #

def _ffprobe_for(ffmpeg: str) -> str | None:
    path = Path(ffmpeg)
    for name in ("ffprobe", "ffprobe.exe"):
        sibling = path.with_name(name)
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe")


def _probe(ffprobe: str | None, source: Path) -> dict:
    if not ffprobe:
        return {}
    try:
        res = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type:format=duration",
             "-of", "json", str(source)],
            capture_output=True, text=True, timeout=120,
        )
        return json.loads(res.stdout or "{}")
    except Exception:
        return {}


# Итоговое выравнивание громкости собранного аудио (после посекционных fade-ов): скачки на стыках исчезают.
AUDIO_NORMALIZERS = {
    "loudnorm": "loudnorm=I=-16:TP=-1.5:LRA=11",
    "dynaudnorm": "dynaudnorm=f=150:g=15:p=0.9:m=10",
}
DEFAULT_NORMALIZE = "loudnorm"

# Режимы звука: "raw" — склейка как есть (только короткие fade на стыках), "leveled" — выравнивание уровней + loudnorm.
AUDIO_MODES = ("raw", "leveled")
DEFAULT_AUDIO_MODE = "raw"
RAW_CROSSFADE = 0.015
_UNSET = object()


def normalize_audio_mode(value) -> str:
    mode = str(value or DEFAULT_AUDIO_MODE).strip().lower()
    return mode if mode in AUDIO_MODES else DEFAULT_AUDIO_MODE


def build_filter_script(intervals: Sequence[tuple[float, float]], *, has_audio: bool,
                        crossfade: float, max_run_gap: float = 20.0, margin: float = 0.5,
                        normalize: str | None = None, gains: Sequence[float] | None = None):
    """Возвращает (input_args_list, filter_script).

    Подряд идущие по времени интервалы объединяются в «запуски», у каждого свой
    вход с быстрым `-ss`, чтобы не декодировать десятки минут между далёкими
    фрагментами.
    """
    runs: list[list[int]] = []
    for idx, (s, e) in enumerate(intervals):
        if runs:
            ps, pe = intervals[runs[-1][-1]]
            if s >= ps and s - pe <= max_run_gap:
                runs[-1].append(idx)
                continue
        runs.append([idx])

    inputs: list[tuple[float, float]] = []
    lines: list[str] = []
    vlabels = [""] * len(intervals)
    alabels = [""] * len(intervals)
    for r, run in enumerate(runs):
        base = max(0.0, intervals[run[0]][0] - margin)
        run_end = max(intervals[i][1] for i in run) + margin
        inputs.append((base, run_end - base))
        n = len(run)
        if n > 1:
            lines.append(f"[{r}:v]split={n}" + "".join(f"[rv{r}_{k}]" for k in range(n)))
            if has_audio:
                lines.append(f"[{r}:a]asplit={n}" + "".join(f"[ra{r}_{k}]" for k in range(n)))
        for k, i in enumerate(run):
            s, e = intervals[i]
            rs, re_ = s - base, e - base
            vin = f"[rv{r}_{k}]" if n > 1 else f"[{r}:v]"
            lines.append(f"{vin}trim=start={rs:.6f}:end={re_:.6f},setpts=PTS-STARTPTS[v{i}]")
            vlabels[i] = f"[v{i}]"
            if has_audio:
                ain = f"[ra{r}_{k}]" if n > 1 else f"[{r}:a]"
                d = e - s
                cf = max(0.0, min(crossfade, d / 2.0))
                fade = ""
                if cf > 0:
                    fade = f",afade=t=in:st=0:d={cf:.4f},afade=t=out:st={d - cf:.6f}:d={cf:.4f}"
                gain = ""
                if gains and i < len(gains) and abs(gains[i]) >= 0.05:
                    gain = f",volume={gains[i]:.2f}dB"
                lines.append(f"{ain}atrim=start={rs:.6f}:end={re_:.6f},asetpts=PTS-STARTPTS{gain}{fade}[a{i}]")
                alabels[i] = f"[a{i}]"
    n_all = len(intervals)
    if has_audio:
        order = "".join(vlabels[i] + alabels[i] for i in range(n_all))
        if normalize:
            lines.append(f"{order}concat=n={n_all}:v=1:a=1[outv][cata]")
            lines.append(f"[cata]{AUDIO_NORMALIZERS.get(normalize, normalize)}[outa]")
        else:
            lines.append(f"{order}concat=n={n_all}:v=1:a=1[outv][outa]")
    else:
        order = "".join(vlabels)
        lines.append(f"{order}concat=n={n_all}:v=1:a=0[outv]")
    return inputs, ";\n".join(lines)


def assemble_clip(
    ffmpeg: str,
    source: Path,
    intervals: Sequence[tuple[float, float]],
    destination: Path,
    *,
    crossfade: float | None = None,
    normalize=_UNSET,
    gains: Sequence[float] | None = None,
    audio_mode: str = DEFAULT_AUDIO_MODE,
    callback: Callable[[int, str], None] | None = None,
) -> dict:
    """audio_mode="raw" (по умолчанию): без gain/loudnorm/dynaudnorm, только fade ~15 мс на каждом куске.

    audio_mode="leveled": поправки громкости кусков (gains), fade 25 мс и итоговый loudnorm (поведение v4).
    """
    audio_mode = normalize_audio_mode(audio_mode)
    if audio_mode == "raw":
        normalize, gains = None, None
        crossfade = RAW_CROSSFADE if crossfade is None else crossfade
    else:
        crossfade = 0.025 if crossfade is None else crossfade
        normalize = DEFAULT_NORMALIZE if normalize is _UNSET else normalize
    intervals = [(float(s), float(e)) for s, e in intervals if e > s]
    if not intervals:
        raise _fail("Нет фрагментов для сборки клипа.")
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffprobe = _ffprobe_for(ffmpeg)
    info = _probe(ffprobe, source)
    streams = [s.get("codec_type") for s in info.get("streams", [])]
    has_audio = ("audio" in streams) if streams else True

    inputs, script = build_filter_script(intervals, has_audio=has_audio, crossfade=crossfade, normalize=normalize,
                                       gains=gains)
    script_path = destination.with_name(destination.stem + ".filter.txt")
    script_path.write_text(script, encoding="utf-8")
    total = assembled_duration(intervals)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
    for base, length in inputs:
        cmd += ["-ss", f"{base:.6f}", "-t", f"{length:.6f}", "-i", str(source)]
    cmd += ["-filter_complex_script", str(script_path), "-map", "[outv]"]
    if has_audio:
        cmd += ["-map", "[outa]"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", str(destination)]

    err_file = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, text=True, errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            m = re.match(r"out_time_(?:us|ms)=(\d+)", line.strip())
            if m and callback and total > 0:
                seconds = int(m.group(1)) / 1_000_000.0
                callback(min(99, int(seconds / total * 100)), "Сборка клипа")
        code = proc.wait()
        err_file.seek(0)
        stderr = err_file.read().decode("utf-8", errors="replace")
    finally:
        err_file.close()
        try:
            script_path.unlink()
        except OSError:
            pass
    if code != 0 or not destination.is_file():
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        raise _fail(f"Не удалось собрать клип из фрагментов (ffmpeg, код {code}).\n{tail}")
    if callback:
        callback(100, "Клип собран")

    measured = total
    out_info = _probe(ffprobe, destination)
    try:
        measured = float(out_info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        pass
    return {"duration": measured, "pieces": len(intervals)}
