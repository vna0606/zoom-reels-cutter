"""Пакетный прогон «эфир целиком -> ролики» одной командой (см. run_reels.py и README, «Запуск на новом эфире»).

Те же функции библиотеки, что и в веб-приложении: transcribe_video -> analyze_transcript (смысловой монтаж) ->
normalize_candidates -> render_highlight_reviewed (рендер + отсмотр готового видео по содержанию и звуку) ->
отчёт report.json и галерея index.html. Возобновляется с кэшей: транскрипт, ответы ИИ (editorial/*.json), готовые ролики
(_state/*.json) пересчитываются только если их нет.
"""
from __future__ import annotations

import html
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from . import editorial
from .assembly import keep_intervals, remap_transcript

DEFAULT_OPTIONS = {
    # broadcast: вертикаль из исходного репозитория — вырезает говорящих спикеров Zoom и рисует графику HyperFrames
    "project_mode": "broadcast", "speaker_count": 2, "visual_preset": "editorial_lime", "playback_speed": 1.0,
    "language": "ru", "model": "small", "provider": "claude",
    "min_duration": 30, "max_duration": 90,
    "max_clips": 0, "all_clips": True,           # без лимита: ролик остаётся, если мысль цельная
    "asr_provider": "auto", "parallel": 3, "tighten": True, "keep_disfluencies": True, "subtitle_style": "pill",
    "chief_editor": True, "coverage": True,
    "video_review": True, "video_review_autofix": True, "review_visual": False,   # отсмотр: только содержание и звук
    "audio_polish": True, "audio_mode": "raw",                                      # звук склейки без gain/loudnorm
}

VERDICT_RU = {"publish": "можно публиковать", "needs_edit": "нужна правка", "reject": "не публиковать", None: "не отсмотрен"}
SCORE_RU = {"hook": "хук", "idea_disclosure": "раскрытие мысли", "flow": "переходы", "visual": "визуал", "audio": "звук"}
CRITIC_RU = {"hook_strength": "хук", "completeness": "полнота", "clarity": "ясность", "standalone": "самодостаточность",
             "single_idea": "одна мысль"}
FORMAT_RU = {"list": "Перечень", "thesis": "Тезис", "story": "История", "advice": "Совет", "contrast": "Контраст"}
STAGE_RU = {"compose": "компоновка", "critic_round1": "критик (1-й раунд)", "critic_round2": "критик (2-й раунд)",
            "repair": "ремонт", "repair_chief": "ремонт по замечанию главреда", "chief_editor": "главный редактор",
            "duration": "длительность", "opening": "начало ролика", "ranking": "лимит роликов", "review": "отсмотр видео"}
MOMENT_RU = {"quote": "цитата", "audience": "реплика зала", "story": "живая история", "claim": "резкое утверждение"}


def build_options(**overrides) -> dict:
    options = dict(DEFAULT_OPTIONS)
    options.update({k: v for k, v in overrides.items() if v is not None})
    options["all_clips"] = bool(options.get("all_clips")) or int(options.get("max_clips", 0)) <= 0
    return options


class RunLog:
    """Журнал прогона: <out>/run.log (дописывается) и stdout."""

    def __init__(self, path: Path, echo: bool = True):
        self.path, self.echo = Path(path), echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print(line, flush=True)


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mm(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def final_text(transcript: dict, candidate: dict) -> str:
    from .highlights import candidate_playback
    intervals = keep_intervals(candidate_playback(candidate), transcript["words"])
    return " ".join(w.get("raw") or w["text"] for w in remap_transcript(transcript, intervals)["words"])


# --------------------------------------------------------------------------- #
# Этапы
# --------------------------------------------------------------------------- #

def stage_transcribe(video: Path, out: Path, options: dict, log: RunLog, timings: dict) -> dict:
    path = out / "transcript.json"
    cached = _read_json(path)
    if cached and cached.get("words"):
        log(f"Транскрипт из кэша: {len(cached['words'])} слов")
        return cached
    from .pipeline import normalize_glossary, transcribe_video
    started = time.monotonic()
    last = [-10]

    def progress(percent: int, message: str) -> None:
        if percent >= last[0] + 5 or percent >= 100:
            last[0] = percent
            log(f"транскрипция {percent}%: {message}")

    from .pipeline import resolve_asr_provider
    engine = resolve_asr_provider(options.get("asr_provider", "auto"))
    log("Транскрипция (точная, Deepgram nova-3, словарь терминов включён)" if engine == "deepgram"
        else f"Транскрипция (точная, локально, модель {options['model']}, словарь терминов включён)")
    transcript = transcribe_video(
        video, out / "_work" / "transcription", options.get("language", "ru"), options["model"], progress,
        mode="accurate", keep_disfluencies=bool(options.get("keep_disfluencies", True)),
        glossary=options.get("asr_glossary") if options.get("asr_glossary") is not None
        else normalize_glossary(None, "ru" if options.get("language", "ru") == "auto" else options.get("language", "ru")),
        asr_provider=options.get("asr_provider", "auto"))
    warning = (transcript.get("transcription") or {}).get("asr_warning")
    if warning:
        log(f"ВНИМАНИЕ: {warning}")
    _write_json(path, transcript)
    timings["transcription"] = round(time.monotonic() - started, 1)
    log(f"Транскрипция готова за {timings['transcription']} с: {len(transcript['words'])} слов")
    return transcript


def stage_editorial(transcript: dict, out: Path, options: dict, log: RunLog, timings: dict) -> tuple[list, dict, str]:
    from .highlights import analyze_transcript, normalize_candidates
    cand_path = out / "candidates.json"
    saved_path = out / "_work" / "editorial" / "editorial_result.json"
    cached = _read_json(cand_path)
    if cached is not None and _read_json(saved_path) is not None:
        saved = _read_json(saved_path)
        log(f"Монтаж из кэша: {len([c for c in cached if not c.get('rejected')])} роликов, "
            f"не прошли проверку {len([c for c in cached if c.get('rejected')])}")
        return cached, saved, options.get("provider", "")
    started = time.monotonic()
    log("Смысловой монтаж: карта тем -> компоновка -> пропущенные моменты -> критик -> ремонт -> главред -> чистка")
    raw, provider, _ = analyze_transcript(transcript, out / "_work", options, lambda p, m: log(f"монтаж {p:>3}%: {m}"))
    duration = float(transcript["words"][-1]["end"])
    candidates = normalize_candidates(raw, duration, options, transcript["words"])
    saved = _read_json(out / "_work" / "editorial" / "editorial_result.json") or {}
    _write_json(cand_path, candidates)
    timings["editorial_total"] = round(time.monotonic() - started, 1)
    for key, value in (saved.get("timings") or {}).items():
        timings[f"editorial_{key}"] = value
    kept = [c for c in candidates if not c.get("rejected")]
    log(f"Монтаж готов за {timings['editorial_total']} с ({provider}): роликов {len(kept)}, "
        f"не прошли проверку {len(candidates) - len(kept)}, "
        f"пропущенных моментов добавлено {saved.get('coverage_added', 0)}, "
        f"спин-оффов {len(saved.get('spinoffs') or [])}, делений длинных {len(saved.get('splits') or [])}")
    return candidates, saved, provider


def _render_one(video: Path, transcript: dict, cand: dict, out_dir: Path, name: str, work_dir: Path,
                options: dict, review: bool) -> dict:
    """Рендер одного кандидата -> name_vertical.mp4 / name_horizontal.mp4 / name_text.txt; состояние в _state."""
    from .highlights import render_highlight, render_highlight_reviewed
    out_dir.mkdir(parents=True, exist_ok=True)
    vertical = out_dir / f"{name}_vertical.mp4"
    staged = out_dir / f".{name}.staged.mp4"
    started = time.monotonic()
    try:
        if review:
            result = render_highlight_reviewed(video, transcript, cand, staged, work_dir, options,
                                               lambda p, m: None, renderer=render_highlight)
        else:
            result = dict(render_highlight(video, transcript, cand, staged, work_dir, options, lambda p, m: None) or {})
        update = result.get("candidate_update")
        final = {**cand, **update} if update else cand
        shutil.move(str(staged), vertical)
        horizontal = Path(result.get("horizontal") or work_dir / "horizontal.mp4")
        if horizontal.is_file():
            shutil.copy(horizontal, out_dir / f"{name}_horizontal.mp4")
        text = final_text(transcript, final)
        (out_dir / f"{name}_text.txt").write_text(text, encoding="utf-8")
    finally:
        staged.unlink(missing_ok=True)
    return {"ok": True, "seconds": round(time.monotonic() - started, 1), "candidate": final, "text": text,
            "review": result.get("review"), "auto_fix": result.get("auto_fix"),
            "review_before_fix": result.get("review_before_fix")}


def stage_render(video: Path, transcript: dict, candidates: list, out: Path, options: dict, log: RunLog,
                 timings: dict) -> None:
    """Сначала основные ролики (рендер + отсмотр), затем отсеянные (рендер без отсмотра, в rejected/).
    По очереди: 8 ГБ ОЗУ. Готовое (state + mp4) пропускается — прогон возобновляется."""
    state_dir = out / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    kept = [c for c in candidates if not c.get("rejected")]
    rejected = [c for c in candidates if c.get("rejected") and c.get("segments")]
    review_on = bool(options.get("video_review", True))
    t_kept, t_rej, t_review = 0.0, 0.0, 0.0
    jobs = [(c, c["id"], out, False) for c in kept] + [(c, "r" + c["id"][1:], out / "rejected", True) for c in rejected]
    for n, (cand, name, folder, is_rejected) in enumerate(jobs, 1):
        state_path = state_dir / f"{name}.json"
        done = (folder / f"{name}_vertical.mp4").is_file()
        state = _read_json(state_path)
        if state and state.get("ok") and done:
            log(f"[{n}/{len(jobs)}] {name} уже готов")
            continue
        label = "отсеянный" if is_rejected else "ролик"
        try:
            state = _render_one(video, transcript, cand, folder, name, out / "_work" / "render" / name, options,
                                review=review_on and not is_rejected)
            _write_json(state_path, state)
            review = state.get("review") or {}
            fix = state.get("auto_fix") or {}
            review_seconds = float((review.get("timings") or {}).get("total") or 0.0)
            if is_rejected:
                t_rej += state["seconds"]
            else:
                t_kept += state["seconds"] - review_seconds
                t_review += review_seconds
            log(f"[{n}/{len(jobs)}] {label} {name}: {state['seconds']} с"
                + (f" · отсмотр: {VERDICT_RU.get(review.get('verdict'))} {review.get('scores')}"
                   f" · правка: {'принята' if fix.get('kept') else ('нет' if not fix else 'отклонена')}" if review else "")
                + f" · {state['candidate']['title']}")
        except Exception as exc:   # один сбойный ролик не должен ронять весь прогон
            traceback.print_exc()
            _write_json(state_path, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "candidate": cand})
            log(f"[{n}/{len(jobs)}] {label} {name}: ОШИБКА {type(exc).__name__}: {exc}")
    timings["render_kept"] = round(t_kept, 1)
    timings["review"] = round(t_review, 1)
    timings["render_rejected"] = round(t_rej, 1)


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #

def _pieces(cand: dict) -> list[dict]:
    return [{"role": s.get("role"), "sentences": [s.get("sentence_from"), s.get("sentence_to")],
             "minutes": [mm(s["start"]), mm(s["end"])]} for s in cand.get("segments") or []]


def build_report(video: Path, out: Path, options: dict, candidates: list, saved: dict, timings: dict) -> dict:
    states = {p.stem: _read_json(p, {}) for p in (out / "_state").glob("*.json")}
    kept = [c for c in candidates if not c.get("rejected")]
    rejected = [c for c in candidates if c.get("rejected")]
    transcript = _read_json(out / "transcript.json", {"words": []})

    def clip_row(c):
        st = states.get(c["id"], {})
        final = st.get("candidate") or c
        review = st.get("review") or {}
        critic = c.get("critic") or {}
        chief = c.get("chief_editor") or {}
        return {
            "id": c["id"], "title": final["title"], "hook_text": final.get("hook_text") or final.get("hook"),
            "thesis": final.get("thesis"), "format": final.get("format"), "topic": final.get("topic"),
            "duration": final["duration"], "source": c.get("source"), "moment": c.get("moment"),
            "spinoff_of": c.get("spinoff_of"), "split_of": c.get("split_of"), "local_fixes": c.get("local_fixes") or [],
            "chronological": editorial.segments_chronological(final.get("segments") or []),
            "pieces": _pieces(final),
            "chief_editor": {k: chief.get(k) for k in ("rank", "decision", "reason", "title_fixed", "original_title",
                                                       "local_fix", "spinoff_of", "coverage_override", "resurrected")
                             if chief.get(k) not in (None, "", [])},
            "critic": {k: critic.get(k) for k in ("hook_strength", "completeness", "clarity", "standalone",
                                                  "single_idea", "problems", "repaired", "trims") if k in critic},
            "video_review": {k: review.get(k) for k in ("status", "verdict", "scores", "issues", "summary", "caption",
                                                        "suggested_fix", "match_ratio", "error")},
            "auto_fix": st.get("auto_fix"),
            "render": {k: st.get(k) for k in ("ok", "seconds", "error")},
            "files": {"vertical": f"{c['id']}_vertical.mp4", "horizontal": f"{c['id']}_horizontal.mp4",
                      "text": f"{c['id']}_text.txt"},
            "final_text": st.get("text", ""),
        }

    def rejected_row(c):
        name = "r" + c["id"][1:]
        st = states.get(name, {})
        has = (out / "rejected" / f"{name}_vertical.mp4").is_file()
        info = c.get("rejected") or {}
        row = {"id": name, "candidate_id": c["id"], "title": c["title"], "hook_text": c.get("hook_text") or c.get("hook"),
               "thesis": c.get("thesis"), "format": c.get("format"), "duration": c["duration"],
               "stage": info.get("stage"), "reason": info.get("reason"), "scores": info.get("scores") or {},
               "pieces": _pieces(c), "chronological": editorial.segments_chronological(c.get("segments") or []),
               "render": {k: st.get(k) for k in ("ok", "seconds", "error")}, "text": st.get("text", "")}
        if has:
            row["files"] = {"vertical": f"rejected/{name}_vertical.mp4", "horizontal": f"rejected/{name}_horizontal.mp4",
                            "text": f"rejected/{name}_text.txt"}
        elif c.get("segments") and transcript.get("words"):
            row["text"] = final_text(transcript, c)
        return row

    rows = [clip_row(c) for c in kept]
    reviews = [r["video_review"] for r in rows]
    count = lambda v: sum(1 for r in reviews if r.get("verdict") == v)
    by_stage: dict[str, int] = {}
    for c in rejected:
        stage = (c.get("rejected") or {}).get("stage", "")
        by_stage[stage] = by_stage.get(stage, 0) + 1
    summary = {
        "video": str(video), "kept": len(rows), "rejected": len(rejected),
        "rejected_by_stage": by_stage,
        "all_chronological": all(r["chronological"] for r in rows) and all(
            editorial.segments_chronological(c.get("segments") or []) for c in rejected),
        "coverage_added": saved.get("coverage_added", 0), "spinoffs": saved.get("spinoffs") or [],
        "splits": saved.get("splits") or [],
        "video_review": {"publish": count("publish"), "needs_edit": count("needs_edit"), "reject": count("reject"),
                         "error": sum(1 for r in reviews if r.get("status") == "error")},
        "auto_fixes": {"applied": sum(1 for r in rows if (r["auto_fix"] or {}).get("applied")),
                       "kept": sum(1 for r in rows if (r["auto_fix"] or {}).get("kept"))},
        "audio_mode": options.get("audio_mode"), "timings_seconds": timings, "options": {
            k: options.get(k) for k in ("provider", "model", "min_duration", "max_duration", "max_clips", "audience")},
    }
    return {"summary": summary, "clips": rows, "rejected": [rejected_row(c) for c in rejected],
            "dropped_log": saved.get("dropped", [])}


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _video(src: str) -> str:
    return f'<video src="{_e(src)}" controls preload="metadata" playsinline></video>'


def render_gallery(report: dict, title: str = "Ролики из эфира") -> str:
    s = report["summary"]
    cards = []
    for c in report["clips"]:
        ch, vr, critic = c.get("chief_editor") or {}, c.get("video_review") or {}, c.get("critic") or {}
        verdict = vr.get("verdict")
        chips = "".join(f"<span>{SCORE_RU.get(k, k)} {v}/10</span>" for k, v in (vr.get("scores") or {}).items())
        issues = "".join(f"<li><b>{mm(i['t'])}</b> <i>{_e(i['kind'])}</i> {_e(i['note'])}</li>" for i in (vr.get("issues") or []))
        critic_chips = "".join(f"<span>{CRITIC_RU[k]} {critic[k]}</span>" for k in CRITIC_RU if critic.get(k) is not None)
        fix = c.get("auto_fix")
        fix_html = ""
        if fix:
            state = ("принята, ролик пересобран" if fix.get("kept") else "не принята (версия не лучше исходной)") \
                if fix.get("applied") else (fix.get("reason") or "не применена")
            fix_html = f'<p class="meta">Авто-правка: {_e(fix.get("description", ""))} — {_e(state)}</p>'
        tags = []
        if c.get("source") == "coverage":
            tags.append("пропущенный момент" + (f": {_e(MOMENT_RU.get(c.get('moment'), c.get('moment')))}" if c.get("moment") else ""))
        if c.get("spinoff_of"):
            tags.append(f"спин-офф ролика «{_e(c['spinoff_of'])}»")
        if c.get("split_of"):
            tags.append(f"часть длинного ролика «{_e(c['split_of'])}»")
        if c.get("local_fixes"):
            tags.append("подрезано: " + ", ".join(f"«{_e(x.get('text'))}»" for x in c["local_fixes"]))
        if ch.get("title_fixed"):
            tags.append(f"название исправлено главредом (было «{_e(ch.get('original_title'))}»)")
        pieces = " · ".join(f"{_e(p['role'])} {p['minutes'][0]}–{p['minutes'][1]}" for p in c["pieces"])
        order = "" if c.get("chronological") else ' <b class="v-reject">порядок записи нарушен</b>'
        cards.append(f"""<article id="{_e(c['id'])}"><div class="media">{_video(c['files']['vertical'])}
<a class="dl" href="{_e(c['files']['horizontal'])}">горизонтальная версия</a></div>
<div class="body"><h2>№{_e(ch.get('rank', '?'))} · {_e(c['title'])}</h2>
<p class="hook"><b>Хук для визуала:</b> «{_e(c.get('hook_text'))}» <span class="meta">(титр на экране, не озвучивается)</span></p>
<p class="meta">{_e(c['id'])} · {FORMAT_RU.get(c.get('format'), _e(c.get('format')))} · {c['duration']:.0f} с · тема: {_e(c.get('topic'))}{order}</p>
<p><b>Тезис:</b> {_e(c.get('thesis'))}</p>
<p class="chief"><b>Главред:</b> {_e(ch.get('decision'))} — {_e(ch.get('reason'))}</p>
{('<p class="meta">' + ' · '.join(tags) + '</p>') if tags else ''}
<p class="meta">Куски (минуты эфира): {pieces}</p>
<div class="chips critic"><b>Критик</b>{critic_chips}</div>
<div class="review"><div class="chips"><b class="v-{_e(verdict)}">Отсмотр: {VERDICT_RU.get(verdict)}</b>{chips}</div>
<p class="meta">{_e(vr.get('summary'))}</p>{('<ul class="issues">' + issues + '</ul>') if issues else ''}{fix_html}
{('<p class="meta">Подпись: ' + _e(vr.get('caption')) + '</p>') if vr.get('caption') else ''}</div>
<details><summary>Текст ролика</summary><p class="text">{_e(c.get('final_text'))}</p></details></div></article>""")
    rej = []
    for r in report["rejected"]:
        files = r.get("files")
        sc = " · ".join(f"{CRITIC_RU.get(k, k)} {v}" for k, v in (r.get("scores") or {}).items())
        rej.append(f"""<article class="rej"><div class="media">{_video(files['vertical']) if files else '<p class="meta">видео нет (нет кусков)</p>'}</div>
<div class="body"><h2>{_e(r['title'])}</h2>
<p class="hook"><b>Хук для визуала:</b> «{_e(r.get('hook_text'))}»</p>
<p class="meta">{_e(r['id'])} · {FORMAT_RU.get(r.get('format'), _e(r.get('format')))} · {r['duration']:.0f} с · стадия: {_e(STAGE_RU.get(r.get('stage'), r.get('stage')))}</p>
<p><b>Причина:</b> {_e(r.get('reason'))}</p><p class="meta">Оценки: {_e(sc) or '—'}</p>
<p class="meta">Тезис: {_e(r.get('thesis'))}</p>
<details><summary>Текст</summary><p class="text">{_e(r.get('text'))}</p></details></div></article>""")
    t = s.get("timings_seconds") or {}
    timing_line = " · ".join(f"{k}: {v} с" for k, v in t.items())
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>
:root{{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--card:#fff;--line:#e4e4df;--ok:#2e7d32;--warn:#b26a00;--bad:#c62828}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ececea;--mute:#9a9a94;--card:#1f1f1d;--line:#33332f;--ok:#7bd88a;--warn:#ffc266;--bad:#ff8a80}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif}}main{{max-width:1040px;margin:0 auto;padding:24px 16px}}
article{{display:grid;grid-template-columns:240px 1fr;gap:20px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}}
video{{width:240px;border-radius:8px;background:#000}}h1{{font-size:22px}}h2{{margin:0 0 4px;font-size:18px}}.hook{{margin:4px 0}}
.meta{{color:var(--mute);font-size:13px;margin:4px 0}}.chips{{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;font-size:12px}}
.chips span,.chips b{{padding:3px 8px;border:1px solid var(--line);border-radius:999px}}.v-publish{{color:var(--ok)}}.v-needs_edit{{color:var(--warn)}}.v-reject{{color:var(--bad)}}
.issues{{margin:6px 0;padding-left:18px;font-size:13px}}.issues i{{color:var(--warn);font-style:normal;margin:0 4px}}.chief{{font-size:13px}}
.text{{font-size:13px;color:var(--mute)}}.dl{{display:block;margin-top:6px;font-size:12px}}
@media (max-width:700px){{article{{grid-template-columns:1fr}}video{{width:100%}}}}</style></head><body><main>
<h1>{_e(title)}: {s['kept']} роликов</h1>
<p class="meta">Не прошли проверку: {s['rejected']} · пропущенных моментов добавлено {s.get('coverage_added', 0)} · спин-оффов {len(s.get('spinoffs') or [])} ·
отсмотр: publish {s['video_review']['publish']}, needs_edit {s['video_review']['needs_edit']}, reject {s['video_review']['reject']} ·
авто-правок принято {s['auto_fixes']['kept']} из {s['auto_fixes']['applied']} · звук: {_e(s.get('audio_mode'))} · порядок записи везде: {'да' if s.get('all_chronological') else 'НЕТ'}</p>
<p class="meta">Время: {_e(timing_line)}</p>
{''.join(cards)}
<h1>Не прошли проверку ({len(report['rejected'])})</h1>
{''.join(rej)}
</main></body></html>"""


def stage_report(video: Path, out: Path, options: dict, candidates: list, saved: dict, timings: dict, log: RunLog) -> dict:
    started = time.monotonic()
    report = build_report(video, out, options, candidates, saved, timings)
    timings["report"] = round(time.monotonic() - started, 1)
    report["summary"]["timings_seconds"] = timings
    _write_json(out / "report.json", report)
    (out / "index.html").write_text(render_gallery(report, f"Ролики: {Path(video).stem}"), encoding="utf-8")
    log("report.json и index.html записаны")
    return report


def run(video: Path, out: Path, options: dict, *, echo: bool = True) -> dict:
    """Полный прогон. Возвращает отчёт."""
    video, out = Path(video).expanduser().resolve(), Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / "run.log", echo=echo)
    timings: dict = _read_json(out / "timings.json", {}) or {}
    started = time.monotonic()
    log(f"Старт: {video.name} -> {out} | провайдер {options.get('provider')}, модель {options.get('model')}, "
        f"{options.get('min_duration')}–{options.get('max_duration')} с, звук {options.get('audio_mode')}")
    transcript = stage_transcribe(video, out, options, log, timings)
    _write_json(out / "timings.json", timings)
    candidates, saved, _provider = stage_editorial(transcript, out, options, log, timings)
    _write_json(out / "timings.json", timings)
    stage_render(video, transcript, candidates, out, options, log, timings)
    _write_json(out / "timings.json", timings)
    report = stage_report(video, out, options, candidates, saved, timings, log)
    timings["total_this_run"] = round(time.monotonic() - started, 1)
    _write_json(out / "timings.json", timings)
    log("Тайминги этапов (с): " + ", ".join(f"{k}={v}" for k, v in timings.items()))
    log(f"Готово за {timings['total_this_run']} с: роликов {report['summary']['kept']}, "
        f"не прошли проверку {report['summary']['rejected']}. Галерея: {out / 'index.html'}")
    return report
