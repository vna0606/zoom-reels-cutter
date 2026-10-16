from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from reel_factory import assembly, editorial

SPECIAL = {3: ["ну", "вот", "мы", "э-э", "мы", "думаем", "что", "важно"]}


def make_transcript(n=20):
    """n предложений по 8 слов; предложение k стартует в (k-1)*3.5 с."""
    words, segments, index = [], [], {}
    for k in range(1, n + 1):
        toks = SPECIAL.get(k) or ["Предложение", "номер", str(k), "про", "важную", "тему", "которая", "интересна"]
        t0 = (k - 1) * 3.5
        first = len(words)
        for j, tok in enumerate(toks):
            start = t0 + j * 0.35
            words.append({"text": tok, "start": round(start, 3), "end": round(start + 0.3, 3), "probability": 0.9})
            index[(k, j)] = len(words) - 1
        segments.append({"start": words[first]["start"], "end": words[-1]["end"], "text": " ".join(toks) + "."})
    return {"language": "ru", "words": words, "segments": segments,
            "text": " ".join(w["text"] for w in words)}, index


class BuildSentencesTest(unittest.TestCase):
    def test_segment_boundaries_with_terminal_punctuation(self):
        tr, _ = make_transcript()
        sents = editorial.build_sentences(tr)
        self.assertEqual(len(sents), 20)
        self.assertEqual(sents[0]["id"], "S0001")
        self.assertEqual((sents[2]["w0"], sents[2]["w1"]), (16, 23))
        self.assertAlmostEqual(sents[0]["start"], 0.0)

    def test_punctuation_in_words(self):
        words = [{"text": t, "start": i * 0.4, "end": i * 0.4 + 0.3, "probability": 1.0}
                 for i, t in enumerate(["Привет,", "мир.", "Как", "дела?", "Хорошо"])]
        sents = editorial.build_sentences({"words": words, "segments": []})
        self.assertEqual([(s["w0"], s["w1"]) for s in sents], [(0, 1), (2, 3), (4, 4)])

    def test_raw_punctuation_used_for_boundaries_and_text(self):
        raws = ["Привет,", "мир.", "Как", "дела?", "Хорошо"]
        words = [{"text": r.strip(",.?"), "raw": r, "start": i * 0.4, "end": i * 0.4 + 0.3, "probability": 1.0}
                 for i, r in enumerate(raws)]
        sents = editorial.build_sentences({"words": words, "segments": []})
        self.assertEqual([(s["w0"], s["w1"]) for s in sents], [(0, 1), (2, 3), (4, 4)])
        self.assertEqual(sents[0]["text"], "Привет, мир.")

    def test_pause_split(self):
        words = [{"text": f"w{i}", "start": i * 0.4 + (3.0 if i >= 3 else 0), "end": i * 0.4 + 0.3 + (3.0 if i >= 3 else 0),
                  "probability": 1.0} for i in range(6)]
        sents = editorial.build_sentences({"words": words, "segments": []}, pause_split=1.2)
        self.assertEqual([(s["w0"], s["w1"]) for s in sents], [(0, 2), (3, 5)])

    def test_long_runon_is_capped_at_largest_pause(self):
        words = []
        t = 0.0
        for i in range(100):
            words.append({"text": f"w{i}", "start": t, "end": t + 0.2, "probability": 1.0})
            t += 0.25 + (0.6 if i == 40 else 0.0)
        sents = editorial.build_sentences({"words": words, "segments": []}, max_words=45)
        self.assertTrue(all(s["w1"] - s["w0"] + 1 <= 45 for s in sents))
        self.assertEqual(sents[0]["w1"], 40)  # режем по самой длинной паузе
        self.assertEqual(sents[-1]["w1"], 99)
        covered = sum(s["w1"] - s["w0"] + 1 for s in sents)
        self.assertEqual(covered, 100)

    def test_no_words_uses_segments(self):
        tr = {"words": [], "segments": [{"start": 0, "end": 3, "text": "Раз два."}, {"start": 3, "end": 6, "text": "Три."}]}
        sents = editorial.build_sentences(tr)
        self.assertEqual(len(sents), 2)
        self.assertEqual((sents[0]["w0"], sents[0]["w1"]), (-1, -1))


SC = {"hook_strength": 8, "completeness": 9, "clarity": 8, "standalone": 9, "single_idea": 9}


def critique_response(prompt, plan, default=None):
    """Строит ответ критика по тексту промпта; plan: {title: {verdict, scores, seam, pieces, problems}}."""
    verdicts = []
    for m in re.finditer(r"=== РОЛИК (\d+): (.*?) ===\n(.*?)(?=\n=== РОЛИК|\Z)", prompt, re.S):
        n, title, block = int(m.group(1)), m.group(2), m.group(3)
        spec = plan.get(title, default or {})
        seams = [{"seam": int(k), "verdict": spec.get("seam", "ok"), "note": ""}
                 for k in re.findall(r"^Стык (\d+):", block, re.M)]
        verdict = {"clip": n, "verdict": spec.get("verdict", "pass"), "scores": dict(SC, **spec.get("scores", {})),
                   "seams": seams, "seams_ok": all(x["verdict"] == "ok" for x in seams),
                   "problems": spec.get("problems", [])}
        if "pieces" in spec:
            verdict["pieces"] = spec["pieces"]
        verdicts.append(verdict)
    return {"verdicts": verdicts}


def _block_pieces(block):
    """Куски ролика из блока промпта: подряд идущие id в разделе ТЕКСТ РОЛИКА."""
    text = block.split("ТЕКСТ РОЛИКА (в порядке воспроизведения):")[1].split("\nСТЫКИ")[0]
    ids = [int(x) for x in re.findall(r"^S(\d{4}) ", text, re.M)]
    runs = []
    for i in ids:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [{"from": f"S{a:04d}", "to": f"S{b:04d}",
             "role": "payoff" if k == len(runs) - 1 else ("hook" if k == 0 and len(runs) > 1 else "point")}
            for k, (a, b) in enumerate(runs)]


def repair_response(prompt, plan):
    """Ответ ремонта: по умолчанию те же куски, оценки 8, стыки ok; plan: {title: {pieces, scores, seam, addressed}}."""
    out = []
    for m in re.finditer(r"=== РОЛИК (\d+): (.*?) ===\n(.*?)(?=\n=== РОЛИК|\Z)", prompt, re.S):
        n, title, block = int(m.group(1)), m.group(2), m.group(3)
        spec = plan.get(title, {})
        pieces = spec.get("pieces") or _block_pieces(block)
        seam_count = max(0, len(pieces) - 1) if "pieces" in spec else len(re.findall(r"^Стык \d+:", block, re.M))
        out.append({"clip": n, "pieces": pieces,
                    "addressed": spec.get("addressed", [{"problem": "p", "fix": "f"}]),
                    "scores": dict({"hook_strength": 8, "completeness": 8, "clarity": 8, "standalone": 8,
                                    "single_idea": 8}, **spec.get("scores", {})),
                    "seams": [{"seam": k, "verdict": spec.get("seam", "ok"), "note": ""} for k in range(1, seam_count + 1)],
                    "seams_ok": spec.get("seam", "ok") == "ok", "problems": spec.get("problems", [])})
    return {"clips": out}


def chief_response(prompt, plan):
    """Ответ главреда: по умолчанию publish всем по порядку; plan: {title: {decision, reason, title}}."""
    decisions = []
    rank = 0
    for m in re.finditer(r"=== КАНДИДАТ (\d+): (.*?) ===", prompt):
        n, title = int(m.group(1)), m.group(2)
        spec = plan.get(title, {})
        decision = spec.get("decision", "publish")
        if decision != "drop":
            rank += 1
        decisions.append({"clip": n, "decision": decision, "reason": spec.get("reason", "ок"),
                          "rank": spec.get("rank", rank if decision != "drop" else 999),
                          "title": spec.get("title", title), "duplicate_of": 0})
    return {"decisions": decisions}


class FakeLLM:
    def __init__(self, index, plan1=None, plan2=None, repair_plan=None, chief_plan=None):
        self.index = index
        self.repair_plan = repair_plan or {}
        self.chief_plan = chief_plan or {}
        self.calls = []
        self.plan1 = plan1 if plan1 is not None else {
            "Клип B": {"verdict": "reject", "scores": {"standalone": 2}, "problems": ["нужен контекст"]},
            "Клип C": {"verdict": "fix", "problems": ["обрыв в начале"],
                       "pieces": [{"from": "S0013", "to": "S0011", "role": "point"},
                                  {"from": "S0018", "to": "S0018", "role": "payoff"},
                                  {"from": "S0999", "to": "S0999", "role": "point"}]},
        }
        self.plan2 = plan2 or {}

    def __call__(self, prompt, schema, work_dir, *, task, provider=None, effort="high", timeout=1800):
        self.calls.append({"task": task, "prompt": prompt, "provider": provider, "effort": effort})
        return getattr(self, "task_" + task)(prompt)

    def task_topic_map(self, prompt):
        return {"topics": [{"title": "Тема", "thesis": "Тезис", "kind": "thesis",
                            "ranges": [{"from": "S0002", "to": "S0008"}], "strength": 80, "appeal": "интересно"}]}

    def task_compose(self, prompt):
        P = lambda a, b, r: {"from": a, "to": b, "role": r}
        base = {"description": "Описание.", "hook_text": "Хук", "rationale": "почему", "thesis": "Один тезис.",
                "format": "thesis"}
        return {"clips": [
            dict(base, title="Клип A", score=90, pieces=[
                P("S0015", "S0015", "hook"), P("S0002", "S0003", "context"), P("S0004", "S0006", "point"),
                P("S0099", "S0100", "point"), P("S0008", "S0007", "payoff")]),
            dict(base, title="Клип B", score=80, pieces=[P("S0016", "S0019", "point")]),
            dict(base, title="Клип C", score=70, pieces=[P("S0010", "S0013", "point")]),
            dict(base, title="Слишком длинный", score=60, pieces=[P("S0001", "S0020", "point")]),
            dict(base, title="Слишком короткий", score=55, pieces=[P("S0017", "S0017", "point")]),
            dict(base, title="Дубль A", score=40, pieces=[P("S0004", "S0008", "point")]),
        ]}

    def task_coverage(self, prompt):
        return {"clips": []}

    def task_critique(self, prompt):
        return critique_response(prompt, self.plan1)

    def task_critique_2(self, prompt):
        return critique_response(prompt, self.plan2)

    def task_repair(self, prompt):
        return repair_response(prompt, self.repair_plan)

    def task_repair_chief(self, prompt):
        return repair_response(prompt, self.repair_plan)

    def task_chief_editor(self, prompt):
        return chief_response(prompt, self.chief_plan)

    def task_tighten(self, prompt):
        i = self.index
        return {"clips": [
            {"clip": 1, "cuts": [
                {"w0": i[(3, 0)], "w1": i[(3, 1)], "reason": "filler"},
                {"w0": i[(3, 2)], "w1": i[(3, 2)], "reason": "false_start"},
                {"w0": i[(19, 0)], "w1": i[(19, 2)], "reason": "aside"},        # вне ролика
                {"w0": i[(2, 5)], "w1": i[(3, 1)], "reason": "aside"},
            ]},
            {"clip": 2, "cuts": []},
        ]}


class ComposeClipsTest(unittest.TestCase):
    def setUp(self):
        self.tr, self.index = make_transcript()
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.llm = FakeLLM(self.index)
        self.progress = []
        self.opts = {"max_clips": 5, "min_duration": 10, "max_duration": 30}
        self.result = editorial.compose_clips(self.tr, self.work, self.opts, llm=self.llm, provider="codex",
                                              progress=lambda p, m: self.progress.append((p, m)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_call_sequence_and_prompts(self):
        self.assertEqual([c["task"] for c in self.llm.calls],
                         ["topic_map", "compose", "coverage", "critique", "critique_2", "chief_editor", "tighten"])
        self.assertTrue(all(c["effort"] == "high" and c["provider"] == "codex" for c in self.llm.calls))
        by = {c["task"]: c["prompt"] for c in self.llm.calls}
        self.assertIn("лучше 4 сильных ролика, чем 10 слабых", by["compose"])
        self.assertIn("S0001 [00:00:00]", by["topic_map"])
        # критик не видит карту тем и оценок композитора
        self.assertNotIn("КАРТА ТЕМ", by["critique"])
        self.assertIn("ДОСТУПНЫЕ ПРЕДЛОЖЕНИЯ ВОКРУГ", by["critique"])
        self.assertIn("Стык 1:", by["critique"])
        self.assertIn("ПРОБЛЕМЫ РАУНДА 1", by["critique_2"])
        self.assertIn("обрыв в начале", by["critique_2"])
        self.assertIn("S0015 ", by["critique"])
        self.assertIn(f"#{self.index[(3, 0)]} ну", by["tighten"])
        self.assertTrue(self.progress and self.progress[-1][0] == 100)

    def test_outputs_and_filtering(self):
        titles = [c["title"] for c in self.result]
        self.assertEqual(titles, ["Клип A", "Клип C"])  # B reject, long/short/dup dropped
        self.assertTrue((self.work / "editorial_result.json").is_file())
        saved = json.loads((self.work / "editorial_result.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved["clips"]), 2)
        dropped = {d["title"]: d for d in saved["dropped"]}
        self.assertEqual(dropped["Клип B"]["stage"], "critic_round1")
        self.assertIn("явный дефект", dropped["Клип B"]["reason"])
        # отсеянный не удаляется: остаётся в rejected с причиной, стадией и оценками, и пригоден к рендеру
        self.assertEqual([c["title"] for c in self.result.rejected].count("Клип B"), 1)
        rej = next(c for c in self.result.rejected if c["title"] == "Клип B")
        self.assertEqual(rej["rejected"]["stage"], "critic_round1")
        self.assertEqual(rej["rejected"]["scores"]["standalone"], 2)
        self.assertTrue(rej["segments"] and rej["duration"] > 0)
        self.assertEqual(len(saved["rejected"]), len(self.result.rejected))
        self.assertEqual(dropped["Клип B"]["scores"]["standalone"], 2)
        self.assertEqual(dropped["Слишком длинный"]["stage"], "compose")
        self.assertIn("Дубль A", dropped)
        # компоновка просит с запасом: max_clips*2 + 1
        compose_prompt = next(c["prompt"] for c in self.llm.calls if c["task"] == "compose")
        self.assertIn("до 11 вертикальных", compose_prompt)

    def test_contract_clip_a(self):
        a = self.result[0]
        for key in ("title", "description", "hook", "score", "rationale", "critic", "segments", "cuts",
                    "start", "end", "duration"):
            self.assertIn(key, a)
        self.assertEqual(a["hook"], "Хук")
        self.assertEqual(a["critic"]["verdict"], "pass")
        # естественный порядок: «хук» из S0015 НЕ выносится вперёд — куски идут как в записи, смежные склеены
        self.assertEqual([(s["sentence_from"], s["sentence_to"]) for s in a["segments"]],
                         [("S0002", "S0008"), ("S0015", "S0015")])
        self.assertTrue(editorial.segments_chronological(a["segments"]))
        self.assertNotIn("hook", [s["role"] for s in a["segments"][1:]])
        # привязка к границам слов
        words = self.tr["words"]
        seg0 = a["segments"][0]
        self.assertEqual(seg0["start"], words[self.index[(2, 0)]]["start"])
        self.assertEqual(seg0["end"], words[self.index[(8, 7)]]["end"])
        self.assertEqual(a["start"], min(s["start"] for s in a["segments"]))
        self.assertEqual(a["end"], max(s["end"] for s in a["segments"]))

    def test_cuts_inside_segments_and_validated(self):
        a = self.result[0]
        reasons = {c["reason"] for c in a["cuts"]}
        self.assertTrue({"filler", "false_start"} <= reasons)
        for c in a["cuts"]:
            self.assertTrue(any(s["start"] - 1e-9 <= c["start"] and c["end"] <= s["end"] + 1e-9 for s in a["segments"]))
        words = self.tr["words"]
        # 'э-э' вырезан автоматически, вне диапазона индексы LLM отброшены
        eh = words[self.index[(3, 3)]]
        self.assertTrue(any(abs(c["start"] - eh["start"]) < 1e-9 for c in a["cuts"]))
        for c in a["cuts"]:
            self.assertFalse(c["start"] >= words[self.index[(19, 0)]]["start"])

    def test_duration_matches_assembly(self):
        for c in self.result:
            ivs = assembly.keep_intervals(c, self.tr["words"])
            self.assertAlmostEqual(c["duration"], assembly.assembled_duration(ivs), places=6)
            self.assertTrue(8.5 <= c["duration"] <= 37.5)
        a = self.result[0]
        raw = sum(s["end"] - s["start"] for s in a["segments"])
        self.assertLess(a["duration"], raw)

    def test_fix_applied_checked_in_round_two_and_bad_ids_dropped(self):
        c = self.result[1]
        self.assertEqual(c["critic"]["verdict"], "fix")
        self.assertEqual(len(c["critic"]["rounds"]), 2)
        self.assertEqual([(s["sentence_from"], s["sentence_to"]) for s in c["segments"]],
                         [("S0011", "S0013"), ("S0018", "S0018")])  # S0999 отброшен, S0018 взят из окна ±15
        self.assertEqual(c["format"], "thesis")
        self.assertEqual(c["thesis"], "Один тезис.")
        self.assertEqual([sm["verdict"] for sm in c["seams"]], ["ok"])
        self.assertEqual(c["cuts"], [])

    def test_missing_words_fallback(self):
        tr = {"language": "ru", "words": [], "segments": [
            {"start": i * 5.0, "end": i * 5.0 + 4.5, "text": f"Предложение номер {i} про важную тему."} for i in range(8)]}

        class L:
            def __call__(s, prompt, schema, work_dir, *, task, provider=None, effort="high", timeout=1800):
                if task == "topic_map":
                    return {"topics": []}
                if task == "compose":
                    return {"clips": [{"title": "X", "thesis": "t", "format": "thesis", "hook_text": "h",
                                       "description": "d", "score": 50, "rationale": "r",
                                       "pieces": [{"from": "S0008", "to": "S0008", "role": "hook"},
                                                  {"from": "S0001", "to": "S0004", "role": "point"}]}]}
                if task == "critique":
                    return critique_response(prompt, {})
                raise AssertionError("tighten не должен вызываться без слов")

        with tempfile.TemporaryDirectory() as d:
            out = editorial.compose_clips(tr, Path(d), {"min_duration": 10, "max_duration": 30, "max_clips": 5}, llm=L())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["segments"][0]["start"], 0.0)   # S0001–S0004 первыми: порядок записи
        self.assertEqual(out[0]["segments"][1]["start"], 35.0)
        self.assertAlmostEqual(out[0]["duration"], 4.5 + 19.5 + 0.0, delta=0.1)


class ComposeRunner:
    def run_compose(self, clips, plan1, opts=None, plan2=None, topics=None, fmt="thesis", repair=None, chief=None):
        tr, index = make_transcript()

        class L(FakeLLM):
            def task_topic_map(s, prompt):
                return {"topics": topics or []}

            def task_compose(s, prompt):
                base = {"description": "d", "hook_text": "h", "rationale": "r", "score": 80, "thesis": "Тезис.",
                        "format": fmt}
                return {"clips": [dict(base, title=t, pieces=pcs) for t, pcs in clips]}

            def task_tighten(s, prompt):
                return {"clips": []}

        llm = L(index, plan1, plan2, repair, chief)
        options = {"max_clips": 5, "min_duration": 20, "max_duration": 60}
        options.update(opts or {})
        with tempfile.TemporaryDirectory() as d:
            result = editorial.compose_clips(tr, Path(d), options, llm=llm)
        return result, llm

    def pieces(self, a, b, role="payoff"):
        return [{"from": f"S{a:04d}", "to": f"S{b:04d}", "role": role}]



class QualityGateTest(ComposeRunner, unittest.TestCase):
    def test_final_duration_below_085_min_is_dropped(self):
        result, _ = self.run_compose(
            [("Короткий", self.pieces(10, 13)), ("Нормальный", self.pieces(1, 9))], {})
        self.assertEqual([c["title"] for c in result], ["Нормальный"])
        self.assertTrue(all(c["duration"] >= 0.85 * 20 for c in result))

    def test_final_window_is_strict_and_hook_payoff_are_never_trimmed(self):
        # 18 предложений собираются в ~54 с: окно thesis max 36/40 строгое (без допуска), а режутся только
        # context/example; кусок из одного payoff резать нельзя -> отсев со стадией duration и причиной
        for max_d in (36, 40):
            dropped, _ = self.run_compose([("Длинный", self.pieces(1, 18))], {}, {"max_duration": max_d, "min_duration": 20})
            self.assertEqual(list(dropped), [])
            self.assertIn(dropped.rejected[0]["rejected"]["stage"], ("compose", "duration"))

    def test_context_is_trimmed_to_fit_window_but_payoff_stays(self):
        pieces = [{"from": "S0001", "to": "S0004", "role": "hook"}, {"from": "S0006", "to": "S0013", "role": "context"},
                  {"from": "S0016", "to": "S0018", "role": "payoff"}]
        result, _ = self.run_compose([("С контекстом", pieces)], {}, {"max_duration": 40, "min_duration": 20})
        self.assertEqual(len(result), 1)
        self.assertLessEqual(result[0]["duration"], 40.0)
        roles = [s["role"] for s in result[0]["segments"]]
        self.assertEqual(roles[0], "hook")
        self.assertEqual(roles[-1], "payoff")

    def test_every_published_clip_is_inside_final_window(self):
        result, _ = self.run_compose([("Нормальный", self.pieces(1, 9)), ("Длинный", self.pieces(1, 18))], {})
        for c in result:
            self.assertTrue(0.85 * 20 <= c["duration"] <= 60, c["duration"])

    def test_list_format_may_run_longer_than_thesis(self):
        # max 50: thesis с 54 с речи не проходит, list (до 1.4*50 = 70 с) проходит
        thesis, _ = self.run_compose([("Длинный", self.pieces(1, 18))], {}, {"max_duration": 50, "min_duration": 20})
        listing, _ = self.run_compose([("Длинный", self.pieces(1, 18))], {}, {"max_duration": 50, "min_duration": 20},
                                      fmt="list")
        self.assertEqual(list(thesis), [])
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["format"], "list")

    def test_only_clear_defects_are_dropped_and_scores_5_6_are_repaired(self):
        for key in ("completeness", "clarity", "standalone", "single_idea"):
            result, _ = self.run_compose([("A", self.pieces(1, 9))], {"A": {"scores": {key: 4}}})
            self.assertEqual(list(result), [], key)
            rej = result.rejected[0]
            self.assertEqual(rej["title"], "A")
            self.assertEqual(rej["rejected"]["stage"], "critic_round1")
            self.assertIn(f"{key} 4<=4", rej["rejected"]["reason"])
            near, llm = self.run_compose([("A", self.pieces(1, 9))], {"A": {"scores": {key: 6}}})
            self.assertEqual(len(near), 1, key)           # 6/10 — не отсев
            self.assertIn("repair", [c["task"] for c in llm.calls])
            self.assertTrue(near[0]["critic"]["repaired"])
            self.assertEqual(near[0]["critic"][key], 8)  # оценка пересчитана после ремонта
        ok, llm = self.run_compose([("A", self.pieces(1, 9))], {"A": {"scores": {
            "completeness": 7, "clarity": 7, "standalone": 7, "single_idea": 7, "hook_strength": 7}}})
        self.assertEqual(len(ok), 1)
        self.assertNotIn("repair", [c["task"] for c in llm.calls])   # идеальный ролик не чинится
        self.assertFalse(ok[0]["critic"]["hook_warning"])

    def test_weak_hook_is_not_dropped_warns_and_chief_rank_decides_order(self):
        weak = {"Слабый хук": {"scores": {"hook_strength": 3}}, "Сильный хук": {"scores": {"hook_strength": 7}}}
        result, _ = self.run_compose(
            [("Слабый хук", self.pieces(1, 9)), ("Сильный хук", self.pieces(11, 19))], weak,
            repair={"Слабый хук": {"scores": {"hook_strength": 4}}},
            chief={"Слабый хук": {"rank": 2}, "Сильный хук": {"rank": 1}})
        self.assertEqual([c["title"] for c in result], ["Сильный хук", "Слабый хук"])
        self.assertEqual([c["chief_editor"]["rank"] for c in result], [1, 2])
        self.assertTrue(result[1]["critic"]["hook_warning"])
        self.assertFalse(result[0]["critic"]["hook_warning"])
        # без главреда (опция выключена) действует прежнее ранжирование критика: слабый хук ниже
        plain, _ = self.run_compose(
            [("Слабый хук", self.pieces(1, 9)), ("Сильный хук", self.pieces(11, 19))], weak,
            opts={"chief_editor": False}, repair={"Слабый хук": {"scores": {"hook_strength": 4}}})
        self.assertEqual([c["title"] for c in plain], ["Сильный хук", "Слабый хук"])

    def test_weak_hook_repaired_by_critic_fix_in_round_one(self):
        # допустимая правка входа: отбросить ведущие предложения (порядок записи не нарушается)
        fix = {"Клип": {"verdict": "fix", "scores": {"hook_strength": 4},
                        "pieces": [{"from": "S0003", "to": "S0009", "role": "hook"}]}}
        result, llm = self.run_compose([("Клип", self.pieces(1, 9))], fix, plan2={"Клип": {}})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["segments"][0]["sentence_from"], "S0003")
        self.assertEqual(len(result[0]["critic"]["rounds"]), 2)

    def test_repair_that_leaves_hard_defect_rejects_but_keeps_candidate(self):
        low = {"A": {"scores": {"single_idea": 5}}}
        result, _ = self.run_compose([("A", self.pieces(1, 9))], low, repair={"A": {"scores": {"single_idea": 3}}})
        self.assertEqual(list(result), [])
        self.assertEqual(result.rejected[0]["rejected"]["stage"], "repair")
        still6, _ = self.run_compose([("A", self.pieces(1, 9))], low, repair={"A": {"scores": {"single_idea": 6}}})
        self.assertEqual(len(still6), 1)   # осталось 6 — не отсев

    def test_missing_payoff_sends_hint_to_critic(self):
        _, llm = self.run_compose([("Без вывода", self.pieces(1, 9, "point"))], {})
        prompt = next(c["prompt"] for c in llm.calls if c["task"] == "critique")
        self.assertIn("ЗАМЕЧАНИЕ ПРОВЕРКИ: последний кусок не имеет роли payoff", prompt)
        result, llm = self.run_compose([("С выводом", self.pieces(1, 9, "payoff"))], {})
        prompt = next(c["prompt"] for c in llm.calls if c["task"] == "critique")
        self.assertNotIn("ЗАМЕЧАНИЕ ПРОВЕРКИ", prompt.split("РОЛИКИ:")[1])
        self.assertEqual(len(result), 1)

    def test_missing_payoff_hint_recorded_in_critic_problems(self):
        result, _ = self.run_compose([("Без вывода", self.pieces(1, 9, "point"))], {})
        self.assertTrue(any("payoff" in p for p in result[0]["critic"]["problems"]))

    def test_jump_seam_goes_to_repair_and_is_dropped_only_if_unresolved(self):
        two = [{"from": "S0001", "to": "S0005", "role": "hook"}, {"from": "S0012", "to": "S0016", "role": "payoff"}]
        result, llm = self.run_compose([("Прыжок", two)], {"Прыжок": {"seam": "jump"}},
                                       repair={"Прыжок": {"seam": "jump"}})
        self.assertEqual(list(result), [])
        self.assertEqual(result.rejected[0]["rejected"]["stage"], "repair")
        self.assertIn("стык jump не снят", result.rejected[0]["rejected"]["reason"])
        prompt = next(c["prompt"] for c in llm.calls if c["task"] == "critique")
        self.assertIn("Стык 1:", prompt)
        self.assertIn("в записи позже на", prompt)
        repair_prompt = next(c["prompt"] for c in llm.calls if c["task"] == "repair")
        self.assertIn("ПРОБЛЕМЫ (обязательно исправить)", repair_prompt)
        self.assertIn("ДОСТУПНЫЕ ПРЕДЛОЖЕНИЯ ВОКРУГ", repair_prompt)
        healed, _ = self.run_compose([("Прыжок", two)], {"Прыжок": {"seam": "jump"}},
                                     repair={"Прыжок": {"pieces": self.pieces(1, 9)}})
        self.assertEqual(len(healed), 1)
        self.assertEqual(len(healed[0]["segments"]), 1)
        ok, _ = self.run_compose([("Прыжок", two)], {"Прыжок": {"seam": "ok"}})
        self.assertEqual(len(ok), 1)
        self.assertEqual(ok[0]["seams"][0]["verdict"], "ok")
        self.assertEqual(ok[0]["seams"][0]["from_sentence"], "S0005")
        self.assertEqual(ok[0]["seams"][0]["to_sentence"], "S0012")

    def test_fix_for_jump_goes_to_round_two_then_near_miss_is_repaired_not_dropped(self):
        two = [{"from": "S0001", "to": "S0005", "role": "hook"}, {"from": "S0012", "to": "S0016", "role": "payoff"}]
        fix = {"Прыжок": {"seam": "jump", "verdict": "fix", "pieces": self.pieces(1, 9)}}
        result, llm = self.run_compose([("Прыжок", two)], fix, plan2={"Прыжок": {}})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["critic"]["verdict"], "fix")
        self.assertEqual(len(result[0]["segments"]), 1)
        self.assertEqual([c["task"] for c in llm.calls].count("critique_2"), 1)
        self.assertNotIn("repair", [c["task"] for c in llm.calls])
        # раунд 2 не прошёл (fix / оценка 5): ролик не отсеивается, а чинится в раунде 3
        for plan2 in ({"Прыжок": {"verdict": "fix"}}, {"Прыжок": {"scores": {"clarity": 5}}}):
            res, llm2 = self.run_compose([("Прыжок", two)], fix, plan2=plan2)
            self.assertEqual(len(res), 1)
            self.assertIn("repair", [c["task"] for c in llm2.calls])
            self.assertTrue(res[0]["critic"]["repaired"])
        dead = self.run_compose([("Прыжок", two)], fix, plan2={"Прыжок": {"scores": {"clarity": 4}}})[0]
        self.assertEqual(list(dead), [])
        self.assertEqual(dead.rejected[0]["rejected"]["stage"], "critic_round2")

    def test_more_than_three_regions_dropped(self):
        far = [{"from": f"S{a:04d}", "to": f"S{a:04d}", "role": "point"} for a in (1, 9, 17, 25)]
        far.append({"from": "S0030", "to": "S0030", "role": "payoff"})
        tr, index = make_transcript(40)
        regions = editorial._count_regions(editorial._normalize_pieces(far, {f"S{n:04d}": n - 1 for n in range(1, 41)}),
                                           editorial.build_sentences(tr), gap=10.0)
        self.assertEqual(regions, 5)
        self.assertEqual(editorial._count_regions(editorial._normalize_pieces(far[:2], {f"S{n:04d}": n - 1 for n in range(1, 41)}), editorial.build_sentences(tr)), 1)

    def test_enumeration_cut_short_flags_hint(self):
        tr, index = make_transcript(30)
        sents = editorial.build_sentences(tr)
        sents[2]["text"] = "Во-первых, это важно."
        sents[10]["text"] = "Второе, это тоже важно."
        sents[14]["text"] = "И последнее, вывод."
        ids = {s["id"]: i for i, s in enumerate(sents)}
        topics = [{"title": "Причины", "kind": "list", "thesis": "t",
                   "ranges": [{"from": "S0001", "to": "S0020"}],
                   "items": [{"from": "S0003", "to": "S0006"}, {"from": "S0011", "to": "S0013"},
                             {"from": "S0015", "to": "S0017"}]}]
        pieces = editorial._normalize_pieces([{"from": "S0003", "to": "S0008", "role": "point"}], ids)
        hint = editorial._enum_hint(pieces, sents, topics, ids)
        self.assertIn("перечень оборван", hint)
        self.assertIn("S0011", hint)
        # без kind=list берутся порядковые слова внутри диапазона темы
        plain = [dict(topics[0], kind="thesis", items=[])]
        hint2 = editorial._enum_hint(pieces, sents, plain, ids)
        self.assertIn("S0011", hint2)
        self.assertIn("S0015", hint2)
        # весь перечень в ролике -> замечания нет
        full = editorial._normalize_pieces([{"from": "S0003", "to": "S0017", "role": "point"}], ids)
        self.assertIsNone(editorial._enum_hint(full, sents, topics, ids))

    def test_enumeration_hint_reaches_critic_prompt(self):
        topics = [{"title": "Причины", "kind": "list", "thesis": "t", "strength": 80,
                   "ranges": [{"from": "S0001", "to": "S0016"}],
                   "items": [{"from": "S0003", "to": "S0006"}, {"from": "S0012", "to": "S0014"}]}]
        _, llm = self.run_compose([("Перечень", self.pieces(3, 9))], {}, topics=topics, fmt="list")
        prompt = next(c["prompt"] for c in llm.calls if c["task"] == "critique")
        self.assertIn("перечень оборван", prompt)


class HookAndOrdinalsTest(unittest.TestCase):
    def sents(self, texts):
        words, segs = [], []
        for k, text in enumerate(texts):
            t0 = k * 4.0
            first = len(words)
            for j, tok in enumerate(text.split()):
                words.append({"text": tok.strip(".,?!"), "raw": tok, "start": t0 + j * 0.3, "end": t0 + j * 0.3 + 0.25,
                              "probability": 1.0})
            segs.append({"start": words[first]["start"], "end": words[-1]["end"], "text": text})
        tr = {"words": words, "segments": segs}
        return tr, editorial.build_sentences(tr)

    def test_hedge_start_detection(self):
        for weak in ("Ну, вот это важно знать всем.", "Это конечно хорошо но есть нюанс.",
                     "Моё мнение такое, что рынок жив.", "Смотрите, тут всё просто.", "Давайте поговорим про деньги."):
            self.assertTrue(editorial._hedge_start(weak), weak)
        for strong in ("Компании сокращают людей, потому что денег нет.", "70 процентов проблем из-за кризиса.",
                       "Почему упал рынок?"):
            self.assertFalse(editorial._hedge_start(strong), strong)

    def test_weak_entry_hint(self):
        tr, sents = self.sents(["Ну вот это конечно хорошо.", "Сокращают потому что денег нет."])
        ids = {s["id"]: i for i, s in enumerate(sents)}
        weak = editorial._normalize_pieces([{"from": "S0001", "to": "S0002", "role": "hook"}], ids)
        self.assertIn("слабый вход", editorial._hook_hint(weak, sents))
        # допустимая правка входа — начать с более сильного предложения (ведущее отброшено), а не переставить кусок
        strong = editorial._normalize_pieces([{"from": "S0002", "to": "S0002", "role": "hook"}], ids)
        self.assertIsNone(editorial._hook_hint(strong, sents))
        # перестановка «сильное предложение вперёд» больше невозможна: куски возвращаются в порядок записи
        reordered = editorial._normalize_pieces(
            [{"from": "S0002", "to": "S0002", "role": "hook"}, {"from": "S0001", "to": "S0001", "role": "point"}], ids)
        self.assertEqual([(p["a"], p["b"]) for p in reordered], [(0, 1)])
        self.assertIn("слабый вход", editorial._hook_hint(reordered, sents))

    def test_ordinal_conflicts_only_word_deleted(self):
        tr, sents = self.sents([
            "Первая причина это кризис.", "Первое это пузырь инвестиций.", "Второй пункт экономия.",
            "Четвертое это найм."])
        ids = {s["id"]: i for i, s in enumerate(sents)}
        pieces = editorial._normalize_pieces([{"from": "S0001", "to": "S0004", "role": "point"}], ids)
        ords = editorial._clip_ordinals(pieces, sents, tr["words"])
        self.assertEqual([n for _, n in ords], [1, 1, 2, 4])
        bad = editorial._ordinal_conflicts(ords)
        self.assertEqual([tr["words"][i]["text"] for i in bad], ["Первое", "Четвертое"])
        hint = editorial._ordinal_hint({"format": "list", "pieces": pieces},
                                       {"sentences": sents, "words": tr["words"]})
        self.assertIn("непоследовательны", hint)
        self.assertIsNone(editorial._ordinal_hint({"format": "thesis", "pieces": pieces},
                                                  {"sentences": sents, "words": tr["words"]}))

    def test_tighten_removes_conflicting_ordinals_and_leading_hedge(self):
        tr, sents = self.sents([
            "Ну вот это важная вещь для всех.", "Первая причина это кризис в мире.", "Первое это пузырь ожиданий.",
            "Второй пункт это экономия денег."])
        ids = {s["id"]: i for i, s in enumerate(sents)}
        pieces = editorial._normalize_pieces([{"from": "S0001", "to": "S0004", "role": "point"}], ids)
        draft = {"raw": {"title": "T"}, "pieces": pieces, "format": "list"}

        def llm(*a, **k):
            return {"clips": []}

        with tempfile.TemporaryDirectory() as d:
            cuts = editorial._tighten([draft], sents, tr["words"], llm, None, Path(d), {})[0]
        removed = {tr["words"][i]["text"]: r for a, b, r in cuts for i in range(a, b + 1)}
        self.assertEqual(removed["Первое"], "ordinal")
        self.assertEqual(removed["Ну"], "filler")
        self.assertEqual(removed["вот"], "filler")
        self.assertNotIn("Первая", removed)
        self.assertNotIn("Второй", removed)

    def test_prompts_carry_hook_and_ordinal_rules(self):
        compose = editorial._compose_prompt("S0001 [00:01, 3с] т", "- т", max_clips=3, min_d=30, max_d=90)
        self.assertIn("ВХОД РОЛИКА", compose)
        self.assertIn("«это конечно хорошо»", compose)
        self.assertIn("РАЗНЫЕ темы", compose)
        critique = editorial._critique_prompt("РОЛИК 1", min_d=30, max_d=90)
        self.assertIn("hook_strength не выше 5", critique)
        self.assertIn("ПОРЯДКОВЫЕ СЛОВА", critique)
        tighten = editorial._tighten_prompt("#1 ну")
        self.assertIn("ordinal", tighten)
        self.assertIn("ПЕРВОГО предложения", tighten)

    def test_ranking_by_critic_scores_then_score(self):
        tr, index = make_transcript()

        class L(FakeLLM):
            def task_topic_map(s, prompt):
                return {"topics": []}

            def task_compose(s, prompt):
                base = {"description": "d", "hook_text": "h", "rationale": "r", "thesis": "T", "format": "thesis"}
                P = lambda a, b: [{"from": f"S{a:04d}", "to": f"S{b:04d}", "role": "payoff"}]
                return {"clips": [dict(base, title="Слабее", score=95, pieces=P(1, 9)),
                                  dict(base, title="Сильнее", score=60, pieces=P(11, 19))]}

            def task_tighten(s, prompt):
                return {"clips": []}

        llm = L(index, {"Слабее": {"scores": {"clarity": 7}}, "Сильнее": {}},
                chief_plan={"Слабее": {"rank": 2}, "Сильнее": {"rank": 1}})
        with tempfile.TemporaryDirectory() as d:
            out = editorial.compose_clips(tr, Path(d), {"max_clips": 1, "min_duration": 20, "max_duration": 60}, llm=llm)
            saved = json.loads((Path(d) / "editorial_result.json").read_text(encoding="utf-8"))
        self.assertEqual([c["title"] for c in out], ["Сильнее"])
        self.assertEqual(saved["dropped"][0]["stage"], "ranking")
        self.assertEqual(saved["dropped"][0]["title"], "Слабее")


class CodeRulesTest(unittest.TestCase):
    def make_ctx(self, texts, gap=0.2):
        sents, t = [], 0.0
        for k, text in enumerate(texts):
            sents.append({"id": f"S{k + 1:04d}", "start": t, "end": t + 3.0, "text": text, "w0": k * 5, "w1": k * 5 + 4})
            t += 3.0 + gap
        return {"sentences": sents, "id_index": {s["id"]: i for i, s in enumerate(sents)}, "topics": [], "words": [],
                "min_d": 6, "max_d": 60, "options": {}}

    def draft(self, a, b, fmt="thesis"):
        return {"raw": {"title": "T"}, "format": fmt, "thesis": "t", "pieces": [{"a": a, "b": b, "role": "payoff"}]}

    def test_continuation_start_extends_back_to_previous_sentence(self):
        ctx = self.make_ctx(["Это история про найм.", "и тут начались проблемы с деньгами.", "Вывод простой.", "Вот так."])
        d = self.draft(1, 2)
        self.assertTrue(editorial._apply_code_rules(d, ctx))
        self.assertEqual([(p["a"], p["b"]) for p in d["pieces"]], [(0, 2)])

    def test_lowercase_fragment_of_split_sentence_extends_but_lowercase_after_period_does_not(self):
        frag = self.make_ctx(["Мы пришли в компанию и увидели что", "рук не хватает совсем.", "Вывод."])
        d = self.draft(1, 2)
        self.assertTrue(editorial._is_continuation_start(1, frag["sentences"]))
        editorial._apply_code_rules(d, frag)
        self.assertEqual(d["pieces"][0]["a"], 0)
        whole = self.make_ctx(["Мы пришли в компанию.", "рук не хватает совсем.", "Вывод."])
        self.assertFalse(editorial._is_continuation_start(1, whole["sentences"]))   # whisper пишет с маленькой после точки
        d2 = self.draft(1, 2)
        self.assertFalse(editorial._apply_code_rules(d2, whole))

    def test_continuation_chain_extended_up_to_three_else_trimmed_forward_else_left_to_tighten(self):
        # растём назад на 1–3 предложения, пока начало не станет самостоятельным
        ctx = self.make_ctx(["Начало мысли про найм.", "и вот тогда", "и потом", "и наконец вывод.", "Итог."])
        d = self.draft(3, 4)
        editorial._apply_code_rules(d, ctx)
        self.assertEqual(d["pieces"][0]["a"], 0)
        # цепочка длиннее трёх назад не лечится: пробуем снять обрывок вперёд
        ctx2 = self.make_ctx(["Начало.", "и раз", "и два", "и три", "и четыре", "Нормальное начало вывода.", "Итог."])
        d2 = self.draft(4, 6)
        editorial._apply_code_rules(d2, ctx2)
        self.assertEqual(d2["pieces"][0]["a"], 5)
        # не лечится ни так ни так: кусок остаётся как есть (ведущее «и» снимет чистка)
        ctx3 = self.make_ctx(["Начало.", "и раз", "и два", "и три", "и четыре", "и пять"])
        d3 = self.draft(5, 5)
        self.assertFalse(editorial._apply_code_rules(d3, ctx3))

    def test_fragment_hook_that_continues_another_piece_is_moved_after_it(self):
        # «рук не хватает» — хвост предложения из другого куска: хук-обрывок ставится вслед за ним, куски склеиваются
        ctx = self.make_ctx(["Тут контекст про найм.", "Это вопрос про найм и про то, что рук,", "рук не хватает совсем.", "Вывод."])
        d = {"raw": {"title": "T"}, "format": "thesis", "thesis": "t",
             "pieces": [{"a": 2, "b": 2, "role": "hook"}, {"a": 0, "b": 1, "role": "context"},
                        {"a": 3, "b": 3, "role": "payoff"}]}
        ctx["min_d"] = 3
        editorial._apply_code_rules(d, ctx)
        self.assertEqual([(p["a"], p["b"]) for p in d["pieces"]], [(0, 3)])
        self.assertEqual(d["pieces"][-1]["role"], "payoff")

    def test_single_fragment_piece_dropped_when_clip_stays_long_enough(self):
        ctx = self.make_ctx(["Начало.", "и раз", "Вот отдельная мысль.", "Вывод."])
        d = {"raw": {"title": "T"}, "format": "thesis", "thesis": "t",
             "pieces": [{"a": 0, "b": 0, "role": "hook"}, {"a": 1, "b": 1, "role": "context"}] }
        d["pieces"][1] = {"a": 1, "b": 1, "role": "context"}
        d["pieces"] = [{"a": 3, "b": 3, "role": "payoff"}, {"a": 1, "b": 1, "role": "context"}, {"a": 2, "b": 2, "role": "point"}]
        ctx["min_d"] = 3
        editorial._apply_code_rules(d, ctx)
        self.assertNotIn(1, [p["a"] for p in d["pieces"]])

    def test_hedge_before_continuation_word_still_counts_as_continuation(self):
        ctx = self.make_ctx(["Растёт конкуренция.", "вот, плюс опять же идёт гонка.", "Вывод."])
        self.assertTrue(editorial._is_continuation_start(1, ctx["sentences"]))
        d = self.draft(1, 2)
        editorial._apply_code_rules(d, ctx)
        self.assertEqual(d["pieces"][0]["a"], 0)
        ctx2 = self.make_ctx(["Растёт конкуренция.", "Вот что я думаю про это.", "Вывод."])
        self.assertFalse(editorial._is_continuation_start(1, ctx2["sentences"]))   # «вот что» — не продолжение
        ctx3 = self.make_ctx(["Растёт конкуренция.", "вот, то есть это важно.", "Вывод."])
        self.assertTrue(editorial._is_continuation_start(1, ctx3["sentences"]))

    def test_extension_does_not_steal_sentence_from_other_piece(self):
        ctx = self.make_ctx(["Начало мысли.", "и продолжение.", "Конец."])
        d = {"raw": {"title": "T"}, "format": "thesis", "thesis": "t",
             "pieces": [{"a": 0, "b": 0, "role": "hook"}, {"a": 1, "b": 2, "role": "payoff"}]}
        editorial._apply_code_rules(d, ctx)
        self.assertEqual([(p["a"], p["b"]) for p in d["pieces"]], [(0, 2)])   # склеились без дублей

    def test_service_tail_is_trimmed_only_if_clip_still_ends_on_complete_sentence(self):
        ctx = self.make_ctx(["Главная мысль.", "Это и есть вывод.", "В чате правильно написали.",
                             "Я про это как-то пост писал."])
        d = self.draft(0, 3)
        editorial._apply_code_rules(d, ctx)
        self.assertEqual(d["pieces"][-1]["b"], 1)
        # без точки перед хвостом хвост остаётся: ролик не заканчивался бы законченной фразой
        ctx2 = self.make_ctx(["Главная мысль.", "Это и есть вывод и", "в чате правильно написали."])
        d2 = self.draft(0, 2)
        self.assertFalse(editorial._apply_code_rules(d2, ctx2))
        # вопрос к зрителям — тоже служебный хвост
        ctx3 = self.make_ctx(["Главная мысль.", "Это и есть вывод.", "А вы как думаете?"])
        d3 = self.draft(0, 2)
        editorial._apply_code_rules(d3, ctx3)
        self.assertEqual(d3["pieces"][-1]["b"], 1)

    def test_service_tail_never_empties_the_clip(self):
        ctx = self.make_ctx(["Подписывайтесь на канал в чате."])
        d = self.draft(0, 0)
        self.assertFalse(editorial._apply_code_rules(d, ctx))


class ChiefEditorTest(ComposeRunner, unittest.TestCase):
    TOPICS = [
        {"title": "Сильная тема", "thesis": "т", "kind": "thesis", "strength": 90,
         "ranges": [{"from": "S0001", "to": "S0009"}]},
        {"title": "Перечень", "thesis": "т", "kind": "list", "strength": 40,
         "ranges": [{"from": "S0011", "to": "S0019"}]},
    ]

    def clips(self):
        return [("Сильный", self.pieces(1, 9)), ("Перечень", self.pieces(11, 19))]

    def test_chief_prompt_carries_topics_candidates_and_decisions_are_recorded(self):
        result, llm = self.run_compose(self.clips(), {}, topics=self.TOPICS,
                                       chief={"Перечень": {"rank": 1, "reason": "ярко"}, "Сильный": {"rank": 2}})
        prompt = next(c["prompt"] for c in llm.calls if c["task"] == "chief_editor")
        self.assertIn("КАРТА ТЕМ", prompt)
        self.assertIn("ОБЯЗАТЕЛЬНА К ПОКРЫТИЮ", prompt)
        self.assertIn("=== КАНДИДАТ 1: Сильный ===", prompt)
        self.assertIn("ОЦЕНКИ КРИТИКА", prompt)
        self.assertIn("выложил бы я это на наш канал", prompt)
        self.assertEqual([c["title"] for c in result], ["Перечень", "Сильный"])   # порядок = ранг главреда
        self.assertEqual(result[0]["chief_editor"]["reason"], "ярко")
        self.assertEqual(result[0]["chief_editor"]["rank"], 1)

    def test_drop_keeps_candidate_in_rejected_and_title_is_fixed(self):
        result, _ = self.run_compose(
            self.clips(), {}, topics=[], chief={"Перечень": {"decision": "drop", "reason": "заезжено"},
                                                "Сильный": {"title": "Новое название"}})
        self.assertEqual([c["title"] for c in result], ["Новое название"])
        self.assertTrue(result[0]["chief_editor"]["title_fixed"])
        self.assertEqual(result[0]["chief_editor"]["original_title"], "Сильный")
        rej = result.rejected[0]
        self.assertEqual((rej["title"], rej["rejected"]["stage"]), ("Перечень", "chief_editor"))
        self.assertIn("заезжено", rej["rejected"]["reason"])
        self.assertTrue(rej["segments"])

    def test_coverage_forces_strong_and_list_topics_back_into_the_issue(self):
        result, _ = self.run_compose(
            self.clips(), {}, topics=self.TOPICS,
            chief={"Перечень": {"decision": "drop", "reason": "скучно"}, "Сильный": {"decision": "drop", "reason": "слабо"}})
        self.assertEqual(sorted(c["title"] for c in result), ["Перечень", "Сильный"])
        self.assertTrue(all(c["chief_editor"].get("coverage_override") for c in result))
        self.assertEqual(result.rejected, [])

    def test_hard_dropped_clip_is_resurrected_for_uncovered_strong_topic(self):
        plan1 = {"Перечень": {"scores": {"completeness": 3}, "problems": ["нет вывода"]}}
        result, llm = self.run_compose(self.clips(), plan1, topics=self.TOPICS)
        prompt = next(c["prompt"] for c in llm.calls if c["task"] == "chief_editor")
        self.assertIn("ВОССТАНОВЛЕН для покрытия темы", prompt)
        self.assertEqual(sorted(c["title"] for c in result), ["Перечень", "Сильный"])
        resurrected = next(c for c in result if c["title"] == "Перечень")
        self.assertIn("resurrected", resurrected["chief_editor"])
        self.assertEqual(result.rejected, [])

    def test_repair_more_gets_one_more_repair_round_and_stays(self):
        result, llm = self.run_compose(
            self.clips(), {}, topics=[],
            chief={"Сильный": {"decision": "repair_more", "reason": "слабое начало"}})
        tasks = [c["task"] for c in llm.calls]
        self.assertEqual(tasks.count("repair_chief"), 1)
        prompt = next(c["prompt"] for c in llm.calls if c["task"] == "repair_chief")
        self.assertIn("замечание главного редактора: слабое начало", prompt)
        self.assertEqual(len(result), 2)

    def test_overlapping_clips_are_deduplicated_keeping_higher_rank(self):
        clips = [("Первый", self.pieces(1, 9)), ("Второй", self.pieces(5, 13))]
        result, _ = self.run_compose(clips, {}, topics=[], chief={"Первый": {"rank": 2}, "Второй": {"rank": 1}})
        self.assertEqual([c["title"] for c in result], ["Второй"])
        self.assertIn("дубль", result.rejected[0]["rejected"]["reason"])

    def test_chief_failure_degrades_to_critic_ranking(self):
        class Boom(FakeLLM):
            def task_chief_editor(self, prompt):
                raise RuntimeError("нет сети")

        tr, index = make_transcript()

        class L(Boom):
            def task_topic_map(s, prompt):
                return {"topics": []}

            def task_compose(s, prompt):
                base = {"description": "d", "hook_text": "h", "rationale": "r", "thesis": "T", "format": "thesis", "score": 70}
                return {"clips": [dict(base, title="Один", pieces=[{"from": "S0001", "to": "S0009", "role": "payoff"}])]}

            def task_tighten(s, prompt):
                return {"clips": []}

        with tempfile.TemporaryDirectory() as d:
            out = editorial.compose_clips(tr, Path(d), {"max_clips": 3, "min_duration": 20, "max_duration": 60},
                                          llm=L(index))
        self.assertEqual([c["title"] for c in out], ["Один"])
        self.assertNotIn("chief_editor", out[0])


class AllClipsModeTest(unittest.TestCase):
    def make(self, n_topics=9):
        tr, index = make_transcript(6 * n_topics + 12)
        topics = [{"title": f"Тема {k}", "thesis": "т", "kind": "list" if k == 1 else "thesis", "strength": 80,
                   "ranges": [{"from": f"S{3 + 6 * k:04d}", "to": f"S{3 + 6 * k + 5:04d}"}]} for k in range(n_topics)]
        calls = []

        class L(FakeLLM):
            def __call__(s, prompt, schema, work_dir, *, task, provider=None, effort="high", timeout=1800):
                calls.append((task, prompt))
                if task == "topic_map":
                    return {"topics": topics}
                if task.startswith("compose"):
                    titles = [t["title"] for t in topics if t["title"] in prompt]
                    out = []
                    for t in titles:
                        k = int(t.split()[-1])
                        a = 3 + 6 * k
                        base = {"description": "d", "hook_text": "h", "rationale": "r", "score": 70, "thesis": "T",
                                "format": "list" if k == 1 else "thesis", "topic": t}
                        out.append(dict(base, title=f"Ролик {k}", pieces=[
                            {"from": f"S{a:04d}", "to": f"S{a + 5:04d}", "role": "payoff"}]))
                        if k == 2:  # тема может дать второй ролик, но он — дубль первого
                            out.append(dict(base, title="Дубль 2", pieces=[
                                {"from": f"S{a:04d}", "to": f"S{a + 4:04d}", "role": "payoff"}]))
                        if k == 3:  # id вне показанной расшифровки не принимаются
                            out.append(dict(base, title="Чужой", pieces=[{"from": "S0066", "to": "S0070", "role": "payoff"}]))
                    return {"clips": out}
                if task.startswith("critique"):
                    return critique_response(prompt, {})
                return {"clips": []}

        with tempfile.TemporaryDirectory() as d:
            result = editorial.compose_clips(
                tr, Path(d), {"all_clips": True, "min_duration": 15, "max_duration": 60}, llm=L(index), provider="claude")
            saved = json.loads((Path(d) / "editorial_result.json").read_text(encoding="utf-8"))
        return result, saved, calls

    def test_batches_cover_every_topic_and_nothing_is_capped(self):
        result, saved, calls = self.make(9)
        compose_tasks = [t for t, _ in calls if t.startswith("compose")]
        self.assertEqual(compose_tasks, ["compose_b1", "compose_b2"])   # 9 тем по 7
        self.assertEqual(len(result), 9)
        self.assertEqual(sorted(c["topic"] for c in result), sorted(f"Тема {k}" for k in range(9)))
        self.assertEqual([c["format"] for c in result if c["topic"] == "Тема 1"], ["list"])
        stages = {d["title"]: d for d in saved["dropped"]}
        self.assertEqual(stages["Дубль 2"]["stage"], "compose")
        self.assertEqual(stages["Дубль 2"]["topic"], "Тема 2")
        self.assertEqual(stages["Чужой"]["stage"], "compose")
        self.assertIn("compose", saved["timings"])
        # пакет видит только свои темы: «Тема 8» нет в первом пакете
        first = next(p for t, p in calls if t == "compose_b1")
        self.assertNotIn("Тема 8", first)
        self.assertIn("ПАКЕТ ТЕМ", first)
        self.assertIn("kind=list ОБЯЗАТЕЛЬНО", first)

    def test_max_clips_zero_is_all_mode_and_critique_is_batched(self):
        result, saved, calls = self.make(14)
        crit = [t for t, _ in calls if t.startswith("critique")]
        self.assertEqual(crit, ["critique", "critique_b2", "critique_b3"])   # 14 роликов по 6
        self.assertEqual(len(result), 14)


class NormalizeTest(unittest.TestCase):
    def test_pieces(self):
        idx = {f"S{n:04d}": n - 1 for n in range(1, 11)}
        got = editorial._normalize_pieces([
            {"from": "S0005", "to": "S0003", "role": "weird"},
            {"from": "S0004", "to": "S0007", "role": "point"},   # пересечение -> остаётся S0006..S0007 и склеивается
            {"from": "S0001", "to": "S0002", "role": "hook"},
            {"from": "nope", "to": "nope2", "role": "hook"},
        ], idx)
        # естественный порядок: куски отсортированы по записи и склеены (раньше «hook» мог стоять после «point»)
        self.assertEqual([(p["a"], p["b"], p["role"]) for p in got], [(0, 6, "hook")])


if __name__ == "__main__":
    unittest.main()
