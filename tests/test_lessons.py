"""Уроки первого эфира, зашитые в библиотеку: естественный порядок, подрезка вместо отсева, длинные сильные ролики,
проход «пропущенные моменты», отсеянные с кусками, сырой звук, hook_text, аудитория, пакетный прогон."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from reel_factory import assembly, batch, editorial, highlights, review
from tests.test_editorial import SC, chief_response, critique_response, make_transcript, repair_response

OPTS = {"max_clips": 5, "min_duration": 20, "max_duration": 60}


def titles_of(prompt, pattern=r"=== РОЛИК (\d+): (.*?) ==="):
    return {int(n): t for n, t in re.findall(pattern, prompt)}


class LLM:
    """Подставной ИИ: ответы по задачам; незаданная задача падает (код обязан это пережить)."""

    def __init__(self, *, topics=None, clips=(), plan1=None, plan2=None, repair=None, chief=None, coverage=None,
                 split=None, trims=None, chief_trims=None, spinoffs=None, fmt="thesis"):
        self.topics, self.clips, self.fmt = topics or [], list(clips), fmt
        self.plan1, self.plan2, self.repair_plan, self.chief_plan = plan1 or {}, plan2 or {}, repair or {}, chief or {}
        self.coverage, self.split = coverage, split
        self.trims, self.chief_trims, self.spinoffs = trims or {}, chief_trims or {}, spinoffs or {}
        self.calls = []

    def __call__(self, prompt, schema, work_dir, *, task, provider=None, effort="high", timeout=1800):
        self.calls.append({"task": task, "prompt": prompt})
        name = "split" if task.startswith("split_") else re.sub(r"_b\d+$", "", task)
        handler = getattr(self, "task_" + name, None)
        if handler is None:
            raise RuntimeError(f"нет ответа для {task}")
        return handler(prompt)

    @property
    def tasks(self):
        return [c["task"] for c in self.calls]

    def prompt(self, task):
        return next(c["prompt"] for c in self.calls if c["task"] == task)

    def task_topic_map(self, prompt):
        return {"topics": self.topics}

    def task_compose(self, prompt):
        base = {"description": "d", "hook_text": "h", "rationale": "r", "score": 80, "thesis": "Тезис.", "format": self.fmt}
        return {"clips": [dict(base, title=t, pieces=p, **extra) for t, p, *rest in self.clips
                          for extra in [rest[0] if rest else {}]]}

    def _with_trims(self, data, prompt, key, mapping, pattern):
        names = titles_of(prompt, pattern)
        for v in data[key]:
            title = names.get(v["clip"])
            if title in mapping:
                v["trims"] = mapping[title]
        return data

    def task_critique(self, prompt):
        data = critique_response(prompt, self.plan1)
        return self._with_trims(data, prompt, "verdicts", self.trims, r"=== РОЛИК (\d+): (.*?) ===")

    def task_critique_2(self, prompt):
        return critique_response(prompt, self.plan2)

    def task_repair(self, prompt):
        return repair_response(prompt, self.repair_plan)

    task_repair_chief = task_repair

    def task_chief_editor(self, prompt):
        data = chief_response(prompt, self.chief_plan)
        names = titles_of(prompt, r"=== КАНДИДАТ (\d+): (.*?) ===")
        for x in data["decisions"]:
            title = names.get(x["clip"])
            if title in self.chief_trims:
                x["trims"] = self.chief_trims[title]
            if title in self.spinoffs:
                x["spinoffs"] = self.spinoffs[title]
        return data

    def task_tighten(self, prompt):
        return {"clips": []}

    def task_coverage(self, prompt):
        if self.coverage is None:
            return {"clips": []}
        return {"clips": self.coverage}

    def task_split(self, prompt):
        if self.split is None:
            raise RuntimeError("split недоступен")
        return {"clips": self.split}


def P(a, b, role="point"):
    return {"from": f"S{a:04d}", "to": f"S{b:04d}", "role": role}


def run(llm, n=30, opts=None):
    tr, index = make_transcript(n)
    options = dict(OPTS)
    options.update(opts or {})
    with tempfile.TemporaryDirectory() as d:
        result = editorial.compose_clips(tr, Path(d), options, llm=llm)
    return result, tr, index


class NaturalOrderTest(unittest.TestCase):
    def test_teaser_from_later_is_not_moved_to_front(self):
        llm = LLM(clips=[("Тизер", [P(15, 15, "hook"), P(2, 8, "point")])])
        result, tr, _ = run(llm)
        self.assertEqual(len(result), 1)
        segs = result[0]["segments"]
        self.assertEqual([(s["sentence_from"], s["sentence_to"]) for s in segs], [("S0002", "S0008"), ("S0015", "S0015")])
        self.assertTrue(editorial.segments_chronological(segs))
        self.assertNotIn("rehook", llm.tasks)           # этап пакетного «перехука» убран из потока

    def test_return_to_same_thought_later_is_allowed(self):
        llm = LLM(clips=[("Возврат", [P(2, 6, "point"), P(12, 16, "payoff")])])
        result, _, _ = run(llm)
        self.assertEqual([(s["sentence_from"], s["sentence_to"]) for s in result[0]["segments"]],
                         [("S0002", "S0006"), ("S0012", "S0016")])

    def test_critic_fix_that_reorders_is_sorted_back(self):
        llm = LLM(clips=[("Клип", [P(1, 9, "payoff")])],
                  plan1={"Клип": {"verdict": "fix", "problems": ["слабый хук"],
                                  "pieces": [P(9, 9, "hook"), P(1, 8, "payoff")]}})
        result, _, _ = run(llm)
        self.assertEqual([(s["sentence_from"], s["sentence_to"]) for s in result[0]["segments"]], [("S0001", "S0009")])

    def test_repair_that_reorders_is_rejected_as_unapplicable(self):
        llm = LLM(clips=[("Клип", [P(1, 9, "payoff")])], plan1={"Клип": {"scores": {"single_idea": 6}}},
                  repair={"Клип": {"pieces": [P(9, 9, "hook"), P(1, 8, "payoff")]}})
        result, _, _ = run(llm)
        self.assertEqual(result[0]["repair"]["status"], "unapplicable")
        self.assertEqual([(s["sentence_from"], s["sentence_to"]) for s in result[0]["segments"]], [("S0001", "S0009")])

    def test_apply_fix_allow_reorder_flag(self):
        tr, _ = make_transcript(20)
        sentences = editorial.build_sentences(tr)
        ctx = {"sentences": sentences, "id_index": {s["id"]: i for i, s in enumerate(sentences)}, "max_d": 60, "min_d": 20,
               "lo_limit": 12}
        d = {"pieces": [{"a": 0, "b": 8, "role": "payoff"}], "format": "thesis"}
        v = {"pieces": [P(9, 9, "hook"), P(1, 8, "payoff")]}
        self.assertIsNone(editorial._apply_fix(v, d, ctx, allow_reorder=False))
        fixed = editorial._apply_fix(v, d, ctx, allow_reorder=True)
        self.assertEqual([(p["a"], p["b"]) for p in fixed], [(0, 8)])
        self.assertTrue(editorial.is_chronological(fixed))

    def test_prompts_never_ask_to_move_hook_to_front(self):
        llm = LLM(clips=[("Клип", [P(1, 9, "payoff")])], plan1={"Клип": {"scores": {"single_idea": 6}}})
        run(llm)
        for task in ("compose", "critique", "repair", "chief_editor"):
            self.assertIn("хронологическ", llm.prompt(task).lower() if task != "chief_editor" else llm.prompt(task).lower())
        self.assertNotIn("из любого места, в том числе из конца эфира", llm.prompt("compose"))

    def test_opening_may_only_extend_backwards(self):
        tr, _ = make_transcript(10)
        sentences = editorial.build_sentences(tr)
        sentences[4] = dict(sentences[4], text="Они были под руководством этого человека.")
        fixed = editorial._fix_opening([{"a": 4, "b": 6, "role": "hook"}], sentences, max_total=100)
        self.assertEqual((fixed[0]["a"], fixed[0]["b"]), (3, 6))


class LocalDefectTest(unittest.TestCase):
    def words_of(self, tr, idx, sentence, a, b):
        return tr["words"][idx[(sentence, a)]]["start"], tr["words"][idx[(sentence, b)]]["end"]

    def test_critic_trim_cuts_words_and_keeps_clip(self):
        trims = {"A": [{"sentence": "S0005", "text": "важную тему которая", "reason": "обрывок"}]}
        llm = LLM(clips=[("A", [P(1, 9, "payoff")])], trims=trims)
        result, tr, idx = run(llm)
        self.assertEqual([c["title"] for c in result], ["A"])
        start, end = self.words_of(tr, idx, 5, 4, 6)
        defect = [c for c in result[0]["cuts"] if c["reason"] == "defect"]
        self.assertEqual(len(defect), 1)
        self.assertAlmostEqual(defect[0]["start"], start, places=3)
        self.assertAlmostEqual(defect[0]["end"], end, places=3)
        self.assertEqual(result[0]["local_fixes"][0]["sentence"], "S0005")
        self.assertEqual(list(result.rejected), [])

    def test_dangling_fragment_at_seam_is_trimmed_not_rejected(self):
        two = [P(1, 5, "point"), P(12, 16, "payoff")]
        trims = {"Стык": [{"sentence": "S0012", "text": "Предложение номер 12 про"}]}
        llm = LLM(clips=[("Стык", two)], plan1={"Стык": {"seam": "jump"}}, trims=trims)
        result, _, _ = run(llm)
        self.assertEqual(len(result), 1)
        self.assertNotIn("repair", llm.tasks)
        self.assertEqual(result[0]["seams"][0]["verdict"], "ok")
        self.assertTrue(any(c["reason"] == "defect" for c in result[0]["cuts"]))
        # без вырезов тот же jump уходит в ремонт и, не снятый, отсеивает ролик
        llm2 = LLM(clips=[("Стык", two)], plan1={"Стык": {"seam": "jump"}}, repair={"Стык": {"seam": "jump"}})
        result2, _, _ = run(llm2)
        self.assertEqual(list(result2), [])
        self.assertEqual(result2.rejected[0]["rejected"]["stage"], "repair")

    def test_chief_drop_with_trims_becomes_publish_with_cut(self):
        llm = LLM(clips=[("A", [P(1, 9, "payoff")])],
                  chief={"A": {"decision": "drop", "reason": "обрывок в кавычках"}},
                  chief_trims={"A": [{"sentence": "S0004", "text": "про важную тему"}]})
        result, _, _ = run(llm)
        self.assertEqual([c["title"] for c in result], ["A"])
        self.assertEqual(result[0]["chief_editor"]["decision"], "publish")
        self.assertTrue(result[0]["chief_editor"]["local_fix"])
        self.assertTrue(any(c["reason"] == "defect" for c in result[0]["cuts"]))

    def test_chief_drop_without_trims_still_drops(self):
        llm = LLM(clips=[("A", [P(1, 9, "payoff")])], chief={"A": {"decision": "drop", "reason": "заезженная мысль"}})
        result, _, _ = run(llm)
        self.assertEqual(list(result), [])
        self.assertEqual(result.rejected[0]["rejected"]["stage"], "chief_editor")

    def test_core_score_le_4_is_hard_reject_even_with_trims(self):
        llm = LLM(clips=[("A", [P(1, 9, "payoff")])], plan1={"A": {"scores": {"completeness": 4}}},
                  trims={"A": [{"sentence": "S0005", "text": "важную тему"}]})
        result, _, _ = run(llm)
        self.assertEqual(list(result), [])
        self.assertEqual(result.rejected[0]["rejected"]["stage"], "critic_round1")

    def test_unfound_phrase_is_recorded_not_fatal(self):
        llm = LLM(clips=[("A", [P(1, 9, "payoff")])], trims={"A": [{"sentence": "S0005", "text": "такого нет в тексте"}]})
        result, _, _ = run(llm)
        self.assertEqual(len(result), 1)
        self.assertFalse([c for c in result[0]["cuts"] if c["reason"] == "defect"])

    def test_locate_phrase(self):
        tr, idx = make_transcript(6)
        sentences = editorial.build_sentences(tr)
        words = tr["words"]
        loc = editorial._locate_phrase(sentences[4], words, "Важную, тему которая!")
        self.assertEqual(loc, (idx[(5, 4)], idx[(5, 6)]))
        self.assertIsNone(editorial._locate_phrase(sentences[4], words, "совсем другая фраза целиком"))


class LongClipTest(unittest.TestCase):
    OPT = {"min_duration": 20, "max_duration": 50}

    def test_caps(self):
        strong, weak = {"completeness": 8, "single_idea": 7}, {"completeness": 8, "single_idea": 6}
        self.assertEqual(editorial.final_cap("list", 90, strong), 144)
        self.assertEqual(editorial.final_cap("story", 90, strong), 144)
        self.assertAlmostEqual(editorial.final_cap("list", 90, weak), 126)
        self.assertEqual(editorial.final_cap("story", 90, weak), 90)
        self.assertEqual(editorial.final_cap("thesis", 90, strong), 90)
        self.assertEqual(editorial.final_cap("list", 90, None), 144)   # ручной кандидат без оценок не режется

    def test_strong_list_runs_up_to_1_6x(self):
        llm = LLM(clips=[("Перечень", [P(1, 23, "payoff")])], fmt="list")
        result, _, _ = run(llm, n=40, opts=self.OPT)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]["duration"], 50 * 1.4 * 1.05)      # дольше прежнего потолка x1.4
        self.assertLessEqual(result[0]["duration"], 50 * 1.6 * 1.05)

    def test_same_list_without_strong_rating_is_not_extended(self):
        weak = {"Перечень": {"scores": {"completeness": 6}}}
        llm = LLM(clips=[("Перечень", [P(1, 23, "payoff")])], fmt="list", plan1=weak, repair=weak)
        result, _, _ = run(llm, n=40, opts=self.OPT)
        self.assertEqual(list(result), [])
        self.assertEqual(result.rejected[0]["rejected"]["stage"], "duration")
        self.assertTrue(result.rejected[0]["segments"])                  # отсеянный остаётся с кусками

    def test_strong_story_runs_up_to_1_6x_too(self):
        llm = LLM(clips=[("История", [P(1, 23, "payoff")])], fmt="story")
        result, _, _ = run(llm, n=40, opts=self.OPT)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]["duration"], 50 * 1.05)

    def test_beyond_cap_is_split_into_two_standalone_clips_by_llm(self):
        split = [{"title": "Часть один", "thesis": "т1", "hook_text": "Первая половина перечня", "format": "list",
                  "pieces": [P(1, 18, "payoff")]},
                 {"title": "Часть два", "thesis": "т2", "hook_text": "Вторая половина перечня", "format": "list",
                  "pieces": [P(19, 36, "payoff")]}]
        llm = LLM(clips=[("Огромный перечень", [P(1, 36, "payoff")])], fmt="list", split=split)
        result, _, _ = run(llm, n=60, opts=self.OPT)
        self.assertEqual(sorted(c["title"] for c in result), ["Часть два", "Часть один"])
        self.assertTrue(all(c["split_of"] == "Огромный перечень" for c in result))
        self.assertTrue(all(editorial.segments_chronological(c["segments"]) for c in result))
        self.assertTrue(all(c["duration"] <= 50 * 1.6 * 1.05 for c in result))
        self.assertTrue(any(t.startswith("split_") for t in llm.tasks))
        self.assertEqual(list(result.rejected), [])                       # не отсеян, а разделён

    def test_split_falls_back_to_deterministic_halves(self):
        llm = LLM(clips=[("Огромный перечень", [P(1, 36, "payoff")])], fmt="list", split=[])
        result, _, _ = run(llm, n=60, opts=self.OPT)
        self.assertEqual(len(result), 2)
        self.assertEqual([c["segments"][-1]["sentence_to"] for c in result], ["S0018", "S0036"])
        self.assertTrue(all("часть" in c["title"] for c in result))

    def test_chief_spinoff_keeps_both_whole_clip_and_part(self):
        spin = {"Большой кейс": [{"title": "Отдельная часть", "thesis": "часть", "hook_text": "Самая сильная часть кейса",
                                   "format": "story", "reason": "самодостаточный кейс", "pieces": [P(11, 22, "payoff")]}]}
        llm = LLM(clips=[("Большой кейс", [P(1, 26, "payoff")])], fmt="story", spinoffs=spin)
        result, _, _ = run(llm, n=40, opts={"min_duration": 20, "max_duration": 60})
        titles = [c["title"] for c in result]
        self.assertEqual(titles, ["Большой кейс", "Отдельная часть"])
        spin_clip = result[1]
        self.assertEqual(spin_clip["spinoff_of"], "Большой кейс")
        self.assertEqual(spin_clip["chief_editor"]["decision"], "spinoff")
        self.assertEqual((spin_clip["segments"][0]["sentence_from"], spin_clip["segments"][-1]["sentence_to"]),
                         ("S0011", "S0022"))
        self.assertEqual([c["chief_editor"]["rank"] for c in result], [1, 2])

    def test_spinoff_must_stay_inside_parent(self):
        spin = {"Большой кейс": [{"title": "Чужая часть", "thesis": "х", "hook_text": "х", "format": "story",
                                   "pieces": [P(30, 38, "payoff")]}]}
        llm = LLM(clips=[("Большой кейс", [P(1, 26, "payoff")])], fmt="story", spinoffs=spin)
        result, _, _ = run(llm, n=40, opts={"min_duration": 20, "max_duration": 60})
        self.assertEqual([c["title"] for c in result], ["Большой кейс"])


class CoveragePassTest(unittest.TestCase):
    TOPICS = [{"title": "Тема", "thesis": "Тезис", "kind": "thesis", "ranges": [{"from": "S0001", "to": "S0009"}],
               "strength": 80, "appeal": "а"}]

    def coverage_clip(self, a, b, title="Реплика зала", moment="audience"):
        return {"title": title, "thesis": "Спикер отвечает залу.", "format": "thesis", "hook_text": "Меня даже за еду не берут",
                "description": "d", "score": 70, "rationale": "живая реплика", "moment": moment, "pieces": [P(a, b, "payoff")],
                "topic": "Тема"}

    def test_missed_moment_becomes_candidate_and_goes_through_critic(self):
        llm = LLM(topics=self.TOPICS, clips=[("Основной", [P(1, 9, "payoff")])],
                  coverage=[self.coverage_clip(25, 33)])
        result, _, _ = run(llm, n=40)
        by = {c["title"]: c for c in result}
        self.assertEqual(set(by), {"Основной", "Реплика зала"})
        self.assertEqual(by["Реплика зала"]["source"], "coverage")
        self.assertEqual(by["Реплика зала"]["moment"], "audience")
        self.assertNotIn("source", by["Основной"])
        tasks = llm.tasks
        self.assertLess(tasks.index("compose"), tasks.index("coverage"))
        self.assertLess(tasks.index("coverage"), tasks.index("critique"))
        critic_prompt = llm.prompt("critique")
        self.assertIn("Реплика зала", critic_prompt)                       # новый кандидат проверяется критиком

    def test_coverage_prompt_marks_covered_sentences_and_has_audience(self):
        llm = LLM(topics=self.TOPICS, clips=[("Основной", [P(1, 9, "payoff")])])
        run(llm, n=40)
        prompt = llm.prompt("coverage")
        self.assertIn("· S0001 ", prompt)
        self.assertIn("\nS0025 ", prompt)
        self.assertNotIn("· S0025 ", prompt)
        self.assertIn("накрутка опыта", prompt)                            # аудитория по умолчанию
        for word in ("цитата", "живая личная история", "комментарий или вопрос из зала", "цифрами"):
            self.assertIn(word, prompt)
        self.assertIn("СТРОГО в хронологическом порядке", prompt)

    def test_already_covered_moment_is_not_duplicated(self):
        llm = LLM(topics=self.TOPICS, clips=[("Основной", [P(1, 9, "payoff")])],
                  coverage=[self.coverage_clip(3, 9, "Дубль")])
        result, _, _ = run(llm, n=40)
        self.assertEqual([c["title"] for c in result], ["Основной"])
        dup = [c for c in result.rejected if c["title"] == "Дубль"]
        self.assertEqual(len(dup), 1)
        self.assertIn("уже покрыт", dup[0]["rejected"]["reason"])
        self.assertTrue(dup[0]["segments"])

    def test_coverage_can_be_disabled_and_failure_is_not_fatal(self):
        llm = LLM(clips=[("Основной", [P(1, 9, "payoff")])])
        result, _, _ = run(llm, opts={"coverage": False})
        self.assertNotIn("coverage", llm.tasks)
        self.assertEqual(len(result), 1)
        broken = LLM(clips=[("Основной", [P(1, 9, "payoff")])])
        broken.task_coverage = lambda prompt: (_ for _ in ()).throw(RuntimeError("упал"))
        result2, _, _ = run(broken)
        self.assertEqual(len(result2), 1)


class OptionsAndContractTest(unittest.TestCase):
    def test_audience_default_and_override(self):
        self.assertIn("накрутка опыта", editorial._audience({}))
        self.assertEqual(editorial._audience({"audience": " врачи "}), "врачи")
        self.assertEqual(editorial._audience({"audience": ""}), "")
        llm = LLM(clips=[("A", [P(1, 9, "payoff")])])
        run(llm, opts={"audience": "бухгалтеры на УСН"})
        self.assertIn("бухгалтеры на УСН", llm.prompt("compose"))
        self.assertIn("бухгалтеры на УСН", llm.prompt("chief_editor"))
        llm2 = LLM(clips=[("A", [P(1, 9, "payoff")])])
        run(llm2, opts={"audience": ""})
        self.assertNotIn("АУДИТОРИЯ СТРИМА", llm2.prompt("compose"))

    def test_default_is_all_clips_without_top_n_cap(self):
        tr, _ = make_transcript(60)

        class Many(LLM):
            def task_topic_map(s, prompt):
                return {"topics": [{"title": f"Т{i}", "thesis": "т", "kind": "thesis", "strength": 60, "appeal": "а",
                                    "ranges": [{"from": f"S{i * 10 + 1:04d}", "to": f"S{i * 10 + 9:04d}"}]} for i in range(5)]}

            def task_compose(s, prompt):
                nums = sorted({int(x) for x in re.findall(r"^S(\d{4}) ", prompt.split("РАСШИФРОВКА:")[1], re.M)})
                starts = [i * 10 + 1 for i in range(5) if i * 10 + 1 in nums]
                return {"clips": [{"title": f"Ролик {a}", "thesis": "т", "format": "thesis", "hook_text": "х",
                                   "description": "d", "score": 50, "rationale": "r", "topic": "",
                                   "pieces": [P(a, a + 8, "payoff")]} for a in starts]}

        llm = Many()
        with tempfile.TemporaryDirectory() as d:
            result = editorial.compose_clips(tr, Path(d), {"min_duration": 20, "max_duration": 60}, llm=llm)
        self.assertEqual(len(result), 5)                                   # без max_clips: никакого топ-N
        cands = highlights.normalize_candidates(list(result) + list(result.rejected), 300.0,
                                                {"min_duration": 20, "max_duration": 60}, tr["words"])
        self.assertEqual(len([c for c in cands if not c.get("rejected")]), 5)

    def test_raw_audio_is_default_everywhere(self):
        self.assertEqual(assembly.DEFAULT_AUDIO_MODE, "raw")
        self.assertEqual(assembly.normalize_audio_mode(None), "raw")
        self.assertEqual(assembly.normalize_audio_mode("loudnorm"), "raw")
        self.assertEqual(batch.DEFAULT_OPTIONS["audio_mode"], "raw")
        self.assertEqual(batch.build_options()["audio_mode"], "raw")
        script, _ = None, None
        _, script = assembly.build_filter_script([(0.0, 2.0), (5.0, 7.0)], has_audio=True, crossfade=assembly.RAW_CROSSFADE)
        self.assertNotIn("loudnorm", script)
        self.assertNotIn("volume=", script)

    def test_batch_defaults_follow_the_lessons(self):
        o = batch.build_options()
        self.assertEqual((o["max_clips"], o["all_clips"], o["chief_editor"], o["coverage"]), (0, True, True, True))
        self.assertEqual((o["video_review"], o["review_visual"], o["video_review_autofix"]), (True, False, True))
        self.assertEqual((o["min_duration"], o["max_duration"], o["model"], o["provider"]), (30, 90, "small", "claude"))
        self.assertFalse(batch.build_options(max_clips=5, all_clips=False)["all_clips"])

    def test_hook_text_is_short_title_for_visual_and_not_audio(self):
        long_hook = "Это очень длинный заголовок хука который не должен быть длиннее восьми слов совсем"
        llm = LLM(clips=[("A", [P(1, 9, "payoff")], {"hook_text": long_hook}), ("Без хука", [P(11, 19, "payoff")], {"hook_text": ""})])
        result, tr, _ = run(llm)
        by = {c["title"]: c for c in result}
        self.assertLessEqual(len(by["A"]["hook_text"].split()), editorial.HOOK_TEXT_MAX_WORDS)
        self.assertEqual(by["A"]["hook_text"], by["A"]["hook"])
        self.assertEqual(by["Без хука"]["hook_text"], "Без хука")          # запасной вариант — название
        # hook_text не входит в куски/звук: сегменты привязаны только к словам эфира
        words = tr["words"]
        for c in result:
            for s in c["segments"]:
                self.assertTrue(any(abs(float(w["start"]) - s["start"]) < 1e-6 for w in words))
        self.assertEqual(editorial.clean_hook_text("«Три причины»", "x"), "Три причины")
        norm = highlights.normalize_candidates([dict(result[0], rejected=None)], 100.0, {"min_duration": 20, "max_duration": 60}, tr["words"])
        self.assertTrue(norm and norm[0]["hook_text"])

    def test_app_options_default_to_all_clips_audience_and_raw_audio(self):
        from tests.test_app_runtime_contract import isolated_app_module
        with isolated_app_module() as app:
            o = app.HIGHLIGHTS._options({})
            self.assertEqual((o["max_clips"], o["all_clips"]), (0, True))
            self.assertIn("накрутка опыта", o["audience"])
            self.assertEqual(o["audio_mode"], "raw")
            self.assertEqual(app.HIGHLIGHTS._options({"audience": "врачи"})["audience"], "врачи")
            self.assertEqual(app.HIGHLIGHTS._options({"audience": ""})["audience"], "")
            self.assertFalse(app.HIGHLIGHTS._options({"max_clips": 5, "all_clips": False})["all_clips"])


class RejectedAndReviewTest(unittest.TestCase):
    def test_rejected_keep_pieces_stage_reason_scores_and_reach_the_report(self):
        llm = LLM(clips=[("Плохой", [P(1, 9, "payoff")]), ("Хороший", [P(11, 19, "payoff")])],
                  plan1={"Плохой": {"scores": {"clarity": 3}, "problems": ["нет связи"]}})
        result, tr, _ = run(llm)
        self.assertEqual([c["title"] for c in result], ["Хороший"])
        rej = result.rejected[0]
        info = rej["rejected"]
        self.assertEqual(info["stage"], "critic_round1")
        self.assertIn("clarity 3<=4", info["reason"])
        self.assertEqual(info["scores"]["clarity"], 3)
        self.assertTrue(rej["segments"] and rej["duration"] > 0 and rej["hook_text"])
        cands = highlights.normalize_candidates(list(result) + list(result.rejected), 100.0,
                                                {"min_duration": 20, "max_duration": 60}, tr["words"])
        self.assertEqual([c["title"] for c in cands if c.get("rejected")], ["Плохой"])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
            report = batch.build_report(Path("x.mp4"), out, batch.build_options(), cands, {}, {"editorial_total": 1.0})
            row = report["rejected"][0]
            self.assertEqual((row["title"], row["stage"]), ("Плохой", "critic_round1"))
            self.assertTrue(row["pieces"] and row["text"])
            self.assertTrue(report["summary"]["all_chronological"])
            page = batch.render_gallery(report)
            self.assertIn("Не прошли проверку", page)
            self.assertIn("Хук для визуала", page)
            self.assertIn("критик (1-й раунд)", page)

    def test_review_fix_never_breaks_natural_order_and_can_cut_phrase(self):
        tr, idx = make_transcript(20)
        sents = editorial.build_sentences(tr)
        words = tr["words"]
        segs = [{"start": words[sents[a - 1]["w0"]]["start"], "end": words[sents[b - 1]["w1"]]["end"], "role": r,
                 "sentence_from": f"S{a:04d}", "sentence_to": f"S{b:04d}"} for a, b, r in ((2, 4, "hook"), (6, 8, "payoff"))]
        cand = {"format": "thesis", "segments": segs, "cuts": [], "title": "T", "thesis": "t"}
        opts = {"min_duration": 8, "max_duration": 60}
        fix = lambda *edits: {"description": "d", "edits": list(edits)}
        upd = review.apply_suggested_fix(cand, fix({"op": "extend_piece_start", "piece": 2, "count": 3}), tr, opts)
        pieces = [(s["sentence_from"], s["sentence_to"]) for s in upd["segments"]]
        self.assertEqual(pieces, [("S0002", "S0008")])                     # расширение назад склеилось, дублей нет
        upd = review.apply_suggested_fix(
            cand, fix({"op": "cut_phrase", "sentence": "S0004", "text": "важную тему которая"}), tr, opts)
        defect = [c for c in upd["cuts"] if c["reason"] == "defect"]
        self.assertEqual(len(defect), 1)
        self.assertAlmostEqual(defect[0]["start"], words[idx[(4, 4)]]["start"], places=3)
        self.assertIsNone(review.apply_suggested_fix(
            cand, fix({"op": "cut_phrase", "sentence": "S0004", "text": "нет такой фразы"}), tr, opts))
        # длительность перепроверяется: слишком короткий остаток — правка отклонена
        self.assertIsNone(review.apply_suggested_fix(
            cand, fix({"op": "drop_piece", "piece": 1}), tr, {"min_duration": 60, "max_duration": 90}))

    def test_review_window_uses_final_cap(self):
        strong = {"completeness": 9, "single_idea": 9}
        captured = {}

        def fake_llm(prompt, schema, work_dir, **kw):
            captured["prompt"] = prompt
            return {"verdict": "publish", "scores": {k: 8 for k in review.SCORE_KEYS}, "issues": [], "summary": "ок"}

        tr, _ = make_transcript(10)
        cand = {"title": "T", "format": "list", "thesis": "t", "critic": strong,
                "segments": [{"start": 0.0, "end": 9.0, "role": "point", "sentence_from": "S0001", "sentence_to": "S0003"}],
                "cuts": []}
        with tempfile.TemporaryDirectory() as d:
            import unittest.mock as mock
            with mock.patch.object(review, "media_duration", return_value=100.0), \
                    mock.patch.object(review, "load_audio", return_value=(__import__("numpy").zeros(16000), 16000)):
                review.review_video(Path(d) / "x.mp4", candidate=cand, transcript=tr, work_dir=Path(d),
                                    options={"min_duration": 30, "max_duration": 90}, llm=fake_llm,
                                    transcribe=lambda v: {"words": []})
        self.assertIn("ДОПУСТИМАЯ ДЛИТЕЛЬНОСТЬ: 26–151 с", captured["prompt"])   # 90 x 1.6 x 1.05


class BatchRunnerTest(unittest.TestCase):
    def candidates(self):
        seg = {"start": 0.0, "end": 9.0, "role": "payoff", "sentence_from": "S0001", "sentence_to": "S0003"}
        base = {"duration": 30.0, "segments": [seg], "cuts": [], "format": "thesis", "thesis": "т", "hook_text": "х"}
        return [dict(base, id="c001", title="Первый", rejected=None), dict(base, id="c002", title="Второй", rejected=None),
                dict(base, id="c003", title="Отсеянный", rejected={"stage": "chief_editor", "reason": "дубль", "scores": {}}),
                dict(base, id="c004", title="Без кусков", segments=[], rejected={"stage": "compose", "reason": "x", "scores": {}})]

    def test_render_covers_kept_and_rejected_with_pieces_and_resumes(self):
        calls = []

        def fake_render(video, transcript, cand, out_dir, name, work_dir, options, review):
            calls.append((name, review))
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{name}_vertical.mp4").write_bytes(b"x")
            return {"ok": True, "seconds": 1.0, "candidate": cand, "text": "текст",
                    "review": {"verdict": "publish", "scores": {"hook": 8}, "timings": {"total": 0.4}} if review else None,
                    "auto_fix": None}

        import unittest.mock as mock
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            log = batch.RunLog(out / "run.log", echo=False)
            timings = {}
            with mock.patch.object(batch, "_render_one", fake_render):
                batch.stage_render(Path("v.mp4"), {"words": []}, self.candidates(), out, batch.build_options(), log, timings)
                self.assertEqual(calls, [("c001", True), ("c002", True), ("r003", False)])   # отсеянный с кусками — без отсмотра
                self.assertTrue((out / "rejected" / "r003_vertical.mp4").is_file())
                self.assertIn("render_kept", timings)
                calls.clear()
                batch.stage_render(Path("v.mp4"), {"words": []}, self.candidates(), out, batch.build_options(), log, {})
                self.assertEqual(calls, [])                                                    # возобновление: всё готово
            self.assertIn("уже готов", (out / "run.log").read_text(encoding="utf-8"))

    def test_cached_transcript_and_editorial_are_reused(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "transcript.json").write_text(json.dumps({"words": [{"text": "а", "start": 0, "end": 1}]}), encoding="utf-8")
            log = batch.RunLog(out / "run.log", echo=False)
            tr = batch.stage_transcribe(Path("v.mp4"), out, batch.build_options(), log, {})
            self.assertEqual(len(tr["words"]), 1)
            (out / "candidates.json").write_text(json.dumps(self.candidates()), encoding="utf-8")
            (out / "_work" / "editorial").mkdir(parents=True)
            (out / "_work" / "editorial" / "editorial_result.json").write_text(json.dumps({"dropped": []}), encoding="utf-8")
            cands, saved, _ = batch.stage_editorial(tr, out, batch.build_options(), log, {})
            self.assertEqual(len(cands), 4)

    def test_run_reels_cli_arguments(self):
        import run_reels
        args = run_reels.parse_args(["/x/a.mp4", "--out", "/tmp/o", "--provider", "codex", "--model", "base",
                                     "--min", "20", "--max", "70", "--no-review"])
        self.assertEqual((args.video, args.out, args.provider, args.model, args.min_duration, args.max_duration, args.no_review),
                         ("/x/a.mp4", "/tmp/o", "codex", "base", 20, 70, True))
        defaults = run_reels.parse_args(["a.mp4"])
        self.assertEqual((defaults.provider, defaults.model, defaults.min_duration, defaults.max_duration, defaults.out),
                         ("claude", "small", 30, 90, None))


if __name__ == "__main__":
    unittest.main()
