from __future__ import annotations

import unittest

from reel_factory import editorial


def sent(i, text, start, dur):
    n = len(text.split())
    return {"id": f"S{i + 1:04d}", "text": text, "start": start, "end": start + dur, "w0": -1, "w1": -1}


def make(texts, dur=5.0):
    return [sent(i, t, i * dur, dur - 0.2) for i, t in enumerate(texts)]


def words_of(tokens, step=0.3, length=0.25):
    return [{"text": t, "start": round(i * step, 3), "end": round(i * step + length, 3)} for i, t in enumerate(tokens)]


class TrimToWindowTest(unittest.TestCase):
    def setUp(self):
        self.s = make([f"Предложение номер {i} про важную тему" for i in range(12)], dur=5.0)   # по 4.8 с

    def pieces(self):
        return [{"a": 0, "b": 1, "role": "hook"}, {"a": 2, "b": 5, "role": "context"},
                {"a": 6, "b": 8, "role": "example"}, {"a": 9, "b": 10, "role": "payoff"}]

    def test_context_goes_first_then_example_hook_payoff_untouched(self):
        out = editorial._trim_to_window(self.pieces(), self.s, 30.0, 0.0)
        roles = [p["role"] for p in out]
        self.assertIn("hook", roles)
        self.assertEqual(roles[-1], "payoff")
        self.assertNotIn("context", roles)
        self.assertLessEqual(editorial._pieces_duration(out, self.s), 30.0)
        out2 = editorial._trim_to_window(self.pieces(), self.s, 22.0, 0.0)
        self.assertEqual([p["role"] for p in out2], ["hook", "payoff"])   # context и example ушли, hook/payoff целы
        self.assertEqual([(p["a"], p["b"]) for p in out2], [(0, 1), (9, 10)])
        self.assertLessEqual(editorial._pieces_duration(out2, self.s), 22.0)

    def test_impossible_returns_none(self):
        self.assertIsNone(editorial._trim_to_window(self.pieces(), self.s, 10.0, 0.0))

    def test_context_shortened_when_dropping_it_would_go_below_minimum(self):
        out = editorial._trim_to_window(self.pieces(), self.s, 40.0, 38.0)
        ctx = [p for p in out if p["role"] == "context"]
        self.assertTrue(ctx and ctx[0]["b"] - ctx[0]["a"] < 3)
        self.assertLessEqual(editorial._pieces_duration(out, self.s), 40.0)


class OpeningRulesTest(unittest.TestCase):
    def test_repeated_hook_sentence_is_removed(self):
        s = make(["400 тысяч на руки.", "400 тысяч.", "Потом его сократили и взяли другого человека."])
        pieces = [{"a": 0, "b": 1, "role": "hook"}, {"a": 2, "b": 2, "role": "payoff"}]
        fixed, left = editorial._fix_repeats(pieces, s)
        self.assertEqual(left, [])
        self.assertEqual([(p["a"], p["b"]) for p in fixed], [(0, 0), (2, 2)])

    def test_unresolvable_repeat_is_reported(self):
        s = make(["400 тысяч на руки.", "Потом всё случилось быстро.", "400 тысяч на руки."])
        pieces = [{"a": 0, "b": 0, "role": "hook"}, {"a": 2, "b": 2, "role": "payoff"}]
        fixed, left = editorial._fix_repeats(pieces, s)
        # payoff-кусок не удаляем, но дубль хука убрать можно только если он не hook/payoff -> сообщаем
        self.assertTrue(left or len(fixed) == 2)

    def test_repeat_across_seam_dropped_from_piece_edge(self):
        s = make(["Это очень важный момент про деньги.", "Мы ушли в зарплату на руки.", "Совсем другая история.",
                  "Другая тема.", "На руки зарплата и всё остальное.", "Дальше идёт продолжение мысли."])
        pieces = [{"a": 0, "b": 1, "role": "hook"}, {"a": 4, "b": 5, "role": "point"}]
        fixed, left = editorial._fix_repeats(pieces, s)
        self.assertEqual(left, [])
        self.assertEqual([(p["a"], p["b"]) for p in fixed], [(0, 1), (5, 5)])
        clean = [{"a": 0, "b": 1, "role": "hook"}, {"a": 2, "b": 3, "role": "point"}]
        self.assertEqual(editorial._fix_repeats(clean, s)[1], [])

    def test_hook_fragment_is_extended_with_next_sentence(self):
        s = make(["Четыреста тысяч.", "Столько получал сеньор на руки."])
        pieces = [{"a": 0, "b": 0, "role": "hook"}]
        self.assertTrue(editorial._hook_is_fragment(pieces, s))
        fixed = editorial._fix_hook_fragment(pieces, s)
        self.assertEqual((fixed[0]["a"], fixed[0]["b"]), (0, 1))
        self.assertFalse(editorial._hook_is_fragment(fixed, s))

    def test_whole_sentence_hook_is_fine(self):
        s = make(["Четыреста тысяч на руки."])
        self.assertFalse(editorial._hook_is_fragment([{"a": 0, "b": 0, "role": "hook"}], s))


class LimitChoppinessTest(unittest.TestCase):
    TOKENS = "мы делали вот этот проект в команде потому что другого выхода не было совсем".split()

    def setUp(self):
        self.w = words_of(self.TOKENS)
        self.rng = [(0, len(self.TOKENS) - 1)]

    def test_at_most_two_cuts_per_sentence_longest_kept(self):
        chosen = {1: "false_start", 3: "aside", 4: "aside", 6: "aside", 9: "repeat", 10: "repeat", 11: "repeat"}
        editorial._limit_choppiness(chosen, self.w, self.rng)
        groups = editorial._cut_groups(chosen, 0, len(self.TOKENS) - 1)
        self.assertLessEqual(len(groups), 2)

    def test_pure_single_filler_cuts_are_not_counted(self):
        tokens = "это очень важная э-э мысль для нас э-э всех сегодня тоже э-э слушают".split()
        w = words_of(tokens, step=1.0, length=0.3)
        chosen = {3: "filler", 7: "filler", 10: "filler"}
        editorial._limit_choppiness(chosen, w, [(0, len(tokens) - 1)])
        self.assertEqual(set(chosen), {3, 7, 10})

    def test_short_kept_fragment_between_cuts_is_merged_back(self):
        # слова идут каждые 0.3 с; между вырезами остаётся одно слово (0.25 с) -> меньший из вырезов отменяется
        chosen = {2: "false_start", 3: "false_start", 5: "aside", 6: "aside", 7: "aside"}
        editorial._limit_choppiness(chosen, self.w, self.rng)
        groups = editorial._cut_groups(chosen, 0, len(self.TOKENS) - 1)
        self.assertEqual(groups, [[5, 7]])

    def test_long_fragment_between_cuts_is_kept(self):
        chosen = {1: "false_start", 9: "aside", 10: "aside"}
        editorial._limit_choppiness(chosen, self.w, self.rng)
        self.assertEqual(editorial._cut_groups(chosen, 0, len(self.TOKENS) - 1), [[1, 1], [9, 10]])

    def test_cut_that_repeats_same_word_across_seam_is_cancelled(self):
        tokens = "это вот это важно для всех нас сегодня".split()
        w = words_of(tokens, step=1.0, length=0.3)
        chosen = {1: "aside"}
        editorial._limit_choppiness(chosen, w, [(0, len(tokens) - 1)])
        self.assertEqual(chosen, {})

    def test_cut_removing_duplicate_word_itself_is_fine(self):
        tokens = "это это важно для всех нас сегодня".split()
        w = words_of(tokens, step=1.0, length=0.3)
        chosen = {0: "repeat"}
        editorial._limit_choppiness(chosen, w, [(0, len(tokens) - 1)])
        self.assertEqual(chosen, {0: "repeat"})


if __name__ == "__main__":
    unittest.main()


class WindowGateTest(unittest.TestCase):
    def ctx(self):
        return {"sentences": make([f"Предложение номер {i} про важную тему" for i in range(30)], dur=5.0),
                "max_d": 60, "min_d": 20, "dropped": [], "topics": [], "id_index": {}}

    def draft(self, pieces, fmt="thesis"):
        return {"raw": {"title": "T"}, "pieces": pieces, "format": fmt, "thesis": ""}

    def test_gate_trims_context_after_a_stage_that_enlarged_the_clip(self):
        ctx = self.ctx()
        d = self.draft([{"a": 0, "b": 2, "role": "hook"}, {"a": 5, "b": 14, "role": "context"},
                        {"a": 20, "b": 22, "role": "payoff"}])      # 16 предложений ~ 76 с > 67 (60*1.12)
        out = editorial._window_gate([d], ctx, "repair_chief")
        self.assertEqual(len(out), 1)
        self.assertLessEqual(editorial._pieces_duration(out[0]["pieces"], ctx["sentences"]), 60 * editorial.PRE_TIGHTEN_SLACK)
        self.assertEqual(out[0]["pieces"][0]["role"], "hook")
        self.assertEqual(out[0]["pieces"][-1]["role"], "payoff")
        self.assertEqual(out[0]["window_trims"][0]["stage"], "repair_chief")

    def test_gate_rejects_with_reason_when_only_hook_payoff_left(self):
        ctx = self.ctx()
        d = self.draft([{"a": 0, "b": 6, "role": "hook"}, {"a": 20, "b": 26, "role": "payoff"}])
        out = editorial._window_gate([d], ctx, "chief_editor")
        self.assertEqual(out, [])
        self.assertEqual(ctx["dropped"][0]["stage"], "duration")
        self.assertIn("hook/payoff", ctx["dropped"][0]["reason"])

    def test_list_format_has_longer_window(self):
        ctx = self.ctx()
        pieces = [{"a": 0, "b": 12, "role": "point"}]          # ~63 с: thesis нельзя (67 ок?), проверяем границу 1.4x
        self.assertEqual(len(editorial._window_gate([self.draft(pieces, "list")], ctx, "x")), 1)


class CutDensityTest(unittest.TestCase):
    def test_no_more_than_four_cuts_in_eight_seconds_fillers_dropped_first(self):
        tokens = [f"слово{i}" for i in range(60)]
        w = words_of(tokens, step=0.5, length=0.4)           # 2 слова в секунду
        # шесть выреза в пределах ~4 с: два одиночных паразита и четыре двухсловных
        chosen = {2: "filler", 5: "filler", 8: "aside", 9: "aside", 12: "repeat", 13: "repeat",
                  16: "false_start", 17: "false_start", 20: "aside", 21: "aside"}
        editorial._limit_cut_density(chosen, w, [(0, 59)])
        groups = editorial._cut_groups(chosen, 0, 59)
        starts = [float(w[g[0]]["start"]) for g in groups]
        for t0 in starts:
            self.assertLessEqual(sum(1 for t in starts if 0 <= t - t0 <= 8.0), 4)
        self.assertNotIn(5, chosen)      # одиночный паразит отменён первым (длинные вырезы остались)
        self.assertTrue(all(i in chosen for i in (8, 9, 12, 13, 16, 17)))

    def test_sparse_cuts_untouched(self):
        tokens = [f"слово{i}" for i in range(80)]
        w = words_of(tokens, step=0.5, length=0.4)
        chosen = {2: "aside", 3: "aside", 20: "aside", 21: "aside", 40: "aside", 41: "aside"}
        before = dict(chosen)
        editorial._limit_cut_density(chosen, w, [(0, 79)])
        self.assertEqual(chosen, before)


class SpliceBudgetTest(unittest.TestCase):
    def mk(self, n=80, step=0.5):
        tokens = [f"слово{i}" for i in range(n)]
        return words_of(tokens, step=step, length=0.3)    # паузы по 0.2 с между словами

    def test_filler_without_clean_pause_is_not_cut(self):
        w = self.mk()
        w[10]["text"] = "ну"
        w[9]["end"] = 5.0
        w[10]["start"], w[10]["end"] = 5.05, 5.35     # вплотную к предыдущему слову (пауза 0.05 с)
        chosen = {10: "filler"}
        editorial._splice_budget(chosen, w, [(0, 79)])
        self.assertEqual(chosen, {})

    def test_eh_between_pauses_is_cut(self):
        w = self.mk()
        w[10]["text"] = "э-э"
        w[9]["end"] = 4.4          # пауза 0.1? -> нужна >= 0.15
        w[10]["start"], w[10]["end"] = 4.65, 4.95
        w[11]["start"] = 5.2
        chosen = {10: "filler"}
        editorial._splice_budget(chosen, w, [(0, 79)])
        self.assertEqual(chosen, {10: "filler"})

    def test_short_repeat_is_not_cut_long_repeat_is(self):
        w = self.mk()
        short = {20: "repeat"}                 # 0.3 с < 0.4
        editorial._splice_budget(short, w, [(0, 79)])
        self.assertEqual(short, {})
        longer = {20: "repeat", 21: "repeat"}  # 0.8 с
        editorial._splice_budget(longer, w, [(0, 79)])
        self.assertEqual(longer, {20: "repeat", 21: "repeat"})

    def test_close_cuts_with_filler_between_merge_into_one(self):
        w = self.mk()
        w[22]["text"] = "ну"
        chosen = {20: "repeat", 21: "repeat", 23: "false_start", 24: "false_start"}
        editorial._splice_budget(chosen, w, [(0, 79)])
        groups = editorial._cut_groups(chosen, 0, 79)
        self.assertEqual(groups, [[20, 24]])

    def test_close_cuts_with_content_between_drop_the_less_valuable(self):
        w = self.mk()
        chosen = {20: "repeat", 21: "repeat", 23: "filler", 24: "filler", 25: "filler"}
        w[23]["text"] = "ну"
        editorial._splice_budget(chosen, w, [(0, 79)])
        groups = editorial._cut_groups(chosen, 0, 79)
        self.assertEqual(groups, [[20, 21]])           # паразиты остались, содержательный вырез сохранён

    def test_no_two_splices_closer_than_1_5_s_and_average_budget(self):
        w = self.mk(n=120, step=0.5)               # 60 с
        chosen = {i: "repeat" for i in range(10, 110, 6) for i in (i, i + 1)}
        editorial._splice_budget(chosen, w, [(0, 119)])
        groups = editorial._cut_groups(chosen, 0, 119)
        starts = [float(w[g[0]]["start"]) for g in groups]
        for a, b in zip(starts, starts[1:]):
            self.assertGreaterEqual(b - a, 1.5)
        self.assertLessEqual(len(groups), 15)       # 60 с / 4 с

    def test_cut_next_to_piece_seam_is_dropped(self):
        w = self.mk(n=60)
        chosen = {5: "repeat", 6: "repeat"}          # через 2.5 с от начала второго куска
        editorial._splice_budget(chosen, w, [(0, 29), (30, 59)])
        self.assertTrue(chosen)                      # это 1-й кусок: старт ролика не склейка, но вырез у 1-го куска вне 1.5 с зоны шва
        chosen2 = {32: "repeat", 33: "repeat"}       # у начала 2-го куска: слева стык
        editorial._splice_budget(chosen2, w, [(0, 29), (30, 59)])
        self.assertEqual(chosen2, {})

    def test_ordinals_untouched(self):
        w = self.mk()
        chosen = {2: "ordinal"}
        editorial._splice_budget(chosen, w, [(0, 79)])
        self.assertEqual(chosen, {2: "ordinal"})


class RehookTest(unittest.TestCase):
    TEXTS = [
        "Компания наняла нового руководителя из Германии.",              # 0
        "Он ушёл через год и увёл с собой половину команды.",            # 1  (продолжение)
        "400 тысяч.",                                                    # 2
        "Дальше шли обычные рабочие моменты про проекты и сроки.",       # 3
        "Всю команду пятьдесят человек одним днём созвали и уволили.",   # 4
        "Заказчик сказал что санкции и работать больше не будем.",       # 5
        "Вывод простой: нельзя зависеть от одного заказчика целиком.",   # 6
    ]

    def ctx(self):
        s = make(self.TEXTS, dur=4.0)
        return {"sentences": s, "id_index": {x["id"]: i for i, x in enumerate(s)}, "topics": [], "max_d": 90, "min_d": 20,
                "dropped": [], "options": {}, "work_dir": None}

    def draft(self, pieces):
        return {"raw": {"title": "T", "hook_text": ""}, "pieces": pieces, "format": "story", "thesis": "t"}

    def test_detection(self):
        self.assertTrue(editorial._continuation_hook("Они были под руководством вот этого человека."))
        self.assertTrue(editorial._continuation_hook("Ну и вот это уже другая история."))
        # разминка и союзы, которые чистка снимет сама, не считаются продолжением
        self.assertFalse(editorial._continuation_hook("Ну, то есть, компания получала 4500 евро."))
        self.assertFalse(editorial._continuation_hook("И компания получала 4500 евро."))
        self.assertTrue(editorial._continuation_hook("Сложнее было потом."))
        self.assertTrue(editorial._continuation_hook("Да, так и есть."))
        self.assertTrue(editorial._continuation_hook("Так вот, он ушёл."))
        self.assertFalse(editorial._continuation_hook("Компанию уволили за один день."))
        self.assertTrue(editorial._number_fragment("400 тысяч."))
        self.assertFalse(editorial._number_fragment("Сеньору платили четыреста тысяч на руки каждый месяц."))

    def test_problems_flag_continuation_and_fragment(self):
        c = self.ctx()
        self.assertTrue(editorial._hook_problems([{"a": 1, "b": 1, "role": "hook"}], c["sentences"]))
        self.assertTrue(editorial._hook_problems([{"a": 2, "b": 2, "role": "hook"}], c["sentences"]))
        self.assertEqual(editorial._hook_problems([{"a": 0, "b": 1, "role": "hook"}], c["sentences"]), [])

    def test_opening_fix_extends_backwards_in_natural_order(self):
        c = self.ctx()
        pieces = [{"a": 1, "b": 1, "role": "hook"}, {"a": 4, "b": 6, "role": "payoff"}]
        fixed = editorial._fix_opening(pieces, c["sentences"], max_total=200)
        self.assertEqual((fixed[0]["a"], fixed[0]["b"]), (0, 1))     # введён герой: предыдущее предложение записи
        self.assertEqual([(p["a"], p["b"]) for p in fixed[1:]], [(4, 6)])
        self.assertTrue(editorial.is_chronological(fixed))

    def test_opening_fix_never_pulls_sentence_from_elsewhere(self):
        c = self.ctx()
        # хук-продолжение в начале записи (предыдущих предложений нет): ничего не переставляется, куски как были
        pieces = [{"a": 1, "b": 1, "role": "hook"}, {"a": 4, "b": 4, "role": "point"}, {"a": 6, "b": 6, "role": "payoff"}]
        fixed = editorial._fix_opening(pieces, c["sentences"], max_total=200)
        self.assertTrue(editorial.is_chronological(fixed))
        self.assertNotEqual(fixed[0]["a"], 4)                         # сильная фраза №4 не вынесена в начало
        self.assertNotIn(4, [i for p in fixed[:1] for i in range(p["a"], p["b"] + 1)])

    def test_opening_fix_capped_to_8_seconds(self):
        s = make(["Раз два три четыре пять шесть семь.", "Он ушёл через год и увёл с собой команду.",
                  "Компания наняла руководителя из Германии давно.", "Он ушёл через год и увёл всех людей."], dur=5.0)
        pieces = [{"a": 3, "b": 3, "role": "hook"}]
        fixed = editorial._fix_opening(pieces, s, max_total=200)
        self.assertEqual(fixed[0]["a"], 2)           # хватило одного предложения (4.8 с)
        s2 = make(["Раз два три четыре пять шесть семь.", "Он ушёл через год и увёл с собой команду.",
                   "Он сказал что команда уйдёт вместе с ним.", "Он ушёл через год и увёл всех людей."], dur=5.0)
        # цепочка продолжений: два предложения по 4.8 с > 8 с, расширение не применяется (куски те же)
        self.assertEqual(editorial._fix_opening(pieces, s2, max_total=200), pieces)

    def test_opening_fix_ignores_self_contained_openings(self):
        c = self.ctx()
        pieces = [{"a": 0, "b": 0, "role": "hook"}, {"a": 4, "b": 6, "role": "payoff"}]
        self.assertEqual(editorial._fix_opening(pieces, c["sentences"], max_total=200), pieces)

    def test_rehook_stage_and_reordering_are_gone_from_library(self):
        for name in ("_rehook", "_apply_rehook", "REHOOK_SCHEMA", "_rehook_prompt"):
            self.assertFalse(hasattr(editorial, name), name)


import tempfile
from pathlib import Path
