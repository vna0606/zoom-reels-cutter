from __future__ import annotations

import unittest

from reel_factory import editorial


class EditorialPromptTests(unittest.TestCase):
    def test_topic_prompt_skips_tech_checks_and_collects_later_returns(self) -> None:
        prompt = editorial._topic_prompt("S0001 [00:00:01] Привет.")
        self.assertIn("проверку связи", prompt)
        self.assertIn("более поздние возвраты", prompt)
        self.assertIn("S0001", prompt)

    def test_compose_prompt_rules(self) -> None:
        prompt = editorial._compose_prompt("S0001 [00:01, 3с] текст", "- тема", max_clips=5, min_d=30, max_d=90)
        self.assertIn("до 5 вертикальных", prompt)
        self.assertIn("от 30 до 90 секунд", prompt)
        for role in ("hook", "context", "point", "example", "payoff"):
            self.assertIn(role, prompt)
        self.assertIn("СТРОГО в хронологическом порядке", prompt)
        self.assertIn("НЕЛЬЗЯ переставлять", prompt)
        self.assertIn("НЕ БОЛЕЕ 8 слов", prompt)
        self.assertIn("неразрешённых отсылок", prompt)
        self.assertIn("меня слышно?", prompt)
        self.assertIn("ОБЯЗАТЕЛЕН: последний кусок", prompt)
        self.assertIn("явную фразу-вывод", prompt)

    def test_compose_prompt_one_idea_lists_and_jumps(self) -> None:
        prompt = editorial._compose_prompt("S0001 [00:01, 3с] т", "- т", max_clips=3, min_d=30, max_d=90)
        self.assertIn("ОДНА МЫСЛЬ НА ОДИН РОЛИК", prompt)
        self.assertIn("ЗАПРЕЩЕНО склеивать тезис с посторонним кейсом", prompt)
        self.assertIn("отдельный ролик формата story", prompt)
        self.assertIn("ВСЕ пункты перечня", prompt)
        self.assertIn("до 144 с", prompt)
        self.assertIn("ОДИН непрерывный отрезок", prompt)
        self.assertIn("ПРИОРИТЕТ ПРИ ВЫБОРЕ", prompt)
        self.assertIn("не более 3 разных областей", prompt)

    def test_compose_prompt_audience_is_context_not_filter(self) -> None:
        prompt = editorial._compose_prompt("S0001 [00:01, 3с] т", "- т", max_clips=3, min_d=30, max_d=90,
                                           audience=editorial.DEFAULT_AUDIENCE)
        self.assertIn("АУДИТОРИЯ СТРИМА", prompt)
        self.assertIn("НЕ фильтр", prompt)
        self.assertIn("накрутка опыта", prompt)
        self.assertNotIn("АУДИТОРИЯ СТРИМА", editorial._compose_prompt("S", "-", max_clips=3, min_d=30, max_d=90))

    def test_critic_repair_chief_prompts_forbid_reordering_and_allow_trims(self) -> None:
        critique = editorial._critique_prompt("РОЛИК 1", min_d=30, max_d=90)
        repair = editorial._repair_prompt("РОЛИК 1", min_d=30, max_d=90)
        chief = editorial._chief_prompt("-", "РОЛИК", audience=editorial.DEFAULT_AUDIENCE)
        for prompt in (critique, repair):
            self.assertIn("trims", prompt)
            self.assertIn("хронологическ", prompt)
            self.assertNotIn("ПЕРВЫМ ставь самую сильную", prompt)
        self.assertNotIn("менять порядок", critique)
        self.assertIn("ПОДРЕЗАЙ, А НЕ ОТБРАСЫВАЙ", critique)
        self.assertIn("ПОДРЕЗАЙ, А НЕ ОТБРАСЫВАЙ", chief)
        self.assertIn("СПИН-ОФФ", chief)
        self.assertIn("АУДИТОРИЯ СТРИМА", chief)

    def test_topic_prompt_asks_kind_and_items(self) -> None:
        prompt = editorial._topic_prompt("S0001 [00:00:01] т")
        self.assertIn("kind", prompt)
        self.assertIn("items", prompt)

    def test_critique_and_tighten_prompts(self) -> None:
        critique = editorial._critique_prompt("РОЛИК 1", min_d=30, max_d=90)
        self.assertIn("standalone", critique)
        self.assertIn("reject", critique)
        self.assertIn("completeness не выше 5", critique)
        self.assertIn("ЗАМЕЧАНИЕ ПРОВЕРКИ", critique)
        self.assertIn("Нельзя использовать любые другие id", critique)
        self.assertIn("single_idea", critique)
        self.assertIn("Оценивай МОНТАЖ", critique)
        self.assertIn("jump", critique)
        tighten = editorial._tighten_prompt("#1 ну")
        self.assertIn("false_start", tighten)
        self.assertIn("не более 20%", tighten)
        self.assertIn("Если сомневаешься — не удаляй", tighten)


if __name__ == "__main__":
    unittest.main()
