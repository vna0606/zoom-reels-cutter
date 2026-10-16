from __future__ import annotations

import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "web" / "app.js"


class FrontendStateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = APP_JS.read_text(encoding="utf-8")

    def block(self, start: str, end: str) -> str:
        start_index = self.source.index(start)
        end_index = self.source.index(end, start_index + len(start))
        return self.source[start_index:end_index]

    def test_project_subtitle_style_hydrates_once_without_overwriting_user_change(self) -> None:
        self.assertIn('highlightSubtitleStyle: $("#highlightSubtitleStyle")', self.source)
        render = self.block(
            "function renderHighlightProject(project)",
            "async function fetchHighlightProject()",
        )

        guard = "if (els.highlightSubtitleStyle.dataset.projectId !== project.id)"
        allowlist = '["pill", "clean", "yellow"].includes(subtitleStyle)'
        set_value = "els.highlightSubtitleStyle.value = subtitleStyle"
        remember_project = "els.highlightSubtitleStyle.dataset.projectId = project.id"
        for token in (
            "const subtitleStyle = project.options?.subtitle_style",
            guard,
            allowlist,
            set_value,
            remember_project,
        ):
            self.assertIn(token, render)
        self.assertLess(render.index(guard), render.index(set_value))
        self.assertLess(render.index(set_value), render.index(remember_project))
        self.assertEqual(render.count(set_value), 1)

    def test_project_reset_forgets_hydration_and_stale_highlight_zip(self) -> None:
        reset = self.block(
            "function resetHighlightProjectUI()",
            "function switchView(view, shouldScroll = true)",
        )

        self.assertIn("delete els.highlightSubtitleStyle.dataset.projectId", reset)
        self.assertIn("delete els.highlightSpeakerCount.dataset.projectId", reset)
        self.assertIn("delete els.highlightVisualPreset.dataset.projectId", reset)
        self.assertIn('els.highlightDownloadAll.removeAttribute("href")', reset)
        self.assertIn('els.highlightDownloadAll.classList.add("hidden")', reset)

    def test_clip_card_shows_structure_critic_and_new_options(self) -> None:
        for token in ("clipStructureHtml(item)", "clipCriticHtml(item)", "вырезано паразитов:",
                      "Хук", "Контекст", "Суть", "Пример", "Вывод",
                      "max_clips:", "asr_glossary: $(\"#asrGlossary\").value", "single_idea", "CLIP_FORMAT_LABELS", "tighten: $(\"#highlightTighten\").checked"):
            self.assertIn(token, self.source)
        index_html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        for token in ('id="asrGlossary"', "Словарь терминов", 'id="maxClips"', 'id="highlightTighten"', "Убирать слова-паразиты и паузы"):
            self.assertIn(token, index_html)

    def test_hook_for_visual_origin_tags_audience_and_all_clips_default(self) -> None:
        index_html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        for token in ("Хук для визуала", "clipHookHtml(item)", "item.hook_text || item.hook", "clipOriginHtml(item)",
                      "audience: $(\"#highlightAudience\").value", "Math.max(0, Math.min(20, Number($(\"#maxClips\").value) || 0))"):
            self.assertIn(token, self.source)
        # карточки отсеянных тоже показывают хук для визуала и кнопку «Всё равно сделать»
        rejected_block = self.source.split("function renderRejectedBlock")[1].split("async function fetchHighlightProject")[0]
        self.assertIn("clipHookHtml(item)", rejected_block)
        self.assertIn("Всё равно сделать", rejected_block)
        for token in ('id="highlightAudience"', 'id="maxClips" type="number" min="0"', 'value="0"',
                      'id="highlightAllClips" type="checkbox" checked'):
            self.assertIn(token, index_html)

    def test_rejected_block_force_button_and_review_card(self) -> None:
        index_html = (APP_JS.parent / "index.html").read_text(encoding="utf-8")
        for token in ('id="rejectedBlock"', "Не прошли проверку", 'id="rejectedGrid"',
                      'id="highlightVideoReview"', "Шеф-редактор отсматривает готовые ролики"):
            self.assertIn(token, index_html)
        for token in ("data-force-clip", "Всё равно сделать", "clipReviewHtml(item)", "clipChiefHtml(item)",
                      "item.result?.review", "video_review: els.highlightVideoReview.checked", "state.forcedClips",
                      "REVIEW_VERDICT_LABELS", "review-issues"):
            self.assertIn(token, self.source)
        render = self.block("function renderHighlightProject(project)", "async function fetchHighlightProject()")
        self.assertIn("renderRejectedBlock(rejectedCandidates", render)
        self.assertIn("formatTimecode(Number(issue.t))", self.source)   # замечания показаны с таймкодами

    def test_missing_zip_removes_stale_href_in_both_workflows(self) -> None:
        batch = self.block("function renderJob(job)", "async function fetchJob()")
        highlights = self.block(
            "function renderHighlightProject(project)",
            "async function fetchHighlightProject()",
        )

        self.assertIn('els.downloadAll.removeAttribute("href")', batch)
        self.assertIn('els.downloadAll.classList.add("hidden")', batch)
        self.assertIn('els.highlightDownloadAll.removeAttribute("href")', highlights)
        self.assertIn('els.highlightDownloadAll.classList.add("hidden")', highlights)

    def test_legacy_broadcast_result_is_not_badged_as_hyperframes_motion(self) -> None:
        label_helper = self.block(
            "function broadcastLayoutLabel(item)",
            'const ANALYSIS_PHASE_ORDER =',
        )
        render = self.block(
            "function renderHighlightProject(project)",
            "async function fetchHighlightProject()",
        )

        self.assertIn('item.result?.visual_engine', label_helper)
        self.assertIn(
            'item.status === "completed" && visualEngine !== "hyperframes"',
            label_helper,
        )
        self.assertIn('return "Старый формат · пересоберите ролик"', label_helper)
        self.assertIn(
            'return "Motion-графика · фиксированные спикеры · субтитры"',
            label_helper,
        )
        self.assertIn("${broadcastLayoutLabel(item)}", render)

    def test_failed_render_with_candidates_keeps_selection_available(self) -> None:
        render = self.block(
            "function renderHighlightProject(project)",
            "async function fetchHighlightProject()",
        )
        self.assertIn(
            'project.status === "failed" && candidates.length > 0',
            render,
        )

    def test_source_download_is_available_independently_of_rendered_download(self) -> None:
        render = self.block(
            "function renderHighlightProject(project)",
            "async function fetchHighlightProject()",
        )

        self.assertIn("item.source_download_url", render)
        self.assertIn("Скачать исходный фрагмент", render)
        self.assertIn("Скачать готовый MP4", render)
        self.assertLess(render.index("item.source_download_url"), render.index("item.download_url"))

    def test_broadcast_render_sends_speaker_count_and_visual_preset(self) -> None:
        render = self.block(
            "async function renderSelectedClips()",
            "async function toggleProcessPause(kind)",
        )

        self.assertIn("speaker_count:", render)
        self.assertIn("Number(els.highlightSpeakerCount.value)", render)
        self.assertIn("visual_preset:", render)
        self.assertIn("els.highlightVisualPreset.value", render)


if __name__ == "__main__":
    unittest.main()
