const WORK_MODES = new Set(["podcast", "broadcast"]);
const savedWorkMode = localStorage.getItem("reelFactoryWorkMode");
const legacyHighlightProjectId = localStorage.getItem("reelFactoryHighlightProject");

const state = {
  files: [],
  jobId: localStorage.getItem("reelFactoryJob"),
  pollTimer: null,
  activeView: localStorage.getItem("reelFactoryView") || "batch",
  workMode: WORK_MODES.has(savedWorkMode) ? savedWorkMode : "podcast",
  highlightSource: "upload",
  podcastHighlightSource: "upload",
  longFile: null,
  longFiles: {podcast: null, broadcast: null},
  podcastYoutubeUrl: "",
  highlightProjectIds: {
    podcast: localStorage.getItem("reelFactoryHighlightProjectPodcast") || legacyHighlightProjectId,
    broadcast: localStorage.getItem("reelFactoryHighlightProjectBroadcast"),
  },
  highlightProjectId: null,
  highlightPollTimer: null,
  selectedClips: new Set(),
  forcedClips: new Set(),   // отсеянные кандидаты, которые пользователь решил «всё равно сделать»
  highlightProject: null,
};
state.highlightProjectId = state.highlightProjectIds[state.workMode];

const $ = (selector) => document.querySelector(selector);
const els = {
  dropzone: $("#dropzone"),
  fileInput: $("#fileInput"),
  fileList: $("#fileList"),
  startButton: $("#startButton"),
  uploadProgress: $("#uploadProgress"),
  uploadBar: $("#uploadBar"),
  uploadPercent: $("#uploadPercent"),
  formError: $("#formError"),
  jobSection: $("#jobSection"),
  jobTitle: $("#jobTitle"),
  jobMessage: $("#jobMessage"),
  jobBar: $("#jobBar"),
  jobPercent: $("#jobPercent"),
  resultGrid: $("#resultGrid"),
  downloadAll: $("#downloadAll"),
  jobPauseButton: $("#jobPauseButton"),
  historyList: $("#historyList"),
  podcastViewNav: $("#podcastViewNav"),
  batchView: $("#batchView"),
  highlightsView: $("#highlightsView"),
  highlightHeroEyebrow: $("#highlightHeroEyebrow"),
  highlightHeroTitle: $("#highlightHeroTitle"),
  highlightHeroDescription: $("#highlightHeroDescription"),
  highlightSourceTitle: $("#highlightSourceTitle"),
  highlightSourceDescription: $("#highlightSourceDescription"),
  highlightSourceSwitch: $("#highlightSourceSwitch"),
  longDropzone: $("#longDropzone"),
  longVideoInput: $("#longVideoInput"),
  longFileTitle: $("#longFileTitle"),
  longFileMeta: $("#longFileMeta"),
  youtubeUrl: $("#youtubeUrl"),
  analyzeButton: $("#analyzeButton"),
  highlightUploadProgress: $("#highlightUploadProgress"),
  highlightUploadBar: $("#highlightUploadBar"),
  highlightUploadPercent: $("#highlightUploadPercent"),
  highlightFormError: $("#highlightFormError"),
  broadcastLayoutPreview: $("#broadcastLayoutPreview"),
  youtubeSubtitleGroup: $("#youtubeSubtitleGroup"),
  highlightPrivacyNote: $("#highlightPrivacyNote"),
  analyzeButtonLabel: $("#analyzeButtonLabel"),
  providerHint: $("#providerHint"),
  highlightProjectSection: $("#highlightProjectSection"),
  highlightProjectEyebrow: $("#highlightProjectEyebrow"),
  highlightModeBadge: $("#highlightModeBadge"),
  highlightProjectTitle: $("#highlightProjectTitle"),
  highlightProjectMessage: $("#highlightProjectMessage"),
  highlightProjectBar: $("#highlightProjectBar"),
  highlightProjectPercent: $("#highlightProjectPercent"),
  highlightPhaseProgress: $("#highlightPhaseProgress"),
  highlightPhaseTitle: $("#highlightPhaseTitle"),
  highlightPhasePercent: $("#highlightPhasePercent"),
  highlightPhaseBar: $("#highlightPhaseBar"),
  highlightPhaseProcessed: $("#highlightPhaseProcessed"),
  highlightPhaseEta: $("#highlightPhaseEta"),
  highlightAnalysisSteps: $("#highlightAnalysisSteps"),
  highlightMotionSteps: $("#highlightMotionSteps"),
  highlightWarning: $("#highlightWarning"),
  selectionToolbar: $("#selectionToolbar"),
  selectionSummary: $("#selectionSummary"),
  renderSelectedButton: $("#renderSelectedButton"),
  renderSelectedLabel: $("#renderSelectedLabel"),
  highlightSubtitleStyle: $("#highlightSubtitleStyle"),
  playbackSpeedGroup: $("#playbackSpeedGroup"),
  highlightPlaybackSpeed: $("#highlightPlaybackSpeed"),
  speakerCountGroup: $("#speakerCountGroup"),
  highlightSpeakerCount: $("#highlightSpeakerCount"),
  visualPresetGroup: $("#visualPresetGroup"),
  highlightVisualPreset: $("#highlightVisualPreset"),
  clipsGrid: $("#clipsGrid"),
  rejectedBlock: $("#rejectedBlock"),
  rejectedGrid: $("#rejectedGrid"),
  rejectedCount: $("#rejectedCount"),
  highlightVideoReview: $("#highlightVideoReview"),
  highlightDownloadAll: $("#highlightDownloadAll"),
  highlightPauseButton: $("#highlightPauseButton"),
  highlightRetryAnalysisButton: $("#highlightRetryAnalysisButton"),
  highlightHistoryList: $("#highlightHistoryList"),
  highlightHistoryTitle: $("#highlightHistoryTitle"),
};

const HIGHLIGHT_MODE_COPY = {
  podcast: {
    documentTitle: "Reel Factory — подкаст в вертикальный формат",
    eyebrow: "LONG VIDEO → SHORTS",
    heroTitle: "Длинный выпуск.<br><em>Серия сильных клипов.</em>",
    heroDescription: "Загрузите видео или вставьте ссылку YouTube. Инструмент получит субтитры с таймкодами, найдёт лучшие моменты и превратит выбранные фрагменты в вертикальные ролики.",
    sourceTitle: "Источник и анализ",
    sourceDescription: "Одно длинное видео или публичная ссылка YouTube",
    dropTitle: "Выберите длинное видео",
    dropMeta: "MP4, MOV, MKV или WebM",
    privacy: "Субтитры и видео проекта сохраняются локально",
    analyze: "Найти лучшие отрывки",
    projectEyebrow: "ПРОЕКТ НАРЕЗКИ",
    historyTitle: "Проекты нарезки",
    historyEmpty: "Проекты появятся здесь.",
    render: "Сделать вертикальные",
  },
  broadcast: {
    documentTitle: "Reel Factory — эфир в вертикальные рилсы",
    eyebrow: "ЭФИР → MOTION REELS",
    heroTitle: "Горизонтальный эфир.<br><em>Смысл в полноэкранной графике.</em>",
    heroDescription: "Загрузите запись эфира из Zoom. Инструмент расшифрует разговор, найдёт ключевые моменты и соберёт для каждого полноэкранную motion-графику. Один или два спикера останутся в постоянных слотах снизу, а субтитры — в безопасной зоне рядом.",
    sourceTitle: "Запись эфира",
    sourceDescription: "Одно горизонтальное видео из Zoom · загрузка с компьютера",
    dropTitle: "Выберите запись эфира",
    dropMeta: "Горизонтальный MP4, MOV, MKV или WebM",
    privacy: "Запись, транскрибация и motion-материалы сохраняются локально",
    analyze: "Найти ключевые моменты",
    projectEyebrow: "ПРОЕКТ ЭФИРА",
    historyTitle: "Проекты эфиров",
    historyEmpty: "Проекты эфиров появятся здесь.",
    render: "Создать motion-ролики",
  },
};

function normalizeProjectMode(value) {
  return value === "broadcast" ? "broadcast" : "podcast";
}

function projectModeOf(project) {
  const explicitMode = project?.project_mode || project?.options?.project_mode;
  if (explicitMode) return normalizeProjectMode(explicitMode);
  if (project?.id && project.id === state.highlightProjectIds.broadcast) return "broadcast";
  return "podcast";
}

function projectModeLabel(mode) {
  return normalizeProjectMode(mode) === "broadcast" ? "Эфир" : "Подкаст";
}

function projectStorageKey(mode) {
  return normalizeProjectMode(mode) === "broadcast"
    ? "reelFactoryHighlightProjectBroadcast"
    : "reelFactoryHighlightProjectPodcast";
}

function saveHighlightProjectId(mode, id) {
  const normalizedMode = normalizeProjectMode(mode);
  state.highlightProjectIds[normalizedMode] = id;
  localStorage.setItem(projectStorageKey(normalizedMode), id);
  if (normalizedMode === "podcast") localStorage.setItem("reelFactoryHighlightProject", id);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["Б", "КБ", "МБ", "ГБ"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "";
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

const CLIP_ROLE_LABELS = {hook: "Хук", context: "Контекст", point: "Суть", example: "Пример", payoff: "Вывод"};
const CLIP_FORMAT_LABELS = {list: "Перечень", thesis: "Тезис", story: "История", advice: "Совет", contrast: "Контраст"};
const CLIP_VERDICT_LABELS = {pass: "Одобрен", fix: "Исправлен", reject: "Отклонён"};
const REVIEW_VERDICT_LABELS = {publish: "Можно публиковать", needs_edit: "Нужна правка", reject: "Не публиковать"};
const REVIEW_SCORE_LABELS = [["hook", "хук"], ["idea_disclosure", "раскрытие мысли"], ["flow", "переходы"], ["visual", "визуал"], ["audio", "звук"]];
const REJECT_STAGE_LABELS = {
  compose: "компоновка", critic_round1: "критик, раунд 1", critic_round2: "критик, раунд 2", repair: "ремонт",
  repair_chief: "ремонт по замечанию главреда", chief_editor: "главный редактор", duration: "длительность", ranking: "лимит роликов",
  opening: "начало ролика",
};
const CLIP_MOMENT_LABELS = {quote: "цитата", audience: "реплика зала", story: "живая история", claim: "резкое утверждение"};

function formatTimecode(seconds) {
  if (!Number.isFinite(seconds)) return "";
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function clipStructureHtml(item) {
  const segments = Array.isArray(item.segments) && item.segments.length
    ? item.segments
    : [{start: item.start, end: item.end, role: "point"}];
  const pieces = segments.map((segment, index) => {
    const role = CLIP_ROLE_LABELS[segment.role] || "Суть";
    return `<li><b>${index + 1}. ${role}</b><span>${formatTimecode(Number(segment.start))}–${formatTimecode(Number(segment.end))}</span></li>`;
  }).join("");
  const cuts = Array.isArray(item.cuts) ? item.cuts : [];
  const removed = cuts.reduce((sum, cut) => sum + Math.max(0, Number(cut.end) - Number(cut.start)), 0);
  const cutLine = cuts.length
    ? `<div class="clip-cuts">вырезано паразитов: ${cuts.length}, −${removed.toFixed(1)} с</div>` : "";
  return `<ol class="clip-structure">${pieces}</ol>${cutLine}`;
}

function clipCriticHtml(item) {
  const critic = item.critic || {};
  const scores = critic.scores || critic;  // оценки лежат прямо в critic (или в critic.scores)
  const labels = [["hook_strength", "хук"], ["completeness", "полнота"], ["clarity", "ясность"], ["standalone", "самодостаточность"], ["single_idea", "одна мысль"]];
  const chips = labels.filter(([key]) => Number.isFinite(Number(scores[key])) && scores[key] !== undefined)
    .map(([key, label]) => `<span>${label} ${Number(scores[key])}/10</span>`).join("");
  const verdict = CLIP_VERDICT_LABELS[critic.verdict];
  const hookWarn = critic.hook_warning ? '<span class="verdict-fix">слабый хук</span>' : "";
  if (!chips && !verdict) return "";
  return `<div class="clip-critic">${hookWarn}${verdict ? `<b class="verdict-${escapeHtml(critic.verdict)}">${verdict}</b>` : ""}${chips}</div>`;
}

function clipChiefHtml(item) {
  const chief = item.chief_editor;
  if (!chief || !chief.rank) return "";
  const title = chief.title_fixed ? ` · название исправлено (было «${escapeHtml(chief.original_title)}»)` : "";
  const flags = chief.coverage_override ? " · оставлен ради покрытия темы" : (chief.resurrected ? " · восстановлен ради покрытия темы" : "");
  const decision = chief.decision === "repair_more" ? " · доработан по замечанию" : "";
  return `<div class="clip-chief"><b>Главред: №${Number(chief.rank)}</b>${chief.reason ? ` — ${escapeHtml(chief.reason)}` : ""}${title}${flags}${decision}</div>`;
}

function clipReviewHtml(item) {
  const review = item.result?.review;
  if (!review) return "";
  const verdictKey = review.verdict || "";
  const verdict = REVIEW_VERDICT_LABELS[verdictKey] || (review.status === "error" ? "Отсмотр не удался" : "");
  const scores = review.scores || {};
  const chips = REVIEW_SCORE_LABELS.filter(([key]) => Number.isFinite(Number(scores[key])) && scores[key] !== undefined)
    .map(([key, label]) => `<span>${label} ${Number(scores[key])}/10</span>`).join("");
  const issues = Array.isArray(review.issues) ? review.issues : [];
  const issueList = issues.length
    ? `<ul class="review-issues">${issues.map(issue => `<li><b>${formatTimecode(Number(issue.t))}</b> <i>${escapeHtml(issue.kind || "")}</i> ${escapeHtml(issue.note || "")}</li>`).join("")}</ul>` : "";
  const fix = item.result?.auto_fix;
  let fixLine = "";
  if (fix) {
    const state = fix.applied ? (fix.kept ? "принята: ролик пересобран" : "не принята: версия не лучше исходной") : (fix.reason || "не применена");
    fixLine = `<p class="review-fix">Авто-правка: ${escapeHtml(fix.description || "")} — ${escapeHtml(state)}</p>`;
  }
  const caption = review.caption ? `<p class="review-caption">Подпись: ${escapeHtml(review.caption)}</p>` : "";
  const summary = review.summary ? `<p class="review-summary">${escapeHtml(review.summary)}</p>` : "";
  return `<div class="clip-review"><div class="clip-critic"><b class="review-${escapeHtml(verdictKey || "none")}">Шеф-редактор: ${escapeHtml(verdict)}</b>${chips}</div>${summary}${issueList}${fixLine}${caption}</div>`;
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatClockDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  return hours
    ? `${hours}:${minutes.toString().padStart(2, "0")}:${rest.toString().padStart(2, "0")}`
    : `${minutes}:${rest.toString().padStart(2, "0")}`;
}

function formatEta(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  if (total < 60) return "меньше минуты";
  const minutes = Math.max(1, Math.round(total / 60));
  if (minutes < 60) return `${minutes} мин`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} ч ${rest} мин` : `${hours} ч`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
}

function addFiles(fileList) {
  const allowed = new Set(["mp4", "mov", "mkv", "webm", "m4v"]);
  const existing = new Set(state.files.map(file => `${file.name}:${file.size}:${file.lastModified}`));
  for (const file of fileList) {
    const extension = file.name.split(".").pop().toLowerCase();
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    if (allowed.has(extension) && !existing.has(key) && state.files.length < 30) {
      state.files.push(file);
      existing.add(key);
    }
  }
  renderSelectedFiles();
}

function renderSelectedFiles() {
  els.fileList.innerHTML = state.files.map((file, index) => `
    <div class="selected-file">
      <span class="file-type">${escapeHtml(file.name.split(".").pop().toUpperCase())}</span>
      <span class="file-copy"><b>${escapeHtml(file.name)}</b><small>${formatBytes(file.size)}</small></span>
      <button class="remove-file" type="button" data-remove="${index}" aria-label="Удалить ${escapeHtml(file.name)}">×</button>
    </div>`).join("");
  els.startButton.disabled = state.files.length === 0;
}

function showError(message) {
  els.formError.textContent = message;
  els.formError.classList.remove("hidden");
}

function clearError() {
  els.formError.classList.add("hidden");
  els.formError.textContent = "";
}

function createJob() {
  if (!state.files.length) return;
  clearError();
  els.startButton.disabled = true;
  els.uploadProgress.classList.remove("hidden");
  const data = new FormData();
  for (const file of state.files) data.append("files", file, file.name);
  data.append("options", JSON.stringify({
    language: $("#language").value,
    model: $("#model").value,
    asr_provider: $("#asrProvider").value,
    subtitle_style: document.querySelector("input[name=subtitleStyle]:checked").value,
  }));
  const request = new XMLHttpRequest();
  request.open("POST", "/api/jobs");
  request.upload.addEventListener("progress", event => {
    if (!event.lengthComputable) return;
    const percent = Math.round(event.loaded / event.total * 100);
    els.uploadBar.value = percent;
    els.uploadPercent.textContent = `${percent}%`;
  });
  request.addEventListener("load", () => {
    els.uploadProgress.classList.add("hidden");
    els.startButton.disabled = false;
    let response;
    try { response = JSON.parse(request.responseText); } catch { response = {}; }
    if (request.status < 200 || request.status >= 300) {
      showError(response.error || "Не удалось создать задание.");
      return;
    }
    state.jobId = response.id;
    localStorage.setItem("reelFactoryJob", response.id);
    state.files = [];
    renderSelectedFiles();
    renderJob(response);
    startPolling();
    els.jobSection.scrollIntoView({behavior: "smooth", block: "start"});
  });
  request.addEventListener("error", () => {
    els.uploadProgress.classList.add("hidden");
    els.startButton.disabled = false;
    showError("Соединение с локальным сервисом прерванo.");
  });
  request.send(data);
}

function statusLabel(status) {
  return ({queued:"В очереди",processing:"В работе",analyzing:"Анализ",awaiting_selection:"Можно выбирать",rendering:"Рендер",completed:"Готово",failed:"Ошибка",partial:"Частично готово",interrupted:"Прервано"})[status] || status;
}

function renderJob(job) {
  els.jobSection.classList.remove("hidden");
  els.jobTitle.textContent = job.paused ? "Обработка на паузе" : (job.status === "completed" ? "Все ролики готовы" : "Обрабатываю видео");
  els.jobMessage.textContent = job.message || "";
  els.jobBar.value = job.progress || 0;
  els.jobPercent.textContent = `${job.progress || 0}%`;
  const pauseAvailable = ["queued", "processing"].includes(job.status);
  els.jobPauseButton.classList.toggle("hidden", !pauseAvailable);
  els.jobPauseButton.classList.toggle("is-paused", Boolean(job.paused));
  els.jobPauseButton.textContent = job.paused ? "Продолжить" : "Поставить на паузу";
  els.jobPauseButton.dataset.paused = job.paused ? "true" : "false";
  if (job.zip_url) {
    els.downloadAll.href = job.zip_url;
    els.downloadAll.classList.remove("hidden");
  } else {
    els.downloadAll.removeAttribute("href");
    els.downloadAll.classList.add("hidden");
  }
  els.resultGrid.innerHTML = job.files.map(item => {
    const result = item.result || {};
    const details = item.status === "completed"
      ? `${formatDuration(result.duration)} · ${formatBytes(result.bytes)} · ${result.words || 0} слов`
      : "";
    return `
      <article class="result-card ${escapeHtml(item.status)}">
        <div class="result-top">
          <span class="result-icon">9:16</span>
          <div class="result-copy">
            <b title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</b>
            <span class="status-line"><i class="status-dot"></i>${escapeHtml(item.message || statusLabel(item.status))}</span>
          </div>
        </div>
        <progress class="result-progress" max="100" value="${item.progress || 0}"></progress>
        ${item.error ? `<p class="file-error">${escapeHtml(item.error)}</p>` : ""}
        ${details ? `<div class="result-details">${escapeHtml(details)}</div>` : ""}
        ${item.download_url ? `<div class="result-actions"><a class="download-button" href="${item.download_url}">Скачать MP4</a></div>` : ""}
      </article>`;
  }).join("");
}

async function fetchJob() {
  if (!state.jobId) return;
  try {
    const response = await fetch(`/api/jobs/${state.jobId}`, {cache: "no-store"});
    if (!response.ok) return;
    const job = await response.json();
    renderJob(job);
    if (["completed", "partial", "failed", "interrupted"].includes(job.status)) {
      stopPolling();
      loadHistory();
    }
  } catch { /* The next polling pass will retry. */ }
}

function startPolling() {
  stopPolling();
  fetchJob();
  state.pollTimer = setInterval(fetchJob, 1500);
}

function stopPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
}

async function loadHistory() {
  try {
    const response = await fetch("/api/jobs", {cache: "no-store"});
    const data = await response.json();
    const jobs = data.jobs || [];
    if (!jobs.length) {
      els.historyList.innerHTML = '<p class="empty-state">Готовые задания появятся здесь.</p>';
      return;
    }
    els.historyList.innerHTML = jobs.map(job => {
      const completed = job.files.filter(file => file.status === "completed").length;
      const created = new Date(job.created_at).toLocaleString("ru-RU", {day:"2-digit",month:"short",hour:"2-digit",minute:"2-digit"});
      return `<div class="history-item">
        <div><b>${created}</b><small>${job.files.length} видео · готово ${completed}</small></div>
        <span class="history-state">${statusLabel(job.status)}</span>
        ${job.zip_url ? `<a class="history-link" href="${job.zip_url}">ZIP ↓</a>` : `<button class="text-button" data-open-job="${job.id}">Открыть</button>`}
      </div>`;
    }).join("");
  } catch {
    els.historyList.innerHTML = '<p class="empty-state">Не удалось загрузить историю.</p>';
  }
}

function applyWorkModeCopy() {
  const copy = HIGHLIGHT_MODE_COPY[state.workMode];
  document.title = copy.documentTitle;
  document.body.dataset.workMode = state.workMode;
  els.highlightHeroEyebrow.textContent = copy.eyebrow;
  els.highlightHeroTitle.innerHTML = copy.heroTitle;
  els.highlightHeroDescription.textContent = copy.heroDescription;
  els.highlightSourceTitle.textContent = copy.sourceTitle;
  els.highlightSourceDescription.textContent = copy.sourceDescription;
  els.highlightPrivacyNote.textContent = copy.privacy;
  els.analyzeButtonLabel.textContent = copy.analyze;
  els.highlightProjectEyebrow.textContent = copy.projectEyebrow;
  els.highlightModeBadge.textContent = projectModeLabel(state.workMode);
  els.highlightHistoryTitle.textContent = copy.historyTitle;
  els.renderSelectedLabel.textContent = copy.render;
  els.highlightSourceSwitch.classList.toggle("hidden", state.workMode === "broadcast");
  els.youtubeSubtitleGroup.classList.toggle("hidden", state.workMode === "broadcast");
  els.broadcastLayoutPreview.classList.toggle("hidden", state.workMode !== "broadcast");
  els.playbackSpeedGroup.classList.toggle("hidden", state.workMode !== "broadcast");
  els.speakerCountGroup.classList.toggle("hidden", state.workMode !== "broadcast");
  els.visualPresetGroup.classList.toggle("hidden", state.workMode !== "broadcast");
}

function resetHighlightProjectUI() {
  els.highlightUploadProgress.classList.add("hidden");
  els.highlightUploadBar.value = 0;
  els.highlightUploadPercent.textContent = "0%";
  clearHighlightError();
  els.highlightProjectSection.classList.add("hidden");
  els.highlightProjectMessage.textContent = "";
  els.highlightProjectBar.classList.remove("indeterminate");
  els.highlightProjectBar.value = 0;
  els.highlightProjectPercent.textContent = "0%";
  els.highlightPhaseProgress.classList.add("hidden");
  els.highlightPhaseBar.classList.remove("indeterminate");
  els.highlightPhaseBar.value = 0;
  els.highlightPhasePercent.textContent = "0%";
  els.highlightPhaseProcessed.textContent = "";
  els.highlightPhaseProcessed.classList.add("hidden");
  els.highlightPhaseEta.textContent = "";
  els.highlightPhaseEta.classList.add("hidden");
  els.highlightAnalysisSteps.classList.remove("hidden");
  els.highlightMotionSteps.classList.add("hidden");
  els.highlightWarning.textContent = "";
  els.highlightWarning.classList.add("hidden");
  els.selectionToolbar.classList.add("hidden");
  delete els.highlightSubtitleStyle.dataset.projectId;
  delete els.highlightPlaybackSpeed.dataset.projectId;
  delete els.highlightSpeakerCount.dataset.projectId;
  delete els.highlightVisualPreset.dataset.projectId;
  els.highlightDownloadAll.removeAttribute("href");
  els.highlightDownloadAll.classList.add("hidden");
  els.highlightPauseButton.classList.add("hidden");
  els.highlightPauseButton.classList.remove("is-paused");
  els.highlightRetryAnalysisButton.classList.add("hidden");
  els.highlightRetryAnalysisButton.disabled = false;
  els.highlightRetryAnalysisButton.textContent = "Повторить поиск моментов";
  els.clipsGrid.innerHTML = "";
  els.rejectedGrid.innerHTML = "";
  els.rejectedBlock.classList.add("hidden");
  delete els.highlightVideoReview.dataset.projectId;
  state.selectedClips.clear();
  state.forcedClips.clear();
  state.highlightProject = null;
  renderSelectionSummary();
}

function switchView(view, shouldScroll = true) {
  state.activeView = view === "highlights" ? "highlights" : "batch";
  localStorage.setItem("reelFactoryView", state.activeView);
  const isPodcast = state.workMode === "podcast";
  els.batchView.classList.toggle("hidden", !isPodcast || state.activeView !== "batch");
  els.highlightsView.classList.toggle("hidden", isPodcast && state.activeView !== "highlights");
  document.querySelectorAll("[data-view]").forEach(button => {
    const active = button.dataset.view === state.activeView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  if (shouldScroll) window.scrollTo({top: 0, behavior: "smooth"});
}

function switchWorkMode(mode, shouldScroll = true) {
  const nextMode = normalizeProjectMode(mode);
  state.longFiles[state.workMode] = state.longFile;
  if (state.workMode === "podcast") state.podcastYoutubeUrl = els.youtubeUrl.value;
  stopHighlightPolling();

  state.workMode = nextMode;
  state.highlightProjectId = state.highlightProjectIds[nextMode];
  state.longFile = state.longFiles[nextMode];
  localStorage.setItem("reelFactoryWorkMode", nextMode);
  els.youtubeUrl.value = nextMode === "podcast" ? state.podcastYoutubeUrl : "";

  resetHighlightProjectUI();
  applyWorkModeCopy();
  els.podcastViewNav.classList.toggle("hidden", nextMode !== "podcast");
  document.querySelectorAll("[data-work-mode]").forEach(button => {
    const active = button.dataset.workMode === nextMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  setLongFile(state.longFile);
  setHighlightSource(nextMode === "broadcast" ? "upload" : state.podcastHighlightSource);
  switchView(state.activeView, false);
  loadHighlightHistory();
  if (state.highlightProjectId) fetchHighlightProject();
  if (shouldScroll) window.scrollTo({top: 0, behavior: "smooth"});
}

function setHighlightSource(source) {
  state.highlightSource = state.workMode === "broadcast"
    ? "upload"
    : (source === "youtube" ? "youtube" : "upload");
  if (state.workMode === "podcast") state.podcastHighlightSource = state.highlightSource;
  document.querySelectorAll("[data-source]").forEach(button => {
    const active = button.dataset.source === state.highlightSource;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  $("#uploadSourcePane").classList.toggle("hidden", state.highlightSource !== "upload");
  $("#youtubeSourcePane").classList.toggle("hidden", state.highlightSource !== "youtube");
  validateHighlightSource();
}

function validateHighlightSource() {
  const ready = state.highlightSource === "upload"
    ? Boolean(state.longFile)
    : /^https?:\/\/(?:www\.)?(?:youtube\.com|youtu\.be)\//i.test(els.youtubeUrl.value.trim());
  els.analyzeButton.disabled = !ready;
}

function setLongFile(file) {
  const extension = file?.name.split(".").pop().toLowerCase();
  if (!file || !["mp4", "mov", "mkv", "webm", "m4v"].includes(extension)) {
    state.longFile = null;
    els.longFileTitle.textContent = HIGHLIGHT_MODE_COPY[state.workMode].dropTitle;
    els.longFileMeta.textContent = HIGHLIGHT_MODE_COPY[state.workMode].dropMeta;
  } else {
    state.longFile = file;
    els.longFileTitle.textContent = file.name;
    els.longFileMeta.textContent = `${formatBytes(file.size)} · нажмите, чтобы заменить`;
  }
  state.longFiles[state.workMode] = state.longFile;
  validateHighlightSource();
}

function showHighlightError(message) {
  els.highlightFormError.textContent = message;
  els.highlightFormError.classList.remove("hidden");
}

function clearHighlightError() {
  els.highlightFormError.textContent = "";
  els.highlightFormError.classList.add("hidden");
}

async function loadProviderStatus() {
  try {
    const response = await fetch("/api/highlights/providers", {cache: "no-store"});
    const providers = await response.json();
    const describe = (name, info) => {
      if (info?.ready) return `${name} готов${info.model ? ` (${info.model})` : ""}`;
      return info?.installed ? `${name} не авторизован` : `${name} не найден`;
    };
    const resolved = ({claude: "Claude", codex: "Codex"})[providers.resolved];
    const auto = resolved ? `Авто → ${resolved}` : "нет готового ИИ-провайдера";
    els.providerHint.textContent = `${describe("Claude", providers.claude)} · ${describe("Codex", providers.codex)} · ${auto}.`;
  } catch {
    els.providerHint.textContent = "Статус ИИ-провайдеров временно недоступен.";
  }
}

function createHighlightProject() {
  clearHighlightError();
  if (state.highlightSource === "upload" && !state.longFile) return;
  if (state.highlightSource === "youtube" && !els.youtubeUrl.value.trim()) return;
  const projectMode = state.workMode;

  const minDuration = Number($("#minDuration").value);
  const maxDuration = Number($("#maxDuration").value);
  if (maxDuration <= minDuration) {
    showHighlightError("Максимальная длительность должна быть больше минимальной.");
    return;
  }
  const data = new FormData();
  data.append("source_kind", state.highlightSource);
  if (state.highlightSource === "upload") data.append("file", state.longFile, state.longFile.name);
  else data.append("youtube_url", els.youtubeUrl.value.trim());
  data.append("options", JSON.stringify({
    provider: $("#highlightProvider").value,
    min_duration: minDuration,
    max_duration: maxDuration,
    max_clips: Math.max(0, Math.min(20, Number($("#maxClips").value) || 0)),   // 0 = все ролики
    tighten: $("#highlightTighten").checked,
    all_clips: $("#highlightAllClips").checked,
    asr_glossary: $("#asrGlossary").value,
    audience: $("#highlightAudience").value,
    language: $("#highlightLanguage").value,
    subtitle_language: $("#youtubeSubtitleLanguage").value,
    model: $("#highlightModel").value,
    asr_provider: $("#highlightAsr").value,
    project_mode: projectMode,
  }));

  els.analyzeButton.disabled = true;
  els.highlightUploadProgress.classList.remove("hidden");
  const request = new XMLHttpRequest();
  request.open("POST", "/api/highlights");
  request.upload.addEventListener("progress", event => {
    if (!event.lengthComputable) return;
    const percent = Math.round(event.loaded / event.total * 100);
    els.highlightUploadBar.value = percent;
    els.highlightUploadPercent.textContent = `${percent}%`;
  });
  request.addEventListener("load", () => {
    if (state.workMode === projectMode) {
      els.highlightUploadProgress.classList.add("hidden");
      validateHighlightSource();
    }
    let project = {};
    try { project = JSON.parse(request.responseText); } catch { /* handled below */ }
    if (request.status < 200 || request.status >= 300) {
      if (state.workMode === projectMode) {
        showHighlightError(project.error || (projectMode === "broadcast" ? "Не удалось создать проект эфира." : "Не удалось создать проект нарезки."));
      }
      return;
    }
    saveHighlightProjectId(projectMode, project.id);
    if (state.workMode !== projectMode) return;
    state.highlightProjectId = project.id;
    state.selectedClips.clear();
    renderHighlightProject(project);
    startHighlightPolling();
    els.highlightProjectSection.scrollIntoView({behavior: "smooth", block: "start"});
  });
  request.addEventListener("error", () => {
    if (state.workMode === projectMode) {
      els.highlightUploadProgress.classList.add("hidden");
      validateHighlightSource();
      showHighlightError("Соединение с локальным сервисом прервано.");
    }
  });
  request.send(data);
}

function clipStatusLabel(status, projectMode = "podcast") {
  if (status === "processing" && normalizeProjectMode(projectMode) === "broadcast") {
    return "Создаю motion-графику";
  }
  return ({suggested:"Готов к выбору",queued:"В очереди",processing:"Создаю вертикальный ролик",completed:"Готово",failed:"Ошибка"})[status] || status;
}

function broadcastLayoutLabel(item) {
  const visualEngine = String(item.result?.visual_engine || "").trim().toLowerCase();
  if (item.status === "completed" && visualEngine !== "hyperframes") {
    return "Старый формат · пересоберите ролик";
  }
  return "Motion-графика · фиксированные спикеры · субтитры";
}

const ANALYSIS_PHASE_ORDER = ["preparing", "transcription", "analysis", "previews"];
const BROADCAST_MOTION_PHASE_ORDER = ["motion_plan", "motion_composition", "motion_render", "finalization"];

function canonicalAnalysisPhase(phase) {
  const value = String(phase || "").trim().toLowerCase();
  if (!value) return null;
  if (value === "completed" || value === "complete" || value === "done") return "completed";
  if (value.includes("preview") || value.includes("thumb")) return "previews";
  if (value.includes("analy") || value.includes("highlight") || value.includes("moment")) return "analysis";
  if (value.includes("transcri") || value.includes("speech") || value.includes("subtitle") || value.includes("model")) return "transcription";
  if (value.includes("prepar") || value.includes("queue") || value.includes("source") || value.includes("download") || value.includes("media") || value.includes("probe")) return "preparing";
  return null;
}

function analysisPhaseLabel(phase, projectMode, fallback) {
  const canonical = canonicalAnalysisPhase(phase);
  if (canonical === "preparing") return projectMode === "broadcast" ? "Подготовка записи эфира" : "Подготовка выпуска";
  if (canonical === "transcription") return "Транскрибация";
  if (canonical === "analysis") return "Поиск ключевых моментов";
  if (canonical === "previews") return "Подготовка превью";
  if (canonical === "completed") return "Анализ завершён";
  return fallback || "Обработка";
}

function canonicalBroadcastMotionPhase(phase) {
  const value = String(phase || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["motion_plan", "visual_plan", "storyboard"].includes(value)) return "motion_plan";
  if (["motion_composition", "motion_graphics", "hyperframes_composition"].includes(value)) return "motion_composition";
  if (["motion_render", "hyperframes_render"].includes(value)) return "motion_render";
  if (["finalization", "finalizing", "speaker_subtitles", "speaker_and_subtitles"].includes(value)) return "finalization";
  return null;
}

function broadcastMotionPhaseLabel(phase, fallback) {
  const canonical = canonicalBroadcastMotionPhase(phase);
  if (canonical === "motion_plan") return "Создаю визуальный план";
  if (canonical === "motion_composition") return "Собираю motion-графику";
  if (canonical === "motion_render") return "Рендерю motion-графику";
  if (canonical === "finalization") return "Добавляю спикеров и субтитры";
  if (["render", "rendering"].includes(String(phase || "").trim().toLowerCase())) return "Рендер ролика";
  return fallback || "Создаю эфирный ролик";
}

function renderAnalysisPhaseProgress(project, projectMode) {
  const phaseProgress = finiteNumber(project.phase_progress);
  const processed = finiteNumber(project.processed_seconds);
  const total = finiteNumber(project.total_seconds);
  const rawEta = finiteNumber(project.eta_seconds);
  const eta = rawEta !== null && rawEta >= 0 ? rawEta : null;
  const hasStructuredProgress = Boolean(project.phase) || phaseProgress !== null || processed !== null || total !== null || eta !== null;
  const motionPhase = canonicalBroadcastMotionPhase(project.phase);
  const broadcastRendering = projectMode === "broadcast"
    && Boolean(project.phase)
    && (project.status === "rendering" || project.error_stage === "rendering" || Boolean(motionPhase));
  const analysisVisible = hasStructuredProgress
    && ["queued", "analyzing", "failed", "interrupted"].includes(project.status)
    && !(projectMode === "broadcast" && project.error_stage === "rendering");
  const visible = broadcastRendering || analysisVisible;
  els.highlightPhaseProgress.classList.toggle("hidden", !visible);
  if (!visible) return;

  const canonical = broadcastRendering ? motionPhase : canonicalAnalysisPhase(project.phase);
  const clampedProgress = phaseProgress === null ? null : Math.max(0, Math.min(100, phaseProgress));
  els.highlightPhaseTitle.textContent = broadcastRendering
    ? broadcastMotionPhaseLabel(project.phase, project.message)
    : analysisPhaseLabel(project.phase, projectMode, project.message);
  if (clampedProgress === null) {
    els.highlightPhaseBar.removeAttribute("value");
    els.highlightPhaseBar.classList.add("indeterminate");
    els.highlightPhasePercent.textContent = "В работе";
  } else {
    els.highlightPhaseBar.classList.remove("indeterminate");
    els.highlightPhaseBar.value = clampedProgress;
    els.highlightPhasePercent.textContent = `${Math.round(clampedProgress)}%`;
  }

  let processedText = "";
  if (!broadcastRendering) {
    if (processed !== null && total !== null && total > 0) {
      processedText = `Обработано ${formatClockDuration(processed)} из ${formatClockDuration(total)}`;
    } else if (processed !== null) {
      processedText = `Обработано ${formatClockDuration(processed)}`;
    } else if (total !== null && total > 0) {
      processedText = `Длительность ${formatClockDuration(total)}`;
    }
  }
  els.highlightPhaseProcessed.textContent = processedText;
  els.highlightPhaseProcessed.classList.toggle("hidden", !processedText);

  const etaText = eta === null || (broadcastRendering && eta <= 0)
    ? ""
    : (eta <= 0 ? "Этап почти завершён" : `Осталось примерно ${formatEta(eta)}`);
  els.highlightPhaseEta.textContent = etaText;
  els.highlightPhaseEta.classList.toggle("hidden", !etaText);

  els.highlightAnalysisSteps.classList.toggle("hidden", broadcastRendering);
  els.highlightMotionSteps.classList.toggle("hidden", !broadcastRendering || !motionPhase);
  const phaseOrder = broadcastRendering ? BROADCAST_MOTION_PHASE_ORDER : ANALYSIS_PHASE_ORDER;
  const stepSelector = broadcastRendering ? "[data-motion-phase]" : "[data-analysis-phase]";
  const activeIndex = canonical === "completed" ? phaseOrder.length : phaseOrder.indexOf(canonical);
  els.highlightPhaseProgress.querySelectorAll("[data-analysis-phase], [data-motion-phase]").forEach(step => {
    step.classList.remove("completed", "active", "failed");
  });
  els.highlightPhaseProgress.querySelectorAll(stepSelector).forEach(step => {
    const stepPhase = broadcastRendering ? step.dataset.motionPhase : step.dataset.analysisPhase;
    const stepIndex = phaseOrder.indexOf(stepPhase);
    step.classList.toggle("completed", activeIndex === phaseOrder.length || (activeIndex >= 0 && stepIndex < activeIndex));
    step.classList.toggle("active", activeIndex >= 0 && stepIndex === activeIndex);
    step.classList.toggle("failed", project.status === "failed" && activeIndex >= 0 && stepIndex === activeIndex);
  });
}

function hasSavedTranscript(project) {
  return project.transcript_saved === true
    || project.transcript_available === true
    || project.has_transcript === true
    || Boolean(project.transcript);
}

function renderSelectionSummary() {
  const count = state.selectedClips.size;
  els.selectionSummary.textContent = `${count} ${count === 1 ? "выбран" : "выбрано"}`;
  els.renderSelectedButton.disabled = count === 0;
}

function renderHighlightProject(project) {
  const projectMode = projectModeOf(project);
  if (projectMode !== state.workMode) return;
  const subtitleStyle = project.options?.subtitle_style;
  if (els.highlightSubtitleStyle.dataset.projectId !== project.id) {
    if (["pill", "clean", "yellow"].includes(subtitleStyle)) {
      els.highlightSubtitleStyle.value = subtitleStyle;
    }
    els.highlightSubtitleStyle.dataset.projectId = project.id;
  }
  if (els.highlightVideoReview.dataset.projectId !== project.id) {
    els.highlightVideoReview.checked = project.options?.video_review !== false;
    els.highlightVideoReview.dataset.projectId = project.id;
  }
  const playbackSpeed = Number(project.options?.playback_speed || 1);
  if (els.highlightPlaybackSpeed.dataset.projectId !== project.id) {
    const speedValue = String(playbackSpeed);
    if ([...els.highlightPlaybackSpeed.options].some(option => option.value === speedValue)) {
      els.highlightPlaybackSpeed.value = speedValue;
    } else {
      els.highlightPlaybackSpeed.value = "1";
    }
    els.highlightPlaybackSpeed.dataset.projectId = project.id;
  }
  const speakerCount = String(project.options?.speaker_count || 2);
  if (els.highlightSpeakerCount.dataset.projectId !== project.id) {
    els.highlightSpeakerCount.value = ["1", "2"].includes(speakerCount) ? speakerCount : "2";
    els.highlightSpeakerCount.dataset.projectId = project.id;
  }
  const visualPreset = String(project.options?.visual_preset || "editorial_lime");
  if (els.highlightVisualPreset.dataset.projectId !== project.id) {
    els.highlightVisualPreset.value = ["editorial_lime", "itbooster"].includes(visualPreset)
      ? visualPreset
      : "editorial_lime";
    els.highlightVisualPreset.dataset.projectId = project.id;
  }
  els.playbackSpeedGroup.classList.toggle("hidden", projectMode !== "broadcast");
  els.speakerCountGroup.classList.toggle("hidden", projectMode !== "broadcast");
  els.visualPresetGroup.classList.toggle("hidden", projectMode !== "broadcast");
  els.highlightProjectSection.classList.remove("hidden");
  const candidates = project.candidates || [];
  const choosing = ["awaiting_selection", "completed", "partial"].includes(project.status)
    || (project.status === "failed" && candidates.length > 0);
  const rendering = project.status === "rendering";
  const activeTitle = projectMode === "broadcast"
    ? (choosing ? "Ключевые моменты эфира найдены" : (rendering ? "Создаю motion-графику для рилсов" : "Анализирую эфир"))
    : (choosing ? "Лучшие моменты найдены" : (rendering ? "Создаю вертикальные ролики" : "Анализирую выпуск"));
  els.highlightProjectTitle.textContent = project.paused ? "Процесс на паузе" : activeTitle;
  els.highlightProjectEyebrow.textContent = HIGHLIGHT_MODE_COPY[projectMode].projectEyebrow;
  els.highlightModeBadge.textContent = projectModeLabel(projectMode);
  els.highlightProjectMessage.textContent = project.message || "";
  const projectProgress = finiteNumber(project.progress);
  if (projectMode === "broadcast" && rendering && projectProgress === null) {
    els.highlightProjectBar.removeAttribute("value");
    els.highlightProjectBar.classList.add("indeterminate");
    els.highlightProjectPercent.textContent = "В работе";
  } else {
    const clampedProjectProgress = Math.max(0, Math.min(100, projectProgress ?? 0));
    els.highlightProjectBar.classList.remove("indeterminate");
    els.highlightProjectBar.value = clampedProjectProgress;
    els.highlightProjectPercent.textContent = `${Math.round(clampedProjectProgress)}%`;
  }
  renderAnalysisPhaseProgress(project, projectMode);
  const pauseAvailable = ["queued", "analyzing", "rendering"].includes(project.status);
  els.highlightPauseButton.classList.toggle("hidden", !pauseAvailable);
  els.highlightPauseButton.classList.toggle("is-paused", Boolean(project.paused));
  els.highlightPauseButton.textContent = project.paused ? "Продолжить" : "Поставить на паузу";
  els.highlightPauseButton.dataset.paused = project.paused ? "true" : "false";
  const canRetryAnalysis = ["failed", "interrupted"].includes(project.status)
    && project.error_stage === "analysis" && hasSavedTranscript(project);
  els.highlightRetryAnalysisButton.classList.toggle("hidden", !canRetryAnalysis);
  if (canRetryAnalysis && !els.highlightRetryAnalysisButton.disabled) {
    els.highlightRetryAnalysisButton.textContent = "Повторить поиск моментов";
  }
  els.highlightWarning.textContent = project.error || project.warning || "";
  els.highlightWarning.classList.toggle("hidden", !project.error && !project.warning);
  els.selectionToolbar.classList.toggle("hidden", !choosing || !candidates.length);
  if (project.zip_url) {
    els.highlightDownloadAll.href = project.zip_url;
    els.highlightDownloadAll.classList.remove("hidden");
  } else {
    els.highlightDownloadAll.removeAttribute("href");
    els.highlightDownloadAll.classList.add("hidden");
  }

  const providerLabel = ({codex:"Codex", claude:"Claude", heuristic:"встроенный анализ"})[project.provider_used] || "";
  state.highlightProject = project;
  const isRejected = item => Boolean(item.rejected) && !item.rejected_forced && !state.forcedClips.has(item.id);
  const mainCandidates = candidates.filter(item => !isRejected(item));
  const rejectedCandidates = candidates.filter(isRejected);
  const cardContext = {projectMode, providerLabel, rendering};
  els.clipsGrid.innerHTML = mainCandidates.map(item => clipCardHtml(item, cardContext)).join("");
  renderRejectedBlock(rejectedCandidates, cardContext);
  if (!candidates.length && project.status === "failed") {
    els.clipsGrid.innerHTML = `<p class="empty-state">${escapeHtml(project.error || "Анализ завершился с ошибкой.")}</p>`;
  }
  renderSelectionSummary();
}

function clipCardHtml(item, {projectMode, providerLabel, rendering}) {

    const checked = state.selectedClips.has(item.id) || item.selected;
    const itemProgress = finiteNumber(item.progress);
    const clampedItemProgress = itemProgress === null ? null : Math.max(0, Math.min(100, itemProgress));
    const clipProgress = item.status === "suggested" ? "" : (projectMode === "broadcast"
      ? `<div class="clip-progress"><span>${escapeHtml(item.message || clipStatusLabel(item.status, projectMode))}</span><b>${clampedItemProgress === null ? "В работе" : `${Math.round(clampedItemProgress)}%`}</b><progress class="${clampedItemProgress === null ? "indeterminate" : ""}" max="100"${clampedItemProgress === null ? "" : ` value="${clampedItemProgress}"`}></progress></div>`
      : `<div class="clip-progress"><span>${escapeHtml(item.message || clipStatusLabel(item.status, projectMode))}</span><b>${item.progress || 0}%</b><progress max="100" value="${item.progress || 0}"></progress></div>`);
    if (item.selected && !state.selectedClips.has(item.id)) state.selectedClips.add(item.id);
    return `<article class="clip-card ${escapeHtml(item.status)} ${checked ? "is-selected" : ""}" data-clip-card="${item.id}">
      <label class="clip-check"><input type="checkbox" data-clip-select="${item.id}" ${checked ? "checked" : ""} ${rendering ? "disabled" : ""}><span>Выбрать</span></label>
      <div class="clip-thumb-wrap">
        <img class="clip-thumb" src="${item.thumbnail_url}" alt="" loading="lazy">
        <span class="clip-score">${item.score}/100</span>
        <span class="clip-time">${formatDuration(item.start)}–${formatDuration(item.end)}</span>
      </div>
      <div class="clip-body">
        <div class="clip-meta">${formatDuration(item.duration)}${providerLabel ? ` · ${providerLabel}` : ""}</div>
        ${projectMode === "broadcast" ? `<div class="clip-layout-mode">${broadcastLayoutLabel(item)}</div>` : ""}
        <h3>${escapeHtml(item.title)}</h3>
        ${item.thesis ? `<p class="clip-thesis"><b>${escapeHtml(CLIP_FORMAT_LABELS[item.format] || "Тезис")}:</b> ${escapeHtml(item.thesis)}</p>` : ""}
        ${clipHookHtml(item)}
        <p>${escapeHtml(item.description || "")}</p>
        ${clipOriginHtml(item)}
        ${item.rejected ? rejectedReasonHtml(item) : ""}
        ${clipChiefHtml(item)}
        ${clipCriticHtml(item)}
        ${clipReviewHtml(item)}
        ${clipStructureHtml(item)}
        ${clipProgress}
        ${item.error ? `<p class="file-error">${escapeHtml(item.error)}</p>` : ""}
        <div class="clip-download-actions">
          ${item.source_download_url ? `<a class="download-button download-button-secondary" href="${item.source_download_url}">Скачать исходный фрагмент</a>` : ""}
          ${item.download_url ? `<a class="download-button" href="${item.download_url}">Скачать готовый MP4</a>` : ""}
        </div>
      </div>
    </article>`;
}

function clipHookHtml(item) {
  const hook = item.hook_text || item.hook;
  return hook ? `<p class="clip-hook"><b>Хук для визуала:</b> «${escapeHtml(hook)}» <small class="meta">(титр на экране, не озвучивается)</small></p>` : "";
}

function clipOriginHtml(item) {
  const tags = [];
  if (item.source === "coverage") tags.push(`пропущенный момент${item.moment ? `: ${escapeHtml(CLIP_MOMENT_LABELS[item.moment] || item.moment)}` : ""}`);
  if (item.spinoff_of) tags.push(`спин-офф ролика «${escapeHtml(item.spinoff_of)}»`);
  if (item.split_of) tags.push(`часть длинного ролика «${escapeHtml(item.split_of)}»`);
  const fixes = (item.local_fixes || []).map(x => `«${escapeHtml(x.text)}»`).join(", ");
  if (fixes) tags.push(`подрезано: ${fixes}`);
  return tags.length ? `<p class="clip-origin meta">${tags.join(" · ")}</p>` : "";
}

function rejectedReasonHtml(item) {
  const info = item.rejected || {};
  const stage = REJECT_STAGE_LABELS[info.stage] || info.stage || "";
  return `<p class="reject-reason"><b>Не прошёл проверку${stage ? ` (${escapeHtml(stage)})` : ""}:</b> ${escapeHtml(info.reason || "")}</p>`;
}

function renderRejectedBlock(list, {providerLabel}) {
  els.rejectedBlock.classList.toggle("hidden", !list.length);
  els.rejectedCount.textContent = String(list.length);
  els.rejectedGrid.innerHTML = list.map(item => `<article class="clip-card rejected" data-rejected-card="${item.id}">
      <div class="clip-thumb-wrap">
        <img class="clip-thumb" src="${item.thumbnail_url}" alt="" loading="lazy">
        <span class="clip-time">${formatDuration(item.start)}–${formatDuration(item.end)}</span>
      </div>
      <div class="clip-body">
        <div class="clip-meta">${formatDuration(item.duration)}${providerLabel ? ` · ${providerLabel}` : ""} · ${escapeHtml(CLIP_FORMAT_LABELS[item.format] || "Тезис")}</div>
        <h3>${escapeHtml(item.title)}</h3>
        ${item.thesis ? `<p class="clip-thesis">${escapeHtml(item.thesis)}</p>` : ""}
        ${clipHookHtml(item)}
        ${rejectedReasonHtml(item)}
        ${clipCriticHtml(item)}
        <div class="clip-download-actions">
          <button class="download-button download-button-secondary" type="button" data-force-clip="${item.id}">Всё равно сделать</button>
        </div>
      </div>
    </article>`).join("");
}

async function fetchHighlightProject() {
  if (!state.highlightProjectId) return;
  const projectId = state.highlightProjectId;
  const requestedMode = state.workMode;
  try {
    const response = await fetch(`/api/highlights/${projectId}`, {cache: "no-store"});
    if (!response.ok) return;
    const project = await response.json();
    if (state.highlightProjectId !== projectId || state.workMode !== requestedMode || projectModeOf(project) !== requestedMode) return;
    renderHighlightProject(project);
    if (["awaiting_selection", "completed", "partial", "failed", "interrupted"].includes(project.status)) {
      stopHighlightPolling();
      loadHighlightHistory();
    }
  } catch { /* retry on next pass */ }
}

function startHighlightPolling() {
  stopHighlightPolling();
  fetchHighlightProject();
  state.highlightPollTimer = setInterval(fetchHighlightProject, 1500);
}

function stopHighlightPolling() {
  if (state.highlightPollTimer) clearInterval(state.highlightPollTimer);
  state.highlightPollTimer = null;
}

async function retryHighlightAnalysis() {
  if (!state.highlightProjectId || els.highlightRetryAnalysisButton.disabled) return;
  const projectId = state.highlightProjectId;
  const requestedMode = state.workMode;
  els.highlightRetryAnalysisButton.disabled = true;
  els.highlightRetryAnalysisButton.textContent = "Запускаю повторно…";
  try {
    const response = await fetch(`/api/highlights/${projectId}/retry-analysis`, {method: "POST"});
    const project = await response.json();
    if (!response.ok) throw new Error(project.error || "Не удалось повторить поиск моментов.");
    if (state.highlightProjectId !== projectId || state.workMode !== requestedMode) return;
    state.selectedClips.clear();
    clearHighlightError();
    renderHighlightProject(project);
    startHighlightPolling();
  } catch (error) {
    if (state.highlightProjectId === projectId && state.workMode === requestedMode) {
      els.highlightWarning.textContent = error.message;
      els.highlightWarning.classList.remove("hidden");
    }
  } finally {
    if (state.highlightProjectId === projectId && state.workMode === requestedMode) {
      els.highlightRetryAnalysisButton.disabled = false;
      els.highlightRetryAnalysisButton.textContent = "Повторить поиск моментов";
    }
  }
}

async function renderSelectedClips() {
  if (!state.highlightProjectId || !state.selectedClips.size) return;
  const projectId = state.highlightProjectId;
  const requestedMode = state.workMode;
  els.renderSelectedButton.disabled = true;
  try {
    const response = await fetch(`/api/highlights/${projectId}/render`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        selected_ids: [...state.selectedClips],
        subtitle_style: $("#highlightSubtitleStyle").value,
        playback_speed: requestedMode === "broadcast" ? Number(els.highlightPlaybackSpeed.value) : 1,
        speaker_count: requestedMode === "broadcast" ? Number(els.highlightSpeakerCount.value) : 1,
        visual_preset: requestedMode === "broadcast" ? els.highlightVisualPreset.value : "editorial_lime",
        video_review: els.highlightVideoReview.checked,
      }),
    });
    const project = await response.json();
    if (!response.ok) throw new Error(project.error || "Не удалось запустить нарезку.");
    if (state.highlightProjectId !== projectId || state.workMode !== requestedMode) return;
    renderHighlightProject(project);
    startHighlightPolling();
  } catch (error) {
    if (state.highlightProjectId === projectId && state.workMode === requestedMode) {
      showHighlightError(error.message);
      renderSelectionSummary();
    }
  }
}

async function toggleProcessPause(kind) {
  const highlights = kind === "highlights";
  const id = highlights ? state.highlightProjectId : state.jobId;
  const button = highlights ? els.highlightPauseButton : els.jobPauseButton;
  if (!id || button.disabled) return;
  const currentlyPaused = button.dataset.paused === "true";
  button.disabled = true;
  try {
    const response = await fetch(`/api/${highlights ? "highlights" : "jobs"}/${id}/${currentlyPaused ? "resume" : "pause"}`, {method: "POST"});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Не удалось изменить состояние процесса.");
    if (highlights) renderHighlightProject(payload);
    else renderJob(payload);
  } catch (error) {
    if (highlights) showHighlightError(error.message);
    else showError(error.message);
  } finally {
    button.disabled = false;
  }
}

async function loadHighlightHistory() {
  const requestedMode = state.workMode;
  const copy = HIGHLIGHT_MODE_COPY[requestedMode];
  try {
    const response = await fetch(`/api/highlights?project_mode=${encodeURIComponent(requestedMode)}`, {cache: "no-store"});
    const data = await response.json();
    if (state.workMode !== requestedMode) return;
    const projects = (data.projects || []).filter(project => projectModeOf(project) === requestedMode);
    if (!projects.length) {
      els.highlightHistoryList.innerHTML = `<p class="empty-state">${escapeHtml(copy.historyEmpty)}</p>`;
      return;
    }
    els.highlightHistoryList.innerHTML = projects.map(project => {
      const projectMode = projectModeOf(project);
      const candidates = project.candidates || [];
      const created = new Date(project.created_at).toLocaleString("ru-RU", {day:"2-digit", month:"short", hour:"2-digit", minute:"2-digit"});
      const ready = candidates.filter(item => item.status === "completed").length;
      return `<div class="history-item">
        <div><div class="history-title-row"><b>${escapeHtml(project.source?.title || project.source?.name || (projectMode === "broadcast" ? "Запись эфира" : "Длинное видео"))}</b><span class="project-mode-badge">${projectModeLabel(projectMode)}</span></div><small>${created} · найдено ${candidates.length} · готово ${ready}</small></div>
        <span class="history-state">${escapeHtml(statusLabel(project.status))}</span>
        <button class="text-button" data-open-highlight="${project.id}" data-project-mode="${projectMode}">Открыть</button>
      </div>`;
    }).join("");
  } catch {
    if (state.workMode === requestedMode) els.highlightHistoryList.innerHTML = '<p class="empty-state">Не удалось загрузить историю.</p>';
  }
}

els.fileInput.addEventListener("change", event => { addFiles(event.target.files); event.target.value = ""; });
els.fileList.addEventListener("click", event => {
  const button = event.target.closest("[data-remove]");
  if (!button) return;
  state.files.splice(Number(button.dataset.remove), 1);
  renderSelectedFiles();
});
["dragenter", "dragover"].forEach(type => els.dropzone.addEventListener(type, event => {
  event.preventDefault(); els.dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => els.dropzone.addEventListener(type, event => {
  event.preventDefault(); els.dropzone.classList.remove("dragging");
}));
els.dropzone.addEventListener("drop", event => addFiles(event.dataTransfer.files));
els.startButton.addEventListener("click", createJob);
els.jobPauseButton.addEventListener("click", () => toggleProcessPause("jobs"));
$("#refreshHistory").addEventListener("click", loadHistory);
els.historyList.addEventListener("click", event => {
  const button = event.target.closest("[data-open-job]");
  if (!button) return;
  state.jobId = button.dataset.openJob;
  localStorage.setItem("reelFactoryJob", state.jobId);
  startPolling();
  els.jobSection.scrollIntoView({behavior: "smooth"});
});

document.querySelectorAll("[data-view]").forEach(button => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});
document.querySelectorAll("[data-work-mode]").forEach(button => {
  button.addEventListener("click", () => switchWorkMode(button.dataset.workMode));
});
document.querySelectorAll("[data-source]").forEach(button => {
  button.addEventListener("click", () => setHighlightSource(button.dataset.source));
});
els.longVideoInput.addEventListener("change", event => {
  setLongFile(event.target.files[0]);
  event.target.value = "";
});
["dragenter", "dragover"].forEach(type => els.longDropzone.addEventListener(type, event => {
  event.preventDefault(); els.longDropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => els.longDropzone.addEventListener(type, event => {
  event.preventDefault(); els.longDropzone.classList.remove("dragging");
}));
els.longDropzone.addEventListener("drop", event => setLongFile(event.dataTransfer.files[0]));
els.youtubeUrl.addEventListener("input", () => {
  if (state.workMode === "podcast") state.podcastYoutubeUrl = els.youtubeUrl.value;
  validateHighlightSource();
});
els.analyzeButton.addEventListener("click", createHighlightProject);
els.clipsGrid.addEventListener("change", event => {
  const checkbox = event.target.closest("[data-clip-select]");
  if (!checkbox) return;
  if (checkbox.checked) state.selectedClips.add(checkbox.dataset.clipSelect);
  else state.selectedClips.delete(checkbox.dataset.clipSelect);
  checkbox.closest(".clip-card")?.classList.toggle("is-selected", checkbox.checked);
  renderSelectionSummary();
});
els.rejectedGrid.addEventListener("click", event => {
  const button = event.target.closest("[data-force-clip]");
  if (!button) return;
  // «Всё равно сделать»: кандидат переезжает в основной список и сразу выбран для рендера
  state.forcedClips.add(button.dataset.forceClip);
  state.selectedClips.add(button.dataset.forceClip);
  if (state.highlightProject) renderHighlightProject(state.highlightProject);
  renderSelectionSummary();
});
$("#selectAllClips").addEventListener("click", () => {
  const boxes = [...els.clipsGrid.querySelectorAll("[data-clip-select]:not(:disabled)")];
  const select = boxes.some(box => !box.checked);
  boxes.forEach(box => {
    box.checked = select;
    if (select) state.selectedClips.add(box.dataset.clipSelect);
    else state.selectedClips.delete(box.dataset.clipSelect);
    box.closest(".clip-card")?.classList.toggle("is-selected", select);
  });
  $("#selectAllClips").textContent = select ? "Снять выбор" : "Выбрать все";
  renderSelectionSummary();
});
els.renderSelectedButton.addEventListener("click", renderSelectedClips);
els.highlightPauseButton.addEventListener("click", () => toggleProcessPause("highlights"));
els.highlightRetryAnalysisButton.addEventListener("click", retryHighlightAnalysis);
$("#refreshHighlightHistory").addEventListener("click", loadHighlightHistory);
els.highlightHistoryList.addEventListener("click", event => {
  const button = event.target.closest("[data-open-highlight]");
  if (!button) return;
  const projectMode = normalizeProjectMode(button.dataset.projectMode);
  if (projectMode !== state.workMode) switchWorkMode(projectMode, false);
  state.highlightProjectId = button.dataset.openHighlight;
  saveHighlightProjectId(projectMode, state.highlightProjectId);
  state.selectedClips.clear();
  resetHighlightProjectUI();
  if (projectMode === "podcast") switchView("highlights", false);
  fetchHighlightProject().then(() => els.highlightProjectSection.scrollIntoView({behavior: "smooth", block: "start"}));
});

renderSelectedFiles();
loadHistory();
if (state.jobId) startPolling();
loadProviderStatus();
switchWorkMode(state.workMode, false);
