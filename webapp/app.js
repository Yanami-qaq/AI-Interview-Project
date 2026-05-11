const apiBase = `${window.location.origin}`;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const speechSynthesisApi = window.speechSynthesis;

const state = {
  sessionId: "",
  role: "",
  recognition: null,
  isRecording: false,
  finalTranscript: "",
  interimTranscript: "",
  recordingStartedAt: 0,
  ttsVoice: null,
  messages: [],
};

const views = {
  setup: document.getElementById("setupView"),
  interview: document.getElementById("interviewView"),
  report: document.getElementById("reportView"),
};

const roleSelect = document.getElementById("roleSelect");
const topicInput = document.getElementById("topicInput");
const difficultySelect = document.getElementById("difficultySelect");
const queryInput = document.getElementById("queryInput");
const resumeInput = document.getElementById("resumeInput");
const startForm = document.getElementById("startForm");
const answerForm = document.getElementById("answerForm");
const answerInput = document.getElementById("answerInput");
const answerButton = document.getElementById("answerButton");
const finishButton = document.getElementById("finishButton");
const restartButton = document.getElementById("restartButton");
const questionText = document.getElementById("questionText");
const questionMeta = document.getElementById("questionMeta");
const guidanceText = document.getElementById("guidanceText");
const sessionIdTag = document.getElementById("sessionIdTag");
const healthStatus = document.getElementById("healthStatus");
const apiBaseLabel = document.getElementById("apiBaseLabel");
const overallScore = document.getElementById("overallScore");
const roundCount = document.getElementById("roundCount");
const weakestMetric = document.getElementById("weakestMetric");
const scoreCards = document.getElementById("scoreCards");
const liveHighlights = document.getElementById("liveHighlights");
const liveImprovements = document.getElementById("liveImprovements");
const roundScore = document.getElementById("roundScore");
const roundConfidence = document.getElementById("roundConfidence");
const roundDimensions = document.getElementById("roundDimensions");
const scoreEvidenceList = document.getElementById("scoreEvidenceList");
const interviewChat = document.getElementById("interviewChat");
const reportNarrative = document.getElementById("reportNarrative");
const reportConversation = document.getElementById("reportConversation");
const convList = document.getElementById("convList");
const actionPlan = document.getElementById("actionPlan");
const resourceList = document.getElementById("resourceList");
const voiceButton = document.getElementById("voiceButton");
const voiceSubmitButton = document.getElementById("voiceSubmitButton");
const voiceClearButton = document.getElementById("voiceClearButton");
const voiceStatus = document.getElementById("voiceStatus");
const voiceTranscript = document.getElementById("voiceTranscript");
const autoSpeakToggle = document.getElementById("autoSpeakToggle");
const stopSpeakButton = document.getElementById("stopSpeakButton");
const historyPanel = document.getElementById("historyPanel");
const historyList = document.getElementById("historyList");
const historyPrevBtn = document.getElementById("historyPrevBtn");
const historyNextBtn = document.getElementById("historyNextBtn");
const historyPageLabel = document.getElementById("historyPageLabel");
const historyState = { sessions: [], page: 0, perPage: 5 };

apiBaseLabel.textContent = apiBase;

function setButtonBusy(button, busy, busyText = "处理中") {
  if (!button) return;
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.dataset.wasDisabled = String(button.disabled);
    button.textContent = busyText;
    button.disabled = true;
    button.classList.add("is-loading");
    return;
  }
  button.textContent = button.dataset.originalText || button.textContent;
  button.disabled = button.dataset.wasDisabled === "true";
  button.classList.remove("is-loading");
}

function showView(name, options = {}) {
  Object.entries(views).forEach(([key, view]) => {
    view.classList.toggle("active", key === name);
    view.setAttribute("aria-hidden", key === name ? "false" : "true");
  });
  document.querySelectorAll(".step-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.step === name);
    const order = ["setup", "interview", "report"];
    item.classList.toggle("complete", order.indexOf(item.dataset.step) < order.indexOf(name));
  });
  document.body.dataset.view = name;
  if (window.location.hash !== `#${name}`) {
    const method = options.replace ? "replaceState" : "pushState";
    window.history[method](null, "", `#${name}`);
  }
  window.scrollTo({ top: 0, behavior: options.instant ? "auto" : "smooth" });
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `请求失败: ${response.status}`);
  }
  return payload;
}

function setHealth(online, text) {
  healthStatus.textContent = text;
  healthStatus.classList.toggle("online", online);
  healthStatus.classList.toggle("offline", !online);
}

function setVoiceStatus(text, mode = "") {
  voiceStatus.textContent = text;
  voiceStatus.classList.toggle("recording", mode === "recording");
  voiceStatus.classList.toggle("success", mode === "success");
}

function pickChineseVoice() {
  if (!speechSynthesisApi) return null;
  const voices = speechSynthesisApi.getVoices();
  return (
    voices.find((voice) => voice.lang?.toLowerCase().startsWith("zh-cn")) ||
    voices.find((voice) => voice.lang?.toLowerCase().startsWith("zh")) ||
    voices[0] ||
    null
  );
}

function stopSpeaking() {
  if (speechSynthesisApi) speechSynthesisApi.cancel();
}

function speakInterviewer(text) {
  if (!speechSynthesisApi || !autoSpeakToggle.checked || !text?.trim()) return;
  stopSpeaking();
  const utterance = new SpeechSynthesisUtterance(text.trim());
  utterance.lang = "zh-CN";
  utterance.rate = 0.95;
  utterance.pitch = 1;
  utterance.volume = 1;
  state.ttsVoice = state.ttsVoice || pickChineseVoice();
  if (state.ttsVoice) utterance.voice = state.ttsVoice;
  speechSynthesisApi.speak(utterance);
}

function currentVoiceText() {
  return `${state.finalTranscript}${state.interimTranscript}`.trim();
}

function renderVoiceTranscript() {
  const text = currentVoiceText();
  voiceTranscript.textContent = text || "暂无转写内容。";
  voiceSubmitButton.disabled = !state.sessionId || !state.finalTranscript.trim();
  voiceClearButton.disabled = !state.finalTranscript.trim() && !state.interimTranscript.trim();
}

function renderList(target, items, fallback) {
  target.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = fallback;
    target.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = typeof item === "string" ? item : item.text || item.description || JSON.stringify(item);
    target.appendChild(li);
  });
}

function confidenceLabel(value) {
  const labels = {
    high: "高置信度",
    medium: "中置信度",
    low: "低置信度",
  };
  return labels[value] || "未评分";
}

function renderRoundScore(evaluation = null) {
  if (!roundScore) return;
  if (!evaluation) {
    roundScore.textContent = "--";
    roundConfidence.textContent = "未评分";
    roundConfidence.className = "tag muted";
    roundDimensions.innerHTML = '<div class="empty-line">暂无评分。</div>';
    renderList(scoreEvidenceList, [], "暂无内容。");
    return;
  }

  roundScore.textContent = evaluation.score ?? "--";
  const confidence = evaluation.confidence || "medium";
  roundConfidence.textContent = confidenceLabel(confidence);
  roundConfidence.className = `tag ${confidence === "high" ? "success" : confidence === "low" ? "recording" : "accent"}`;

  const rubric = evaluation.rubric || {};
  const dimensions = evaluation.dimensions || {};
  roundDimensions.innerHTML = "";
  Object.entries(dimensions).forEach(([key, value]) => {
    const item = document.createElement("div");
    item.className = "dimension-row";
    const ratio = Math.max(0.04, Math.min((Number(value) || 0) / 10, 1));
    item.style.setProperty("--score-ratio", ratio.toFixed(2));
    item.innerHTML = `
      <div>
        <strong>${rubric[key]?.label || key}</strong>
        <span>${value}/10</span>
      </div>
      <i></i>
    `;
    roundDimensions.appendChild(item);
  });

  const matched = (evaluation.point_checks || []).filter((item) => item.matched).slice(0, 3);
  const missing = (evaluation.point_checks || []).filter((item) => !item.matched).slice(0, 3);
  const relevance = evaluation.relevance || {};
  const evidenceItems = [
    ...(relevance.level && relevance.level !== "relevant"
      ? [`相关性：${relevance.reason || "回答与当前题目关联度不足。"}`]
      : []),
    ...(evaluation.llm_review ? [`面试官评价：${evaluation.llm_review}`] : []),
    ...matched.map((item) => `已覆盖：${item.point}`),
    ...missing.map((item) => `待补充：${item.point}`),
    ...(evaluation.evidence || []).slice(0, 2).map((item) => `证据：${item}`),
  ];
  renderList(scoreEvidenceList, evidenceItems, "暂无内容。");
}

function renderChat() {
  interviewChat.innerHTML = "";
  if (!state.messages.length) {
    const empty = document.createElement("div");
    empty.className = "chat-empty";
    empty.textContent = "开始面试后，对话记录会显示在这里。";
    interviewChat.appendChild(empty);
    return;
  }
  state.messages.forEach((message) => {
    const row = document.createElement("div");
    row.className = `chat-message ${message.role}`;
    const speaker = document.createElement("div");
    speaker.className = "chat-speaker";
    speaker.textContent = message.role === "candidate" ? "我" : "AI 面试官";
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.textContent = message.text;
    row.appendChild(speaker);
    row.appendChild(bubble);
    interviewChat.appendChild(row);
  });
  interviewChat.scrollTop = interviewChat.scrollHeight;
}

function appendChatMessage(role, text) {
  const cleaned = (text || "").trim();
  if (!cleaned) return;
  state.messages.push({ role, text: cleaned });
  renderChat();
}

function updateLastInterviewerMessage(text) {
  const cleaned = (text || "").trim();
  if (!cleaned) return;
  const bubbles = interviewChat.querySelectorAll(".chat-message.interviewer .chat-bubble");
  const last = bubbles[bubbles.length - 1];
  if (last) last.textContent = cleaned;
  for (let i = state.messages.length - 1; i >= 0; i--) {
    if (state.messages[i].role === "interviewer") {
      state.messages[i].text = cleaned;
      break;
    }
  }
  interviewChat.scrollTop = interviewChat.scrollHeight;
}

function resetChat() {
  state.messages = [];
  renderChat();
}

function renderScoreCards(cards = []) {
  scoreCards.innerHTML = "";
  if (!cards.length) {
    scoreCards.innerHTML = `
      <div class="metric-item">
        <div class="metric-label">等待报告生成</div>
        <div class="metric-score"><strong>--</strong><span>/10</span></div>
      </div>
    `;
    return;
  }
  cards.forEach((card) => {
    const div = document.createElement("div");
    div.className = "metric-item";
    const score = Number(card.score) || 0;
    const maxScore = Number(card.max_score) || 10;
    div.style.setProperty("--score-ratio", Math.max(0.04, Math.min(score / maxScore, 1)).toFixed(2));
    div.innerHTML = `
      <div class="metric-label">${card.label}</div>
      <div class="metric-score">
        <strong>${card.score}</strong>
        <span>/${card.max_score}</span>
      </div>
    `;
    scoreCards.appendChild(div);
  });
}

function renderConversation(conversation) {
  if (!conversation || conversation.length === 0) {
    reportConversation.hidden = true;
    return;
  }
  convList.innerHTML = "";
  conversation.forEach((item, index) => {
    const isIntro = item.question_id === "SELF_INTRO";
    const round = document.createElement("div");
    round.className = "conv-round";

    const qTurn = document.createElement("div");
    qTurn.className = "conv-turn";
    qTurn.innerHTML = `<span class="conv-label">面试官</span><p>${item.question || "—"}</p>`;

    const aTurn = document.createElement("div");
    aTurn.className = "conv-turn answer";
    aTurn.innerHTML = `<span class="conv-label">我的回答</span><p>${item.answer || "—"}</p>`;

    round.appendChild(qTurn);
    round.appendChild(aTurn);

    if (item.interviewer_message) {
      const divider = document.createElement("div");
      divider.className = "conv-divider";
      const footer = document.createElement("div");
      footer.className = "conv-footer";
      const resp = document.createElement("div");
      resp.className = "conv-turn";
      resp.style.flex = "1";
      resp.innerHTML = `<span class="conv-label">面试官回应</span><p>${item.interviewer_message}</p>`;
      footer.appendChild(resp);
      if (item.scored && item.score != null) {
        const scoreEl = document.createElement("span");
        scoreEl.className = "conv-score tag";
        scoreEl.textContent = `${item.score} 分`;
        footer.appendChild(scoreEl);
      }
      round.appendChild(divider);
      round.appendChild(footer);
    }

    convList.appendChild(round);
  });
  reportConversation.hidden = false;
}

function renderReportView(viewModel) {
  overallScore.textContent = `${viewModel.summary.overall_score}/10`;
  roundCount.textContent = viewModel.summary.rounds;
  weakestMetric.textContent = viewModel.improvements[0]?.text || "继续练习";
  reportNarrative.textContent = viewModel.summary.text;
  renderScoreCards(viewModel.score_cards);
  renderList(
    liveHighlights,
    (viewModel.highlights || []).map((item) => item.text || item),
    "暂无亮点。"
  );
  renderList(
    liveImprovements,
    (viewModel.improvements || []).map((item) => item.text || item),
    "暂无改进建议。"
  );

  if (viewModel.expression?.enabled) {
    const expressionText = Object.entries(viewModel.expression.average_metrics || {})
      .map(([name, value]) => `${name}: ${value}/10`)
      .join("，");
    if (expressionText) {
      reportNarrative.textContent = `${viewModel.summary.text} 表达分析：${expressionText}。`;
    }
  }

  renderList(
    actionPlan,
    viewModel.action_plan.map((item) => `${item.step}. ${item.text}`),
    "暂无内容。"
  );
  renderList(
    resourceList,
    viewModel.recommended_resources.map((item) => `${item.title}: ${item.description}`),
    "暂无内容。"
  );
  renderConversation(viewModel.conversation || []);
}

function enableAnswering(enabled) {
  answerInput.disabled = !enabled;
  answerButton.disabled = !enabled;
  finishButton.disabled = !enabled;
  voiceButton.disabled = !enabled || !SpeechRecognition;
  voiceSubmitButton.disabled = !enabled || !state.finalTranscript.trim();
  voiceClearButton.disabled = !enabled || !currentVoiceText();
}

function resetRoundInput() {
  answerInput.value = "";
  state.finalTranscript = "";
  state.interimTranscript = "";
  renderVoiceTranscript();
}

async function generateReportAndShow() {
  if (!state.sessionId) return;
  stopSpeaking();
  setButtonBusy(finishButton, true, "正在生成总结");
  try {
    const payload = await requestJson("/report/view", {
      method: "POST",
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    renderReportView(payload.view_model);
    enableAnswering(false);
    showView("report");
  } catch (error) {
    alert(error.message);
  } finally {
    setButtonBusy(finishButton, false);
  }
}

async function renderAnswerResult(payload, candidateText = "") {
  appendChatMessage("candidate", candidateText);
  const decision = payload.interviewer_decision || { action: "continue", message: payload.follow_up };
  appendChatMessage("interviewer", decision.message || payload.follow_up);
  if (decision.action === "switch_question" && decision.next_question) {
    questionMeta.textContent = `${decision.next_question.topic} / ${decision.next_question.difficulty || "未知难度"}`;
    questionText.textContent = decision.next_question.question;
    guidanceText.textContent = "请作答。";
  } else {
    questionText.textContent = decision.message || payload.follow_up;
    guidanceText.textContent = decision.action === "finish" ? "面试已结束。" : "请继续作答。";
  }
  resetRoundInput();
  speakInterviewer(decision.message || payload.follow_up);
  if (decision.action === "finish" || decision.should_finish) {
    enableAnswering(false);
    window.setTimeout(() => {
      generateReportAndShow();
    }, 900);
  }
}

async function submitAnswerStream(answer) {
  appendChatMessage("candidate", answer);
  resetRoundInput();

  const response = await fetch(`${apiBase}/interview/answer/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, answer }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
    throw new Error(payload.error || `请求失败: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let preliminaryShown = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      if (!part.trim().startsWith("data: ")) continue;
      const data = part.trim().slice(6);
      if (data === "[DONE]") return;

      let event;
      try { event = JSON.parse(data); } catch { continue; }
      if (event.type === "error") throw new Error(event.error || "响应出错");

      const msg = (event.interviewer_decision?.message || event.follow_up || "").trim();
      const decision = event.interviewer_decision || { action: "continue", message: event.follow_up };

      if (event.type === "preliminary") {
        appendChatMessage("interviewer", msg);
        speakInterviewer(msg);
        preliminaryShown = true;
      } else if (event.type === "final") {
        if (preliminaryShown) {
          updateLastInterviewerMessage(msg);
        } else {
          appendChatMessage("interviewer", msg);
          speakInterviewer(msg);
        }
        if (decision.action === "switch_question" && decision.next_question) {
          questionMeta.textContent = `${decision.next_question.topic} / ${decision.next_question.difficulty || "未知难度"}`;
          questionText.textContent = decision.next_question.question;
          guidanceText.textContent = "请作答。";
        } else {
          questionText.textContent = msg;
          guidanceText.textContent = decision.action === "finish" ? "面试已结束。" : "请继续作答。";
        }
        renderRoundScore(event.evaluation);
        enableAnswering(!!state.sessionId);
        if (decision.action === "finish" || decision.should_finish) {
          enableAnswering(false);
          setTimeout(generateReportAndShow, 900);
        }
      }
    }
  }
}

function setupSpeechRecognition() {
  if (!SpeechRecognition) {
    setVoiceStatus("当前浏览器不支持");
    voiceTranscript.textContent = "不可用";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "zh-CN";
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    stopSpeaking();
    state.isRecording = true;
    state.recordingStartedAt = Date.now();
    voiceButton.textContent = "停止录音";
    setVoiceStatus("正在聆听", "recording");
  };

  recognition.onresult = (event) => {
    let interim = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript;
      if (event.results[index].isFinal) {
        state.finalTranscript += transcript;
      } else {
        interim += transcript;
      }
    }
    state.interimTranscript = interim;
    answerInput.value = currentVoiceText();
    renderVoiceTranscript();
  };

  recognition.onerror = (event) => {
    state.isRecording = false;
    voiceButton.textContent = "开始录音";
    const message = event.error === "not-allowed" ? "麦克风权限被拒绝" : `识别异常: ${event.error}`;
    setVoiceStatus(message);
  };

  recognition.onend = () => {
    state.isRecording = false;
    state.interimTranscript = "";
    voiceButton.textContent = "开始录音";
    if (state.finalTranscript.trim()) {
      setVoiceStatus("已生成转写", "success");
    } else if (state.sessionId) {
      setVoiceStatus("等待开始");
    }
    renderVoiceTranscript();
  };

  state.recognition = recognition;
}

function formatSessionDate(iso) {
  if (!iso) return "未知时间";
  try {
    const d = new Date(iso);
    return `${d.getMonth() + 1}月${d.getDate()}日 ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  } catch {
    return iso;
  }
}

function renderHistoryPage() {
  const { sessions, page, perPage } = historyState;
  const totalPages = Math.ceil(sessions.length / perPage);
  const pageItems = sessions.slice(page * perPage, page * perPage + perPage);

  historyList.innerHTML = "";
  pageItems.forEach((s) => {
    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
      <div class="history-meta">
        <strong>${s.role_label || s.role}</strong>
        <span>${formatSessionDate(s.created_at)}</span>
      </div>
      <div class="history-stats">
        <span>${s.rounds} 轮</span>
        <span class="history-score">${s.score != null ? s.score + " 分" : "未评分"}</span>
      </div>
      <button class="ghost-button" type="button" style="width:100%">查看报告</button>
    `;
    item.querySelector("button").addEventListener("click", async () => {
      await viewHistoryReport(s.session_id);
    });
    historyList.appendChild(item);
  });

  historyPrevBtn.disabled = page === 0;
  historyNextBtn.disabled = page >= totalPages - 1;
  historyPageLabel.textContent = totalPages > 1 ? `${page + 1} / ${totalPages}` : "";
}

function renderHistory(sessions) {
  if (!sessions || sessions.length === 0) {
    historyList.innerHTML = '<div class="history-empty">暂无历史记录，完成一次面试后即可在此查看。</div>';
    historyPrevBtn.disabled = true;
    historyNextBtn.disabled = true;
    historyPageLabel.textContent = "";
    historyPanel.hidden = false;
    return;
  }
  historyState.sessions = sessions;
  historyState.page = 0;
  historyPanel.hidden = false;
  renderHistoryPage();
}

async function viewHistoryReport(sessionId) {
  try {
    const payload = await requestJson("/report/view", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    });
    renderReportView(payload.view_model);
    showView("report");
  } catch (error) {
    alert(error.message);
  }
}

async function loadHistory() {
  try {
    const payload = await requestJson("/history", { method: "GET" });
    renderHistory(payload.sessions || []);
  } catch {
    historyPanel.hidden = true;
  }
}

async function loadHealth() {
  try {
    const payload = await requestJson("/health", { method: "GET" });
    setHealth(true, `服务在线 · ${payload.roles.length} 个岗位`);
  } catch (error) {
    setHealth(false, `服务未连接 · ${error.message}`);
  }
}

async function loadRoles() {
  try {
    const payload = await requestJson("/roles", { method: "GET" });
    roleSelect.innerHTML = "";
    payload.roles.forEach((role) => {
      const option = document.createElement("option");
      option.value = role.role;
      option.textContent = role.role_label;
      roleSelect.appendChild(option);
    });
  } catch (error) {
    roleSelect.innerHTML = '<option value="">服务连接后加载岗位</option>';
    roleSelect.disabled = true;
  }
}

startForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setButtonBusy(startForm.querySelector("button[type='submit']"), true, "正在准备面试");
  try {
    const payload = await requestJson("/interview/start", {
      method: "POST",
      body: JSON.stringify({
        role: roleSelect.value,
        topic: topicInput.value.trim(),
        difficulty: difficultySelect.value,
        question_query: queryInput.value.trim(),
        resume_text: resumeInput.value.trim(),
      }),
    });

    state.sessionId = payload.session_id;
    state.role = payload.role;
    resetChat();
    resetRoundInput();
    sessionIdTag.textContent = payload.session_id;
    questionMeta.textContent = `${payload.question.topic} / ${payload.question.difficulty || "未知难度"}`;
    questionText.textContent = payload.question.question;
    guidanceText.textContent = payload.fallback_reason || "请作答。";
    if (payload.opening_message) {
      appendChatMessage("interviewer", payload.opening_message);
    }
    appendChatMessage("interviewer", payload.question.question);
    renderRoundScore(null);
    setVoiceStatus(SpeechRecognition ? "等待开始" : "当前浏览器不支持");
    enableAnswering(true);
    showView("interview");
    speakInterviewer([payload.opening_message, payload.question.question].filter(Boolean).join(" "));
  } catch (error) {
    alert(error.message);
  } finally {
    setButtonBusy(startForm.querySelector("button[type='submit']"), false);
    roleSelect.disabled = false;
  }
});

answerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.sessionId) return;
  const answer = answerInput.value.trim();
  if (!answer) {
    alert("请先输入或录制回答。");
    return;
  }
  setButtonBusy(answerButton, true, "正在评估回答");
  enableAnswering(false);
  try {
    await submitAnswerStream(answer);
  } catch (error) {
    alert(error.message);
    enableAnswering(!!state.sessionId);
  } finally {
    setButtonBusy(answerButton, false);
  }
});

voiceButton.addEventListener("click", () => {
  if (!state.recognition || !state.sessionId) return;
  if (state.isRecording) {
    state.recognition.stop();
    return;
  }
  state.interimTranscript = "";
  try {
    state.recognition.start();
  } catch (error) {
    setVoiceStatus("录音启动失败");
  }
});

autoSpeakToggle.addEventListener("change", () => {
  if (!autoSpeakToggle.checked) {
    stopSpeaking();
  } else if (state.sessionId && questionText.textContent.trim()) {
    speakInterviewer(questionText.textContent);
  }
});

stopSpeakButton.addEventListener("click", () => {
  stopSpeaking();
});

voiceClearButton.addEventListener("click", () => {
  resetRoundInput();
  setVoiceStatus("等待开始");
  enableAnswering(!!state.sessionId);
});

voiceSubmitButton.addEventListener("click", async () => {
  if (!state.sessionId) return;
  if (state.isRecording && state.recognition) {
    state.recognition.stop();
  }
  const transcript = state.finalTranscript.trim();
  if (!transcript) {
    alert("还没有可提交的语音转写内容。");
    return;
  }
  setButtonBusy(voiceSubmitButton, true, "正在提交语音");
  try {
    const durationMs = state.recordingStartedAt ? Date.now() - state.recordingStartedAt : 0;
    const payload = await requestJson("/interview/answer-audio", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        text_hint: transcript,
        duration_ms: durationMs,
        language: "zh-CN",
        source: "browser_web_speech",
      }),
    });
    setVoiceStatus("语音回答已提交", "success");
    await renderAnswerResult(payload, transcript);
  } catch (error) {
    alert(error.message);
  } finally {
    setButtonBusy(voiceSubmitButton, false);
    enableAnswering(!!state.sessionId);
  }
});

finishButton.addEventListener("click", async () => {
  if (!state.sessionId) return;
  await generateReportAndShow();
});

restartButton.addEventListener("click", () => {
  stopSpeaking();
  state.sessionId = "";
  state.role = "";
  resetChat();
  resetRoundInput();
  enableAnswering(false);
  sessionIdTag.textContent = "无 Session";
  questionMeta.textContent = "未开始";
  questionText.textContent = "开始面试后，这里会出现当前问题。";
  guidanceText.textContent = "请作答。";
  renderRoundScore(null);
  showView("setup");
  loadHistory();
});

window.addEventListener("hashchange", () => {
  const target = window.location.hash.replace("#", "");
  if (!["setup", "interview", "report"].includes(target)) return;
  if (target !== "setup" && !state.sessionId) {
    showView("setup", { replace: true, instant: true });
    return;
  }
  showView(target, { replace: true, instant: true });
});

historyPrevBtn.addEventListener("click", () => {
  if (historyState.page > 0) {
    historyState.page--;
    renderHistoryPage();
  }
});

historyNextBtn.addEventListener("click", () => {
  const totalPages = Math.ceil(historyState.sessions.length / historyState.perPage);
  if (historyState.page < totalPages - 1) {
    historyState.page++;
    renderHistoryPage();
  }
});

async function bootstrap() {
  renderScoreCards([]);
  renderChat();
  setupSpeechRecognition();
  if (speechSynthesisApi) {
    state.ttsVoice = pickChineseVoice();
    speechSynthesisApi.onvoiceschanged = () => {
      state.ttsVoice = pickChineseVoice();
    };
  } else {
    autoSpeakToggle.checked = false;
    autoSpeakToggle.disabled = true;
    stopSpeakButton.disabled = true;
  }
  renderVoiceTranscript();
  renderRoundScore(null);
  enableAnswering(false);
  await loadHealth();
  await loadRoles();
  await loadHistory();
  const initialView = window.location.hash.replace("#", "");
  showView(["setup", "interview", "report"].includes(initialView) ? initialView : "setup", {
    replace: true,
    instant: true,
  });
}

bootstrap();
