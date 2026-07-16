const PROFILE_URL = "/assets/data/daniel-profile.json";
const ASSET_VERSION = "3";

const els = {
  runtimePill: document.querySelector("#runtime-pill"),
  runtimeLabel: document.querySelector("#runtime-label"),
  localTime: document.querySelector("#local-time"),
  portrait: document.querySelector("#portrait"),
  portraitState: document.querySelector("#portrait-state"),
  canvas: document.querySelector("#voice-wave"),
  loader: document.querySelector("#model-loader"),
  modelStatus: document.querySelector("#model-status"),
  modelDetail: document.querySelector("#model-detail"),
  modelProgress: document.querySelector("#model-progress"),
  progressTrack: document.querySelector(".progress-track"),
  loadButton: document.querySelector("#load-model"),
  chatLog: document.querySelector("#chat-log"),
  form: document.querySelector("#prompt-form"),
  input: document.querySelector("#prompt-input"),
  sendButton: document.querySelector("#send-button"),
  micButton: document.querySelector("#mic-button"),
  clearButton: document.querySelector("#clear-chat"),
  voiceOutput: document.querySelector("#voice-output"),
  latency: document.querySelector("#latency-label"),
};

const state = {
  worker: null,
  profile: null,
  systemPrompt: "",
  modelReady: false,
  modelLoading: false,
  generating: false,
  pendingPrompt: null,
  assistantNode: null,
  streamedText: "",
  conversation: [],
  recognition: null,
  speaking: false,
  backend: "webgpu",
};

function initializeIcons() {
  if (window.lucide) window.lucide.createIcons();
  else window.setTimeout(initializeIcons, 100);
}

function setPortraitState(next, label) {
  els.portrait.dataset.state = next;
  els.portraitState.textContent = label || next.toUpperCase();
}

function setRuntime(label, ready = false) {
  els.runtimeLabel.textContent = label;
  els.runtimePill.classList.toggle("is-ready", ready);
}

function updateClock() {
  const value = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Seoul",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date());
  els.localTime.textContent = `${value} KST`;
}

function selectProfileContext(profile, prompt = "") {
  const query = prompt.toLowerCase();
  const context = { identity: profile.identity, links: profile.links };

  if (/open.?source|hugging|sam2|molmo|transformers|오픈.?소스/.test(query)) {
    context.open_source = profile.open_source;
  } else if (/toss|bank|document|authentication|agent|토스|은행|문서|인증|에이전트/.test(query)) {
    context.current_work = profile.current_work;
  } else if (/superb|multimodal|ground|training|gpu|dataset|cvpr|멀티모달|학습|데이터/.test(query)) {
    context.previous_work = profile.previous_work;
  } else if (/paper|publication|research|mobilehuman|zero|논문|연구/.test(query)) {
    context.research = profile.research;
    context.previous_work = profile.previous_work;
  } else if (/education|school|kaist|postech|uiuc|학력|학교/.test(query)) {
    context.education = profile.education;
  } else {
    context.current_work = profile.current_work;
    context.previous_work = { company: profile.previous_work.company, title: profile.previous_work.title };
    context.open_source = { contributions: profile.open_source.contributions.slice(0, 2) };
    context.research = profile.research;
  }
  return context;
}

function buildSystemPrompt(profile, prompt = "") {
  const focusedContext = selectProfileContext(profile, prompt);
  return [
    "You are Daniel OS, the personal AI portfolio of Sangbum Daniel Choi.",
    "Answer in the same language as the visitor, in at most 100 words.",
    "Use only the verified facts below. Never infer industries, adoption, impact, definitions, or acronym expansions.",
    "If a requested fact is missing, say that the portfolio does not contain that information.",
    "When useful, provide a relevant profile link using Markdown.",
    "Do not pretend to be the real Daniel. Say you are his browser-native portfolio assistant.",
    "Verified facts for this question:",
    JSON.stringify(focusedContext),
  ].join("\n");
}

async function loadProfile() {
  const response = await fetch(PROFILE_URL);
  if (!response.ok) throw new Error("Could not load profile context.");
  state.profile = await response.json();
  state.systemPrompt = buildSystemPrompt(state.profile);
}

function createMessage(role, text = "") {
  const article = document.createElement("article");
  article.className = `message message--${role}`;
  const now = new Intl.DateTimeFormat("en", { hour: "2-digit", minute: "2-digit" }).format(new Date());
  article.innerHTML = `
    <div class="message__meta"><span>${role === "assistant" ? "DANIEL AI" : "VISITOR"}</span><time>${now}</time></div>
    <div class="message__body"></div>
  `;
  const body = article.querySelector(".message__body");
  renderMessage(body, text);
  els.chatLog.appendChild(article);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
  return article;
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function renderMessage(node, text) {
  const safe = escapeHtml(text || "")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
  node.innerHTML = safe;
}

function routeCommand(prompt) {
  const command = prompt.trim().toLowerCase();
  const routes = {
    "/resume": "/files/resume/daniel_choi_resume_clean.pdf",
    "/cv": "/cv/",
    "/papers": "/publications/",
    "/github": "https://github.com/SangbumChoi",
    "/linkedin": "https://www.linkedin.com/in/daniel-choi-86648216b/",
  };
  if (!routes[command]) return false;
  window.open(routes[command], command === "/cv" || command === "/papers" ? "_self" : "_blank", "noopener");
  return true;
}

function groundedAnswer(prompt) {
  if (!state.profile) return null;
  const query = prompt.toLowerCase();
  const korean = /[가-힣]/.test(prompt);
  const links = state.profile.links;

  if (/who (is|are) daniel|who (is|are) sangbum|about (daniel|sangbum)|introduce (daniel|yourself)|다니엘.*누구|상범.*누구|자기.?소개|소개해/.test(query)) {
    return korean
      ? "Sangbum Daniel Choi는 서울에서 일하는 AI research and systems engineer입니다. 6년 이상 멀티모달 모델 학습, 데이터 설계, 평가, 오픈소스 통합, 온디바이스 배포와 프로덕션 ML 인프라를 경험했으며 현재 Toss Bank의 Data Scientist입니다."
      : "Sangbum Daniel Choi is an AI research and systems engineer in Seoul with 6+ years of experience in multimodal training, data design, evaluation, open-source integration, edge deployment, and production ML infrastructure. He currently works as a Data Scientist at Toss Bank.";
  }

  if (/link|링크|resume|cv|github|linkedin|paper|publication|논문/.test(query)) {
    return korean
      ? `검증된 링크입니다: [이력서](${links.resume}), [온라인 CV](${links.cv}), [논문](${links.publications}), [GitHub](${links.github}), [LinkedIn](${links.linkedin}). /resume, /papers, /github 명령도 사용할 수 있습니다.`
      : `Verified links: [resume](${links.resume}), [online CV](${links.cv}), [publications](${links.publications}), [GitHub](${links.github}), and [LinkedIn](${links.linkedin}). You can also type /resume, /papers, or /github.`;
  }

  if (/why.*open|open.?source.*why|philosophy|오픈.?소스.*이유|철학/.test(query)) {
    return korean
      ? "Daniel은 오픈소스가 기술의 공통 기반을 넓힌다고 믿습니다. 모델을 누구나 검토하고 재현하고 사용할 수 있게 하면, 다른 엔지니어가 그 위에서 배우고 개선하며 최초 기여자가 예상하지 못한 결과를 만들 수 있기 때문입니다."
      : state.profile.open_source.philosophy;
  }

  if (/open.?source|hugging|sam2|molmo|transformers|오픈.?소스/.test(query)) {
    return korean
      ? "Daniel은 Hugging Face Transformers에 40개 이상의 PR을 기여했습니다. SAM2 통합을 주도했고, Molmo2 지원을 열어 Molmo2-4B 체크포인트를 공개했으며, RT-DETR·ViTPose·DETA 학습·DINOv3 유틸리티·분산 학습 수정·테스트·문서화에도 기여했습니다."
      : "Daniel has contributed 40+ pull requests to Hugging Face Transformers. He led the SAM2 integration, opened Molmo2 support and published a Molmo2-4B checkpoint, and contributed to RT-DETR, ViTPose, DETA training, DINOv3 utilities, distributed training fixes, tests, and documentation.";
  }

  if (/toss|bank|document|authentication|agent|토스|은행|문서|인증|에이전트/.test(query)) {
    return korean
      ? "Toss Bank에서 Daniel은 내부 LLM을 사용하는 온프레미스 AI 에이전트, 얼굴·신분증 인증, 그리고 약 10억 파라미터 VLM 기반의 end-to-end 문서 추출 파이프라인을 개발합니다. 분류·회전·레이아웃·OCR/표·키값 추출 단계별 평가를 설계했고, 자동 처리 가능한 결과 기준 exact match 61%를 달성했습니다."
      : "At Toss Bank, Daniel works on an on-premise AI agent using internally deployed LLMs, face and ID-card authentication, and an end-to-end document extraction pipeline with an approximately 1B-parameter VLM. He designed stage-level evaluation and reached a 61% exact-match baseline for automation-ready outputs.";
  }

  if (/superb|multimodal|ground|training|gpu|dataset|cvpr|멀티모달|학습|데이터/.test(query)) {
    return korean
      ? "SuperbAI에서 Daniel은 110만 장의 caption·box·mask 데이터로 visual-grounding 모델의 멀티모달 사전학습과 text-image alignment를 주도했습니다. CVPR 2025 IOD 2위와 FSOD 4위를 기록했고, AWS Batch 멀티 GPU 학습 및 TensorRT/Triton 서빙으로 PyTorch 대비 처리량을 5배 높였습니다."
      : "At SuperbAI, Daniel led multimodal pre-training and text-image alignment for a visual-grounding model using 1.1 million images with captions, boxes, and masks. The work placed 2nd in a CVPR 2025 IOD challenge and 4th in FSOD. He also built AWS Batch multi-GPU training and TensorRT/Triton serving with 5x the throughput of pure PyTorch.";
  }

  if (/paper|publication|research|mobilehuman|\bzero\b|논문|연구/.test(query)) {
    return korean
      ? `대표 연구는 2025년 visual grounding 논문 [ZERO](${state.profile.research[0].url})와 CVPR Workshop 2021의 모바일 3D 자세 추정 논문 [MobileHumanPose](${state.profile.research[1].url})입니다. 전체 목록은 [publications](${links.publications})에서 볼 수 있습니다.`
      : `Daniel's representative research includes [ZERO](${state.profile.research[0].url}), a 2025 visual-grounding paper, and [MobileHumanPose](${state.profile.research[1].url}), published at a CVPR Workshop in 2021. See the complete [publication list](${links.publications}).`;
  }

  if (/education|school|kaist|postech|uiuc|학력|학교/.test(query)) {
    return korean
      ? "Daniel은 KAIST 전기전자공학 전문석사, POSTECH 전자전기공학 학사 학위를 받았고 UIUC에서 교환학생으로 공부했습니다."
      : "Daniel earned a professional master's degree in Electrical Engineering from KAIST and a bachelor's degree in Electrical Engineering from POSTECH. He also studied at UIUC as an exchange student.";
  }
  return null;
}

function deliverGroundedAnswer(answer) {
  const node = createMessage("assistant", answer);
  node.dataset.source = "profile-index";
  state.conversation.push({ role: "assistant", content: answer });
  els.latency.textContent = "PROFILE INDEX · grounded · local";
  setPortraitState("idle", state.modelReady ? "LOCAL MODEL READY" : "PROFILE INDEX READY");
  if (els.voiceOutput.checked) speak(answer);
}

function updateProgress(payload = {}) {
  const progress = Number(payload.progress);
  if (Number.isFinite(progress)) {
    const pct = Math.max(0, Math.min(100, progress));
    els.modelProgress.style.width = `${pct}%`;
    els.progressTrack.setAttribute("aria-valuenow", String(Math.round(pct)));
  }
  if (payload.file) els.modelDetail.textContent = `Downloading ${payload.file}`;
  if (payload.status === "ready") els.modelDetail.textContent = "Compiling local inference session";
}

async function initWorker() {
  if (state.worker || state.modelLoading) return;
  state.modelLoading = true;
  state.backend = "gpu" in navigator ? "webgpu" : "wasm";
  try {
    await navigator.storage?.persist?.();
  } catch (_) {
    // Persistent storage is an optimization; inference still works without it.
  }
  els.loadButton.disabled = true;
  els.modelStatus.textContent = "Loading personalized LFM2";
  els.modelDetail.textContent = state.backend === "webgpu" ? "q4 · WebGPU · ~294 MB · cached" : "q4 · WASM fallback · ~294 MB · cached";
  setRuntime(`${state.backend} / loading`);

  state.worker = new Worker(`/assets/js/lfm-worker.js?v=${ASSET_VERSION}`, { type: "module" });
  state.worker.addEventListener("message", handleWorkerMessage);
  state.worker.addEventListener("error", (event) => handleModelError(event.message));
  state.worker.postMessage({ type: "load", device: state.backend });
}

function handleWorkerMessage(event) {
  const data = event.data || {};
  if (data.type === "progress") updateProgress(data.payload);
  if (data.type === "fallback") {
    state.backend = "wasm";
    els.modelDetail.textContent = data.message;
  }
  if (data.type === "ready") {
    state.modelReady = true;
    state.modelLoading = false;
    state.backend = data.runtime;
    els.modelProgress.style.width = "100%";
    els.progressTrack.setAttribute("aria-valuenow", "100");
    els.modelStatus.textContent = "Personalized LFM2 ready";
    els.modelDetail.textContent = `q4 · ${data.runtime.toUpperCase()} · runs locally`;
    els.loadButton.innerHTML = '<i data-lucide="check" aria-hidden="true"></i><span>Ready</span>';
    setRuntime(`${data.runtime} / private`, true);
    setPortraitState("idle", "LOCAL MODEL READY");
    initializeIcons();
    if (state.pendingPrompt) {
      const queued = state.pendingPrompt;
      state.pendingPrompt = null;
      generateAnswer(queued);
    }
  }
  if (data.type === "token") {
    state.streamedText += data.text || "";
    if (state.assistantNode) renderMessage(state.assistantNode.querySelector(".message__body"), state.streamedText);
    els.chatLog.scrollTop = els.chatLog.scrollHeight;
  }
  if (data.type === "complete") completeGeneration(data);
  if (data.type === "error") handleModelError(data.message);
}

function handleModelError(message) {
  state.generating = false;
  state.modelLoading = false;
  els.sendButton.disabled = false;
  els.loadButton.disabled = false;
  els.modelStatus.textContent = "Local model unavailable";
  els.modelDetail.textContent = message || "Use a recent Chromium browser with WebGPU.";
  setRuntime("runtime error");
  setPortraitState("idle", "FALLBACK MODE");
  if (state.assistantNode) {
    state.assistantNode.classList.remove("is-streaming");
    renderMessage(state.assistantNode.querySelector(".message__body"), "The local model could not start in this browser. Profile links and deterministic answers remain available.");
  }
}

function generateAnswer(prompt) {
  if (!state.modelReady) {
    deliverGroundedAnswer(mockSynthesisAnswer(prompt));
    return;
  }

  state.generating = true;
  els.chatLog.querySelectorAll('[data-loading-notice="true"]').forEach((node) => node.remove());
  state.streamedText = "";
  els.sendButton.disabled = true;
  setPortraitState("thinking", "THINKING LOCALLY");
  state.assistantNode = createMessage("assistant", "");
  state.assistantNode.classList.add("is-streaming");

  const recent = state.conversation.slice(-8);
  state.worker.postMessage({
    type: "generate",
    device: state.backend,
    messages: [
      { role: "system", content: buildSystemPrompt(state.profile, prompt) },
      ...recent,
    ],
  });
}

function mockSynthesisAnswer(prompt) {
  const korean = /[가-힣]/.test(prompt);
  return korean
    ? "현재는 Daniel의 검증된 프로필 정보를 기반으로 즉시 답하는 가벼운 데모 모드입니다. 이 질문에 대한 구체적인 정보는 프로필에 아직 없습니다. 실제 LFM2 로컬 추론은 위의 'Enable local AI' 버튼을 눌러 선택적으로 실행할 수 있습니다."
    : "This is the lightweight demo mode, answering instantly from Daniel's verified profile. The portfolio does not yet contain a specific answer to that question. You can optionally run local LFM2 inference by selecting Enable local AI above.";
}

function completeGeneration(data) {
  const answer = guardModelIdentity(data.text || state.streamedText || "I could not generate a response.");
  state.generating = false;
  els.sendButton.disabled = false;
  if (state.assistantNode) {
    state.assistantNode.classList.remove("is-streaming");
    renderMessage(state.assistantNode.querySelector(".message__body"), answer);
  }
  state.conversation.push({ role: "assistant", content: answer });
  els.latency.textContent = `${data.runtime.toUpperCase()} · ${(data.elapsed / 1000).toFixed(1)}s · local`;
  setPortraitState("idle", "LOCAL MODEL READY");
  if (els.voiceOutput.checked) speak(answer);
}

function guardModelIdentity(text) {
  return text.trim()
    .replace(/^As (?:Sangbum )?Daniel Choi,?\s*/i, "As Daniel's browser-native portfolio assistant, ")
    .replace(/^I am (?:Sangbum )?Daniel Choi[,.]?\s*/i, "I am Daniel's browser-native portfolio assistant. ");
}

function submitPrompt(rawPrompt) {
  const prompt = rawPrompt.trim();
  if (!prompt || state.generating) return;
  if (routeCommand(prompt)) return;
  window.speechSynthesis?.cancel();
  createMessage("user", prompt);
  state.conversation.push({ role: "user", content: prompt });
  els.input.value = "";
  resizeComposer();
  const grounded = groundedAnswer(prompt);
  if (grounded) {
    deliverGroundedAnswer(grounded);
    return;
  }
  generateAnswer(prompt);
}

function speak(text) {
  if (!("speechSynthesis" in window) || !text) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1"));
  const korean = /[가-힣]/.test(text);
  utterance.lang = korean ? "ko-KR" : "en-US";
  utterance.rate = 1.02;
  utterance.pitch = 0.96;
  const voices = window.speechSynthesis.getVoices();
  utterance.voice = voices.find((voice) => voice.lang === utterance.lang) || null;
  utterance.onstart = () => {
    state.speaking = true;
    setPortraitState("speaking", "SPEAKING");
  };
  utterance.onend = utterance.onerror = () => {
    state.speaking = false;
    setPortraitState("idle", state.modelReady ? "LOCAL MODEL READY" : "STANDING BY");
  };
  window.speechSynthesis.speak(utterance);
}

function initSpeechRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    els.micButton.disabled = true;
    els.micButton.title = "Speech recognition is not available in this browser";
    return;
  }
  state.recognition = new Recognition();
  state.recognition.continuous = false;
  state.recognition.interimResults = true;
  state.recognition.lang = navigator.language?.startsWith("ko") ? "ko-KR" : "en-US";
  state.recognition.onstart = () => {
    els.micButton.classList.add("is-listening");
    setPortraitState("listening", "LISTENING");
  };
  state.recognition.onresult = (event) => {
    const transcript = Array.from(event.results).map((result) => result[0].transcript).join("");
    els.input.value = transcript;
    resizeComposer();
    if (event.results[event.results.length - 1].isFinal) submitPrompt(transcript);
  };
  state.recognition.onend = () => {
    els.micButton.classList.remove("is-listening");
    if (!state.generating && !state.speaking) setPortraitState("idle", state.modelReady ? "LOCAL MODEL READY" : "STANDING BY");
  };
  state.recognition.onerror = () => state.recognition.onend();
}

function resizeComposer() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 140)}px`;
}

function initWaveform() {
  const ctx = els.canvas.getContext("2d");
  let phase = 0;
  function draw() {
    const rect = els.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (els.canvas.width !== width || els.canvas.height !== height) {
      els.canvas.width = width;
      els.canvas.height = height;
    }
    ctx.clearRect(0, 0, width, height);
    const active = els.portrait.dataset.state;
    const amplitude = active === "speaking" ? 0.78 : active === "listening" ? 0.58 : active === "thinking" ? 0.28 : 0.12;
    const bars = Math.max(28, Math.floor(rect.width / 9));
    const gap = width / bars;
    ctx.fillStyle = active === "listening" ? "#ff6b57" : "#b7f24a";
    for (let i = 0; i < bars; i += 1) {
      const envelope = 0.25 + Math.sin((i / bars) * Math.PI) * 0.75;
      const wave = Math.abs(Math.sin(i * 0.52 + phase) * Math.cos(i * 0.17 - phase * 0.7));
      const barHeight = Math.max(2 * dpr, height * amplitude * envelope * (0.22 + wave * 0.78));
      ctx.globalAlpha = 0.42 + wave * 0.58;
      ctx.fillRect(i * gap, (height - barHeight) / 2, Math.max(1, gap * 0.28), barHeight);
    }
    ctx.globalAlpha = 1;
    phase += active === "speaking" ? 0.17 : 0.055;
    requestAnimationFrame(draw);
  }
  draw();
}

function bindEvents() {
  els.loadButton.addEventListener("click", initWorker);
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrompt(els.input.value);
  });
  els.input.addEventListener("input", resizeComposer);
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitPrompt(els.input.value);
    }
  });
  els.micButton.addEventListener("click", () => {
    if (!state.recognition) return;
    try { state.recognition.start(); } catch (_) { state.recognition.stop(); }
  });
  els.clearButton.addEventListener("click", () => {
    state.conversation = [];
    window.speechSynthesis?.cancel();
    els.chatLog.querySelectorAll(".message:not(:first-child)").forEach((node) => node.remove());
    setPortraitState("idle", state.modelReady ? "LOCAL MODEL READY" : "STANDING BY");
  });
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => submitPrompt(button.dataset.prompt));
  });
}

async function boot() {
  initializeIcons();
  updateClock();
  window.setInterval(updateClock, 30_000);
  initWaveform();
  initSpeechRecognition();
  bindEvents();

  state.backend = "gpu" in navigator ? "webgpu" : "wasm";
  setRuntime("demo / private", true);
  els.modelStatus.textContent = "Profile demo ready";
  els.modelDetail.textContent = state.backend === "webgpu"
    ? "instant grounded answers · personalized WebGPU model on demand"
    : "instant grounded answers · personalized WASM model on demand";

  try {
    await loadProfile();
  } catch (error) {
    els.modelDetail.textContent = error.message;
  }
}

boot();
