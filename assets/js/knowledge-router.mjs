const PROFILE_NAME_PATTERN = /\b(?:daniel|sangbum|sangbum daniel choi|choi)\b|최상범|상범|다니엘/i;
const PROFILE_PRONOUN_PATTERN = /\b(?:he|him|his|you|your)\b|그가|그의|본인|당신|너의/i;
const PROFILE_SCOPE_PATTERN = /\b(?:portfolio|resume|cv|career|experience|publications?|education|startup history|current (?:role|employer)|(?:open.?source|hugging face) contributions?)\b|포트폴리오|이력서|경력|논문|학력|오픈.?소스 기여/i;
const DEFINITION_PATTERN = /^\s*(?:what|who|where)\s+(?:is|are|was|were)\b|^\s*(?:define|explain)\b|^\s*(?:what does|where does)\b|무엇|뭐야|뭔가요|어디|설명(?:해|해줘|해주세요)/i;
const PRIVATE_PATTERN = /\b(?:bank account|account number|height|how tall|tallness|stature|weight|how old|current age|exact age|relationship status|girlfriend|boyfriend|spouse|wife|husband|family|home address|phone number|salary|exact birthday|social security|passport)\b|계좌|키|신장|몸무게|정확한 나이|현재 나이|몇 살|연애|여자친구|남자친구|배우자|가족|집 주소|전화번호|연봉|생일|주민등록|여권/i;
const VISITOR_IDENTITY_PATTERN = /^\s*(?:who am i|do you know who i am)\b|나는 누구|내가 누구/i;
const GENERAL_LOOKUP_PATTERN = /^\s*(?!(?:what|who)\s+should\b)(?:what|who|where|when|how|why)\b|^\s*(?:(?:can|could|would) you )?(?:define|explain|tell me about|teach me about)\b|^\s*(?:give|provide) me (?:an? )?(?:primer|overview|introduction|definition|explanation) (?:of|on|to)\b|^\s*i (?:want|need|would like) to (?:know|learn) about\b|무엇|뭐야|뭔가요|어디|누가|언제|어떻게|왜|설명(?:해|해줘|해주세요)/i;
const AMBIGUOUS_WORK_FOLLOWUP_PATTERN = /^\s*(?:how|what)\s+about(?:\s+(?:at|in|with))?\s+(?:(?:his|your|the)\s+)?(?:work|job|company|role)\s*[?.!]*$/i;
const PROFILE_WORK_PATTERN = /\b(?:(?:his|your|daniel(?:'s)?)\s+(?:work|job|company|role|employer)|work experience|company experience|current job)\b|\b(?:at|in|for)\s+(?:his|your|the)\s+compan(?:y|ies)\b|(?:그의|다니엘의|최상범의)\s*(?:회사|직장|업무|일)|회사에서\s*(?:무엇|뭘|어떤 일)|직장\s*경력/i;

function normalize(value = "") {
  return value.toLowerCase().replace(/[–—_]/g, "-").replace(/\s+/g, " ").trim();
}

export function detectAnswerLanguage(prompt = "") {
  return /[가-힣]/.test(prompt) ? "ko" : "en";
}

export function findKnownEntity(prompt, knowledge) {
  const query = normalize(prompt);
  const entities = knowledge?.entities || [];
  const candidates = entities.flatMap((entity) =>
    [entity.name, ...(entity.aliases || [])].map((alias) => ({
      entity,
      alias: normalize(alias),
    })),
  ).sort((left, right) => right.alias.length - left.alias.length);

  return candidates.find(({ alias }) => {
    if (!alias) return false;
    const escaped = alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, "i").test(query);
  })?.entity || null;
}

function asksAboutDaniel(prompt) {
  const subject = prompt.replace(
    /^\s*(?:can|could|would) you (?:define|explain|tell me about|teach me about)\s+/i,
    "",
  );
  return PROFILE_NAME_PATTERN.test(subject)
    || PROFILE_PRONOUN_PATTERN.test(subject)
    || PROFILE_SCOPE_PATTERN.test(subject);
}

function usesEntityPronoun(prompt) {
  return /\b(?:it|that|this model|this method|this university|this project)\b|그것|그 모델|그 방법|그 학교|그 프로젝트/i.test(prompt);
}

export function classifyKnowledgeIntent(prompt, knowledge, options = {}) {
  const entity = findKnownEntity(prompt, knowledge)
    || (usesEntityPronoun(prompt)
      ? (knowledge?.entities || []).find((item) => item.id === options.lastEntityId) || null
      : null);
  const profileRelated = asksAboutDaniel(prompt);

  if (VISITOR_IDENTITY_PATTERN.test(prompt)) return { type: "profile", entity: null };
  if (AMBIGUOUS_WORK_FOLLOWUP_PATTERN.test(prompt)) {
    return { type: "profile_clarification", entity: null };
  }
  if (PRIVATE_PATTERN.test(prompt) && (profileRelated || PROFILE_PRONOUN_PATTERN.test(prompt))) {
    return { type: "sensitive_personal", entity: null };
  }
  if (entity && profileRelated) return { type: "profile_entity", entity };
  if (entity && (DEFINITION_PATTERN.test(prompt) || !profileRelated)) {
    return { type: "entity_definition", entity };
  }
  if (PROFILE_WORK_PATTERN.test(prompt)) return { type: "profile_work", entity: null };
  if (profileRelated) return { type: "profile", entity: null };
  if (GENERAL_LOOKUP_PATTERN.test(prompt)) return { type: "external_knowledge", entity: null };
  return { type: "profile", entity: null };
}

function sourceMarkdown(entity, relation = false) {
  const sources = relation
    ? [
      { label: "Daniel's verified profile", url: "/assets/data/daniel-profile.json" },
      ...(entity.sources || []),
    ]
    : (entity.sources || []);
  return sources
    .slice(0, 2)
    .map((source) => `[${source.label}](${source.url})`)
    .join(" · ");
}

export function buildEntityAnswer(entity, language = "en", relation = false) {
  const field = relation ? `portfolio_relation_${language}` : `definition_${language}`;
  const fallback = relation ? entity.portfolio_relation_en : entity.definition_en;
  const answer = entity[field] || fallback;
  const sources = sourceMarkdown(entity, relation);
  return sources ? `${answer}\n\nSource: ${sources}` : answer;
}

export function externalSearchTerm(prompt) {
  const candidate = prompt
    .trim()
    .replace(/^\s*how\s+(?:does|do|did|is|are)\s+(.+?)\s+work[?.!]*$/i, "$1")
    .replace(/^\s*who\s+(?:wrote|created|invented|founded|painted|developed|discovered)\s+/i, "")
    .replace(/^\s*(?:what|who|where)\s+(?:is|are|was|were)\s+/i, "")
    .replace(/^\s*(?:(?:can|could|would) you )?(?:define|explain|tell me about|teach me about)\s+/i, "")
    .replace(/^\s*(?:give|provide) me (?:an? )?(?:primer|overview|introduction|definition|explanation) (?:of|on|to)\s+/i, "")
    .replace(/^\s*i (?:want|need|would like) to (?:know|learn) about\s+/i, "")
    .replace(/^\s*(?:무엇이|무엇은|뭐야|뭔가요|어디에|어디야|설명해줘|설명해주세요)\s*/i, "")
    .replace(/^\s*(?:a|an|the)\s+/i, "")
    .replace(/[?.!]+$/g, "")
    .trim()
    .slice(0, 160);
  const normalizedCandidate = normalize(candidate);
  if (
    !normalizedCandidate
    || /^(?:(?:how|what) about )?(?:in |at )?(?:the |his |your )?(?:work|job|company|role|it|this|that|there)$/.test(normalizedCandidate)
  ) {
    return "";
  }
  return candidate;
}

export function profileWorkClarificationResponse(language = "en") {
  return language === "ko"
    ? "어느 경력을 뜻하는지 알려주세요: 현재 Toss Bank 업무, 이전 SuperbAI 업무, 또는 Team ISLAND 창업 경험 중에서 답할 수 있습니다."
    : "Which work do you mean: Daniel's current role at Toss Bank, his previous role at SuperbAI, or his startup work at Team ISLAND?";
}

export function buildProfileWorkAnswer(profile, language = "en") {
  const currentCompany = profile?.current_work?.company || "Toss Bank";
  const previousCompany = profile?.previous_work?.company || "SuperbAI";
  const startup = profile?.career_timeline?.startup?.company || "Team ISLAND";
  const product = profile?.products?.zzazz?.name || "ZZAZZ";
  return language === "ko"
    ? `Daniel의 회사 경력은 세 축입니다. 현재 ${currentCompany}에서는 온프레미스 LLM 에이전트, 얼굴·신분증 인증, end-to-end VLM 문서 추출을 개발합니다. ${previousCompany}에서는 멀티모달 사전학습, 데이터 큐레이션, GPU 학습·서빙을 주도했습니다. ${startup}의 공동 창업자 겸 CTO로서는 모바일 영상 편집 앱 ${product}와 온디바이스 비전 기술을 개발했습니다.`
    : `Daniel's company work spans three verified roles. At ${currentCompany}, he develops an on-premise LLM agent, face and ID-card authentication, and end-to-end VLM document extraction. At ${previousCompany}, he led multimodal pre-training, data curation, and GPU training and serving. As ${startup}'s co-founder and CTO, he built the mobile video editor ${product} and its on-device vision stack.`;
}

function conciseExtract(extract, maxLength = 560) {
  const sentences = extract.match(/[^.!?]+[.!?]+/g) || [extract];
  let result = "";
  for (const sentence of sentences.slice(0, 3)) {
    if (`${result} ${sentence}`.trim().length > maxLength) break;
    result = `${result} ${sentence}`.trim();
  }
  return result || `${extract.slice(0, maxLength).trim()}…`;
}

export async function fetchWikipediaEvidence(searchTerm, language = "en", fetchImpl = fetch) {
  if (!searchTerm || searchTerm.length < 2) return null;
  const locale = language === "ko" ? "ko" : "en";
  const common = {
    action: "query",
    prop: "extracts|info",
    exintro: "1",
    explaintext: "1",
    inprop: "url",
    redirects: "1",
    origin: "*",
    format: "json",
  };
  const request = async (params) => {
    const response = await fetchImpl(
      `https://${locale}.wikipedia.org/w/api.php?${new URLSearchParams(params)}`,
    );
    if (!response.ok) return null;
    const payload = await response.json();
    return Object.values(payload?.query?.pages || {}).find((page) => !page.missing) || null;
  };
  let page = await request({ ...common, titles: searchTerm });
  if (!page?.extract) {
    page = await request({
      ...common,
      generator: "search",
      gsrsearch: searchTerm,
      gsrlimit: "1",
    });
  }
  if (!page?.extract || !page?.fullurl) return null;
  return {
    title: page.title,
    extract: conciseExtract(page.extract),
    url: page.fullurl,
  };
}

export function buildExternalEvidenceAnswer(evidence, language = "en") {
  const prefix = language === "ko" ? `${evidence.title}: ` : "";
  const label = language === "ko" ? "Wikipedia 출처" : "Wikipedia source";
  const safeUrl = evidence.url.replace(/\(/g, "%28").replace(/\)/g, "%29");
  return `${prefix}${evidence.extract}\n\n[${label}](${safeUrl})`;
}

export function privateInformationResponse(language = "en", prompt = "") {
  if (/\b(?:height|how tall|tallness|stature)\b|키|신장/i.test(prompt)) {
    return language === "ko"
      ? "이 포트폴리오에는 Daniel의 키에 관한 검증된 정보가 없습니다. 경력 연수와 신체 정보를 연결하거나 키를 추측하지 않습니다."
      : "The portfolio does not contain verified information about Daniel's height. I will not infer a physical measurement from his years of professional experience.";
  }
  if (/\b(?:how old|current age|exact age)\b|정확한 나이|현재 나이|몇 살/i.test(prompt)) {
    return language === "ko"
      ? "공개된 출생연도는 1997년이지만 정확한 생일은 공개되지 않았으므로 현재의 정확한 나이를 확정하지 않습니다."
      : "Daniel's verified public birth year is 1997, but his exact birthday is not published, so I cannot verify one exact current age.";
  }
  return language === "ko"
    ? "이 포트폴리오는 Daniel의 비공개 개인정보를 제공하거나 추측하지 않습니다. 공개된 경력, 연구, 프로젝트에 대해서는 답할 수 있습니다."
    : "This portfolio does not disclose or guess Daniel's private personal information. I can answer questions about his published work, research, and projects.";
}
