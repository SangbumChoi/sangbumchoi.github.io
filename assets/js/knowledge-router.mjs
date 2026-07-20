const PROFILE_NAME_PATTERN = /\b(?:daniel|sangbum|sangbum daniel choi|choi)\b|최상범|상범|다니엘/i;
const PROFILE_PRONOUN_PATTERN = /\b(?:he|him|his|you|your)\b|그가|그의|본인|당신|너의/i;
const PROFILE_SCOPE_PATTERN = /\b(?:portfolio|resume|cv|career|experience|publications?|education|startup history|current (?:role|employer)|(?:open.?source|hugging face) contributions?)\b|포트폴리오|이력서|경력|논문|학력|오픈.?소스 기여/i;
const DEFINITION_PATTERN = /^\s*(?:what|who|where)\s+(?:is|are|was|were)\b|^\s*(?:define|explain)\b|^\s*(?:what does|where does)\b|무엇|뭐야|뭔가요|어디|설명(?:해|해줘|해주세요)/i;
const PRIVATE_PATTERN = /\b(?:bank account|account number|height|weight|relationship status|girlfriend|boyfriend|spouse|wife|husband|family|home address|phone number|salary|exact birthday|social security|passport)\b|계좌|키|몸무게|연애|여자친구|남자친구|배우자|가족|집 주소|전화번호|연봉|생일|주민등록|여권/i;
const GENERAL_LOOKUP_PATTERN = /^\s*(?!(?:what|who)\s+should\b)(?:what|who|where|when)\b|^\s*(?:define|explain|tell me about|teach me about)\b|무엇|뭐야|뭔가요|어디|누가|언제|설명(?:해|해줘|해주세요)/i;

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
  return PROFILE_NAME_PATTERN.test(prompt)
    || PROFILE_PRONOUN_PATTERN.test(prompt)
    || PROFILE_SCOPE_PATTERN.test(prompt);
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

  if (PRIVATE_PATTERN.test(prompt) && (profileRelated || PROFILE_PRONOUN_PATTERN.test(prompt))) {
    return { type: "sensitive_personal", entity: null };
  }
  if (entity && profileRelated) return { type: "profile_entity", entity };
  if (entity && (DEFINITION_PATTERN.test(prompt) || !profileRelated)) {
    return { type: "entity_definition", entity };
  }
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
  return prompt
    .trim()
    .replace(/^\s*who\s+(?:wrote|created|invented|founded|painted|developed|discovered)\s+/i, "")
    .replace(/^\s*(?:what|who|where)\s+(?:is|are|was|were)\s+/i, "")
    .replace(/^\s*(?:define|explain|tell me about)\s+/i, "")
    .replace(/^\s*(?:무엇이|무엇은|뭐야|뭔가요|어디에|어디야|설명해줘|설명해주세요)\s*/i, "")
    .replace(/^\s*(?:a|an|the)\s+/i, "")
    .replace(/[?.!]+$/g, "")
    .trim()
    .slice(0, 160);
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

export function privateInformationResponse(language = "en") {
  return language === "ko"
    ? "이 포트폴리오는 Daniel의 비공개 개인정보를 제공하거나 추측하지 않습니다. 공개된 경력, 연구, 프로젝트에 대해서는 답할 수 있습니다."
    : "This portfolio does not disclose or guess Daniel's private personal information. I can answer questions about his published work, research, and projects.";
}
