import test from "node:test";
import assert from "node:assert/strict";
import { ConversationCollector, followUpFor } from "../assets/js/conversation-feedback.mjs";

function setup(fetcher) {
  const requests = [];
  const statuses = [];
  const collector = new ConversationCollector({ endpoint: "http://localhost/api/conversations", onStatus: s => statuses.push(s),
    fetcher: fetcher || (async (url, options) => {
      const payload = JSON.parse(options.body);
      requests.push(payload);
      return { ok: true, json: async () => ({ saved: true, turn_id: payload.turn_id }) };
    }) });
  return { collector, requests, statuses };
}
const turn = { question: "What is ZERO?", answer: "A visual-grounding model.", source: "entity-index" };

test("keeps all transcript data local until consent, then saves whole session", async () => {
  const { collector, requests } = setup();
  collector.record(turn);
  collector.record({ ...turn, question: "Who built it?" });
  await collector.sync();
  assert.equal(requests.length, 0);
  await collector.setConsent(true);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].session_id, requests[1].session_id);
  assert.equal(requests[0].consent, true);
  await collector.sync();
  assert.equal(requests.length, 2);
});

test("ratings reuse the turn ID; revoking consent stops new uploads", async () => {
  const { collector, requests } = setup();
  const id = collector.record(turn);
  await collector.setConsent(true);
  collector.rate(id, "not_helpful");
  await collector.sync();
  assert.equal(requests[1].turn_id, requests[0].turn_id);
  assert.equal(requests[1].rating, "not_helpful");
  collector.setConsent(false);
  collector.record(turn);
  await collector.sync();
  assert.equal(requests.length, 2);
  const session = collector.sessionId;
  collector.reset();
  assert.notEqual(collector.sessionId, session);
  assert.equal(collector.consent, false);
});

test("an empty opted-in session still uploads its first future answer", async () => {
  const { collector, requests } = setup();
  await collector.setConsent(true);
  collector.record(turn);
  await collector.sync();
  assert.equal(requests.length, 1);
});

test("failed sends stay retryable without a background retry loop", async () => {
  let attempts = 0;
  const { collector, statuses } = setup(async (url, options) => {
    attempts++;
    if (attempts < 2) throw new Error("offline");
    return { ok: true, json: async () => ({ saved: true, turn_id: JSON.parse(options.body).turn_id }) };
  });
  collector.record(turn);
  await collector.setConsent(true);
  assert.equal(attempts, 1);
  assert.match(statuses.at(-1), /Could not save/);
  await collector.sync();
  assert.equal(attempts, 2);
  assert.equal(statuses.at(-1), "Conversation shared");
});

test("a rating changed during upload is eventually saved", async () => {
  let resolveFirst;
  const seen = [];
  const { collector } = setup(async (url, options) => {
    const payload = JSON.parse(options.body);
    seen.push(payload);
    if (seen.length === 1) await new Promise(resolve => { resolveFirst = resolve; });
    return { ok: true, json: async () => ({ saved: true, turn_id: payload.turn_id }) };
  });
  const id = collector.record(turn);
  const saving = collector.setConsent(true);
  await Promise.resolve();
  collector.rate(id, "helpful");
  resolveFirst();
  await saving;
  assert.equal(seen.length, 2);
  assert.equal(seen[1].rating, "helpful");
});

test("revoking consent drops queued sends and keeps the status off", async () => {
  let finishFirst;
  let requests = 0;
  const { collector, statuses } = setup(async (url, options) => {
    requests++;
    await new Promise(resolve => { finishFirst = resolve; });
    return { ok: true, json: async () => ({ saved: true, turn_id: JSON.parse(options.body).turn_id }) };
  });
  collector.record(turn);
  collector.record({ ...turn, question: "Another question" });
  const saving = collector.setConsent(true);
  await Promise.resolve();
  collector.setConsent(false);
  finishFirst();
  await saving;
  assert.equal(requests, 1);
  assert.equal(statuses.at(-1), "Sharing off");
});

test("a failed rating update does not loop", async () => {
  let requests = 0;
  const { collector } = setup(async (url, options) => {
    requests++;
    if (requests > 1) throw new Error("offline");
    return { ok: true, json: async () => ({ saved: true, turn_id: JSON.parse(options.body).turn_id }) };
  });
  const id = collector.record(turn);
  await collector.setConsent(true);
  collector.rate(id, "helpful");
  await collector.sync();
  assert.equal(requests, 2);
});

test("default browser fetch keeps its global receiver", async () => {
  const original = globalThis.fetch;
  let called = false;
  globalThis.fetch = async function (url, options) {
    assert.equal(this, globalThis);
    called = true;
    return { ok: true, json: async () => ({ saved: true, turn_id: JSON.parse(options.body).turn_id }) };
  };
  try {
    const collector = new ConversationCollector({ endpoint: "http://localhost/api/conversations" });
    collector.record(turn);
    await collector.setConsent(true);
    assert.equal(called, true);
  } finally { globalThis.fetch = original; }
});

test("follow-ups progress instead of repeating the same question", () => {
  for (const question of ["What is SAM2?", "What did Daniel contribute to SAM2?", "What research has Daniel published?", "Explain Daniel's document AI experience."]) {
    const next = followUpFor(question);
    assert.notEqual(next.prompt.toLowerCase(), question.toLowerCase());
    assert.ok(next.href.startsWith("/"));
  }
});
