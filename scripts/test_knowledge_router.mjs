import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  buildEntityAnswer,
  buildExternalEvidenceAnswer,
  buildProfileWorkAnswer,
  classifyKnowledgeIntent,
  externalSearchTerm,
  fetchWikipediaEvidence,
  findKnownEntity,
  privateInformationResponse,
  profileWorkClarificationResponse,
} from "../assets/js/knowledge-router.mjs";

const knowledge = JSON.parse(
  await readFile(new URL("../assets/data/daniel-entity-knowledge.json", import.meta.url), "utf8"),
);
const profile = JSON.parse(
  await readFile(new URL("../assets/data/daniel-profile.json", import.meta.url), "utf8"),
);

test("routes a model definition to external entity knowledge", () => {
  const route = classifyKnowledgeIntent("what is RT-DETR", knowledge);
  assert.equal(route.type, "entity_definition");
  assert.equal(route.entity.id, "rt-detr");
});

test("routes a Daniel contribution question to profile evidence", () => {
  const route = classifyKnowledgeIntent("What did Daniel contribute to RT-DETR?", knowledge);
  assert.equal(route.type, "profile_entity");
  assert.equal(route.entity.id, "rt-detr");
});

test("does not confuse a UIUC location question with Daniel's education", () => {
  const route = classifyKnowledgeIntent("where is UIUC", knowledge);
  assert.equal(route.type, "entity_definition");
  assert.match(buildEntityAnswer(route.entity), /Champaign-Urbana/);
});

test("routes Daniel's UIUC history to the profile relation", () => {
  const route = classifyKnowledgeIntent("When did Daniel study at UIUC?", knowledge);
  assert.equal(route.type, "profile_entity");
  assert.match(buildEntityAnswer(route.entity, "en", true), /August to December 2018/);
});

test("resolves portfolio entity follow-ups", () => {
  const route = classifyKnowledgeIntent("What did he do with it?", knowledge, {
    lastEntityId: "vitpose",
  });
  assert.equal(route.type, "profile_entity");
  assert.equal(route.entity.id, "vitpose");
});

test("routes private-person requests before retrieval", () => {
  const route = classifyKnowledgeIntent("What is his bank account number?", knowledge);
  assert.equal(route.type, "sensitive_personal");
});

test("treats height wording as private without confusing it with experience", () => {
  const prompt = "How tall is he?";
  const route = classifyKnowledgeIntent(prompt, knowledge);
  const answer = privateInformationResponse("en", prompt);
  assert.equal(route.type, "sensitive_personal");
  assert.match(answer, /verified information about Daniel's height/);
  assert.doesNotMatch(answer, /six|6\+|years old/i);
});

test("keeps exact-age questions separate from years of experience", () => {
  const prompt = "How old is he?";
  const route = classifyKnowledgeIntent(prompt, knowledge);
  const answer = privateInformationResponse("en", prompt);
  assert.equal(route.type, "sensitive_personal");
  assert.match(answer, /1997/);
  assert.match(answer, /exact birthday is not published/);
  assert.doesNotMatch(answer, /6\+|years of experience/i);
});

test("routes unseen definitions to external search", () => {
  const route = classifyKnowledgeIntent("What is a graph neural network?", knowledge);
  assert.equal(route.type, "external_knowledge");
  assert.equal(externalSearchTerm("What is a graph neural network?"), "graph neural network");
});

test("routes neutral factual questions that are not definitions to search", () => {
  const route = classifyKnowledgeIntent("Who wrote Pride and Prejudice?", knowledge);
  assert.equal(route.type, "external_knowledge");
  assert.equal(externalSearchTerm("Who wrote Pride and Prejudice?"), "Pride and Prejudice");
});

test("does not treat generic work verbs as Daniel profile references", () => {
  const route = classifyKnowledgeIntent("Who created Python?", knowledge);
  assert.equal(route.type, "external_knowledge");
  assert.equal(externalSearchTerm("Who created Python?"), "Python");
});

test("routes second-person portfolio questions to Daniel's profile", () => {
  const route = classifyKnowledgeIntent("What did you build at Toss Bank?", knowledge);
  assert.equal(route.type, "profile");
});

test("routes standalone leadership questions to Daniel's profile", () => {
  const english = classifyKnowledgeIntent("Tell me about the leadership experience", knowledge);
  const korean = classifyKnowledgeIntent("리더십과 팀 규모를 설명해 줘", knowledge);
  assert.equal(english.type, "profile");
  assert.equal(korean.type, "profile");
  assert.equal(profile.leadership.maximum_people_led_simultaneously, 8);
});

test("keeps visitor identity questions on the identity-safe model path", () => {
  const route = classifyKnowledgeIntent("Who am I?", knowledge);
  assert.equal(route.type, "profile");
});

test("routes how-does-it-work questions to public knowledge", () => {
  const route = classifyKnowledgeIntent("How does DHT work?", knowledge);
  assert.equal(route.type, "external_knowledge");
  assert.equal(externalSearchTerm("How does DHT work?"), "DHT");
});

test("routes polite explanation requests to public knowledge", () => {
  const route = classifyKnowledgeIntent("Could you explain federated learning?", knowledge);
  assert.equal(route.type, "external_knowledge");
  assert.equal(externalSearchTerm("Could you explain federated learning?"), "federated learning");
});

test("routes broad company-work questions to the complete work index", () => {
  const route = classifyKnowledgeIntent("Could you explain your work?", knowledge);
  const answer = buildProfileWorkAnswer(profile);
  assert.equal(route.type, "profile_work");
  assert.match(answer, /Toss Bank/);
  assert.match(answer, /SuperbAI/);
  assert.match(answer, /Team ISLAND/);
});

test("handles ungrammatical company-work questions as profile work", () => {
  const route = classifyKnowledgeIntent("what did he did in his company", knowledge);
  assert.equal(route.type, "profile_work");
});

test("asks for clarification instead of searching an ambiguous work follow-up", () => {
  const prompt = "how about in the work";
  const route = classifyKnowledgeIntent(prompt, knowledge);
  assert.equal(route.type, "profile_clarification");
  assert.equal(externalSearchTerm(prompt), "");
  assert.match(profileWorkClarificationResponse("en"), /Toss Bank/);
  assert.match(profileWorkClarificationResponse("en"), /SuperbAI/);
});

test("routes indirect primer requests to public knowledge", () => {
  const prompt = "Give me a primer on graph neural networks.";
  const route = classifyKnowledgeIntent(prompt, knowledge);
  assert.equal(route.type, "external_knowledge");
  assert.equal(externalSearchTerm(prompt), "graph neural networks");
});

test("known aliases are matched case-insensitively", () => {
  assert.equal(findKnownEntity("Explain VIT POSE", knowledge).id, "vitpose");
});

test("Wikipedia evidence is converted into a cited answer", async () => {
  const evidence = await fetchWikipediaEvidence("contrastive learning", "en", async () => ({
    ok: true,
    async json() {
      return {
        query: {
          pages: {
            1: {
              title: "Contrastive learning",
              extract: "Contrastive learning is a machine learning technique. It learns representations by comparing examples.",
              fullurl: "https://en.wikipedia.org/wiki/Contrastive_learning",
            },
          },
        },
      };
    },
  }));
  const answer = buildExternalEvidenceAnswer(evidence);
  assert.match(answer, /machine learning technique/);
  assert.match(answer, /Wikipedia source/);
});

test("Wikipedia retrieval falls back to ranked search when an exact title is absent", async () => {
  let calls = 0;
  const evidence = await fetchWikipediaEvidence("Pride and Prejudice author", "en", async () => {
    calls += 1;
    return {
      ok: true,
      async json() {
        return calls === 1
          ? { query: { pages: { "-1": { title: "Pride and Prejudice author", missing: true } } } }
          : {
            query: {
              pages: {
                2: {
                  title: "Pride and Prejudice",
                  extract: "Pride and Prejudice is a novel by Jane Austen.",
                  fullurl: "https://en.wikipedia.org/wiki/Pride_and_Prejudice",
                },
              },
            },
          };
      },
    };
  });
  assert.equal(calls, 2);
  assert.equal(evidence.title, "Pride and Prejudice");
});

test("Wikipedia citations encode parentheses for Markdown rendering", () => {
  const answer = buildExternalEvidenceAnswer({
    title: "Example",
    extract: "Example text.",
    url: "https://en.wikipedia.org/wiki/Example_(topic)",
  });
  assert.match(answer, /Example_%28topic%29/);
});
