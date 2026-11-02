export class ConversationCollector {
  constructor({ endpoint = "", fetcher = (...args) => globalThis.fetch(...args), onStatus = () => {} } = {}) {
    this.endpoint = endpoint;
    this.fetcher = fetcher;
    this.onStatus = onStatus;
    this.reset();
  }

  reset() {
    this.controller?.abort();
    this.epoch = (this.epoch || 0) + 1;
    this.sessionId = crypto.randomUUID();
    this.turns = new Map();
    this.consent = false;
    this.running = null;
    this.onStatus("Not shared");
  }

  record({ question, answer, source = "", runtime = "", appVersion = "" }) {
    if (!question?.trim() || !answer?.trim()) return null;
    const id = crypto.randomUUID();
    this.turns.set(id, {
      payload: { schema_version: 1, consent: true, session_id: this.sessionId, turn_id: id,
        question, answer, source, runtime, app_version: appVersion, rating: null },
      revision: 1, savedRevision: 0,
    });
    this.sync();
    return id;
  }

  rate(id, rating) {
    const turn = this.turns.get(id);
    if (!turn || ![null, "helpful", "not_helpful"].includes(rating)) return;
    turn.payload.rating = rating;
    turn.revision += 1;
    this.sync();
  }

  setConsent(value) {
    this.consent = Boolean(value);
    if (!this.consent) {
      this.controller?.abort();
      this.epoch += 1;
      this.running = null;
      this.onStatus("Sharing off");
      return;
    }
    return this.sync();
  }

  sync() {
    if (!this.consent) return Promise.resolve();
    if (!this.endpoint) {
      this.onStatus("Sharing is unavailable on this site.");
      return Promise.resolve();
    }
    if (this.running) return this.running;
    const epoch = this.epoch;
    const controller = new AbortController();
    this.controller = controller;
    const signal = controller.signal;
    let completed = false;
    this.running = Promise.resolve().then(async () => {
      try {
        this.onStatus("Sharing...");
        for (const turn of this.turns.values()) {
          if (!this.consent || this.epoch !== epoch) return;
          if (turn.savedRevision === turn.revision) continue;
          // Capture a revision so a rating changed in flight is sent next.
          const revision = turn.revision;
          const timeout = setTimeout(() => controller.abort(), 10000);
          try {
            const response = await this.fetcher(this.endpoint, {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify(turn.payload), signal, credentials: "omit",
            });
            if (!response.ok) throw new Error("Save failed");
            const receipt = await response.json();
            if (receipt.saved !== true || receipt.turn_id !== turn.payload.turn_id) throw new Error("Invalid receipt");
          } finally { clearTimeout(timeout); }
          turn.savedRevision = revision;
        }
        completed = true;
        if (this.epoch === epoch) this.onStatus(this.turns.size ? "Conversation shared" : "Sharing on");
      } catch (error) {
        if (this.epoch === epoch) {
          console.warn("Conversation sharing failed:", error.message);
          this.onStatus("Could not save. Please retry.");
        }
      } finally {
        if (this.epoch === epoch) this.running = null;
      }
    });
    return this.running.then(() => {
      if (completed && this.consent && this.epoch === epoch && [...this.turns.values()].some(t => t.savedRevision < t.revision)) return this.sync();
    });
  }
}

export function followUpFor(question = "") {
  if (/sam\s*2/i.test(question)) return /what is|how.*work|explain/i.test(question)
    ? { label: "Explore visual grounding", prompt: "What is ZERO?", href: "/publications/", link: "View publications" }
    : { label: "What is SAM2?", prompt: "What is SAM2?", href: "/github/", link: "Explore open source" };
  if (/hugging|open.source|contribut/i.test(question)) return {
    label: "How did SAM2 work?", prompt: "What did Daniel contribute to SAM2?", href: "/github/", link: "Explore open source",
  };
  if (/document/i.test(question)) return {
    label: "Explore mobile authentication", prompt: "What did Daniel build for mobile ID verification?", href: "/cv/", link: "View experience",
  };
  if (/bank|toss|kyc/i.test(question)) return {
    label: "Explore document AI", prompt: "Explain Daniel's end-to-end document AI work at Toss Bank.", href: "/cv/", link: "View experience",
  };
  if (/research|publish|\bzero\b/i.test(question)) return {
    label: "From research to production", prompt: "What did Daniel build at SuperbAI?", href: "/cv/", link: "View experience",
  };
  return { label: "Explore his research", prompt: "What research has Daniel published?", href: "/publications/", link: "View publications" };
}
