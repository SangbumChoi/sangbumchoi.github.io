import { env, pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "danelcsb/daniel-lfm2-350m-ONNX";
const MODEL_REVISION = "1f7e797b18fbffe712ef829f9460b29e3591450b";

env.allowLocalModels = false;
env.allowRemoteModels = true;

let generator = null;
let runtime = null;
let runtimeDtype = null;
let loadingPromise = null;

function serializeError(error) {
  if (error && typeof error === "object") {
    return {
      name: error.name || "Error",
      message: error.message || String(error),
      stack: error.stack || "",
      cause: error.cause ? String(error.cause) : "",
    };
  }
  return {
    name: typeof error,
    message: typeof error === "number" ? `ONNX Runtime failed with code ${error}.` : String(error),
    stack: "",
    cause: "",
  };
}

async function disposeGenerator() {
  const activeGenerator = generator;
  generator = null;
  runtime = null;
  runtimeDtype = null;
  loadingPromise = null;
  if (activeGenerator?.dispose) await activeGenerator.dispose();
}

async function createGenerator(device = "webgpu", dtype = "q4") {
  const startedAt = performance.now();
  const options = {
    dtype,
    device,
    revision: MODEL_REVISION,
    progress_callback: (payload) => self.postMessage({ type: "progress", payload }),
  };

  self.postMessage({ type: "stage", stage: "initializing", device });
  generator = await pipeline("text-generation", MODEL_ID, options);
  runtime = device;
  runtimeDtype = dtype;

  self.postMessage({
    type: "ready",
    runtime,
    dtype: runtimeDtype,
    model: MODEL_ID,
    initElapsed: Math.round(performance.now() - startedAt),
  });
}

async function ensureGenerator(device, dtype) {
  if (generator && runtime === device && runtimeDtype === dtype) return Promise.resolve();
  if (generator) await disposeGenerator();
  if (!loadingPromise) {
    loadingPromise = createGenerator(device, dtype).finally(() => {
      loadingPromise = null;
    });
  }
  return loadingPromise;
}

self.addEventListener("message", async (event) => {
  const { type, messages, device, dtype } = event.data || {};

  if (type === "dispose") {
    try {
      await disposeGenerator();
      self.postMessage({ type: "disposed" });
    } catch (error) {
      self.postMessage({ type: "error", phase: "dispose", error: serializeError(error) });
    }
    return;
  }

  if (type === "load") {
    try {
      await ensureGenerator(device || "webgpu", dtype || "q4");
    } catch (error) {
      self.postMessage({
        type: "error",
        phase: "load",
        device: device || "webgpu",
        error: serializeError(error),
      });
    }
    return;
  }

  if (type !== "generate") return;

  try {
    await ensureGenerator(device || "webgpu", dtype || "q4");
    const startedAt = performance.now();
    let streamedText = "";
    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk) => {
        streamedText += chunk;
        self.postMessage({ type: "token", text: chunk });
      },
    });

    const output = await generator(messages, {
      max_new_tokens: 256,
      do_sample: false,
      repetition_penalty: 1.05,
      streamer,
    });

    const generated = output?.[0]?.generated_text;
    const finalText = Array.isArray(generated)
      ? generated.at(-1)?.content || streamedText
      : streamedText || String(generated || "");

    self.postMessage({
      type: "complete",
      text: finalText.trim(),
      runtime,
      elapsed: Math.round(performance.now() - startedAt),
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      phase: "generate",
      device: runtime || device || "webgpu",
      error: serializeError(error),
    });
  }
});
