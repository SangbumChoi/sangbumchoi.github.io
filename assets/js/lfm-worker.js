import { env, pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "SangbumChoi/sangbumchoi.github.io";
const MODEL_REVISION = "3c17b3ab590bc854df861310adc7a54d6ac96e4d";
const MODEL_PATH = "models/daniel-lfm2-350m-ONNX";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.remoteHost = "https://media.githubusercontent.com/media/";
env.remotePathTemplate = `{model}/${MODEL_REVISION}/${MODEL_PATH}/`;

let generator = null;
let runtime = null;
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

async function createGenerator(device = "webgpu") {
  const options = {
    dtype: "q4",
    device,
    revision: MODEL_REVISION,
    progress_callback: (payload) => self.postMessage({ type: "progress", payload }),
  };

  self.postMessage({ type: "stage", stage: "initializing", device });
  generator = await pipeline("text-generation", MODEL_ID, options);
  runtime = device;

  self.postMessage({ type: "ready", runtime, model: MODEL_ID });
}

function ensureGenerator(device) {
  if (generator) return Promise.resolve();
  if (!loadingPromise) loadingPromise = createGenerator(device);
  return loadingPromise;
}

self.addEventListener("message", async (event) => {
  const { type, messages, device } = event.data || {};

  if (type === "load") {
    try {
      await ensureGenerator(device || "webgpu");
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
    await ensureGenerator(device || "webgpu");
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
      max_new_tokens: 150,
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
