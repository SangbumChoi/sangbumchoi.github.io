import { env, pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "SangbumChoi/sangbumchoi.github.io";
const MODEL_REVISION = "f1ad101660c858eb65357a6c0088a516c0b84f62";
const MODEL_PATH = "models/daniel-lfm2-350m-ONNX";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.remoteHost = "https://media.githubusercontent.com/media/";
env.remotePathTemplate = `{model}/{revision}/${MODEL_PATH}/`;

let generator = null;
let runtime = null;
let loadingPromise = null;

async function createGenerator(preferredDevice = "webgpu") {
  const options = {
    dtype: "q4",
    device: preferredDevice,
    revision: MODEL_REVISION,
    progress_callback: (payload) => self.postMessage({ type: "progress", payload }),
  };

  try {
    generator = await pipeline("text-generation", MODEL_ID, options);
    runtime = preferredDevice;
  } catch (error) {
    if (preferredDevice !== "webgpu") throw error;
    self.postMessage({ type: "fallback", message: "WebGPU initialization failed. Falling back to WASM." });
    generator = await pipeline("text-generation", MODEL_ID, {
      ...options,
      dtype: "q4",
      device: "wasm",
    });
    runtime = "wasm";
  }

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
        message: error?.message || String(error),
        stack: error?.stack || "",
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
      message: error?.message || String(error),
      stack: error?.stack || "",
    });
  }
});
