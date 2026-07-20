export const LLM_WEIGHT_BYTES = 289_140_736;

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function featureList(features) {
  if (!features) return [];
  try {
    return Array.from(features, String).sort();
  } catch (_) {
    return [];
  }
}

function readAdapterInfo(adapter) {
  const info = adapter?.info || {};
  return {
    architecture: info.architecture || "",
    description: info.description || "",
    device: info.device || "",
    vendor: info.vendor || "",
    isFallbackAdapter: Boolean(info.isFallbackAdapter || adapter?.isFallbackAdapter),
  };
}

export async function probeRuntimeCapabilities(navigatorObject = globalThis.navigator) {
  const hardwareConcurrency = finiteNumber(navigatorObject?.hardwareConcurrency);
  const deviceMemoryGB = finiteNumber(navigatorObject?.deviceMemory);
  const capabilities = {
    webgpu: Boolean(navigatorObject?.gpu),
    adapterAvailable: false,
    adapterInfo: readAdapterInfo(null),
    features: [],
    limits: {},
    hardwareConcurrency,
    deviceMemoryGB,
    crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
    probeError: "",
  };

  if (!capabilities.webgpu) return capabilities;

  try {
    const adapter = await navigatorObject.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return capabilities;
    capabilities.adapterAvailable = true;
    capabilities.adapterInfo = readAdapterInfo(adapter);
    capabilities.features = featureList(adapter.features);
    const limitNames = [
      "maxBufferSize",
      "maxStorageBufferBindingSize",
      "maxComputeWorkgroupStorageSize",
      "maxComputeInvocationsPerWorkgroup",
    ];
    capabilities.limits = Object.fromEntries(limitNames.map((name) => [name, finiteNumber(adapter.limits?.[name])]));
  } catch (error) {
    capabilities.probeError = error?.message || String(error);
  }

  return capabilities;
}

export function chooseRuntimePolicy(capabilities = {}) {
  const webgpuAvailable = Boolean(capabilities.webgpu && capabilities.adapterAvailable);
  const memory = finiteNumber(capabilities.deviceMemoryGB);
  const cores = finiteNumber(capabilities.hardwareConcurrency);
  const fallbackAdapter = Boolean(capabilities.adapterInfo?.isFallbackAdapter);
  const constrainedMemory = memory !== null && memory <= 4;
  const constrainedCpu = cores !== null && cores <= 4;
  const shaderF16 = (capabilities.features || []).includes("shader-f16");

  let tier = "balanced";
  if (!webgpuAvailable) tier = "compatibility";
  else if (fallbackAdapter || constrainedMemory || constrainedCpu) tier = "low";
  else if ((memory === null || memory >= 8) && (cores === null || cores >= 8) && shaderF16) tier = "high";

  const backend = webgpuAvailable ? "webgpu" : "wasm";
  const autoLoadModel = webgpuAvailable && !fallbackAdapter && !constrainedMemory && !constrainedCpu;
  const releaseAfterIdleMs = tier === "low" || tier === "compatibility" ? 90_000 : 0;

  return {
    tier,
    backend,
    autoLoadModel,
    releaseAfterIdleMs,
    maxResidentGpuModels: 1,
    executionOrder: ["stt", "llm", "tts"],
    llm: {
      dtype: "q4",
      weightBytes: LLM_WEIGHT_BYTES,
      load: autoLoadModel ? "eager" : "on-demand",
    },
    speech: {
      sttRuntime: "browser-speech-recognition",
      ttsRuntime: "browser-speech-synthesis",
      usesWebGPU: false,
      futureSttCandidates: ["q8", "encoder-q8-decoder-q4"],
      futureTtsCandidates: ["q8", "fp16"],
    },
    reasons: [
      !webgpuAvailable ? "no-webgpu-adapter" : "webgpu-adapter",
      fallbackAdapter ? "fallback-adapter" : null,
      constrainedMemory ? "device-memory-at-most-4gb" : null,
      constrainedCpu ? "at-most-4-logical-cores" : null,
    ].filter(Boolean),
  };
}

export function formatWeightSize(bytes = LLM_WEIGHT_BYTES) {
  return `${Math.round(bytes / 1_000_000)} MB`;
}

class ModelResidencyCoordinator {
  constructor() {
    this.active = null;
    this.operation = Promise.resolve();
  }

  acquire(name, release) {
    return this.enqueue(async () => {
      if (this.active?.name === name) return;
      if (this.active) await this.active.release();
      this.active = { name, release };
    });
  }

  release(name) {
    return this.enqueue(async () => {
      if (this.active?.name !== name) return;
      const active = this.active;
      this.active = null;
      await active.release();
    });
  }

  enqueue(operation) {
    this.operation = this.operation.then(operation, operation);
    return this.operation;
  }
}

// Every future WebGPU speech or language worker must acquire this coordinator.
export const modelResidencyCoordinator = new ModelResidencyCoordinator();
