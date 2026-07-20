import assert from "node:assert/strict";
import test from "node:test";

import {
  LLM_WEIGHT_BYTES,
  chooseRuntimePolicy,
  formatWeightSize,
  probeRuntimeCapabilities,
} from "../assets/js/runtime-policy.mjs";

const capableAdapter = {
  webgpu: true,
  adapterAvailable: true,
  adapterInfo: { isFallbackAdapter: false },
  features: [],
  limits: {},
};

test("uses on-demand WASM when WebGPU is unavailable", () => {
  const policy = chooseRuntimePolicy({
    webgpu: false,
    adapterAvailable: false,
    hardwareConcurrency: 8,
    deviceMemoryGB: 8,
  });
  assert.equal(policy.tier, "compatibility");
  assert.equal(policy.backend, "wasm");
  assert.equal(policy.autoLoadModel, false);
  assert.equal(policy.maxResidentGpuModels, 1);
});

test("keeps a real WebGPU adapter but avoids eager loading on a low-spec device", () => {
  const policy = chooseRuntimePolicy({
    ...capableAdapter,
    hardwareConcurrency: 4,
    deviceMemoryGB: 4,
  });
  assert.equal(policy.tier, "low");
  assert.equal(policy.backend, "webgpu");
  assert.equal(policy.autoLoadModel, false);
  assert.equal(policy.llm.load, "on-demand");
  assert.equal(policy.releaseAfterIdleMs, 90_000);
});

test("treats a software fallback adapter as low tier", () => {
  const policy = chooseRuntimePolicy({
    ...capableAdapter,
    adapterInfo: { isFallbackAdapter: true },
    hardwareConcurrency: 12,
    deviceMemoryGB: 8,
  });
  assert.equal(policy.tier, "low");
  assert.equal(policy.autoLoadModel, false);
  assert.ok(policy.reasons.includes("fallback-adapter"));
});

test("eager-loads Q4 on a balanced hardware adapter", () => {
  const policy = chooseRuntimePolicy({
    ...capableAdapter,
    hardwareConcurrency: 8,
    deviceMemoryGB: 8,
  });
  assert.equal(policy.tier, "balanced");
  assert.equal(policy.backend, "webgpu");
  assert.equal(policy.autoLoadModel, true);
  assert.equal(policy.llm.dtype, "q4");
  assert.deepEqual(policy.executionOrder, ["stt", "llm", "tts"]);
});

test("uses shader-f16 only as a high-tier capability signal", () => {
  const policy = chooseRuntimePolicy({
    ...capableAdapter,
    features: ["shader-f16"],
    hardwareConcurrency: 10,
    deviceMemoryGB: 8,
  });
  assert.equal(policy.tier, "high");
  assert.equal(policy.llm.dtype, "q4");
});

test("does not mistake an unavailable memory hint for zero memory", () => {
  const policy = chooseRuntimePolicy({
    ...capableAdapter,
    features: ["shader-f16"],
    hardwareConcurrency: 8,
    deviceMemoryGB: null,
  });
  assert.equal(policy.tier, "high");
  assert.equal(policy.autoLoadModel, true);
});

test("reports the pinned browser weight size", () => {
  assert.equal(LLM_WEIGHT_BYTES, 289_140_736);
  assert.equal(formatWeightSize(), "289 MB");
});

test("probes adapter capabilities without creating a GPUDevice", async () => {
  let deviceRequested = false;
  const navigatorMock = {
    hardwareConcurrency: 8,
    deviceMemory: 8,
    gpu: {
      async requestAdapter() {
        return {
          info: { vendor: "test-vendor" },
          features: new Set(["shader-f16"]),
          limits: { maxBufferSize: 268_435_456 },
          requestDevice() {
            deviceRequested = true;
          },
        };
      },
    },
  };
  const capabilities = await probeRuntimeCapabilities(navigatorMock);
  assert.equal(capabilities.adapterAvailable, true);
  assert.equal(capabilities.adapterInfo.vendor, "test-vendor");
  assert.deepEqual(capabilities.features, ["shader-f16"]);
  assert.equal(deviceRequested, false);
});
